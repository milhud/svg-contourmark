import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from contourmark import attacks
from contourmark.geometry import GeometryError, dense_points, load_document, parse_path_data, resample, serialize_compact, split_segment
from contourmark.spectral import SpectralParameters, chernoff_log_p, detect, embed

KEY = bytes(range(32))
WRONG = bytes(range(1, 33))
ICON = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2">
<!-- license comment kept -->
<path d="M2 9.5a5.5 5.5 0 0 1 9.591-3.676.56.56 0 0 0 .818 0A5.49 5.49 0 0 1 22 9.5c0 2.29-1.5 4-3 5.5l-5.492 5.313a2 2 0 0 1-3 .019L5 15c-1.5-1.5-3-3.2-3-5.5"/>
<circle cx="12" cy="12" r="3"/>
<path d="M4 20c3-2 5-2 8 0s5 2 8 0"/>
</svg>"""


@pytest.fixture(scope="module")
def marked():
    return embed(ICON, KEY)


def test_parser_handles_compact_arcs_and_shorthand():
    subpaths = parse_path_data("M2 9.5a5.5 5.5 0 0 1 9.591-3.676.56.56 0 0 0 .818 0s1 1 2 0t3 0q1 1 2 0z")
    kinds = [segment.kind for segment in subpaths[0].segments]
    assert kinds[:2] == ["A", "A"] and "C" in kinds and "Q" in kinds and subpaths[0].closed


def test_compact_serialization_round_trips():
    data = "M8.5 5.5a.5.5 0 0 0-1 0v3.362l-1.429 2.38a.5.5 0 1 0 .858.515l1.5-2.5A.5.5 0 0 0 8.5 9z"
    before = parse_path_data(data)
    after = parse_path_data(serialize_compact(before, 3))
    for a, b in zip(before, after):
        assert np.abs(dense_points(a) - dense_points(b)).max() < 1e-9


def test_exact_split_preserves_geometry():
    for data in ("M0 0C20 -15 80 15 100 0", "M0 0A10 5 30 1 0 10 10", "M0 0Q5 9 10 0"):
        subpath = parse_path_data(data)[0]
        halves = type(subpath)(split_segment(subpath.segments[0]), False)
        a, _ = resample(dense_points(subpath), 300, False)
        b, _ = resample(dense_points(halves), 300, False)
        assert np.abs(a - b).max() < 2e-3


def test_marked_detected_and_unmarked_or_wrong_key_not(marked):
    svg, report = marked
    assert report["marked_contours"] >= 2
    assert detect(svg, KEY)["detected"]
    assert not detect(svg, WRONG)["detected"]
    assert not detect(ICON, KEY)["detected"]
    assert b"license comment kept" in svg


def test_detection_needs_no_side_information(marked):
    svg, report = marked
    # The report is informational: detection uses only the SVG and the key.
    assert "key" not in str(report).lower() or "mac" not in report
    assert detect(svg, KEY)["log10_p_value"] < -6


@pytest.mark.parametrize("name", ["reorder", "reverse", "restart", "subdivide", "merge_paths", "split_subpaths", "rotate_30", "scale_0.37", "mirror", "group_transform", "translate_5pct", "round_2dp"])
def test_invariances(marked, name):
    svg, _ = marked
    attacked = attacks.standard_suite()[name](svg)
    assert detect(attacked, KEY)["log10_p_value"] < -6, name


def test_distortion_is_small(marked):
    _, report = marked
    for contour in report["contours"]:
        assert contour["curve_rms_over_radius"] < 0.02


def test_tail_bound_is_valid_against_monte_carlo():
    rng = np.random.default_rng(0)
    weights = np.array([1.0, 1.0, 2.0, 0.5, 1.5, 1.0])
    draws = (weights * np.cos(2 * np.pi * rng.random((400_000, len(weights))))).sum(axis=1)
    for threshold in (1.0, 3.0, 5.0):
        empirical = (draws >= threshold).mean()
        bound = math.exp(chernoff_log_p(threshold, weights))
        assert bound >= empirical * 0.97
        assert bound <= max(4 * empirical, 1e-4)


def test_null_p_values_are_super_uniform_over_keys():
    # For a fixed unmarked document, p-values over random keys must satisfy
    # P[p <= a] <= a (checked loosely for a small sample).
    values = [detect(ICON, os.urandom(32))["p_value"] for _ in range(60)]
    assert np.mean(np.array(values) <= 0.1) <= 0.25


def test_rejects_dtd():
    with pytest.raises(GeometryError):
        embed(b'<!DOCTYPE svg [<!ENTITY x "y">]><svg xmlns="http://www.w3.org/2000/svg"/>', KEY)


def test_invisible_contours_are_ignored(marked):
    svg, _ = marked
    root = ET.fromstring(svg)
    for element in root.iter():
        if element.tag.endswith("path"):
            element.set("display", "none")
    assert detect(ET.tostring(root), KEY)["log10_p_value"] > -1
