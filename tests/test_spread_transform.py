"""Spread-transform dither modulation (``SpectralParameters.spread > 0``)."""

import os
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from contourmark import attacks
from contourmark.geometry import GeometryError, load_document
from contourmark.spectral import (
    SpectralParameters,
    capacity,
    detect,
    embed,
    observe,
    spread_carriers,
    spread_dithers,
    spread_matrix,
)

KEY = bytes(range(32))
WRONG = bytes(range(1, 33))
SMALL = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2">
<path d="M2 9.5a5.5 5.5 0 0 1 9.591-3.676.56.56 0 0 0 .818 0A5.49 5.49 0 0 1 22 9.5c0 2.29-1.5 4-3 5.5l-5.492 5.313a2 2 0 0 1-3 .019L5 15c-1.5-1.5-3-3.2-3-5.5"/>
<circle cx="12" cy="12" r="3"/>
<path d="M4 20c3-2 5-2 8 0s5 2 8 0"/>
</svg>"""
# STDM carries only ``spread`` terms per contour, so it needs more contours
# than plain QIM to reach 1e-6: eight distinct shapes here.
ICON = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" fill="none" stroke="black" stroke-width="1">
<path d="M8 22a14 14 0 0 1 24-9a1.5 1.5 0 0 0 2 0a14 14 0 0 1 24 9c0 6-4 10-8 14l-14 13a5 5 0 0 1-7 0l-13-13c-4-4-8-8-8-14"/>
<circle cx="78" cy="20" r="12"/>
<path d="M5 62c8-5 13-5 20 0s13 5 20 0"/>
<ellipse cx="75" cy="52" rx="18" ry="8"/>
<path d="M10 75c2-8 12-10 18-4s14 2 16 8s-6 14-16 12s-20-8-18-16z"/>
<path d="M55 70q10-12 20 0t20 0"/>
<path d="M58 88c4-9 10-9 14 0s10 9 14 0s6-6 9-2"/>
<path d="M50 5c6 2 9 8 6 14s-10 8-14 3s-2-13 8-17z"/>
<rect x="4" y="84" width="40" height="12" rx="5"/>
</svg>"""
PARAMS = SpectralParameters(spread=2)


@pytest.fixture(scope="module")
def marked():
    return embed(ICON, KEY, PARAMS)


def test_spread_zero_is_the_unchanged_default():
    default_svg, default_report = embed(SMALL, KEY)
    explicit_svg, explicit_report = embed(SMALL, KEY, SpectralParameters(spread=0))
    assert default_svg == explicit_svg
    assert default_report == explicit_report
    assert asdict(SpectralParameters())["spread"] == 0
    # No STDM-only fields leak into the plain QIM report.
    assert all("mask_stable" not in contour for contour in default_report["contours"])
    assert detect(default_svg, KEY) == detect(default_svg, KEY, SpectralParameters(spread=0))
    # Value recorded from the code before STDM was added.
    assert detect(default_svg, KEY)["log10_p_value"] == pytest.approx(-19.443168421905778, abs=0.05)
    # The spread-spectrum ablation ignores ``spread``.
    ss = SpectralParameters(scheme="ss")
    assert embed(SMALL, KEY, ss)[0] == embed(SMALL, KEY, SpectralParameters(scheme="ss", spread=2))[0]


def test_parameter_validation():
    with pytest.raises(GeometryError):
        SpectralParameters(spread=9).validate()
    with pytest.raises(GeometryError):
        SpectralParameters(spread=-1).validate()


def test_projection_rows_are_orthonormal_and_indexed_by_coefficient():
    full = np.ones(8, dtype=bool)
    matrix = spread_matrix(KEY, "c:0:0", full, 3)
    assert matrix.shape == (3, 8)
    assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
    mask = np.array([True, False, True, True, False, True, False, False])
    restricted = spread_matrix(KEY, "c:0:0", mask, 3)
    assert restricted.shape == (3, 4)
    assert np.allclose(restricted @ restricted.T, np.eye(3), atol=1e-12)
    # The first row is the same per-index Gaussian row, restricted and rescaled.
    assert np.allclose(restricted[0], matrix[0][mask] / np.linalg.norm(matrix[0][mask]))
    # Fewer selected coefficients than rows: use as many rows as coefficients.
    assert spread_matrix(KEY, "c:0:0", np.arange(8) < 2, 3).shape == (2, 2)
    # Different keys and seeds give different rows; dithers lie in (0, 1).
    assert not np.allclose(matrix, spread_matrix(WRONG, "c:0:0", full, 3))
    assert not np.allclose(matrix, spread_matrix(KEY, "c:0:1", full, 3))
    dither = spread_dithers(KEY, "c:0:0", 3)
    assert ((dither > 0) & (dither < 1)).all()


def test_marked_detected_and_unmarked_or_wrong_key_not(marked):
    svg, report = marked
    assert report["marked_contours"] >= 6
    assert report["parameters"]["spread"] == 2
    result = detect(svg, KEY, PARAMS)
    assert result["detected"] and result["log10_p_value"] < -6
    assert not detect(svg, WRONG, PARAMS)["detected"]
    assert not detect(ICON, KEY, PARAMS)["detected"]
    # A plain-QIM verifier does not read an STDM mark.
    assert not detect(svg, KEY)["detected"]


def test_marked_projections_sit_on_the_keyed_lattice(marked):
    svg, _ = marked
    observations = observe(load_document(svg), PARAMS)
    alignments = []
    for observation in observations:
        values, step = spread_carriers(observation, KEY, PARAMS)
        assert len(values) == min(PARAMS.spread, int(observation.selected.sum()))
        assert step == pytest.approx(PARAMS.delta * np.sqrt(observation.selected.sum() / len(values)))
        offsets = spread_dithers(KEY, observation.seed, PARAMS.spread)[: len(values)]
        alignments.extend(np.cos(2 * np.pi * (values / step - offsets)))
    assert np.mean(alignments) > 0.9


def test_capacity_counts_projections_not_coefficients():
    plain = capacity(ICON)
    spread = capacity(ICON, PARAMS)
    assert spread["usable_contours"] == plain["usable_contours"]
    assert 0 < spread["coefficients"] <= PARAMS.spread * spread["usable_contours"]
    assert spread["coefficients"] < plain["coefficients"]


def test_distortion_is_small(marked):
    _, report = marked
    for contour in report["contours"]:
        assert contour["curve_rms_over_radius"] < 0.02


def test_null_p_values_are_super_uniform_over_keys():
    # For a fixed unmarked document, p-values over random keys must satisfy
    # P[p <= a] <= a (checked loosely for a small sample).  Also for a
    # document marked under a different key.
    values = np.array([detect(ICON, os.urandom(32), PARAMS)["p_value"] for _ in range(60)])
    assert np.mean(values <= 0.1) <= 0.25
    assert values.min() > 1e-5
    foreign = embed(ICON, WRONG, PARAMS)[0]
    values = np.array([detect(foreign, os.urandom(32), PARAMS)["p_value"] for _ in range(60)])
    assert np.mean(values <= 0.1) <= 0.25
    assert values.min() > 1e-5


@pytest.mark.parametrize("name", ["reorder", "rotate_30", "subdivide", "svgo_default"])
def test_invariances(marked, name):
    svg, _ = marked
    if name == "svgo_default" and (shutil.which("node") is None or not (Path(__file__).parents[1] / "node_modules" / "svgo").exists()):
        pytest.skip("node or the SVGO development dependency is not installed")
    attacked = attacks.standard_suite()[name](svg)
    assert detect(attacked, KEY, PARAMS)["log10_p_value"] < -6, name
