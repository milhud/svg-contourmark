"""Regression tests for the counterexamples in docs/review-2026-09-30."""

import hashlib
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from contourmark import geosample, spectral
from contourmark.geometry import load_document
from contourmark.geosample import GeoParameters, GeoWatermark, document_descriptors

NS = "{http://www.w3.org/2000/svg}"
KEY = bytes(range(32))
ICON = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2">
<path d="M2 9.5a5.5 5.5 0 0 1 9.591-3.676.56.56 0 0 0 .818 0A5.49 5.49 0 0 1 22 9.5c0 2.29-1.5 4-3 5.5l-5.492 5.313a2 2 0 0 1-3 .019L5 15c-1.5-1.5-3-3.2-3-5.5"/>
<circle cx="12" cy="12" r="3"/>
<path d="M4 20c3-2 5-2 8 0s5 2 8 0"/>
</svg>"""
LABELS = [("v", 1, 1, 1), ("v", 2, 2, 2)]


def wrap_in_clip(source: bytes, clip_d: str | None) -> bytes:
    root = ET.fromstring(source)
    children = list(root)
    for child in children:
        root.remove(child)
    clip = ET.SubElement(ET.SubElement(root, NS + "defs"), NS + "clipPath", {"id": "c"})
    if clip_d:
        ET.SubElement(clip, NS + "path", {"d": clip_d})
    group = ET.SubElement(root, NS + "g", {"clip-path": "url(#c)", "fill": "none", "stroke": "black", "stroke-width": "2"})
    for child in children:
        group.append(child)
    return ET.tostring(root)


# --- P0: sequence-level distribution preservation -------------------------


def _two_step(reuse: str, trials: int = 6000) -> tuple[float, float]:
    repeated = adaptive = 0
    for i in range(trials):
        key = hashlib.sha256(str(i).encode()).digest()
        watermark = GeoWatermark(key, seed=i, reuse=reuse)
        first = watermark._gumbel(LABELS, [0.5, 0.5]).index
        repeated += watermark._gumbel(LABELS, [0.5, 0.5]).index == first
        watermark = GeoWatermark(key, seed=i, reuse=reuse)
        first = watermark._gumbel(LABELS, [0.5, 0.5]).index
        adaptive += watermark._gumbel(LABELS, [0.9, 0.1] if first == 0 else [0.5, 0.5]).index == 0
    return repeated / trials, adaptive / trials


def test_masked_reuse_preserves_the_two_step_law():
    # Ordinary sampling: identical contests repeat 50% of the time; the
    # adaptive second step picks A with 0.5*0.9 + 0.5*0.5 = 0.7.
    repeated, adaptive = _two_step("mask")
    assert abs(repeated - 0.5) < 0.03
    assert abs(adaptive - 0.7) < 0.03


def test_naive_reuse_is_biased_and_kept_only_as_an_ablation():
    repeated, adaptive = _two_step("allow")
    assert repeated > 0.95  # the same keyed scores always repeat the choice
    assert abs(adaptive - 0.5) < 0.03  # versus 0.7 for the model


def test_reset_starts_a_new_drawing():
    watermark = GeoWatermark(KEY, seed=0)
    first = watermark._gumbel(LABELS, [0.5, 0.5]).index
    watermark.reset()
    assert watermark._gumbel(LABELS, [0.5, 0.5]).index == first  # keyed again


# --- P0: visible-content binding ------------------------------------------


@pytest.fixture(scope="module")
def marked_icon():
    return spectral.embed(ICON, KEY)[0]


def test_empty_clip_is_not_detected_by_either_detector(marked_icon):
    assert spectral.detect(marked_icon, KEY)["status"] == "detected"
    hidden = wrap_in_clip(marked_icon, None)
    result = spectral.detect(hidden, KEY)
    assert not result["detected"] and result["status"] == "indeterminate"
    assert result["excluded_contours"] >= 2 and "clip-path" in result["excluded_reasons"]
    geo = geosample.detect(hidden, KEY)
    assert not geo["detected"] and geo["status"] == "indeterminate"


@pytest.mark.skipif(shutil.which("rsvg-convert") is None, reason="needs librsvg")
def test_render_assisted_visibility_resolves_clips(marked_icon):
    empty = spectral.detect(wrap_in_clip(marked_icon, None), KEY, visibility="render")
    assert empty["status"] == "not_detected"  # resolved: nothing is drawn
    benign = spectral.detect(wrap_in_clip(marked_icon, "M0 0H24V24H0Z"), KEY, visibility="render")
    assert benign["status"] == "detected"  # a full-canvas clip hides nothing


def test_css_rules_make_the_decision_indeterminate(marked_icon):
    root = ET.fromstring(marked_icon)
    style = ET.Element(NS + "style")
    style.text = "path{display:none}"
    root.insert(0, style)
    result = spectral.detect(ET.tostring(root), KEY)
    assert not result["detected"] and result["status"] == "indeterminate"
    assert "css" in result["unresolved_document_features"]


def test_off_canvas_geometry_is_not_scored():
    document = load_document(b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><path d="M1 1L5 5L1 5Z"/><path d="M20 20L30 30L20 30Z"/></svg>')
    assert [c.scorable for c in document.contours] == [True, False]


# --- P1: representation-level descriptor equality -------------------------


def _descriptors(d: str):
    return document_descriptors(f'<svg xmlns="http://www.w3.org/2000/svg"><path d="{d}"/></svg>'.encode(), GeoParameters())


def test_degree_elevation_does_not_change_descriptors():
    assert _descriptors("M0 0 Q6 12 12 0 C15 3 24 -6 30 0") == _descriptors("M0 0 C4 8 8 8 12 0 C15 3 24 -6 30 0")


def test_straight_cubic_matches_line_vertex_descriptor():
    line = [d for d in _descriptors("M0 0 L10 0 L10 8 L3 12")[0] if d[0] == "v"]
    cubic = [d for d in _descriptors("M0 0 L10 0 C10 2.6667 10 5.3333 10 8 L3 12")[0] if d[0] == "v"]
    assert line == cubic


# --- spectral embedder: seed stability is enforced -------------------------


def test_embed_reports_status_and_never_ships_an_unstable_seed():
    _, report = spectral.embed(ICON, KEY)
    assert all(c["status"] in {"marked", "skipped_unstable_seed", "skipped_distortion"} for c in report["contours"])
    assert all(c["seed_stable"] for c in report["contours"] if c["status"] == "marked")
    assert report["marked_contours"] + report["skipped_contours"] == len(report["contours"])
