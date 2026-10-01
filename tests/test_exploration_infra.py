"""Tests for the exploration infrastructure: metrics, capacity, sampler variants, extended attacks."""

import hashlib
import shutil

import numpy as np
import pytest

from contourmark import attacks, metrics, spectral
from contourmark.geosample import GeoWatermark, detect
from contourmark.spectral import SpectralParameters
from contourmark.toygen import toy_drawing

KEY = bytes(range(32))
ICON = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2">
<path d="M3 12C3 6 8 3 12 3s9 3 9 9-5 9-9 9-9-3-9-9z"/>
<path d="M7 12c1.5-3 3.5-3 5 0s3.5 3 5 0"/>
<circle cx="12" cy="17" r="2.5"/>
</svg>"""
LINES = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path stroke="black" d="M5 12h14"/><text x="2" y="20">hi</text></svg>'


@pytest.fixture(scope="module")
def marked():
    return spectral.embed(ICON, KEY)[0]


# --- metrics ---------------------------------------------------------------


def test_boundary_distance_is_zero_for_identity_and_small_for_marked(marked):
    # Identity measures the sampling floor (about 0.01% of the diagonal), not zero.
    assert metrics.boundary_distance(ICON, ICON)["fraction_of_diagonal"]["hausdorff"] < 2e-4
    distance = metrics.boundary_distance(ICON, marked)
    assert 5e-4 < distance["fraction_of_diagonal"]["hausdorff"] < 0.02
    assert distance["pixels"]["24"]["hausdorff"] < 0.5  # sub-pixel at native icon size


def test_boundary_distance_is_symmetric_about_missing_geometry():
    fewer = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M3 12C3 6 8 3 12 3s9 3 9 9-5 9-9 9-9-3-9-9z"/></svg>'
    forward = metrics.boundary_distance(ICON, fewer)["units"]["hausdorff"]
    backward = metrics.boundary_distance(fewer, ICON)["units"]["hausdorff"]
    assert forward == pytest.approx(backward) and forward > 1  # the removed shapes are noticed either way


def test_embedding_keeps_smooth_joins_smooth(marked):
    report = metrics.smoothness(ICON, marked)
    assert report["marked_soft_kinks"] <= report["original_soft_kinks"] + 1


def test_editability_reports_structure_changes(marked):
    report = metrics.editability(ICON, marked)
    assert report["contours_before"] == report["contours_after"] == 3
    assert report["primitives_lost"] == {"circle": 1}  # the circle became a path


@pytest.mark.skipif(shutil.which("rsvg-convert") is None, reason="needs librsvg")
def test_render_fidelity_uses_two_backgrounds(marked):
    report = metrics.render_fidelity(ICON, marked, sizes=(64,))["64"]
    assert report["silhouette_iou"] > 0.9
    assert {"white", "dark"} <= set(report)


# --- capacity --------------------------------------------------------------


def test_capacity_outcomes():
    assert spectral.capacity(ICON)["outcome"] == "supported"
    report = spectral.capacity(LINES)
    assert report["outcome"] == "unsupported" and report["unscored_content"] == {"text": 1}
    single = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M4 4L20 4L12 18Z"/></svg>'
    assert spectral.capacity(single)["outcome"] in {"insufficient_capacity", "unsupported"}


def test_capacity_bound_is_reached_by_no_real_mark(marked):
    attainable = spectral.capacity(marked)["attainable_log10_p"]
    assert attainable <= spectral.detect(marked, KEY)["log10_p_value"] + 1e-6


# --- spectral ablation -----------------------------------------------------


def test_spread_spectrum_ablation_has_a_valid_null():
    params = SpectralParameters(scheme="ss")
    values = [spectral.detect(ICON, hashlib.sha256(bytes([i])).digest(), params)["p_value"] for i in range(40)]
    assert np.mean(np.array(values) <= 0.1) <= 0.25


# --- sampler variants ------------------------------------------------------


def test_bias_mode_is_detected_by_the_green_statistic_and_reports_its_cost():
    watermark = GeoWatermark(KEY, seed=3, mode="bias", delta=4.0)
    state, stats = toy_drawing(watermark, seed=3, shapes=5, vertices=14, sigma=1.5)
    assert detect(state.svg(), KEY, statistic="green")["log10_p_value"] < -6
    assert np.mean(stats["kl"]) > 0.1  # the bias sampler is not distribution-preserving
    plain, _ = toy_drawing(None, seed=3, shapes=5, vertices=14, sigma=1.5)
    assert detect(plain.svg(), KEY, statistic="green")["log10_p_value"] > -3


def test_toy_generator_entropy_follows_sigma():
    low = toy_drawing(GeoWatermark(KEY, seed=1), seed=1, shapes=2, vertices=8, sigma=0.45)[1]
    high = toy_drawing(GeoWatermark(KEY, seed=1), seed=1, shapes=2, vertices=8, sigma=1.5)[1]
    assert np.mean(low["entropy"]) < np.mean(high["entropy"])


# --- extended attacks ------------------------------------------------------


@pytest.mark.parametrize("name", ["to_cubics", "subdivide_uneven"])
def test_exact_rewrites_do_not_change_spectral_detection(marked, name):
    clean = spectral.detect(marked, KEY)["log10_p_value"]
    attacked = spectral.detect(attacks.extended_suite()[name](marked), KEY)["log10_p_value"]
    assert attacked == pytest.approx(clean, abs=1.0)


def test_clip_half_cuts_through_contours():
    clipped = attacks.clip_half(ICON)
    from contourmark.geometry import load_document
    xs = np.concatenate([attacks.dense_points(c.subpath).real for c in load_document(clipped).contours])
    assert xs.max() <= 12.0 + 1e-6 and len(load_document(clipped).contours) >= 2
