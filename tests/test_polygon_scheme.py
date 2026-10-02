"""The control-polygon ("polygon") scheme of the inference-time watermark."""

import hashlib

import numpy as np
import pytest

from contourmark import attacks
from contourmark.geosample import GeoParameters, GeoWatermark, PathTracker, detect, knot_descriptor, pair_key, polygon_keys, Segment2
from contourmark.toygen import toy_drawing

KEY = bytes(range(32))
POLYGON = GeoParameters(scheme="polygon")


@pytest.fixture(scope="module")
def marked():
    return toy_drawing(GeoWatermark(KEY, POLYGON, seed=2), seed=2, shapes=5, vertices=14, sigma=0.7)[0].svg()


def test_knot_descriptor_is_symmetric_under_reversal_mirror_and_similarity():
    a, b, c = 0j, 10 + 2j, 14 + 9j
    base = knot_descriptor(a, b, c, True, POLYGON)
    assert base == knot_descriptor(c, b, a, True, POLYGON)  # reversal swaps the neighbours
    assert base == knot_descriptor(a.conjugate(), b.conjugate(), c.conjugate(), True, POLYGON)  # mirror
    turn = np.exp(0.7j) * 3.1
    assert base == knot_descriptor(a * turn + 5, b * turn + 5, c * turn + 5, True, POLYGON)
    assert base != knot_descriptor(a, b, c, False, POLYGON)  # on-curve and control points are separate domains


def test_pair_key_is_unordered():
    x, y = ("k", 1, 3, 4), ("k", 0, 7, 2)
    assert pair_key(x, y) == pair_key(y, x)
    assert pair_key(None, x) == ("s", x)


def test_sampler_and_detector_compute_the_same_keys():
    # Every keyed choice the sampler makes must be among the keys the detector reads.
    watermark = GeoWatermark(KEY, POLYGON, seed=5)
    chosen: list = []
    original = watermark._gumbel

    def spy(descriptors, probabilities):
        choice = original(descriptors, probabilities)
        if choice.descriptor is not None:
            chosen.append(choice.descriptor)
        return choice

    watermark._gumbel = spy
    state, _ = toy_drawing(watermark, seed=5, shapes=3, vertices=10, sigma=0.7)
    from contourmark.geosample import document_descriptors

    detected = {d for contour in document_descriptors(state.svg(), POLYGON) for d in contour}
    assert len(chosen) > 30
    recovered = sum(d in detected for d in chosen) / len(chosen)
    assert recovered > 0.9, recovered


def test_polygon_scheme_detects_at_low_entropy_where_vertex_scheme_does_not():
    polygon, vertex = [], []
    for seed in range(6):
        svg = toy_drawing(GeoWatermark(KEY, POLYGON, seed=seed), seed, shapes=5, vertices=14, sigma=0.45)[0].svg()
        polygon.append(detect(svg, KEY, POLYGON)["log10_p_value"])
        svg = toy_drawing(GeoWatermark(KEY, seed=seed), seed, shapes=5, vertices=14, sigma=0.45)[0].svg()
        vertex.append(detect(svg, KEY)["log10_p_value"])
    assert np.median(polygon) < -8 < np.median(vertex)


def test_unmarked_and_wrong_key_are_not_detected(marked):
    plain = toy_drawing(None, seed=2, shapes=5, vertices=14, sigma=0.7)[0].svg()
    assert detect(marked, KEY, POLYGON)["detected"]
    assert detect(plain, KEY, POLYGON)["log10_p_value"] > -3
    assert detect(marked, bytes(range(1, 33)), POLYGON)["log10_p_value"] > -3


def test_null_is_super_uniform_over_keys():
    plain = toy_drawing(None, seed=7, shapes=5, vertices=14, sigma=0.7)[0].svg()
    values = np.array([detect(plain, hashlib.sha256(bytes([i])).digest(), POLYGON)["p_value"] for i in range(200)])
    assert (values <= 0.1).mean() <= 0.17 and (values <= 0.01).mean() <= 0.035


@pytest.mark.parametrize("name", ["svgo_default", "scour_default", "picosvg", "round_1dp", "round_0dp", "rotate_30", "scale_0.37", "mirror",
                                  "translate_5pct", "group_transform", "reorder", "reverse", "restart", "merge_paths", "split_subpaths"])
def test_polygon_scheme_invariances(marked, name):
    clean = detect(marked, KEY, POLYGON)["log10_p_value"]
    attacked = detect(attacks.standard_suite()[name](marked), KEY, POLYGON)["log10_p_value"]
    assert attacked < -6 and attacked < 0.7 * clean, (name, clean, attacked)


def test_known_limits_of_the_polygon_scheme(marked):
    # The control polygon is not intrinsic: subdivision destroys it.
    assert detect(attacks.standard_suite()["subdivide"](marked), KEY, POLYGON)["log10_p_value"] > -6
    assert detect(attacks.standard_suite()["aspect_1.2"](marked), KEY, POLYGON)["log10_p_value"] > -6


def test_straight_runs_are_merged_before_keys():
    line = lambda a, b: Segment2(a, b, 0.0, True)  # noqa: E731
    split = [line(0j, 5 + 0j), line(5 + 0j, 10 + 0j), line(10 + 0j, 10 + 8j), line(10 + 8j, 2 + 12j)]
    merged = [line(0j, 10 + 0j), line(10 + 0j, 10 + 8j), line(10 + 8j, 2 + 12j)]
    assert polygon_keys(split, POLYGON) == polygon_keys(merged, POLYGON)


def test_tracker_uses_the_samplers_parameters():
    from contourmark.point_token_models import DecodeState, IconShopGrammar, watermarked_step

    state = DecodeState(IconShopGrammar())
    assert isinstance(state.tracker, PathTracker)
    watermark = GeoWatermark(KEY, GeoParameters(scheme="polygon", angle_step=3.0), seed=0)
    logits = np.full(6 + 200 * 200, -np.inf)
    logits[6:20] = 0.0
    state.feed(3)
    watermarked_step(state, logits, watermark, np.random.default_rng(0))
    assert state.tracker.params.angle_step == 3.0
