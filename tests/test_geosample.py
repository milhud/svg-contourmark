import hashlib
import math

import numpy as np
import pytest

from contourmark import attacks
from contourmark.geosample import GeoWatermark, PathTracker, detect
from contourmark.point_token_models import (
    DecodeState,
    GeoWatermarkLogitsProcessor,
    IconShopGrammar,
    OmniSVGGrammar,
    truncate,
    watermarked_step,
)

KEY = hashlib.sha256(b"geosample-test-key").digest()
WRONG = hashlib.sha256(b"geosample-wrong-key").digest()
GRAMMAR = IconShopGrammar()
VOCAB = GRAMMAR.offset + GRAMMAR.grid * GRAMMAR.grid
_Q = np.arange(GRAMMAR.grid * GRAMMAR.grid)
_X, _Y = _Q % GRAMMAR.grid, _Q // GRAMMAR.grid


def point_logits(target: complex, sigma: float = 3.0, window: float = 10.0) -> np.ndarray:
    """Toy model: Gaussian over grid points near ``target``; masked elsewhere."""
    logits = np.full(VOCAB, -np.inf)
    distance2 = (_X - target.real) ** 2 + (_Y - target.imag) ** 2
    logits[GRAMMAR.offset:] = np.where(distance2 <= window ** 2, -distance2 / (2 * sigma ** 2), -np.inf)
    return logits


def toy_drawing(watermark, seed, shapes=5, vertices=14):
    """A point-token "generator" drafting noisy polygons with lines and curves."""
    rng = np.random.default_rng(seed)
    state = DecodeState(GRAMMAR)

    def put(token):
        state.feed(token)

    def draw(target):
        token, _ = watermarked_step(state, point_logits(target), watermark, rng, top_p=1.0)
        put(token)

    for _ in range(shapes):
        cx, cy = 30 + 140 * rng.random(), 30 + 140 * rng.random()
        radius = 12 + 18 * rng.random()
        points = [complex(cx + radius * (1 + 0.3 * rng.standard_normal()) * math.cos(2 * math.pi * k / vertices),
                          cy + radius * (1 + 0.3 * rng.standard_normal()) * math.sin(2 * math.pi * k / vertices)) for k in range(vertices)]
        put(3)
        draw(points[0])
        draw(points[0])
        for k in range(1, vertices + 1):
            end, start = points[k % vertices], points[k - 1]
            if rng.random() < 0.5:
                put(4)
            else:
                put(5)
                draw((2 * start + end) / 3 + 3j)
                draw((start + 2 * end) / 3 - 3j)
            draw(end)
    return state


@pytest.fixture(scope="module")
def marked_svg():
    return toy_drawing(GeoWatermark(KEY, seed=1), seed=1).svg()


def test_keyed_choice_preserves_the_categorical_distribution():
    tracker = PathTracker()
    tracker.move_to(0j)
    tracker.line_to(40 + 0j)
    tracker.line_to(40 + 40j)
    ends = [
        0 + 40j, 1 + 40j,  # share a descriptor group (nearly identical corner)
        10 + 60j, 70 + 70j, 40 + 90j, 20 + 75j, 60 + 50j, 5 + 20j,
        40 + 80j,  # continues the straight run: vertex-free
        None,  # non-point token: vertex-free
    ]
    probabilities = np.array([0.15, 0.05, 0.1, 0.1, 0.05, 0.2, 0.1, 0.1, 0.1, 0.05])
    trials = 6000
    counts = np.zeros(len(ends))
    for trial in range(trials):
        key = hashlib.sha256(trial.to_bytes(4, "big")).digest()
        counts[GeoWatermark(key, seed=trial).choose(tracker, ends, probabilities).index] += 1
    expected = probabilities * trials
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    assert chi2 < 27.9, (chi2, counts)  # chi-square 9 dof, p ~ 0.001


def test_detector_null_is_super_uniform():
    values = []
    for seed in (2, 3):
        svg = toy_drawing(None, seed=seed).svg()
        for index in range(150):
            values.append(detect(svg, hashlib.sha256(f"null-{seed}-{index}".encode()).digest())["p_value"])
    values = np.array(values)
    assert np.mean(values <= 0.1) <= 0.15
    assert np.mean(values <= 0.01) <= 0.03


def test_marked_detected_unmarked_and_wrong_key_not(marked_svg):
    result = detect(marked_svg, KEY)
    assert result["distinct_vertices"] >= 40
    assert result["detected"] and result["log10_p_value"] < -6
    assert detect(marked_svg, WRONG)["log10_p_value"] > -3
    assert detect(toy_drawing(None, seed=1).svg(), KEY)["log10_p_value"] > -3


ROBUST = ["reorder", "reverse", "restart", "rotate_30", "scale_0.37", "mirror", "translate_5pct", "group_transform",
          "svgo_default", "scour_default", "round_2dp", "merge_paths", "split_subpaths"]


@pytest.mark.parametrize("name", ROBUST)
def test_detection_survives(marked_svg, name):
    attacked = attacks.standard_suite()[name](marked_svg)
    assert detect(attacked, KEY)["log10_p_value"] < -6, name


def test_subdivision_is_degraded_not_erased(marked_svg):
    # Subdivision adds smooth vertices and halves chords and handles; vertex
    # descriptors change but tangent angles at original vertices survive, so
    # the evidence weakens rather than disappearing.
    clean = detect(marked_svg, KEY)["log10_p_value"]
    attacked = detect(attacks.standard_suite()["subdivide"](marked_svg), KEY)["log10_p_value"]
    assert clean < attacked < 0


def test_aspect_stretch_is_a_known_failure(marked_svg):
    # A non-uniform stretch changes angles: not similarity-invariant.
    attacked = attacks.standard_suite()["aspect_1.2"](marked_svg)
    assert detect(attacked, KEY)["log10_p_value"] > -6


def test_omnisvg_grammar_decodes_and_reports_endpoint_slots():
    grammar = OmniSVGGrammar(move=10, line=11, curve=12, close=13, end=14, coordinate_start=100)

    def pt(x, y):
        return 100 + y * 200 + x

    state = DecodeState(grammar)
    assert state.endpoint_slot() == (False, [])
    for token in (10, pt(5, 5)):
        state.feed(token)
    assert state.endpoint_slot() == (False, [])  # after a completed move
    state.feed(11)
    assert state.endpoint_slot() == (True, [])  # line endpoint
    state.feed(pt(50, 5))
    state.feed(12)
    assert state.endpoint_slot()[0] is False  # first control point
    state.feed(pt(60, 20))
    assert state.endpoint_slot()[0] is False  # second control point
    state.feed(pt(60, 40))
    assert state.endpoint_slot() == (True, [60 + 20j, 60 + 40j])
    state.feed(pt(50, 50))
    state.feed(13)
    assert state.endpoint_slot() == (False, [])
    assert grammar.is_end(14)
    assert [name for name, _ in state.subpaths[0]] == ["M", "L", "C", "Z"]
    assert state.tracker.current == 5 + 5j  # Z returns to the subpath start
    svg = state.svg().decode()
    assert "M5 5L50 5C60 20 60 40 50 50Z" in svg


def test_iconshop_grammar_point_mapping():
    assert GRAMMAR.point(6) == 0j
    assert GRAMMAR.point(6 + 3 * 200 + 7) == 7 + 3j
    assert GRAMMAR.point(5) is None and GRAMMAR.command(5) == "C"
    assert GRAMMAR.is_end(0)


def test_truncate_drops_zero_probability_tokens():
    logits = np.array([0.0, -np.inf, 1.0, -np.inf, 0.5])
    ids, probs = truncate(logits)
    assert set(ids.tolist()) == {0, 2, 4}
    assert probs.sum() == pytest.approx(1.0)


def test_hf_logits_processor_returns_one_hot():
    torch = pytest.importorskip("torch")
    grammar = OmniSVGGrammar(move=10, line=11, curve=12, close=13, end=14, coordinate_start=100, grid=20)

    def pt(x, y):
        return 100 + y * 20 + x

    prompt = [1, 2, 3]
    at_endpoint = prompt + [10, pt(2, 2), 11, pt(15, 2), 11]
    after_move_token = prompt + [10]
    processor = GeoWatermarkLogitsProcessor(grammar, GeoWatermark(KEY, seed=0), prompt_length=len(prompt), top_p=0.9)
    generator = torch.Generator().manual_seed(0)
    vocab = 100 + 20 * 20
    for history in (at_endpoint, after_move_token):
        scores = torch.randn(2, vocab, generator=generator)
        scores[:, 100:] += 3.0  # make grid points the plausible continuations
        out = processor(torch.tensor([history, history]), scores.clone())
        for row in range(2):
            finite = torch.isfinite(out[row])
            assert int(finite.sum()) == 1
            chosen = int(torch.nonzero(finite)[0])
            ids, _ = truncate(scores[row].numpy(), top_p=0.9)
            assert chosen in set(ids.tolist())


def test_step_with_non_point_candidates_stays_keyed_and_unbiased():
    # At an endpoint slot whose truncated set also contains a command token,
    # the step must still go through the keyed choice (not fall back), with
    # the command token competing as a vertex-free group.
    from contourmark import geosample

    base = DecodeState(GRAMMAR)
    for token in (3, 6, 6, 4, 6 + 40, 4):  # M (0,0)(0,0), L (40,0), L -> endpoint slot
        base.feed(token)
    assert base.endpoint_slot() == (True, [])
    logits = np.full(VOCAB, -np.inf)
    candidates = {4: 0.2, 6 + 40 + 40 * 200: 0.3, 6 + 10 * 200: 0.25, 6 + 70 + 30 * 200: 0.25}
    for token, probability in candidates.items():
        logits[token] = math.log(probability)
    calls = []
    original = geosample.GeoWatermark.choose

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    geosample.GeoWatermark.choose = spy
    try:
        counts = {token: 0 for token in candidates}
        trials = 3000
        for trial in range(trials):
            state = DecodeState(GRAMMAR)
            for token in (3, 6, 6, 4, 6 + 40, 4):
                state.feed(token)
            key = hashlib.sha256(b"step" + trial.to_bytes(4, "big")).digest()
            token, _ = watermarked_step(state, logits, GeoWatermark(key, seed=trial), np.random.default_rng(trial))
            counts[token] += 1
    finally:
        geosample.GeoWatermark.choose = original
    assert len(calls) == trials
    for token, probability in candidates.items():
        assert abs(counts[token] / trials - probability) < 0.035, counts
