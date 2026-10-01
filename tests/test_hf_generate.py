"""End-to-end: the watermark runs inside transformers' ``model.generate``.

A tiny randomly initialized GPT-2 stands in for an OmniSVG-style point-token
model (commands + one token per grid point).  A grammar processor keeps the
stream well-formed; the geosample processor then makes the keyed choice.
"""

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from contourmark.geosample import GeoWatermark, detect  # noqa: E402
from contourmark.point_token_models import DecodeState, GeoWatermarkLogitsProcessor, OmniSVGGrammar  # noqa: E402

GRID = 24
GRAMMAR = OmniSVGGrammar(move=1, line=2, curve=3, close=4, end=0, coordinate_start=8, grid=GRID)
VOCAB = 8 + GRID * GRID
KEY = hashlib.sha256(b"hf-generate-test").digest()


class GrammarProcessor:
    """Allow only grammatical continuations: M p (L p | C p p p)* Z, repeated."""

    def __init__(self, prompt_length: int):
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, -float("inf"))
        points = list(range(GRAMMAR.coordinate_start, GRAMMAR.coordinate_start + GRID * GRID))
        for row in range(scores.shape[0]):
            state = DecodeState(GRAMMAR)
            history = input_ids[row, self.prompt_length:].tolist()
            for token in history:
                state.feed(token)
            last = GRAMMAR.command(history[-1]) if history else None
            if last in ("M", "L", "C"):
                allowed = points  # a command must be followed by its points
            elif state.pending in ("M", "L", "C") and 0 < len(state.points) < GRAMMAR.arity[state.pending]:
                allowed = points  # finish the current command
            elif state.pending is None or state.pending == "Z":
                allowed = [GRAMMAR.move] if len(history) < 140 else [GRAMMAR.end]
            else:
                segments = len(state.subpaths[-1]) - 1 if state.subpaths else 0
                allowed = [GRAMMAR.line, GRAMMAR.curve] + ([GRAMMAR.close] if segments >= 4 else [])
                if len(history) >= 140:
                    allowed = [GRAMMAR.close]
            mask[row, allowed] = scores[row, allowed]
        return mask


def run(watermark: GeoWatermark | None, seed: int) -> bytes:
    torch.manual_seed(seed)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=VOCAB, n_positions=256, n_embd=32, n_layer=1, n_head=2))
    model.eval()
    prompt = torch.tensor([[GRAMMAR.move]])
    processors = [GrammarProcessor(prompt_length=0)]
    if watermark is not None:
        processors.append(GeoWatermarkLogitsProcessor(GRAMMAR, watermark, prompt_length=0, top_p=1.0, seed=seed))
    output = model.generate(
        prompt, do_sample=True, max_new_tokens=150, top_k=0, top_p=1.0, temperature=1.0,
        logits_processor=transformers.LogitsProcessorList(processors), pad_token_id=0, eos_token_id=0,
    )
    state = DecodeState(GRAMMAR)
    for token in output[0].tolist():
        if token == GRAMMAR.end:
            break
        state.feed(token)
    return state.svg()


def test_generate_with_processor_is_marked_and_plain_is_not():
    marked = [run(GeoWatermark(KEY, seed=s), s) for s in range(3)]
    plain = [run(None, s) for s in range(3)]
    marked_p = [detect(svg, KEY)["log10_p_value"] for svg in marked]
    plain_p = [detect(svg, KEY)["log10_p_value"] for svg in plain]
    # A random model on a 24x24 grid spreads mass widely, so keyed choices
    # carry strong evidence; unmarked outputs follow the exact null.
    assert np.median(marked_p) < -6, marked_p
    assert max(plain_p) > -3, plain_p
