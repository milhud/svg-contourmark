"""Review probes for geosample.py / point_token_models.py (CPU, no torch).

Run against a snapshot of HEAD (25c7dc1) so concurrent edits do not matter:
    PYTHONPATH=<snapshot>/src OMP_NUM_THREADS=1 .venv/bin/python docs/review-2026-10-02/probe_geosample.py [A B C D E]
"""
from __future__ import annotations

import hashlib
import itertools
import math
import sys
from collections import Counter

import numpy as np
from scipy.stats import chi2

from contourmark.geosample import GeoWatermark, detect, document_descriptors, GeoParameters, keyed_uniform
from contourmark.point_token_models import GeoWatermarkLogitsProcessor, IconShopGrammar
from contourmark.toygen import toy_drawing

which = set(sys.argv[1:]) or set("ABCDE")


def key(i: int, tag: str = "k") -> bytes:
    return hashlib.sha256(f"{tag}{i}".encode()).digest()


# ---------------------------------------------------------------------------
# A. Joint law of an adaptive multi-step process, straight through _gumbel.
# Candidates per step: a mix of keyed groups (some with two member tokens),
# repeated descriptors across steps, and free (None) candidates.  Step
# probabilities depend on the full history.
D = [("v", i, 0, 0) for i in range(4)]
LAYOUT = [D[0], D[0], D[1], D[2], None, None, D[3]]  # token index -> descriptor


def step_probs(history: tuple[int, ...]) -> np.ndarray:
    rng = np.random.default_rng(1000 + sum((h + 1) * 7 ** i for i, h in enumerate(history)))
    p = rng.dirichlet(np.full(len(LAYOUT), 0.6))
    return p


def exact_joint(steps: int) -> dict[tuple[int, ...], float]:
    out = {}
    for seq in itertools.product(range(len(LAYOUT)), repeat=steps):
        p = 1.0
        for t in range(steps):
            p *= step_probs(seq[:t])[seq[t]]
        out[seq] = p
    return out


def run_joint(reuse: str, trials: int, steps: int = 3) -> tuple[float, float, float]:
    target = exact_joint(steps)
    counts: Counter = Counter()
    for i in range(trials):
        w = GeoWatermark(key(i, reuse), seed=i, reuse=reuse)
        history: tuple[int, ...] = ()
        for _ in range(steps):
            history += (w._gumbel(list(LAYOUT), step_probs(history)).index,)
        counts[history] += 1
    # pool cells with expectation < 5
    stat, dof, pooled_o, pooled_e = 0.0, 0, 0.0, 0.0
    for seq, p in target.items():
        e = p * trials
        if e < 5:
            pooled_o += counts[seq]; pooled_e += e
            continue
        stat += (counts[seq] - e) ** 2 / e; dof += 1
    if pooled_e > 0:
        stat += (pooled_o - pooled_e) ** 2 / pooled_e; dof += 1
    tv = 0.5 * sum(abs(counts[s] / trials - p) for s, p in target.items())
    return stat, dof - 1, tv


if "A" in which:
    for reuse in ("mask", "allow"):
        stat, dof, tv = run_joint(reuse, 150_000)
        print(f"A joint-law reuse={reuse}: chi2={stat:.1f} dof={dof} p={chi2.sf(stat, dof):.3g} TV={tv:.4f}")


# ---------------------------------------------------------------------------
# B. GeoWatermarkLogitsProcessor row construction (no torch needed for _row).
if "B" in which:
    template = GeoWatermark(key(0), mode="bias", delta=4.0, gamma=0.25)
    proc = GeoWatermarkLogitsProcessor(IconShopGrammar(), template, prompt_length=0, seed=5)
    row_w, row_rng = proc._row(0, fresh=True)
    print(f"B template mode/delta/gamma = {template.mode}/{template.delta}/{template.gamma};"
          f" row sampler = {row_w.mode}/{row_w.delta}/{row_w.gamma}")
    first = (row_w.rng.random(), float(row_rng.random()))
    row_w2, row_rng2 = proc._row(0, fresh=True)  # what a second generate() call does at its first step
    second = (row_w2.rng.random(), float(row_rng2.random()))
    print(f"B unkeyed randomness, 1st generate() call: {first}")
    print(f"B unkeyed randomness, 2nd generate() call: {second}  identical={first == second}")


# ---------------------------------------------------------------------------
# C. Null calibration of detect() over many keys on fixed unmarked drawings.
if "C" in which:
    levels = [1e-1, 1e-2, 1e-3]
    for statistic in ("gamma", "green"):
        ps = []
        for doc in range(4):
            state, _ = toy_drawing(None, 500 + doc, shapes=3 + doc, vertices=10)
            svg = state.svg()
            for i in range(6000):
                ps.append(detect(svg, key(i, f"null{doc}"), statistic=statistic)["p_value"])
        ps = np.array(ps)
        print(f"C null statistic={statistic} n={len(ps)}: " + ", ".join(f"P[p<={a:g}]={np.mean(ps <= a):.5f}" for a in levels))


# ---------------------------------------------------------------------------
# D. How many of the sampler's keyed (first-look, winning) descriptors does
# the detector actually recover from the emitted SVG?  And how much of the
# detector's distinct set was never keyed (pure null dilution)?
if "D" in which:
    import contourmark.geosample as G

    recovered, keyed_total, diluting, distinct_total = 0, 0, 0, 0
    for seed in range(20):
        w = GeoWatermark(key(1), seed=seed)
        wins: list[tuple] = []
        original = w._gumbel

        def spy(descriptors, probabilities, _orig=original, _w=w):
            before = set(_w.queried)
            choice = _orig(descriptors, probabilities)
            if choice.descriptor is not None and choice.descriptor not in before:
                wins.append(choice.descriptor)
            return choice

        w._gumbel = spy  # type: ignore[method-assign]
        state, _ = toy_drawing(w, seed)
        seen = {d for c in document_descriptors(state.svg(), w.params) for d in c}
        recovered += sum(d in seen for d in set(wins)); keyed_total += len(set(wins))
        diluting += len(seen - set(wins)); distinct_total += len(seen)
    print(f"D keyed first-look winners recovered by the detector: {recovered}/{keyed_total} = {recovered / keyed_total:.3f}")
    print(f"D detector distinct descriptors that were never a keyed winner: {diluting}/{distinct_total} = {diluting / distinct_total:.3f}")


# ---------------------------------------------------------------------------
# E. "keyed" flag returned by the sampler: counts masked (unkeyed) choices too.
if "E" in which:
    w = GeoWatermark(key(2), seed=0)
    d = [("v", 1, 0, 0), ("v", 2, 0, 0)]
    first = w._gumbel(d, [0.5, 0.5])
    second = w._gumbel(d, [0.5, 0.5])  # both descriptors already queried: fresh uniforms only
    print(f"E first step keyed={first.keyed} groups={first.groups}; second step (all masked) keyed={second.keyed} groups={second.groups}")


# ---------------------------------------------------------------------------
# F. "Straight cubic <-> line" is NOT invariant for the tangent descriptor.
# Scour and SVGO rewrite a cubic whose controls lie on its chord as a line.
if "F" in which:
    def svg(d: str) -> bytes:
        return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><path d="{d}"/></svg>'.encode()

    as_cubic = svg("M10 10C15 10 22 10 40 10C60 12 70 30 60 50C50 70 20 60 10 40Z")
    as_line = svg("M10 10L40 10C60 12 70 30 60 50C50 70 20 60 10 40Z")
    a = document_descriptors(as_cubic, GeoParameters())[0]
    b = document_descriptors(as_line, GeoParameters())[0]
    print(f"F straight edge written as C: {sorted(a, key=str)}")
    print(f"F same edge written as L:     {sorted(b, key=str)}")
    print(f"F descriptor multisets equal: {sorted(a, key=str) == sorted(b, key=str)}")


# ---------------------------------------------------------------------------
# G. Processor state across generate() calls when the first history is not
# empty (decoder start token, or prompt_length too small): `fresh` never
# fires, so the previous drawing's `queried` set masks the next drawing.
# Uses torch tensors only; no model is loaded.
if "G" in which:
    import torch

    grammar = IconShopGrammar()
    vocab = grammar.offset + grammar.grid * grammar.grid

    def tok(x: int, y: int) -> int:
        return grammar.offset + y * grammar.grid + x

    def drawing(proc: GeoWatermarkLogitsProcessor, start: list[int]) -> int:
        """Feed a fixed prefix, then let the processor pick 12 line endpoints from 40 candidates each."""
        ids = start + [3, tok(50, 50), tok(50, 50), 4, tok(90, 55)]
        rng = np.random.default_rng(7)
        for _ in range(12):
            ids.append(4)
            scores = torch.full((1, vocab), -1e9)
            cand = [tok(int(x), int(y)) for x, y in rng.integers(20, 180, size=(40, 2))]
            scores[0, cand] = 0.0
            out = proc(torch.tensor([ids]), scores)
            ids.append(int(out.argmax()))
        return len(proc.rows[0][0].queried)

    for start, plen, label in (([], 0, "history empty at first step"), ([1], 0, "history starts with a start token (prompt_length too small by one)")):
        proc = GeoWatermarkLogitsProcessor(grammar, GeoWatermark(key(9)), prompt_length=plen, seed=1)
        first = drawing(proc, start)
        before_second = len(proc.rows[0][0].queried)
        # first step of a second generate() call: only the start token (if any) is in the history
        proc(torch.tensor([start + [3]] if start else torch.zeros((1, 0), dtype=torch.long).tolist()).reshape(1, -1).long(), torch.zeros((1, vocab)))
        print(f"G {label}: queried after drawing 1 = {first}; carried into drawing 2 = {len(proc.rows[0][0].queried)}")
