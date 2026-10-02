"""Review probes for spectral.py (CPU).  Run from the repository root:

    PYTHONPATH=<snapshot>/src OMP_NUM_THREADS=1 .venv/bin/python docs/review-2026-10-02/probe_spectral.py [A B C D E]
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from contourmark import metrics, spectral
from contourmark.spectral import SpectralParameters, chernoff_log_p

ROOT = Path(__file__).resolve().parents[2]
KEY = hashlib.sha256(b"review-spectral").digest()
which = set(sys.argv[1:]) or set("ABCDE")

# ---------------------------------------------------------------------------
# A. Tail bound versus Monte Carlo for adversarial weight vectors.
if "A" in which:
    rng = np.random.default_rng(0)
    cases = {
        "n=1": np.array([1.0]),
        "n=1 w=0.3": np.array([0.3]),
        "n=2 equal": np.ones(2),
        "8 equal": np.ones(8),
        "64 equal": np.ones(64),
        "1000 tiny(1e-3)+one 5": np.concatenate([np.full(1000, 1e-3), [5.0]]),
        "200 tiny(0.02)+one 3": np.concatenate([np.full(200, 0.02), [3.0]]),
        "geometric 2^-k, 12 terms": 2.0 ** -np.arange(12),
        "16 distinct irrational": np.sqrt(np.arange(2, 18)) / 3,
        "3, 1e-13 (rounds to 0)": np.array([3.0, 1e-13]),
    }
    draws = 2_000_000
    worst = math.inf
    for name, w in cases.items():
        sums = np.zeros(draws)
        for weight in w:  # one weight at a time keeps memory flat
            sums += weight * np.cos(2 * np.pi * rng.random(draws))
        for q in (0.9, 0.99, 0.999, 0.9999):
            t = float(np.quantile(sums, q))
            if t <= 0:
                continue
            mc = float((sums >= t).mean())
            hi = mc + 3 * math.sqrt(mc * (1 - mc) / draws)
            lo = mc - 3 * math.sqrt(mc * (1 - mc) / draws)
            bound = math.exp(chernoff_log_p(t, w))
            chern = math.exp(chernoff_log_p(t, w, exact=False))
            worst = min(worst, bound / mc)
            flag = "  <-- BELOW MC" if bound < lo else ""
            print(f"A {name:26s} t={t:8.4f} MC={mc:.3e} bound={bound:.3e} ratio={bound / mc:6.2f} (chernoff only {chern / mc:7.1f}x){flag}")
    # n=1 has a closed form: P[w cos(2 pi U) >= t] = arccos(t / w) / pi
    for t in (0.5, 0.9, 0.99, 0.999999):
        print(f"A n=1 closed form t={t}: true={math.acos(t) / math.pi:.4e} bound={math.exp(chernoff_log_p(t, np.array([1.0]))):.4e}")
    print(f"A smallest bound/MC ratio over all cases: {worst:.3f}")

# ---------------------------------------------------------------------------
# B. capacity(): what "attainable" means, and whether "supported" predicts detection.
if "B" in which:
    for n in (1, 2, 3, 5, 8):
        w = np.ones(n)
        print(f"B attainable log10 p for {n} unit coefficients (statistic == total): {chernoff_log_p(float(w.sum()), w) / math.log(10) + math.log10(2):.1f}")
    rows = [json.loads(line) for line in (ROOT / "experiments/results/blind/v3/test_spectral.jsonl").read_text().splitlines() if line.strip()]
    missed = [r for r in rows if r["methods"]["spectral"]["attacks"]["identity"]["log10_p"] > -6]
    outcomes: Counter = Counter()
    attain = []
    for r in missed:
        try:
            c = spectral.capacity((ROOT / r["path"]).read_bytes())
        except Exception as exc:  # noqa: BLE001
            outcomes[f"error {type(exc).__name__}"] += 1
            continue
        outcomes[c["outcome"]] += 1
        if c["outcome"] == "supported":
            attain.append(c["attainable_log10_p"])
    print(f"B stored v3 run: {len(rows)} assets, {len(missed)} not detected clean at 1e-6")
    print(f"B capacity() outcome on the ORIGINALS of those misses (current code): {dict(outcomes)}")
    if attain:
        print(f"B 'supported' misses: median attainable log10 p = {np.median(attain):.1f}, max = {max(attain):.1f}")
    # <use> instances are counted by capacity() but never embedded.
    blob = "M60 30C90 20 120 40 110 70C105 95 125 110 100 130C75 150 40 130 35 100C30 75 35 40 60 30Z"
    use = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 160"><defs><path id="p" d="{blob}"/></defs><use href="#p"/></svg>'.encode()
    cap = spectral.capacity(use)
    marked, report = spectral.embed(use, KEY)
    print(f"B <use>-only document: capacity outcome={cap['outcome']} coefficients={cap['coefficients']}; embed marked_contours={report['marked_contours']} "
          f"eligible={report['eligible_contours']} skipped={report['skipped_contours']}; detect log10 p={spectral.detect(marked, KEY)['log10_p_value']:.2f}")

# ---------------------------------------------------------------------------
# C. Spread-spectrum "ablation": best p-value the detector can ever report.
if "C" in which:
    for n in (8, 16, 24, 32, 64):
        # (i) every selected magnitude equal and every keyed sign +1 (cannot be arranged by an embedder; an upper limit)
        best = -n / 2 + math.log(2)
        # (ii) infinite embedding strength: +sign magnitudes large and equal, -sign magnitudes pushed to 0 (half of each)
        infinite = -(n / 2) ** 2 / (2 * (n / 2)) + math.log(2)
        print(f"C spread-spectrum, {n} selected coefficients: best possible log10 p = {best / math.log(10):.2f}; at unbounded strength = {infinite / math.log(10):.2f} (threshold -6)")
    rows = [json.loads(line) for line in (ROOT / "experiments/results/blind/v3/test_spectral.jsonl").read_text().splitlines() if line.strip()]
    selected = np.array([r["methods"]["spectral"].get("selected", 0) for r in rows])
    print(f"C stored v3 test run: selected coefficients per asset median={np.median(selected):.0f}, share with >= 58 (needed at unbounded strength) = {(selected >= 58).mean():.2f}, share with >= 30 = {(selected >= 30).mean():.2f}")

# ---------------------------------------------------------------------------
# D. Smooth-join constraint with quadratic segments (one control point serves two joins).
if "D" in which:
    def smooth_quads(n: int = 10, r: float = 40.0) -> str:
        on = [complex(80 + r * (1 + 0.25 * math.sin(3 * a)) * math.cos(a), 80 + r * (1 + 0.25 * math.sin(3 * a)) * math.sin(a)) for a in np.linspace(0, 2 * math.pi, n + 1)[:-1]]
        # controls reflected so each join is G1 (T-style chain); close the chain by solving the last control from reflection
        c = on[0] + (on[1] - on[0]) * 0.5 + 6j
        d = f"M{on[0].real:.4f} {on[0].imag:.4f}Q{c.real:.4f} {c.imag:.4f} {on[1].real:.4f} {on[1].imag:.4f}"
        for i in range(1, n):
            d += f"T{on[(i + 1) % n].real:.4f} {on[(i + 1) % n].imag:.4f}"
        return d + "Z"

    quad = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 160"><path d="{smooth_quads()}"/></svg>'.encode()
    marked, report = spectral.embed(quad, KEY)
    s = metrics.smoothness(quad, marked)
    print(f"D quadratic T-chain: marked_contours={report['marked_contours']} detect log10 p={spectral.detect(marked, KEY)['log10_p_value']:.1f}")
    print(f"D smoothness report: {json.dumps(s)}")
    cubic = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black"><path d="M3 12C3 6 8 3 12 3s9 3 9 9-5 9-9 9-9-3-9-9z"/></svg>'
    marked_c, _ = spectral.embed(cubic, KEY)
    print(f"D cubic control: {json.dumps(metrics.smoothness(cubic, marked_c))}")

# ---------------------------------------------------------------------------
# E. Seed-stability retry: how often the report says marked while the serialized file's seed differs.
if "E" in which:
    items = json.loads((ROOT / "experiments/corpus/dev.json").read_text())["items"][:120]
    statuses: Counter = Counter()
    drift = total = 0
    for item in items:
        source = (ROOT / item["path"]).read_bytes()
        try:
            marked, report = spectral.embed(source, KEY)
        except Exception as exc:  # noqa: BLE001
            statuses[f"embed error {type(exc).__name__}"] += 1
            continue
        after = {o.index: o.seed for o in spectral.observe(spectral.load_document(marked), SpectralParameters())}
        for c in report["contours"]:
            statuses[c["status"]] += 1
            if c["status"] == "marked":
                total += 1
                drift += after.get(c["contour"]) != c["seed"]
    print(f"E dev[:120] contour statuses: {dict(statuses)}")
    print(f"E marked contours whose seed in the SERIALIZED file differs from the embedding seed (or contour not observed): {drift}/{total}")
