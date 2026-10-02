"""Recompute, from stored result files only, the numbers the review disputes or confirms.

    .venv/bin/python docs/review-2026-10-02/claims_audit.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binom

R = Path(__file__).resolve().parents[2] / "experiments/results"


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


print("== geosample on IconShop: is detection 'unchanged' / 'exact' under optimizers? (marked samples, *_attacks_local.jsonl)")
for run in ("colab_run1", "colab_run1_repeat"):
    for cfg in ("iconshop_p05", "iconshop_p09"):
        rows = lines(R / "geosample" / run / f"{cfg}_attacks_local.jsonl")
        marked = [r for r in rows if r["marked"]]
        for attack in ("svgo_default", "scour_default", "svgo_p1", "round_0dp", "rotate_30", "scale_0.37"):
            delta = np.array([r["attacks"][attack]["log10_p"] - r["attacks"]["identity"]["log10_p"] for r in marked])
            changed = delta[np.abs(delta) > 1e-6]
            flips = sum((r["attacks"][attack]["log10_p"] <= -6) != (r["attacks"]["identity"]["log10_p"] <= -6) for r in marked)
            worst = f"{changed.max():+.2f}" if len(changed) else "n/a"
            print(f"  {run:18s} {cfg} {attack:14s} p-value changed in {len(changed):3d}/120, largest loss {worst} decades, decisions flipped at 1e-6: {flips}")

print("\n== geosample: counts behind the conditional percentages")
for run in ("colab_run1", "colab_run1_repeat"):
    for cfg in ("iconshop_p05", "iconshop_p09"):
        s = json.loads((R / "geosample" / run / f"{cfg}_attacks_local.summary.json").read_text())["attacks"]
        print(f"  {run:18s} {cfg} clean {s['identity']['tpr_1e-6']['hits']}/120; " + "; ".join(
            f"{a} {s[a]['conditional_1e-6']['hits']}/{s[a]['conditional_1e-6']['n']}" for a in ("rotate_30", "scale_0.37", "group_transform", "compose_grid", "delete_50pct", "svgo_p1")))

print("\n== geosample key-randomized nulls (36,000 tests each)")
for path in sorted((R / "geosample").glob("colab_run1*/**/iconshop_p0*_null.json")):
    n = json.loads(path.read_text())
    print(f"  {str(path.relative_to(R / 'geosample')):62s} hits@1e-3={n['levels']['0.001']['hits']:3d} @1e-5={n['levels']['1e-05']['hits']} min log10 p={n['min_log10_p']:.2f}")
print(f"  P[>=46 hits | 36000, 1e-3] = {binom.sf(45, 36000, 1e-3):.3f};  P[min p <= 10^-5.99 | 36000 tests] <= {36000 * 10 ** -5.99:.3f}")

print("\n== spectral frontier: the spread-spectrum 'ablation' rows (blind/frontier.json)")
frontier = json.loads((R / "blind/frontier.json").read_text())["summary"]
for corpus, table in frontier.items():
    for name, v in table.items():
        if name.startswith(("arc_ss", "spectral@0.004", "fd_vertex@0.4")):
            print(f"  {corpus:14s} {name:16s} n={v['n']:2d} p95 boundary distance={100 * v['p95_boundary_median']:.3f}% of diagonal, clean TPR={v['clean_tpr_1e-6']:.2f}")

print("\n== spectral main run (blind/v3/test.analysis.json)")
a = json.loads((R / "blind/v3/test.analysis.json").read_text())
print(f"  clean {a['clean_detectable']}/{a['assets']}; reverse survival {a['conditional_survival']['reverse']['rate']:.4f}; misses {a['clean_failures']}")
null = json.loads((R / "blind/v2/null_calibration.json").read_text())
print(f"  spectral null (v2 file; none stored for v3): tests={null['tests']} min log10 p={null['min_log10_p']:.3f} (below the 1e-6 decision threshold: {null['min_log10_p'] <= -6})")
