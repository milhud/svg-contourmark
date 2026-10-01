"""Robustness-distortion frontiers for every method (development split).

A comparison at one hand-picked strength per method says little: methods
differ in how much they move the drawing.  This harness sweeps each method's
strength, measures the *same* geometric distortion for all of them (symmetric
boundary distance as a fraction of the diagonal), and records detection clean
and under a few transformations.  The summary then compares methods at
matched distortion.

Methods and strength parameters:

* ``spectral``     QIM on arc-length radial spectra; lattice step delta.
* ``arc_ss``       same carrier, additive spread spectrum (ablation); delta.
* ``fd_vertex``    vertex Fourier-descriptor adaptation; multiplicative strength.
* ``numeric_lsb``  coordinate parity; quantum.

``--positive-control`` adds synthetic long, dense closed polylines (about 600
vertices), the regime the vertex Fourier-descriptor method was designed for.

Use ``--smoke`` for a seconds-long run.  Tune on dev only; freeze the chosen
operating points before any test-split run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np

from contourmark import attacks as attack_lib
from contourmark import vector_baselines as vb
from contourmark.metrics import boundary_distance
from contourmark.spectral import SpectralParameters, detect, embed

ROOT = Path(__file__).resolve().parents[1]
GRIDS = {
    "spectral": [0.001, 0.002, 0.004, 0.008, 0.016],
    "arc_ss": [0.004, 0.016, 0.064],
    "fd_vertex": [0.02, 0.05, 0.1, 0.2, 0.4],
    "numeric_lsb": [0.001, 0.01, 0.1],
}
ATTACKS = ["svgo_default", "round_2dp", "rotate_30", "subdivide", "noise_0.1pct"]


def positive_control(index: int) -> bytes:
    """A long noisy closed polyline: the vertex-FD method's home ground."""
    rng = np.random.default_rng(1000 + index)
    n = 600
    theta = np.linspace(0, 2 * math.pi, n, endpoint=False)
    # Broadband outline (energy at every frequency, as in digitized map lines).
    radius = 70 + sum(rng.normal(0, 8 / k) * np.cos(k * theta + rng.uniform(0, 6.28)) for k in range(2, 280))
    points = 100 + radius * np.cos(theta) + 1j * (100 + radius * np.sin(theta))
    d = "M" + "L".join(f"{p.real:.3f} {p.imag:.3f}" for p in points) + "Z"
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"><path d="{d}"/></svg>'.encode()


def functions(method: str, strength: float):
    if method in ("spectral", "arc_ss"):
        params = SpectralParameters(delta=strength, scheme="ss" if method == "arc_ss" else "qim")
        return (lambda s, k: embed(s, k, params)[0]), (lambda s, k: detect(s, k, params)["log10_p_value"])
    if method == "fd_vertex":
        return (lambda s, k: vb.fd_vertex_embed(s, k, strength)), (lambda s, k: vb.fd_vertex_detect(s, k)["log10_p_value"])
    if method == "numeric_lsb":
        return (lambda s, k: vb.numeric_lsb_embed(s, k, quantum=strength)), (lambda s, k: vb.numeric_lsb_detect(s, k, quantum=strength)["log10_p_value"])
    raise SystemExit(f"unknown method {method}")


def run(task: tuple[str, str, bytes, str, float]) -> dict:
    name, source_set, source, method, strength = task
    key = hashlib.sha256(f"frontier|{name}".encode()).digest()
    embed_fn, detect_fn = functions(method, strength)
    record = {"asset": name, "set": source_set, "method": method, "strength": strength}
    try:
        marked = embed_fn(source, key)
        record["clean"] = detect_fn(marked, key)
        record["null"] = detect_fn(source, key)
        distance = boundary_distance(source, marked)["fraction_of_diagonal"]
        record["p95_boundary"], record["hausdorff"] = distance["p95"], distance["hausdorff"]
        record["byte_ratio"] = len(marked) / len(source)
        suite = attack_lib.standard_suite()
        record["attacks"] = {}
        for attack in ATTACKS:
            try:
                record["attacks"][attack] = detect_fn(suite[attack](marked), key)
            except Exception as exc:
                record["attacks"][attack] = None
                record.setdefault("attack_errors", {})[attack] = str(exc)[:120]
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return record


def summarize(records: list[dict]) -> dict:
    summary: dict = {}
    for source_set in sorted({r["set"] for r in records}):
        table = {}
        for method in GRIDS:
            for strength in sorted({r["strength"] for r in records if r["method"] == method}):
                rows = [r for r in records if r["set"] == source_set and r["method"] == method and r["strength"] == strength]
                good = [r for r in rows if "clean" in r]
                if not good:
                    continue
                entry = {
                    "n": len(rows), "errors": len(rows) - len(good),
                    "p95_boundary_median": float(np.median([r["p95_boundary"] for r in good])),
                    "hausdorff_median": float(np.median([r["hausdorff"] for r in good])),
                    "byte_ratio_median": float(np.median([r["byte_ratio"] for r in good])),
                    "clean_tpr_1e-6": float(np.mean([r["clean"] <= -6 for r in good])),
                    "null_fpr_1e-2": float(np.mean([r["null"] <= -2 for r in good])),
                }
                for attack in ATTACKS:
                    values = [r["attacks"][attack] for r in good if r["attacks"].get(attack) is not None]
                    entry[f"{attack}_tpr_1e-6"] = float(np.mean([v <= -6 for v in values])) if values else None
                    entry[f"{attack}_attempted"] = len(values)
                table[f"{method}@{strength:g}"] = entry
        summary[source_set] = table
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=ROOT / "experiments/corpus/dev.json")
    parser.add_argument("--per-source", type=int, default=10)
    parser.add_argument("--methods", nargs="+", default=list(GRIDS))
    parser.add_argument("--positive-control", type=int, default=6, help="number of synthetic long polylines (0 disables)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/results/blind/frontier.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.per_source, args.positive_control = 1, 1
    chosen: dict[str, list] = {}
    for item in json.loads(args.corpus.read_text())["items"]:
        group = chosen.setdefault(item["source"], [])
        if len(group) < args.per_source:
            group.append(item)
    assets = [(item["path"], "icons", (ROOT / item["path"]).read_bytes()) for group in chosen.values() for item in group]
    assets += [(f"positive-control-{i}", "long-polylines", positive_control(i)) for i in range(args.positive_control)]
    grids = {m: (GRIDS[m][1:3] if args.smoke else GRIDS[m]) for m in args.methods}
    tasks = [(name, source_set, data, method, strength) for name, source_set, data in assets for method, grid in grids.items() for strength in grid]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        records = list(pool.imap_unordered(run, tasks, chunksize=2))
    summary = summarize(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "records": records}, indent=1) + "\n")
    for source_set, table in summary.items():
        print(f"\n== {source_set} ==")
        print(f"{'method@strength':22s} {'p95 %diag':>9s} {'bytes x':>7s} {'clean':>6s} {'svgo':>6s} {'2dp':>6s} {'rot':>6s} {'subdiv':>6s} {'noise':>6s}")
        for name, e in table.items():
            cell = lambda k: "   —  " if e[k] is None else f"{e[k]:6.2f}"  # noqa: E731
            print(f"{name:22s} {100 * e['p95_boundary_median']:9.4f} {e['byte_ratio_median']:7.2f} {e['clean_tpr_1e-6']:6.2f} {cell('svgo_default_tpr_1e-6')} {cell('round_2dp_tpr_1e-6')} {cell('rotate_30_tpr_1e-6')} {cell('subdivide_tpr_1e-6')} {cell('noise_0.1pct_tpr_1e-6')}")


if __name__ == "__main__":
    main()
