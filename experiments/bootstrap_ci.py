"""Asset-level bootstrap confidence intervals for evaluate_blind.py results.

Attacks on the same asset are not independent observations, and neither are
methods compared on the same assets.  This resamples *assets* (optionally
within source) and reports percentile intervals for:

* clean detection rate and per-attack detection rate, per method;
* conditional survival (among clean-detected assets), per method;
* paired differences between two methods on the same resampled assets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

THRESHOLD = -6.0


def load(paths: list[Path]) -> list[dict]:
    merged: dict[str, dict] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                merged.setdefault(record["path"], {**record, "methods": {}})["methods"].update(record["methods"])
    return list(merged.values())


def matrix(records: list[dict], method: str, attacks: list[str]) -> np.ndarray:
    """assets x attacks: 1 detected, 0 not detected, nan not attempted."""
    out = np.full((len(records), len(attacks)), np.nan)
    for i, record in enumerate(records):
        entry = record["methods"].get(method, {}).get("attacks", {})
        for j, attack in enumerate(attacks):
            value = entry.get(attack)
            if value is not None and "attack_error" not in value:
                out[i, j] = float(value.get("log10_p", 0.0) <= THRESHOLD)
    return out


def interval(samples: np.ndarray) -> list[float]:
    return [float(np.nanpercentile(samples, 2.5)), float(np.nanpercentile(samples, 97.5))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--methods", nargs="+", default=["spectral", "numeric_lsb"])
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--stratify", action="store_true", help="resample within each source")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = load(args.results)
    attacks = sorted({a for r in records for e in r["methods"].values() for a in e.get("attacks", {})})
    rng = np.random.default_rng(args.seed)
    n = len(records)
    sources = np.array([r["source"] for r in records])
    data = {m: matrix(records, m, attacks) for m in args.methods}
    clean = attacks.index("identity")

    def draw() -> np.ndarray:
        if not args.stratify:
            return rng.integers(0, n, n)
        return np.concatenate([rng.choice(np.flatnonzero(sources == s), int((sources == s).sum())) for s in np.unique(sources)])

    rates = {m: np.empty((args.resamples, len(attacks))) for m in args.methods}
    conditional = {m: np.empty((args.resamples, len(attacks))) for m in args.methods}
    for b in range(args.resamples):
        index = draw()
        for m in args.methods:
            sample = data[m][index]
            with np.errstate(invalid="ignore"):
                rates[m][b] = np.nanmean(sample, axis=0)
                detected = sample[sample[:, clean] == 1]
                conditional[m][b] = np.nanmean(detected, axis=0) if len(detected) else np.nan
    report: dict = {"assets": n, "resamples": args.resamples, "stratified": args.stratify, "methods": {}}
    for m in args.methods:
        with np.errstate(invalid="ignore"):
            point = np.nanmean(data[m], axis=0)
            detected = data[m][data[m][:, clean] == 1]
            point_conditional = np.nanmean(detected, axis=0) if len(detected) else np.full(len(attacks), np.nan)
        report["methods"][m] = {
            attack: {"rate": float(point[j]), "ci95": interval(rates[m][:, j]),
                     "conditional": float(point_conditional[j]), "conditional_ci95": interval(conditional[m][:, j])}
            for j, attack in enumerate(attacks) if not np.isnan(point[j])
        }
    if len(args.methods) >= 2:
        a, b = args.methods[:2]
        report["paired_difference"] = {
            "methods": [a, b],
            "attacks": {attack: {"difference": float(np.nanmean(data[a][:, j]) - np.nanmean(data[b][:, j])), "ci95": interval(rates[a][:, j] - rates[b][:, j])}
                        for j, attack in enumerate(attacks) if attack in report["methods"][a] and attack in report["methods"][b]},
        }
    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + "\n")
    for m in args.methods:
        print(f"\n{m}: rate [95% CI] | conditional on clean [95% CI]")
        for attack in ("identity", "svgo_default", "round_2dp", "rotate_30", "compose_grid", "revectorize_1024"):
            if attack in report["methods"][m]:
                e = report["methods"][m][attack]
                print(f"  {attack:18s} {e['rate']:.3f} [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}] | {e['conditional']:.3f} [{e['conditional_ci95'][0]:.3f}, {e['conditional_ci95'][1]:.3f}]")
    if "paired_difference" in report:
        a, b = report["paired_difference"]["methods"]
        print(f"\npaired difference {a} - {b}:")
        for attack in ("identity", "svgo_default", "round_2dp", "rotate_30"):
            e = report["paired_difference"]["attacks"].get(attack)
            if e:
                print(f"  {attack:18s} {e['difference']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
