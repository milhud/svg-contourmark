"""Summarize evaluate_geosample.py output: detection, nulls, and preservation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from summarize_blind import ATTACK_GROUPS, wilson


def rate(values: list[float], threshold: float) -> dict:
    hits = sum(v <= threshold for v in values)
    return {"rate": hits / len(values) if values else float("nan"), "hits": hits, "n": len(values), "ci95": list(wilson(hits, len(values)))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for path in args.results for line in path.read_text().splitlines() if line.strip()]
    marked = [r for r in records if r["marked"]]
    plain = [r for r in records if not r["marked"]]
    attacks = ["identity"] + [a for group in ATTACK_GROUPS.values() for a in group]
    attacks += sorted({a for r in records for a in r["attacks"]} - set(attacks))
    clean_ok = [r for r in marked if r["attacks"].get("identity", {}).get("log10_p", 0) <= -6]
    summary: dict = {"marked": len(marked), "plain": len(plain), "clean_detectable_1e-6": len(clean_ok), "attacks": {}}
    # End-to-end denominators: truncated generations are kept in every rate
    # and reported, not silently treated as finished icons.
    summary["generation"] = {
        "marked_truncated": sum(bool(r.get("truncated")) for r in marked),
        "plain_truncated": sum(bool(r.get("truncated")) for r in plain),
        "marked_without_descriptors": sum(r["distinct_vertices"] == 0 for r in marked),
        "run_ids": sorted({str(r.get("run_id")) for r in records}),
    }
    for attack in attacks:
        values = [r["attacks"][attack]["log10_p"] for r in marked if "log10_p" in r["attacks"].get(attack, {})]
        if not values:
            continue
        conditional = [r["attacks"][attack]["log10_p"] for r in clean_ok if "log10_p" in r["attacks"].get(attack, {})]
        attempted = [r["attacks"][attack] for r in marked if attack in r["attacks"]]
        summary["attacks"][attack] = {
            # Attack failures (missing tool, crash) are reported, not hidden:
            # rates are over completed attacks only.
            "attempted": len(attempted),
            "attack_errors": sum("attack_error" in a for a in attempted),
            "detect_errors": sum("detect_error" in a for a in attempted),
            "tpr_1e-6": rate(values, -6), "tpr_1e-3": rate(values, -3),
            "conditional_1e-6": rate(conditional, -6) if conditional else None,
            "median_log10_p": float(np.median(values)),
        }
    nulls = {
        "plain_with_key": [r["attacks"]["identity"]["log10_p"] for r in plain if "identity" in r["attacks"]],
        "marked_wrong_key": [r["wrong_key_log10_p"] for r in marked],
        "plain_wrong_key": [r["wrong_key_log10_p"] for r in plain],
    }
    summary["nulls"] = {name: {f"fpr_1e-{k}": rate(values, -k) for k in (2, 3, 6)} for name, values in nulls.items() if values}
    distinct = np.array([r["distinct_vertices"] for r in marked])
    summary["distinct_vertices_marked"] = {"median": float(np.median(distinct)), "p10": float(np.percentile(distinct, 10)), "p90": float(np.percentile(distinct, 90))} if len(distinct) else {}
    bins = [0, 10, 20, 40, 80, 10_000]
    by_bin = {}
    for low, high in zip(bins[:-1], bins[1:]):
        group = [r["attacks"]["identity"]["log10_p"] for r in marked if low <= r["distinct_vertices"] < high and "identity" in r["attacks"]]
        if group:
            by_bin[f"{low}-{high - 1}"] = {"n": len(group), "median_log10_p": float(np.median(group)), "tpr_1e-6": rate(group, -6)["rate"], "tpr_1e-3": rate(group, -3)["rate"]}
    summary["log10_p_by_distinct_vertices"] = by_bin
    preservation = {}
    for field in ("tokens", "vertices", "distinct_vertices", "paths", "contours"):
        a = [r[field] for r in plain if field in r]
        b = [r[field] for r in marked if field in r]
        if len(a) >= 2 and len(b) >= 2:
            test = mannwhitneyu(a, b, alternative="two-sided")
            preservation[field] = {"plain_median": float(np.median(a)), "marked_median": float(np.median(b)), "mannwhitney_p": float(test.pvalue)}
    summary["preservation"] = preservation

    generation = summary["generation"]
    lines = [f"# Geosample evaluation ({len(marked)} marked, {len(plain)} plain samples)", "",
             f"Truncated at the token limit: {generation['marked_truncated']} marked, {generation['plain_truncated']} plain. "
             f"Marked samples with no descriptor: {generation['marked_without_descriptors']}. All are kept in the denominators.", "",
             "Rates are over completed attacks; `errors` counts attacks that could not run (attack/detect).", "",
             "| Attack | TPR@1e-6 | TPR@1e-3 | conditional@1e-6 | median log10 p | attempted | errors |", "|---|---|---|---|---|---|---|"]
    for attack, info in summary["attacks"].items():
        conditional = "—" if info["conditional_1e-6"] is None else f"{info['conditional_1e-6']['rate']:.2f}"
        lines.append(f"| {attack} | {info['tpr_1e-6']['rate']:.2f} | {info['tpr_1e-3']['rate']:.2f} | {conditional} | {info['median_log10_p']:.1f} | {info['attempted']} | {info['attack_errors']}/{info['detect_errors']} |")
    lines += ["", "## Nulls", "", "| Null | n | FPR@1e-2 | FPR@1e-3 | FPR@1e-6 |", "|---|---|---|---|---|"]
    for name, info in summary["nulls"].items():
        lines.append(f"| {name} | {info['fpr_1e-2']['n']} | {info['fpr_1e-2']['rate']:.3f} | {info['fpr_1e-3']['rate']:.3f} | {info['fpr_1e-6']['rate']:.3f} |")
    lines += ["", "## Clean detection vs distinct vertices (marked)", "", "| distinct vertices | n | median log10 p | TPR@1e-6 | TPR@1e-3 |", "|---|---|---|---|---|"]
    for label, info in by_bin.items():
        lines.append(f"| {label} | {info['n']} | {info['median_log10_p']:.1f} | {info['tpr_1e-6']:.2f} | {info['tpr_1e-3']:.2f} |")
    lines += ["", "## Distribution preservation (plain vs marked, Mann-Whitney U)", "", "| Statistic | plain median | marked median | p |", "|---|---|---|---|"]
    for field, info in preservation.items():
        lines.append(f"| {field} | {info['plain_median']:.1f} | {info['marked_median']:.1f} | {info['mannwhitney_p']:.3f} |")
    text = "\n".join(lines) + "\n"
    if args.markdown:
        args.markdown.write_text(text)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1) + "\n")
    print(text)


if __name__ == "__main__":
    main()
