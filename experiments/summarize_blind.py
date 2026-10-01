"""Summarize evaluate_blind.py JSON lines into tables and a JSON summary."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

THRESHOLD = -6.0  # log10 p decision threshold (FPR 1e-6)
ATTACK_GROUPS = {
    "Optimizers": ["svgo_default", "svgo_multipass", "svgo_p2", "scour_default", "scour_p3", "picosvg"],
    "Precision": ["round_2dp", "round_1dp", "svgo_p1", "round_0dp"],
    "Similarity": ["translate_5pct", "scale_0.37", "scale_3", "rotate_30", "rotate_90", "mirror", "group_transform"],
    "Non-similarity": ["aspect_1.2"],
    "Structure": ["reorder", "reverse", "restart", "subdivide", "merge_paths", "split_subpaths"],
    "Removal/crop": ["delete_25pct", "delete_50pct", "crop_half", "compose_grid"],
    "Noise": ["noise_0.1pct", "noise_0.3pct", "noise_1pct"],
    "Re-drawing": ["polyline_0.05pct", "polyline_0.2pct", "revectorize_512", "revectorize_1024"],
}


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def rate(values: list[float], threshold: float = THRESHOLD) -> dict:
    hits = sum(v <= threshold for v in values)
    low, high = wilson(hits, len(values))
    return {"rate": hits / len(values) if values else float("nan"), "n": len(values), "hits": hits, "ci95": [low, high]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    # Later files override earlier ones per (asset, method), so a method can be
    # re-run into its own file and merged.
    merged: dict[str, dict] = {}
    for path in args.results:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            base = merged.setdefault(record["path"], {**record, "methods": {}})
            base["methods"].update(record["methods"])
    records = list(merged.values())
    methods = sorted({m for r in records for m in r["methods"]}, key=["spectral", "fd_vertex", "bezier_split", "numeric_lsb", "xml_comment", "metadata_element"].index)
    attacks = sorted({a for r in records for e in r["methods"].values() for a in e.get("attacks", {})})
    summary: dict = {"items": len(records), "threshold_log10": THRESHOLD, "methods": {}}
    lines = [f"# Blind watermark evaluation ({len(records)} assets, decision threshold p <= 1e-6)", ""]
    for method in methods:
        entries = [r["methods"].get(method, {}) for r in records]
        applicable = [e for e in entries if "attacks" in e]
        info: dict = {"embedded": len(applicable), "embed_failures": len(entries) - len(applicable)}
        clean = [e["attacks"]["identity"]["log10_p"] for e in applicable if "identity" in e["attacks"]]
        info["clean_detection"] = rate(clean)
        for label in ("null_original", "null_wrong_key"):
            values = [e[label] for e in applicable if label in e]
            info[label] = {f"p<=1e-{k}": rate(values, -k) for k in (2, 3, 6)}
        per_attack = {}
        for attack in attacks:
            values = [e["attacks"][attack].get("log10_p", 0.0) for e in applicable if attack in e["attacks"] and "attack_error" not in e["attacks"][attack]]
            if values:
                per_attack[attack] = rate(values)
        info["attacks"] = per_attack
        fid = [e["fidelity"] for e in applicable if "fidelity" in e]
        if fid:
            info["fidelity_median"] = {k: float(np.median([f[k] for f in fid])) for k in fid[0]}
            info["fidelity_p05"] = {k: float(np.percentile([f[k] for f in fid], 5 if "psnr" in k or "ssim" in k else 95)) for k in fid[0]}
        for size_key in ("bytes", "gzip_bytes", "svgo_bytes"):
            base = {"bytes": "bytes", "gzip_bytes": "source_gzip_bytes", "svgo_bytes": "source_svgo_bytes"}[size_key]
            ratios = []
            for record, entry in zip(records, entries):
                if size_key in entry:
                    denominator = record["bytes"] if base == "bytes" else entry.get(base)
                    if denominator:
                        ratios.append(entry[size_key] / denominator)
            if ratios:
                info[f"{size_key}_ratio_median"] = float(np.median(ratios))
                info[f"{size_key}_ratio_p90"] = float(np.percentile(ratios, 90))
        times = [e["embed_seconds"] for e in applicable if "embed_seconds" in e]
        if times:
            info["embed_seconds_median"] = float(np.median(times))
            info["detect_seconds_median"] = float(np.median([e["detect_seconds"] for e in applicable if "detect_seconds" in e]))
        if method == "spectral":
            by_source = defaultdict(list)
            for record, entry in zip(records, entries):
                if "attacks" in entry:
                    by_source[record["source"]].append(entry)
            info["by_source"] = {
                source: {
                    "clean": rate([e["attacks"]["identity"]["log10_p"] for e in group]),
                    "svgo_default": rate([e["attacks"]["svgo_default"]["log10_p"] for e in group if "svgo_default" in e["attacks"]]),
                    "round_2dp": rate([e["attacks"]["round_2dp"]["log10_p"] for e in group if "round_2dp" in e["attacks"]]),
                    "median_clean_log10_p": float(np.median([e["attacks"]["identity"]["log10_p"] for e in group])),
                    "median_marked_contours": float(np.median([e.get("marked_contours", 0) for e in group])),
                    "psnr_1024_median": float(np.median([e["fidelity"]["psnr_1024"] for e in group if "fidelity" in e])),
                }
                for source, group in sorted(by_source.items())
            }
            info["max_curve_displacement_over_diagonal_median"] = float(np.median([e.get("max_curve_displacement_over_diagonal", 0) for e in applicable]))
            selected = sum(e.get("selected", 0) for e in applicable)
            info["unmarked_selected_fraction"] = sum(e.get("unmarked_selected", 0) for e in applicable) / max(selected, 1)
        summary["methods"][method] = info

    # Markdown tables
    lines.append("## Detection rate (TPR at p <= 1e-6) by attack")
    lines.append("")
    lines.append("| Attack | " + " | ".join(methods) + " |")
    lines.append("|---" * (len(methods) + 1) + "|")
    ordered = ["identity"] + [a for group in ATTACK_GROUPS.values() for a in group]
    for attack in ordered + [a for a in attacks if a not in ordered]:
        cells = []
        for method in methods:
            entry = summary["methods"][method]["attacks"].get(attack)
            cells.append("—" if entry is None else f"{entry['rate']:.2f}")
        lines.append(f"| {attack} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Nulls, fidelity, size")
    lines.append("")
    lines.append("| Method | embedded | FPR@1e-2 orig | FPR@1e-6 orig | FPR@1e-6 wrong key | PSNR@1024 med | SSIM@1024 med | bytes x | SVGO bytes x | gzip x |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for method in methods:
        info = summary["methods"][method]
        fidelity = info.get("fidelity_median", {})
        lines.append(
            f"| {method} | {info['embedded']} | {info['null_original']['p<=1e-2']['rate']:.3f} | {info['null_original']['p<=1e-6']['rate']:.3f} | "
            f"{info['null_wrong_key']['p<=1e-6']['rate']:.3f} | {fidelity.get('psnr_1024', float('nan')):.1f} | {fidelity.get('ssim_1024', float('nan')):.4f} | "
            f"{info.get('bytes_ratio_median', float('nan')):.2f} | {info.get('svgo_bytes_ratio_median', float('nan')):.2f} | {info.get('gzip_bytes_ratio_median', float('nan')):.2f} |"
        )
    if "spectral" in summary["methods"] and "by_source" in summary["methods"]["spectral"]:
        lines += ["", "## Spectral watermark by source", "", "| Source | n | clean TPR | SVGO TPR | round-2dp TPR | median log10 p | median marked contours | PSNR@1024 |", "|---|---|---|---|---|---|---|---|"]
        for source, info in summary["methods"]["spectral"]["by_source"].items():
            lines.append(f"| {source} | {info['clean']['n']} | {info['clean']['rate']:.2f} | {info['svgo_default']['rate']:.2f} | {info['round_2dp']['rate']:.2f} | {info['median_clean_log10_p']:.1f} | {info['median_marked_contours']:.0f} | {info['psnr_1024_median']:.1f} |")
    text = "\n".join(lines) + "\n"
    if args.markdown:
        args.markdown.write_text(text)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1) + "\n")
    print(text)


if __name__ == "__main__":
    main()
