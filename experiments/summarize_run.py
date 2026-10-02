"""One table for a whole run directory (all sampler configurations).

Reads ``<name>_attacks*.summary.json``, ``<name>/samples*.run.json`` and
``clip_quality.json`` from a run directory written by colab/run_single_gpu.sh
or colab/run2.sh, and prints one row per configuration: sampler, top-p,
detection, survival under key attacks, false positives, truncation, and CLIP
prompt-retrieval accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_identity import detector_settings

ATTACKS = ["svgo_default", "round_1dp", "rotate_30", "scale_0.37", "subdivide", "compose_grid", "noise_0.1pct"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--suffix", default="_attacks", help="summary file suffix, e.g. _attacks_local")
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    quality = {}
    clip_path = args.run / "clip_quality.json"
    if clip_path.exists():
        quality = {Path(k).name: v for k, v in json.loads(clip_path.read_text())["runs"].items()}
    rows = []
    for summary_path in sorted(args.run.glob(f"*{args.suffix}.summary.json")):
        name = summary_path.name[: -len(f"{args.suffix}.summary.json")]
        summary = json.loads(summary_path.read_text())
        settings = detector_settings(args.run / name)
        sampler = settings["sampler"]
        label = sampler.get("mode", "gumbel") + ("-" + sampler.get("reuse", "mask") if sampler.get("mode", "gumbel") == "gumbel" else f"-{sampler.get('delta'):g}")
        label = settings["params"].get("scheme", "vertex") + "/" + label
        identity = summary["attacks"].get("identity", {})
        row = {
            "name": name, "sampler": label, "top_p": settings["top_p"], "key": settings["key_label"],
            "marked": summary["marked"], "plain": summary["plain"],
            "truncated_marked": summary.get("generation", {}).get("marked_truncated"),
            "distinct_median": summary.get("distinct_vertices_marked", {}).get("median"),
            "tpr_1e-6": identity.get("tpr_1e-6", {}).get("rate") if identity.get("tpr_1e-6") else None,
            "tpr_1e-3": identity.get("tpr_1e-3", {}).get("rate") if identity.get("tpr_1e-3") else None,
            "median_log10_p": identity.get("median_log10_p"),
            "attacks": {a: (summary["attacks"][a]["tpr_1e-6"]["rate"] if summary["attacks"].get(a, {}).get("tpr_1e-6") else None) for a in ATTACKS},
            "attack_errors": sum(v.get("attack_errors", 0) for v in summary["attacks"].values()),
            "fpr_plain_1e-3": summary.get("nulls", {}).get("plain_with_key", {}).get("fpr_1e-3", {}).get("rate"),
            "fpr_wrong_key_1e-3": summary.get("nulls", {}).get("marked_wrong_key", {}).get("fpr_1e-3", {}).get("rate"),
            "clip_top1_marked": quality.get(name, {}).get("marked", {}).get("top1", {}).get("mean"),
            "clip_top1_plain": quality.get(name, {}).get("plain", {}).get("top1", {}).get("mean"),
        }
        rows.append(row)
    rows.sort(key=lambda r: (r["sampler"], r["top_p"] or 0, r["key"]))
    cell = lambda v, f="{:.2f}": "—" if v is None else f.format(v)  # noqa: E731
    lines = [f"# Run summary: {args.run.name}", "",
             "Detection columns are the fraction of marked samples with p <= 1e-6 (clean, then after each transformation).", "",
             "| Config | sampler | top-p | key | n | clean 1e-6 | clean 1e-3 | SVGO | round 1dp | rotate | scale | subdivide | compose | noise 0.1% | FPR plain 1e-3 | CLIP top-1 marked / plain | truncated | descriptors | attack errors |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        a = r["attacks"]
        lines.append(
            f"| {r['name']} | {r['sampler']} | {cell(r['top_p'], '{:g}')} | {r['key']} | {r['marked']} | {cell(r['tpr_1e-6'])} | {cell(r['tpr_1e-3'])} | "
            f"{cell(a['svgo_default'])} | {cell(a['round_1dp'])} | {cell(a['rotate_30'])} | {cell(a['scale_0.37'])} | {cell(a['subdivide'])} | {cell(a['compose_grid'])} | {cell(a['noise_0.1pct'])} | "
            f"{cell(r['fpr_plain_1e-3'], '{:.3f}')} | {cell(r['clip_top1_marked'])} / {cell(r['clip_top1_plain'])} | {cell(r['truncated_marked'], '{}')} | {cell(r['distinct_median'], '{:.0f}')} | {r['attack_errors']} |")
    text = "\n".join(lines) + "\n"
    if args.markdown:
        args.markdown.write_text(text)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n")
    print(text)


if __name__ == "__main__":
    main()
