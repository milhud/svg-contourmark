"""Paper figures from evaluate_blind.py results (static PDF for LaTeX)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_blind import ATTACK_GROUPS

# Reference categorical slots 1-3 (validated all-pairs for three series).
SERIES = {"spectral": "#2a78d6", "numeric_lsb": "#eb6834", "xml_comment": "#1baf7a"}
LABELS = {"spectral": "ContourMark (blind spectral)", "numeric_lsb": "Coordinate LSB", "xml_comment": "XML comment / metadata"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelsize=7)
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)


def group_rates(records: list[dict], method: str) -> list[float]:
    rates = []
    for attacks in ATTACK_GROUPS.values():
        hits = total = 0
        for record in records:
            entry = record["methods"].get(method, {})
            if "attacks" not in entry or entry["attacks"].get("identity", {}).get("log10_p", 0) > -6:
                continue  # conditional on clean detectability
            for attack in attacks:
                value = entry["attacks"].get(attack, {})
                if "log10_p" in value:
                    total += 1
                    hits += value["log10_p"] <= -6
        rates.append(hits / total if total else 0.0)
    return rates


def figure_attacks(records: list[dict], output: Path) -> None:
    groups = list(ATTACK_GROUPS)
    fig, axis = plt.subplots(figsize=(3.4, 3.2))
    height = 0.26
    y = np.arange(len(groups))
    for offset, method in enumerate(SERIES):
        rates = group_rates(records, method)
        axis.barh(y + (offset - 1) * height, rates, height=height * 0.9, color=SERIES[method], label=LABELS[method], edgecolor="white", linewidth=0.5)
    axis.set_yticks(y, groups, color=INK, fontsize=7)
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("Detection rate at p ≤ 1e-6 (clean-detectable assets)", fontsize=7, color=MUTED)
    style(axis)
    axis.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=1, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def figure_null(records: list[dict], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(3.4, 2.4))
    for label, key, color in (("Unmarked original, owner key", "null_original", SERIES["spectral"]), ("Marked, wrong key", "null_wrong_key", SERIES["numeric_lsb"])):
        values = np.array([10 ** r["methods"]["spectral"][key] for r in records if key in r["methods"].get("spectral", {})])
        values = np.sort(values)
        empirical = np.arange(1, len(values) + 1) / len(values)
        axis.step(values, empirical, where="post", color=color, linewidth=2, label=f"{label} (n={len(values)})")
    grid = np.logspace(-4, 0, 50)
    axis.plot(grid, grid, color=MUTED, linewidth=1, linestyle="--", label="Uniform (exact calibration)")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1e-4, 1)
    axis.set_ylim(1e-3, 1)
    axis.set_xlabel("p-value threshold α", fontsize=7, color=MUTED)
    axis.set_ylabel("Fraction with p ≤ α", fontsize=7, color=MUTED)
    style(axis)
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.legend(fontsize=6, frameon=False, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    merged: dict[str, dict] = {}
    for path in args.results:
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                merged.setdefault(record["path"], {**record, "methods": {}})["methods"].update(record["methods"])
    records = list(merged.values())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_attacks(records, args.out_dir / "attack_groups.pdf")
    figure_attacks(records, args.out_dir / "attack_groups.png")
    figure_null(records, args.out_dir / "null_calibration.pdf")
    figure_null(records, args.out_dir / "null_calibration.png")
    print("wrote figures to", args.out_dir)


if __name__ == "__main__":
    main()
