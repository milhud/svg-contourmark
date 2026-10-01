"""Dev-split sweep of capacity parameters (clean detection, SVGO, distortion)."""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

from contourmark.attacks import _SVGO, round_absolute
from contourmark.spectral import SpectralParameters, detect, embed

ROOT = Path(__file__).resolve().parents[1]
KEY = bytes(range(32))
CONFIGS = {
    "default": {},
    "k12": {"k_count": 12},
    "gain0.3": {"min_gain": 0.3, "embed_margin": 0.7},
    "k12_gain0.3": {"k_count": 12, "min_gain": 0.3, "embed_margin": 0.7},
    "delta0.006": {"delta": 0.006},
    "k12_delta0.006": {"k_count": 12, "delta": 0.006},
}


def run(task):
    path, name = task
    params = SpectralParameters(**CONFIGS[name])
    source = (ROOT / path).read_bytes()
    try:
        marked, report = embed(source, KEY, params)
    except Exception:
        return name, None
    rms = [c["curve_rms_over_radius"] for c in report["contours"]]
    return name, {
        "clean": detect(marked, KEY, params)["log10_p_value"],
        "svgo": detect(_SVGO(marked), KEY, params)["log10_p_value"],
        "round2": detect(round_absolute(2)(marked), KEY, params)["log10_p_value"],
        "rms": float(np.median(rms)) if rms else 0.0,
        "bytes": len(marked) / len(source),
    }


def main() -> None:
    items = json.loads((ROOT / "experiments/corpus/dev.json").read_text())["items"]
    tasks = [(item["path"], name) for item in items for name in CONFIGS]
    results: dict[str, list] = {name: [] for name in CONFIGS}
    with mp.get_context("spawn").Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
        for name, value in pool.imap_unordered(run, tasks, chunksize=4):
            if value:
                results[name].append(value)
    report = {}
    for name, rows in results.items():
        report[name] = {key: float(np.mean([r[key] <= -6 for r in rows])) for key in ("clean", "svgo", "round2")}
        report[name].update({"median_rms": float(np.median([r["rms"] for r in rows])), "median_bytes": float(np.median([r["bytes"] for r in rows])), "n": len(rows)})
        print(name, report[name], flush=True)
    (ROOT / "experiments/results/blind/dev_capacity_sweep.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
