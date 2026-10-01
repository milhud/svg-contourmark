"""Distortion/robustness trade-off across lattice steps (dev split)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

rows = {}
for path in sorted(Path("results/blind").glob("dev_delta_*.jsonl")):
    delta = float(path.stem.split("_")[-1])
    entries = [json.loads(l)["methods"]["spectral"] for l in path.read_text().splitlines() if l.strip()]
    entries = [e for e in entries if "attacks" in e]
    detectable = [e for e in entries if e["attacks"]["identity"]["log10_p"] <= -6]
    row = {"n": len(entries), "clean": len(detectable) / len(entries)}
    for attack in ("svgo_default", "svgo_p2", "round_2dp", "round_1dp", "noise_0.1pct", "noise_0.3pct", "polyline_0.2pct", "revectorize_1024", "aspect_1.2"):
        row[attack] = float(np.mean([e["attacks"][attack]["log10_p"] <= -6 for e in detectable if attack in e["attacks"]]))
    row["psnr_1024"] = float(np.median([e["fidelity"]["psnr_1024"] for e in entries if "fidelity" in e]))
    row["ssim_1024"] = float(np.median([e["fidelity"]["ssim_1024"] for e in entries if "fidelity" in e]))
    row["max_disp_diag"] = float(np.median([e.get("max_curve_displacement_over_diagonal", 0) for e in entries]))
    row["svgo_bytes_x"] = float(np.median([e["svgo_bytes"] / e["source_svgo_bytes"] for e in entries if "svgo_bytes" in e]))
    rows[delta] = row
Path("results/blind/dev_delta_summary.json").write_text(json.dumps(rows, indent=1) + "\n")
keys = list(next(iter(rows.values())))
print("delta | " + " | ".join(keys))
for delta, row in rows.items():
    print(f"{delta} | " + " | ".join(f"{row[k]:.3f}" if isinstance(row[k], float) else str(row[k]) for k in keys))
