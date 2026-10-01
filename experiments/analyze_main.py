"""Secondary statistics for the paper from the main test run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from summarize_blind import ATTACK_GROUPS, wilson

records = [json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
spectral = [(r, r["methods"]["spectral"]) for r in records if "attacks" in r["methods"].get("spectral", {})]
detectable = [(r, e) for r, e in spectral if e["attacks"]["identity"]["log10_p"] <= -6]
out: dict = {"assets": len(spectral), "clean_detectable": len(detectable)}
out["conditional_survival"] = {}
for attack in [a for g in ATTACK_GROUPS.values() for a in g]:
    values = [e["attacks"][attack].get("log10_p", 0) <= -6 for _, e in detectable if attack in e["attacks"]]
    hits = sum(values)
    out["conditional_survival"][attack] = {"rate": hits / len(values), "ci95": wilson(hits, len(values)), "n": len(values)}
failures = [(r, e) for r, e in spectral if e["attacks"]["identity"]["log10_p"] > -6]
out["clean_failures"] = {
    "count": len(failures),
    "no_marked_contour": sum(e.get("marked_contours", 0) == 0 for _, e in failures),
    "one_marked_contour": sum(e.get("marked_contours", 0) == 1 for _, e in failures),
    "median_log10_p": float(np.median([e["attacks"]["identity"]["log10_p"] for _, e in failures])),
    "p_below_1e-3": sum(e["attacks"]["identity"]["log10_p"] <= -3 for _, e in failures),
    "by_source": {s: sum(r["source"] == s for r, _ in failures) for s in sorted({r["source"] for r, _ in spectral})},
}
disp = np.array([e.get("max_curve_displacement_over_diagonal", 0) for _, e in spectral])
# Pixels at a square render of side S: diagonal = S*sqrt(2) when the drawing spans the canvas.
out["max_displacement_px"] = {f"render_{s}": {"median": float(np.median(disp) * s * 2 ** 0.5), "p95": float(np.percentile(disp, 95) * s * 2 ** 0.5)} for s in (64, 256, 1024)}
out["max_displacement_over_diagonal"] = {"median": float(np.median(disp)), "p95": float(np.percentile(disp, 95))}
out["timing_seconds"] = {
    "embed_median": float(np.median([e["embed_seconds"] for _, e in spectral])),
    "embed_p95": float(np.percentile([e["embed_seconds"] for _, e in spectral], 95)),
    "detect_median": float(np.median([e["detect_seconds"] for _, e in spectral])),
    "detect_p95": float(np.percentile([e["detect_seconds"] for _, e in spectral], 95)),
}
out["bytes"] = {
    "raw_ratio_median": float(np.median([e["bytes"] / r["bytes"] for r, e in spectral])),
    "svgo_ratio_median": float(np.median([e["svgo_bytes"] / e["source_svgo_bytes"] for _, e in spectral if "svgo_bytes" in e])),
    "added_bytes_after_svgo_median": float(np.median([e["svgo_bytes"] - e["source_svgo_bytes"] for _, e in spectral if "svgo_bytes" in e])),
    "gzip_ratio_median": float(np.median([e["gzip_bytes"] / e["source_gzip_bytes"] for _, e in spectral])),
}
out["selection_mismatch_fraction"] = sum(e.get("unmarked_selected", 0) for _, e in spectral) / max(1, sum(e.get("selected", 0) for _, e in spectral))
out["ssim_256_median"] = float(np.median([e["fidelity"]["ssim_256"] for _, e in spectral if "fidelity" in e]))
out["psnr_256_median"] = float(np.median([e["fidelity"]["psnr_256"] for _, e in spectral if "fidelity" in e]))
Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "conditional_survival"}, indent=1))
print({k: round(v["rate"], 3) for k, v in out["conditional_survival"].items()})
