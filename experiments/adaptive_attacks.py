"""Informed (Kerckhoffs) removal attacks against the blind spectral watermark.

The attacker knows the algorithm and its public parameters but not the key:

* ``remark_random_key``: re-embed with a fresh random key.  Every carried
  coefficient is moved to a random lattice, which is the natural keyless
  re-quantization attack on QIM.
* ``remark_2x`` / ``remark_4x``: the same with a 2x / 4x lattice step,
  which moves each coefficient further (more distortion, more removal).

For each attack we record the owner's detection and the attack's own render
distortion relative to the marked file, so removal cost can be compared with
embedding cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from contourmark.spectral import SpectralParameters, detect, embed
from evaluate_blind import fidelity, item_key

ROOT = Path(__file__).resolve().parents[1]


def run(item: dict) -> dict:
    source = (ROOT / item["path"]).read_bytes()
    key = item_key(item["path"])
    params = SpectralParameters()
    marked, _ = embed(source, key, params)
    record = {"path": item["path"], "source": item["source"], "clean": detect(marked, key, params)["log10_p_value"]}
    record["embedding_fidelity"] = fidelity(source, marked)
    attacker_key = hashlib.sha256(b"attacker|" + item["path"].encode()).digest()
    # The attacker does not honour the embedder's own displacement cap.
    uncapped = {"max_relative_displacement": 10.0}
    attacks = {
        "remark_random_key": lambda: embed(marked, attacker_key, SpectralParameters(**uncapped))[0],
        "remark_2x": lambda: embed(marked, attacker_key, SpectralParameters(delta=2 * params.delta, **uncapped))[0],
        "remark_4x": lambda: embed(marked, attacker_key, SpectralParameters(delta=4 * params.delta, **uncapped))[0],
    }
    for name, attack in attacks.items():
        try:
            attacked = attack()
            record[name] = {"log10_p": detect(attacked, key, params)["log10_p_value"], "fidelity_vs_marked": fidelity(marked, attacked)}
        except Exception as exc:
            record[name] = {"error": str(exc)[:200]}
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    items = json.loads(args.corpus.read_text())["items"]
    chosen: dict[str, list] = {}
    for item in items:
        chosen.setdefault(item["source"], [])
        if len(chosen[item["source"]]) < args.per_source:
            chosen[item["source"]].append(item)
    tasks = [item for group in chosen.values() for item in group]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        records = list(pool.imap_unordered(run, tasks))
    detectable = [r for r in records if r["clean"] <= -6]
    summary = {"items": len(records), "clean_detectable": len(detectable)}
    for name in ("remark_random_key", "remark_2x", "remark_4x"):
        rows = [r[name] for r in detectable if "log10_p" in r.get(name, {})]
        summary[name] = {
            "survival_rate_at_1e-6": float(np.mean([r["log10_p"] <= -6 for r in rows])) if rows else None,
            "median_log10_p": float(np.median([r["log10_p"] for r in rows])) if rows else None,
            "attack_psnr_1024_median": float(np.median([r["fidelity_vs_marked"]["psnr_1024"] for r in rows])) if rows else None,
        }
    summary["embedding_psnr_1024_median"] = float(np.median([r["embedding_fidelity"]["psnr_1024"] for r in detectable]))
    args.output.write_text(json.dumps({"summary": summary, "records": records}, indent=1) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
