"""Informed (Kerckhoffs) removal attacks against the blind spectral watermark.

The attacker knows the algorithm and its public parameters but not the key:

* ``remark_random_key``: re-embed plain QIM with a fresh random key.  Every
  carried coefficient is moved to a random lattice, which is the natural
  keyless re-quantization attack on QIM.
* ``remark_2x`` / ``remark_4x``: the same with a 2x / 4x lattice step,
  which moves each coefficient further (more distortion, more removal).
* ``remark_same_scheme`` / ``remark_same_scheme_2x``: re-embed with the
  owner's own public parameters (``--params``, e.g. spread-transform dither
  modulation) and a random key, at 1x / 2x the owner's step.  With the
  default parameters the first is the same attack as ``remark_random_key``.

For each attack we record the owner's detection and the attack's own render
and boundary distortion relative to the marked file, so removal cost can be
compared with embedding cost.  Two benign transforms (SVGO defaults,
two-decimal rounding) are recorded alongside, with how often they change a
contour's (seed, selection mask) pair, which the spread-transform scheme
depends on.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import multiprocessing as mp
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from contourmark import attacks as attack_lib
from contourmark.geometry import load_document
from contourmark.metrics import boundary_distance
from contourmark.spectral import SpectralParameters, detect, embed, observe
from evaluate_blind import fidelity, item_key

ROOT = Path(__file__).resolve().parents[1]
ATTACKERS = ("remark_random_key", "remark_2x", "remark_4x", "remark_same_scheme", "remark_same_scheme_2x")
BENIGN = ("svgo_default", "round_2dp")


def _p95(original: bytes, changed: bytes) -> float | None:
    try:
        return boundary_distance(original, changed)["fraction_of_diagonal"]["p95"]
    except Exception:
        return None


def _owner(source: bytes, key: bytes, params: SpectralParameters) -> dict:
    """Owner's detection: p-value and the weighted mean lattice alignment (1 = on the lattice, 0 = unmarked)."""
    result = detect(source, key, params)
    return {"log10_p": result["log10_p_value"], "alignment": result["statistic"] / result["max_statistic"] if result["max_statistic"] > 0 else None}


def _carriers(source: bytes, params: SpectralParameters) -> collections.Counter:
    return collections.Counter((o.seed, tuple(bool(v) for v in o.selected)) for o in observe(load_document(source), params))


def run(task: tuple[dict, dict]) -> dict:
    item, overrides = task
    source = (ROOT / item["path"]).read_bytes()
    key = item_key(item["path"])
    params = SpectralParameters(**overrides)
    try:
        marked, report = embed(source, key, params)
    except Exception as exc:
        return {"path": item["path"], "source": item["source"], "error": str(exc)[:200]}
    clean = _owner(marked, key, params)
    record = {"path": item["path"], "source": item["source"], "clean": clean["log10_p"], "clean_alignment": clean["alignment"]}
    record["embedding_fidelity"] = fidelity(source, marked)
    record["embedding_boundary_p95"] = _p95(source, marked)
    record["contour_status"] = dict(collections.Counter(c["status"] for c in report["contours"]))
    attacker_key = hashlib.sha256(b"attacker|" + item["path"].encode()).digest()
    # The attacker does not honour the embedder's own displacement cap.
    uncapped = {"max_relative_displacement": 10.0}
    attacks = {
        "remark_random_key": lambda: embed(marked, attacker_key, SpectralParameters(**uncapped))[0],
        "remark_2x": lambda: embed(marked, attacker_key, SpectralParameters(delta=2 * params.delta, **uncapped))[0],
        "remark_4x": lambda: embed(marked, attacker_key, SpectralParameters(delta=4 * params.delta, **uncapped))[0],
        "remark_same_scheme": lambda: embed(marked, attacker_key, replace(params, **uncapped))[0],
        "remark_same_scheme_2x": lambda: embed(marked, attacker_key, replace(params, delta=2 * params.delta, **uncapped))[0],
    }
    for name, attack in attacks.items():
        try:
            attacked = attack()
            record[name] = {
                **_owner(attacked, key, params),
                "fidelity_vs_marked": fidelity(marked, attacked),
                "boundary_p95_vs_marked": _p95(marked, attacked),
            }
        except Exception as exc:
            record[name] = {"error": str(exc)[:200]}
    suite = attack_lib.standard_suite()
    before = _carriers(marked, params)
    for name in BENIGN:
        try:
            attacked = suite[name](marked)
            kept = sum((before & _carriers(attacked, params)).values())
            record[name] = {**_owner(attacked, key, params), "carriers": sum(before.values()), "carriers_unchanged": kept}
        except Exception as exc:
            record[name] = {"error": str(exc)[:200]}
    return record


def _median(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--params", type=str, default="{}", help='JSON overrides for the owner\'s SpectralParameters, e.g. \'{"spread": 2}\'')
    args = parser.parse_args()
    overrides = json.loads(args.params)
    owner = SpectralParameters(**overrides)
    owner.validate()
    items = json.loads(args.corpus.read_text())["items"]
    chosen: dict[str, list] = {}
    for item in items:
        chosen.setdefault(item["source"], [])
        if len(chosen[item["source"]]) < args.per_source:
            chosen[item["source"]].append(item)
    tasks = [(item, overrides) for group in chosen.values() for item in group]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        records = list(pool.imap_unordered(run, tasks))
    records.sort(key=lambda r: r["path"])
    embedded = [r for r in records if "clean" in r]
    threshold = float(np.log10(owner.threshold))
    detectable = [r for r in embedded if r["clean"] <= threshold]
    statuses: collections.Counter = collections.Counter()
    for r in embedded:
        statuses.update(r["contour_status"])
    summary = {
        "owner_parameters": asdict(owner), "items": len(records), "embed_errors": len(records) - len(embedded),
        "clean_detectable": len(detectable), "clean_detection_rate": len(detectable) / len(records) if records else None,
        "median_clean_log10_p": _median([r["clean"] for r in embedded]),
        "median_clean_alignment": _median([r["clean_alignment"] for r in detectable]),
        "contour_status": dict(statuses),
    }
    for name in ATTACKERS + BENIGN:
        rows = [r[name] for r in detectable if "log10_p" in r.get(name, {})]
        summary[name] = {
            "evaluated": len(rows),
            "survival_rate": float(np.mean([r["log10_p"] <= threshold for r in rows])) if rows else None,
            "median_log10_p": _median([r["log10_p"] for r in rows]),
            "median_alignment": _median([r["alignment"] for r in rows]),
        }
        if name in ATTACKERS:
            summary[name]["attack_psnr_1024_median"] = _median([r["fidelity_vs_marked"]["psnr_1024"] for r in rows])
            summary[name]["attack_boundary_p95_median"] = _median([r["boundary_p95_vs_marked"] for r in rows])
            # Backwards-compatible key used by earlier result files.
            summary[name]["survival_rate_at_1e-6"] = summary[name]["survival_rate"]
        else:
            total = sum(r["carriers"] for r in rows)
            summary[name]["seed_and_mask_unchanged"] = sum(r["carriers_unchanged"] for r in rows) / total if total else None
    summary["embedding_psnr_1024_median"] = _median([r["embedding_fidelity"]["psnr_1024"] for r in detectable])
    summary["embedding_boundary_p95_median"] = _median([r["embedding_boundary_p95"] for r in detectable])
    summary["embedding_boundary_p95_median_all"] = _median([r["embedding_boundary_p95"] for r in embedded])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "records": records}, indent=1) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
