"""Large-scale null calibration of the blind detector.

For each unmarked corpus document, geometric observations are computed once
and scored under many independent random keys.  Under H0 the reported p-value
must satisfy P[p <= a] <= a for every a.  This checks the implementation of
the exact bound (and the PRF idealization) far below the 1e-2 level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from contourmark.geometry import load_document
from contourmark.spectral import SpectralParameters, observe, score

ROOT = Path(__file__).resolve().parents[1]
LEVELS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]


def run(task: tuple[str, int]) -> list[float]:
    path, keys = task
    params = SpectralParameters()
    observations = observe(load_document((ROOT / path).read_bytes()), params)
    if not observations:
        return []
    values = []
    for index in range(keys):
        key = hashlib.sha256(f"null|{path}|{index}".encode()).digest()
        values.append(score(observations, key, params)["log10_p_value"])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--keys", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = json.loads(args.corpus.read_text())["items"]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        per_document = pool.map(run, [(item["path"], args.keys) for item in items])
    values = np.array([v for doc in per_document for v in doc])
    documents = sum(1 for doc in per_document if doc)
    report = {"documents_with_carriers": documents, "tests": int(values.size), "keys_per_document": args.keys, "levels": {}}
    for level in LEVELS:
        hits = int((values <= np.log10(level)).sum())
        report["levels"][str(level)] = {"rate": hits / values.size, "hits": hits, "bound": level}
    report["min_log10_p"] = float(values.min())
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
