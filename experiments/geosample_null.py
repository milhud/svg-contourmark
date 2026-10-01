"""Key-randomized null calibration for the geosample (inference-time) detector.

Vertex descriptors are extracted once per unmarked document and re-scored
under many independent keys with exactly the statistic used by
``contourmark.geosample.detect`` (global Gamma test and Bonferroni
per-contour test, combined with a factor of two).  Under H0 the reported
p-value must satisfy P[p <= a] <= a.

Inputs: a corpus manifest (``--corpus``, e.g. corpus/test.json of human
icons) and/or a samples directory written by iconshop_geosample.py
(``--samples-dir``; only plain, unmarked samples are used).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.special import gammaincc  # noqa: E402

from contourmark.geosample import GeoParameters, detect, document_descriptors, keyed_uniform  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LEVELS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]


def score(per_contour: list[list[tuple[int, int, int]]], key: bytes) -> float:
    """log10 p-value, identical to geosample.detect for precomputed descriptors."""
    distinct = sorted({d for contour in per_contour for d in contour})
    if not distinct:
        return 0.0
    scores = {d: -math.log(1 - keyed_uniform(key, d)) for d in distinct}
    global_p = float(gammaincc(len(distinct), sum(scores.values())))
    contour_ps = [float(gammaincc(len(u), sum(scores[d] for d in u))) for u in (set(c) for c in per_contour) if u]
    minimum = min(1.0, len(contour_ps) * min(contour_ps)) if contour_ps else 1.0
    return math.log10(max(min(1.0, 2 * min(global_p, minimum)), 1e-300))


def run(task: tuple[str, int]) -> tuple[str, list[float], int]:
    path, keys = task
    source = Path(path).read_bytes()
    try:
        per_contour = document_descriptors(source, GeoParameters())
    except Exception:
        return path, [], 0
    distinct = len({d for contour in per_contour for d in contour})
    if distinct == 0:
        return path, [], 0
    values = [score(per_contour, hashlib.sha256(f"geosample-null|{path}|{i}".encode()).digest()) for i in range(keys)]
    return path, values, distinct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--samples-dir", type=Path)
    parser.add_argument("--keys", type=int, default=300)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths: list[str] = []
    if args.corpus:
        paths += [str(ROOT / item["path"]) for item in json.loads(args.corpus.read_text())["items"]]
    if args.samples_dir:
        for line in (args.samples_dir / "samples.jsonl").read_text().splitlines():
            record = json.loads(line)
            if not record["marked"] and (args.samples_dir / record["file"]).exists():
                paths.append(str(args.samples_dir / record["file"]))
    if not paths:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"documents": 0, "tests": 0, "note": "no plain samples in this run"}, indent=1) + "\n")
        print("no plain samples; nothing to score")
        return
    # Consistency check: the fast re-scorer must equal the shipped detector.
    for path in paths[:5]:
        key = hashlib.sha256(b"consistency").digest()
        fast = score(document_descriptors(Path(path).read_bytes(), GeoParameters()), key)
        slow = detect(Path(path).read_bytes(), key)["log10_p_value"]
        assert abs(fast - slow) < 1e-9, (path, fast, slow)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(run, [(path, args.keys) for path in paths], chunksize=4)
    values = np.array([v for _, doc, _ in results for v in doc])
    distinct = [d for _, doc, d in results if doc]
    report = {
        "documents": len(paths),
        "documents_with_vertices": len(distinct),
        "distinct_vertices_median": float(np.median(distinct)) if distinct else 0.0,
        "keys_per_document": args.keys,
        "tests": int(values.size),
        "levels": {},
        "min_log10_p": float(values.min()) if values.size else 0.0,
    }
    for level in LEVELS:
        hits = int((values <= math.log10(level)).sum())
        report["levels"][str(level)] = {"rate": hits / values.size if values.size else float("nan"), "hits": hits, "bound": level}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
