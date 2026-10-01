"""Attack-suite evaluation of the inference-time (geosample) watermark.

Reads ``samples.jsonl`` and the SVG files written by iconshop_geosample.py,
applies every keyless attack in contourmark.attacks.standard_suite to each
marked and plain sample, and records the geosample detector's log10 p-value.
Composition partners are three other plain samples from the same run.
Output is one JSON line per sample; reruns resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

from contourmark import attacks as attack_lib  # noqa: E402
from contourmark.geometry import load_document  # noqa: E402
from contourmark.geosample import GeoParameters, detect as _detect  # noqa: E402
from run_identity import detector_settings, evaluation_key  # noqa: E402


def _context(directory: str):
    """Key, parameters and statistic recorded when the run was generated."""
    settings = detector_settings(Path(directory))
    params = GeoParameters(**settings["params"])
    key, wrong = evaluation_key(settings["key_label"]), evaluation_key(settings["key_label"], wrong=True)
    detect = lambda source, k: _detect(source, k, params, statistic=settings["statistic"], gamma=settings["gamma"])  # noqa: E731
    return key, wrong, detect


def run(task: tuple[dict, str, list[bytes], list[str] | None]) -> dict:
    record, directory, partners, only = task
    KEY, WRONG, detect = _context(directory)
    source = (Path(directory) / record["file"]).read_bytes()
    suite = attack_lib.standard_suite(partners or None)
    if only:
        suite = {name: suite[name] for name in only if name in suite}
    try:
        document = load_document(source)
        paths = len({id(c.element) for c in document.contours})
        contours = len(document.contours)
    except Exception:
        paths = contours = 0
    clean = detect(source, KEY)
    out = {
        "file": record["file"], "prompt": record["prompt"], "seed": record["seed"], "marked": record["marked"],
        "tokens": len(record.get("tokens", [])), "keyed_steps": record.get("keyed_steps", 0),
        "truncated": bool(record.get("truncated", False)), "run_id": record.get("run_id"),
        "paths": paths, "contours": contours,
        "vertices": clean.get("vertices", 0), "distinct_vertices": clean["distinct_vertices"],
        "wrong_key_log10_p": detect(source, WRONG)["log10_p_value"], "attacks": {},
    }
    for name, attack in suite.items():
        try:
            attacked = attack(source)
        except Exception as exc:
            out["attacks"][name] = {"attack_error": f"{type(exc).__name__}: {exc}"[:160]}
            continue
        try:
            result = detect(attacked, KEY)
            out["attacks"][name] = {"log10_p": result["log10_p_value"], "distinct_vertices": result["distinct_vertices"]}
        except Exception as exc:
            out["attacks"][name] = {"log10_p": 0.0, "distinct_vertices": 0, "detect_error": f"{type(exc).__name__}: {exc}"[:160]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attacks", nargs="*")
    args = parser.parse_args()
    records = [json.loads(line) for line in (args.samples_dir / "samples.jsonl").read_text().splitlines() if line.strip()]
    records = [r for r in records if (args.samples_dir / r["file"]).exists()]
    done = set()
    if args.output.exists():
        done = {json.loads(line)["file"] for line in args.output.read_text().splitlines() if line.strip()}
    plain = [r for r in records if not r["marked"]]
    tasks = []
    for index, record in enumerate(records):
        if record["file"] in done:
            continue
        others = [p for p in plain if p["prompt"] != record["prompt"]] or [p for p in plain if p["file"] != record["file"]]
        partners = [(args.samples_dir / others[(index + k * 7) % len(others)]["file"]).read_bytes() for k in range(3)] if others else []
        tasks.append((record, str(args.samples_dir), partners, args.attacks))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool, args.output.open("a") as stream:
        for count, result in enumerate(pool.imap_unordered(run, tasks, chunksize=1), start=1):
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            if count % 20 == 0 or count == len(tasks):
                print(f"{count}/{len(tasks)} samples, {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
