"""Corpus-scale evaluation of the blind spectral watermark and baselines.

For every corpus item and method: embed, measure fidelity and size, run the
keyless attack suite, and record the detector's log10 p-value.  Nulls are
measured on the unmarked original with the item key and on the marked file
with a wrong key.  Output is JSON lines, one per item; reruns resume.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import multiprocessing as mp
import re
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from contourmark import attacks as attack_lib
from contourmark import baselines as simple
from contourmark import vector_baselines as vb
from contourmark.geometry import load_document
from contourmark.spectral import SpectralParameters, detect, embed

ROOT = Path(__file__).resolve().parents[1]
MASTER = hashlib.sha256(b"contourmark-evaluation-master-key-v1").digest()
_NUMBER = re.compile(rb"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[Ee][-+]?\d+)?")


def item_key(path: str, label: str = "owner", shared: bool = False) -> bytes:
    """Per-asset keys by default; ``shared`` models one production key for all assets."""
    return hashlib.sha256(MASTER + (f"|{label}|shared" if shared else f"|{label}|{path}").encode()).digest()


# ---------------------------------------------------------------------------
# Fidelity


def render(source: bytes, size: int) -> np.ndarray:
    png = attack_lib.render_png(source, size)
    return np.asarray(Image.open(io.BytesIO(png)).convert("L"), dtype=np.float64) / 255.0


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = gaussian_filter(a, 1.5), gaussian_filter(b, 1.5)
    var_a = gaussian_filter(a * a, 1.5) - mu_a ** 2
    var_b = gaussian_filter(b * b, 1.5) - mu_b ** 2
    cov = gaussian_filter(a * b, 1.5) - mu_a * mu_b
    value = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return float(value.mean())


def fidelity(original: bytes, marked: bytes) -> dict:
    out = {}
    for size in (256, 1024):
        a, b = render(original, size), render(marked, size)
        diff = np.abs(a - b)
        mse = float((diff ** 2).mean())
        out[f"mae_{size}"] = float(diff.mean())
        out[f"psnr_{size}"] = 99.0 if mse == 0 else float(10 * math.log10(1 / mse))
        out[f"ssim_{size}"] = ssim(a, b)
        out[f"max_pixel_diff_{size}"] = float(diff.max())
    return out


# ---------------------------------------------------------------------------
# Methods: (embed(source, key) -> marked, detect(svg, key) -> log10 p)


def _numeric_chips(source: bytes) -> int:
    count = sum(len(_NUMBER.findall(m.group(2))) for m in re.finditer(rb"\bd\s*=\s*(['\"])(.*?)\1", source, re.DOTALL))
    return max(8, min(64, count))


def _numeric_embed(source: bytes, key: bytes) -> bytes:
    return simple.numeric_lsb_embed(source, key, "eval", chips=_numeric_chips(source))


def _numeric_detect(source: bytes, key: bytes, chips: int) -> float:
    result = simple.numeric_lsb_detect(source, key, "eval", chips=chips)
    return math.log10(max(result["p_value"], 1e-300))


def _comment_embed(source: bytes, key: bytes) -> bytes:
    return simple.metadata_embed(source, hashlib.sha256(key).hexdigest()[:32])


def _comment_detect(source: bytes, key: bytes) -> float:
    return -300.0 if simple.metadata_detect(source, hashlib.sha256(key).hexdigest()[:32]) else 0.0


def methods(params: SpectralParameters, visibility: str = "strict") -> dict:
    def spectral_detect(source: bytes, key: bytes) -> float:
        result = detect(source, key, params, visibility=visibility)
        # Evidence that cannot be bound to visible content does not count.
        return result["log10_p_value"] if result["status"] != "indeterminate" or result["p_value"] > params.threshold else 0.0

    return {
        "spectral": (lambda s, k: embed(s, k, params, visibility=visibility)[0], spectral_detect),
        "fd_vertex": (vb.fd_vertex_embed, lambda s, k: vb.fd_vertex_detect(s, k)["log10_p_value"]),
        "bezier_split": (vb.bezier_split_embed, lambda s, k: vb.bezier_split_detect(s, k)["log10_p_value"]),
        "numeric_lsb": (vb.numeric_lsb_embed, lambda s, k: vb.numeric_lsb_detect(s, k)["log10_p_value"]),
        "xml_comment": (_comment_embed, _comment_detect),
        "metadata_element": (vb.metadata_element_embed, lambda s, k: vb.metadata_element_detect(s, k)["log10_p_value"]),
    }


# ---------------------------------------------------------------------------


def run_item(task: tuple[dict, list[bytes], dict]) -> dict:
    item, others, config = task
    params = SpectralParameters(**config.get("params", {}))
    source = (ROOT / item["path"]).read_bytes()
    shared = config.get("key_mode") == "shared"
    key = item_key(item["path"], shared=shared)
    wrong = item_key(item["path"], "wrong", shared=shared)
    suite = attack_lib.extended_suite(others) if config.get("suite") == "extended" else attack_lib.standard_suite(others)
    if config.get("attacks"):
        suite = {name: suite[name] for name in config["attacks"]}
    record: dict = {"path": item["path"], "source": item["source"], "style": item["style"], "bytes": len(source), "methods": {}}
    visibility = config.get("visibility", "strict")
    selected = config.get("methods") or list(methods(params))
    for name in selected:
        embed_fn, detect_fn = methods(params, visibility)[name]
        entry: dict = {}
        started = time.perf_counter()
        try:
            marked = embed_fn(source, key)
        except Exception as exc:  # method not applicable to this asset
            entry["embed_error"] = f"{type(exc).__name__}: {exc}"[:200]
            record["methods"][name] = entry
            continue
        entry["embed_seconds"] = time.perf_counter() - started
        entry["bytes"] = len(marked)
        entry["gzip_bytes"] = len(gzip.compress(marked, 9))
        entry["source_gzip_bytes"] = len(gzip.compress(source, 9))
        try:
            entry["svgo_bytes"] = len(attack_lib._SVGO(marked))
            entry["source_svgo_bytes"] = len(attack_lib._SVGO(source))
        except Exception:
            pass
        try:
            entry["fidelity"] = fidelity(source, marked)
        except Exception as exc:
            entry["fidelity_error"] = str(exc)[:200]
        if name == "spectral":
            try:
                report = embed(source, key, params, visibility=visibility)[1]
                entry["marked_contours"] = report["marked_contours"]
                entry["eligible_contours"] = report["eligible_contours"]
                entry["total_contours"] = report["total_contours"]
                scale = load_document(source).scale() or 1.0
                marked_only = [c for c in report["contours"] if c.get("status", "marked") == "marked"]
                entry["skipped_contours"] = report.get("skipped_contours", 0)
                entry["skipped_unstable_seed"] = sum(c.get("status") == "skipped_unstable_seed" for c in report["contours"])
                entry["skipped_distortion"] = sum(c.get("status") == "skipped_distortion" for c in report["contours"])
                entry["max_curve_displacement_over_diagonal"] = max((c["curve_max_displacement"] for c in marked_only), default=0.0) / scale
                entry["unmarked_selected"] = sum(c["verifier_unmarked"] for c in report["contours"])
                entry["selected"] = sum(c["verifier_coefficients"] for c in report["contours"])
            except Exception:
                pass
        started = time.perf_counter()
        try:
            entry["null_original"] = detect_fn(source, key)
        except Exception as exc:
            entry["null_original_error"] = str(exc)[:200]
        entry["detect_seconds"] = time.perf_counter() - started
        try:
            entry["null_wrong_key"] = detect_fn(marked, wrong)
        except Exception as exc:
            entry["null_wrong_key_error"] = str(exc)[:200]
        results = {}
        for attack_name, attack in suite.items():
            try:
                attacked = attack(marked)
            except Exception as exc:
                results[attack_name] = {"attack_error": f"{type(exc).__name__}: {exc}"[:160]}
                continue
            try:
                results[attack_name] = {"log10_p": detect_fn(attacked, key)}
            except Exception as exc:
                results[attack_name] = {"log10_p": 0.0, "detect_error": f"{type(exc).__name__}: {exc}"[:160]}
        entry["attacks"] = results
        record["methods"][name] = entry
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--methods", nargs="*")
    parser.add_argument("--attacks", nargs="*")
    parser.add_argument("--params", type=json.loads, default={})
    parser.add_argument("--visibility", choices=["strict", "render"], default="strict")
    parser.add_argument("--key-mode", choices=["per-asset", "shared"], default="per-asset", help="shared: one key for every asset, as in deployment")
    parser.add_argument("--suite", choices=["standard", "extended"], default="standard")
    args = parser.parse_args()
    items = json.loads(args.corpus.read_text())["items"]
    if args.limit:
        by_source: dict[str, list] = {}
        for item in items:
            by_source.setdefault(item["source"], []).append(item)
        items = [item for group in by_source.values() for item in group[: args.limit]]
    done = set()
    if args.output.exists():
        done = {json.loads(line)["path"] for line in args.output.read_text().splitlines() if line.strip()}
    config = {"methods": args.methods, "attacks": args.attacks, "params": args.params, "visibility": args.visibility, "key_mode": args.key_mode, "suite": args.suite}
    tasks = []
    for index, item in enumerate(items):
        if item["path"] in done:
            continue
        # Composition partners: three other items from different sources.
        partners = [items[(index + offset * 97) % len(items)] for offset in (1, 2, 3)]
        tasks.append((item, [(ROOT / p["path"]).read_bytes() for p in partners], config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool, args.output.open("a") as stream:
        for count, record in enumerate(pool.imap_unordered(run_item, tasks, chunksize=1), start=1):
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if count % 25 == 0 or count == len(tasks):
                print(f"{count}/{len(tasks)} items, {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
