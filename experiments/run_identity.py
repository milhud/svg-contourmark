"""Immutable identities for resumable experiments (no model dependency)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_run(log: Path, configuration: dict) -> str:
    """Refuse to append to a log made with another or unknown configuration."""
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    run_id = hashlib.sha256(encoded.encode()).hexdigest()
    manifest = log.with_suffix(".run.json")
    payload = {"run_id": run_id, "configuration": configuration}
    if manifest.exists():
        if json.loads(manifest.read_text()) != payload:
            raise ValueError(f"run configuration changed; use a new output directory: {manifest}")
    else:
        if log.exists() and log.stat().st_size:
            raise ValueError(f"cannot resume legacy log without run identity: {log}")
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("x") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return run_id


def evaluation_key(label: str = "evaluation", wrong: bool = False) -> bytes:
    """Public, deterministic benchmark keys (not secrets)."""
    if label == "evaluation":
        return hashlib.sha256(b"iconshop-geosample-wrong-key" if wrong else b"iconshop-geosample-evaluation-key").digest()
    return hashlib.sha256(f"iconshop-geosample-{'wrong-' if wrong else ''}key-{label}".encode()).digest()


def detector_settings(samples_dir: Path) -> dict:
    """How to detect the samples in a run directory, from its run manifest.

    Returns ``params`` (keyword arguments for GeoParameters), ``statistic``,
    ``gamma`` and ``key_label``.  Runs made before a setting existed get that
    setting's value at the time (for example ``bin_offset`` 0.0).
    """
    manifests = sorted(Path(samples_dir).glob("samples*.run.json"))
    configuration = json.loads(manifests[0].read_text())["configuration"] if manifests else {}
    params = {"bin_offset": 0.0, **configuration.get("watermark", {})}
    sampler = configuration.get("sampler", {})
    return {
        "params": params,
        "statistic": "green" if sampler.get("mode") == "bias" else "gamma",
        "gamma": sampler.get("gamma", 0.5),
        "key_label": configuration.get("key_label", "evaluation"),
        "sampler": sampler or {"mode": "gumbel", "reuse": "mask"},
        "top_p": configuration.get("top_p"),
    }
