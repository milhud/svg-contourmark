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
