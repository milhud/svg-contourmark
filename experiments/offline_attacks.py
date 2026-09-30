"""Reproducible, keyless removal probes against the manifest-free prototype.

The geometry attacks do not inspect the watermark key. The fixed key below is
only for reproducible generation and evaluation, and is not a production key.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from svgpathtools import CubicBezier, Path as SVGPath, parse_path

from contourmark.blind import BlindGenerationSession, detect_blind
from contourmark.core import WatermarkError


ROOT = Path(__file__).resolve().parents[1]
KEY = bytes(range(32))
ASSET_ID = "offline-attack-fixture-v1"


def toy_asset() -> bytes:
    proposal_data = json.loads(subprocess.check_output([sys.executable, str(ROOT / "examples/make_blind_proposals.py")]))
    session = BlindGenerationSession(KEY, proposal_data["viewBox"], ASSET_ID)
    for step in proposal_data["steps"]:
        session.add_step([candidate["d"] for candidate in step["candidates"]], step["attributes"])
    return session.finish()


def moved(source: bytes, dx: float, dy: float = 0.0, alternating: bool = False) -> bytes:
    root = ET.fromstring(source)
    for index, path in enumerate(root.iter("{http://www.w3.org/2000/svg}path")):
        sign = -1 if alternating and index % 2 else 1
        path.set("d", parse_path(path.get("d", "")).translated(complex(sign * dx, sign * dy)).d())
    return ET.tostring(root)


def perturbed_controls(source: bytes, amount: float) -> bytes:
    """Keyless shape attack: move cubic controls while retaining endpoints."""
    root = ET.fromstring(source)
    for path_index, element in enumerate(root.iter("{http://www.w3.org/2000/svg}path")):
        rebuilt = []
        for segment_index, segment in enumerate(parse_path(element.get("d", ""))):
            if isinstance(segment, CubicBezier):
                sign = -1 if (path_index + segment_index) % 2 else 1
                delta = complex(sign * amount, amount)
                rebuilt.append(CubicBezier(segment.start, segment.control1 + delta, segment.control2 - delta, segment.end))
            else:
                rebuilt.append(segment)
        element.set("d", SVGPath(*rebuilt).d())
    return ET.tostring(root)


def deleted(source: bytes, count: int) -> bytes:
    root = ET.fromstring(source)
    root[:] = list(root)[count:]
    return ET.tostring(root)


def reordered(source: bytes) -> bytes:
    root = ET.fromstring(source)
    root[:] = list(root)[::-1]
    return ET.tostring(root)


def optimized(source: bytes, command: list[str]) -> bytes:
    with tempfile.TemporaryDirectory() as directory:
        before = Path(directory) / "before.svg"
        after = Path(directory) / "after.svg"
        before.write_bytes(source)
        argv = [item.format(input=str(before), output=str(after)) for item in command]
        subprocess.run(argv, check=True, capture_output=True)
        return after.read_bytes()


def assess(source: bytes) -> dict:
    try:
        result = detect_blind(source, KEY, ASSET_ID)
        return {name: result[name] for name in ("detected", "p_value", "usable_contours", "skipped_contours")}
    except WatermarkError as exc:
        return {"detected": False, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/results/offline_attacks.json")
    args = parser.parse_args()
    source = toy_asset()
    attacks: dict[str, bytes] = {
        "original": source,
        "reorder_paths": reordered(source),
        "delete_first_8_of_32": deleted(source, 8),
        "translate_all_0.5_unit": moved(source, 0.5),
        "translate_all_1_unit": moved(source, 1.0),
        "translate_all_2_units": moved(source, 2.0),
        "alternating_translate_1_unit": moved(source, 1.0, alternating=True),
        "control_perturb_0.25_unit": perturbed_controls(source, 0.25),
        "control_perturb_0.5_unit": perturbed_controls(source, 0.5),
        "control_perturb_0.625_unit": perturbed_controls(source, 0.625),
        "control_perturb_0.75_unit": perturbed_controls(source, 0.75),
        "control_perturb_1_unit": perturbed_controls(source, 1.0),
        "control_perturb_1.5_unit": perturbed_controls(source, 1.5),
        "control_perturb_2_unit": perturbed_controls(source, 2.0),
        "control_perturb_3_unit": perturbed_controls(source, 3.0),
        "control_perturb_4_unit": perturbed_controls(source, 4.0),
        "control_perturb_6_unit": perturbed_controls(source, 6.0),
        "control_perturb_8_unit": perturbed_controls(source, 8.0),
        "control_perturb_12_unit": perturbed_controls(source, 12.0),
    }
    scour = ROOT / ".venv/bin/scour"
    if scour.exists():
        attacks["scour"] = optimized(source, [str(scour), "-i", "{input}", "-o", "{output}", "--indent=none"])
    svgo = ROOT / "node_modules/.bin/svgo"
    if svgo.exists():
        attacks["svgo"] = optimized(source, [str(svgo), "{input}", "-o", "{output}"])
    report = {
        "asset": "32 synthetic cubic strokes, equal-weight alternatives",
        "viewBox": "0 0 200 200",
        "key_policy": "fixed public test key; attacks independent of key",
        "attacks": {name: {**assess(svg), "svg_bytes": len(svg)} for name, svg in attacks.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
