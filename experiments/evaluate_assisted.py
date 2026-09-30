"""Apply fixed, keyless edits to a real generator output and record verification."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

from contourmark.core import WatermarkError, read_key
from contourmark.inference import _path_points, verify_generation
from offline_attacks import deleted, moved, optimized, perturbed_controls, reordered


ROOT = Path(__file__).resolve().parents[1]


def paired_geometry_rms(reference: bytes, candidate: bytes) -> float | None:
    before = [element.get("d", "") for element in ET.fromstring(reference).iter("{http://www.w3.org/2000/svg}path")]
    after = [element.get("d", "") for element in ET.fromstring(candidate).iter("{http://www.w3.org/2000/svg}path")]
    if len(before) != len(after):
        return None
    squared = []
    for left, right in zip(before, after):
        a, b = _path_points(left), _path_points(right)
        squared.extend(abs(x - y) ** 2 for x, y in zip(a, b))
    return math.sqrt(sum(squared) / len(squared)) / math.hypot(256, 256)


def render_metrics(reference: bytes, candidate: bytes, size: int = 512) -> dict[str, float]:
    def render(source: bytes) -> Image.Image:
        result = subprocess.run(
            ["rsvg-convert", "--width", str(size), "--height", str(size), "--background-color", "white"],
            input=source,
            check=True,
            capture_output=True,
        )
        return Image.open(io.BytesIO(result.stdout)).convert("RGB")

    before, after = render(reference), render(candidate)
    difference = ImageChops.difference(before, after)
    statistics = ImageStat.Stat(difference)
    mae = sum(statistics.mean) / (3 * 255)
    rms = math.sqrt(sum(value * value for value in statistics.rms) / 3) / 255
    pixels = difference.get_flattened_data() if hasattr(difference, "get_flattened_data") else difference.getdata()
    changed = sum(1 for pixel in pixels if pixel != (0, 0, 0)) / (size * size)
    return {"render_mae": mae, "render_rmse": rms, "changed_pixel_fraction": changed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("svg", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.svg.read_bytes()
    manifest = json.loads(args.manifest.read_text())
    key = read_key(args.key)
    attacks = {
        "original": source,
        "reorder_paths": reordered(source),
        "delete_first_4": deleted(source, 4),
        "translate_all_0.25_unit": moved(source, 0.25),
        "translate_all_0.5_unit": moved(source, 0.5),
        "translate_all_1_unit": moved(source, 1.0),
        "alternating_translate_0.5_unit": moved(source, 0.5, alternating=True),
        "control_perturb_0.25_unit": perturbed_controls(source, 0.25),
        "control_perturb_0.5_unit": perturbed_controls(source, 0.5),
        "control_perturb_0.625_unit": perturbed_controls(source, 0.625),
        "control_perturb_0.75_unit": perturbed_controls(source, 0.75),
        "control_perturb_1_unit": perturbed_controls(source, 1.0),
    }
    scour = ROOT / ".venv/bin/scour"
    if scour.exists():
        attacks["scour"] = optimized(source, [str(scour), "-i", "{input}", "-o", "{output}", "--indent=none"])
    svgo = ROOT / "node_modules/.bin/svgo"
    if svgo.exists():
        attacks["svgo"] = optimized(source, [str(svgo), "{input}", "-o", "{output}"])
    results = {}
    for name, candidate in attacks.items():
        try:
            check = verify_generation(candidate, manifest, key)
            results[name] = {
                "detected": check["detected"],
                "matched_steps": check["matched_steps"],
                "total_steps": check["total_steps"],
                "conditional_p_value": check["conditional_p_value"],
                "visible_contours": check["visible_contours"],
            }
        except WatermarkError as exc:
            results[name] = {"detected": False, "error": str(exc)}
        results[name]["svg_bytes"] = len(candidate)
        results[name]["paired_geometry_rms_fraction"] = None if name == "reorder_paths" else paired_geometry_rms(source, candidate)
        results[name].update(render_metrics(source, candidate))
    report = {
        "svg": str(args.svg.resolve().relative_to(ROOT)),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "attacks": results,
        "note": "Single asset, fixed edits, and 512px librsvg pixel metrics; not a perceptual study or superiority claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
