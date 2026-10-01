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
    svg_name = str(args.svg.resolve().relative_to(ROOT))
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
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
        "control_perturb_1.5_unit": perturbed_controls(source, 1.5),
        "control_perturb_2_unit": perturbed_controls(source, 2.0),
        "control_perturb_3_unit": perturbed_controls(source, 3.0),
        "control_perturb_4_unit": perturbed_controls(source, 4.0),
        "control_perturb_6_unit": perturbed_controls(source, 6.0),
        "control_perturb_8_unit": perturbed_controls(source, 8.0),
        "control_perturb_12_unit": perturbed_controls(source, 12.0),
    }
    if len(manifest.get("steps", [])) >= 32:
        for count in (8, 16, 24, 32):
            if count < len(manifest["steps"]):
                attacks[f"delete_first_{count}"] = deleted(source, count)
    scour = ROOT / ".venv/bin/scour"
    if scour.exists():
        attacks["scour"] = optimized(source, [str(scour), "-i", "{input}", "-o", "{output}", "--indent=none"])
    svgo = ROOT / "node_modules/.bin/svgo"
    if svgo.exists():
        attacks["svgo"] = optimized(source, [str(svgo), "{input}", "-o", "{output}"])
    results = {}
    if args.output.exists():
        checkpoint = json.loads(args.output.read_text())
        if checkpoint.get("status") == "in_progress" and checkpoint.get("svg") == svg_name and checkpoint.get("manifest_sha256") == manifest_sha256:
            results = checkpoint.get("attacks", {})
            print(f"resumed {len(results)}/{len(attacks)} attack results", flush=True)

    def write_report(status: str) -> None:
        report = {
            "status": status,
            "svg": svg_name,
            "manifest_sha256": manifest_sha256,
            "attacks": results,
            "note": "Single asset, fixed edits, and 512px librsvg pixel metrics; not a perceptual study or superiority claim.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    write_report("in_progress")
    for name, candidate in attacks.items():
        if name in results:
            continue
        try:
            check = verify_generation(candidate, manifest, key)
            results[name] = {
                "detected": check["detected"],
                "conditional_detected": check["conditional_detected"],
                "matched_steps": check["matched_steps"],
                "recognized_steps": check["recognized_steps"],
                "total_steps": check["total_steps"],
                "conditional_p_value": check["conditional_p_value"],
                "conservative_p_value": check["conservative_p_value"],
                "visible_contours": check["visible_contours"],
            }
        except WatermarkError as exc:
            results[name] = {"detected": False, "error": str(exc)}
        results[name]["svg_bytes"] = len(candidate)
        results[name]["paired_geometry_rms_fraction"] = None if name == "reorder_paths" else paired_geometry_rms(source, candidate)
        results[name].update(render_metrics(source, candidate))
        write_report("in_progress")
    write_report("completed")
    report = json.loads(args.output.read_text())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
