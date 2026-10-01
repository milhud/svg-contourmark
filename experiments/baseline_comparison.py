"""Run simple baselines and ContourMark under the same benign rewrites."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from contourmark.baselines import metadata_detect, metadata_embed, numeric_lsb_detect, numeric_lsb_embed
from contourmark.core import read_key
from contourmark.inference import verify_generation
from offline_attacks import optimized


ROOT = Path(__file__).resolve().parents[1]


def rounded(source: bytes, decimals: int) -> bytes:
    """Round numbers only inside path data, preserving XML syntax."""
    text = source.decode()
    number = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.\d*)(?:[Ee][-+]?\d+)?")

    def path_replacement(attribute: re.Match[str]) -> str:
        data = number.sub(lambda item: f"{float(item.group()):.{decimals}f}".rstrip("0").rstrip("."), attribute.group(2))
        return f'd={attribute.group(1)}{data}{attribute.group(1)}'

    return re.sub(r"\bd\s*=\s*(['\"])(.*?)\1", path_replacement, text, flags=re.DOTALL).encode()


def variants(source: bytes) -> dict[str, bytes]:
    results = {"original": source, "round_path_2_decimals": rounded(source, 2)}
    scour = ROOT / ".venv/bin/scour"
    if scour.exists():
        results["scour"] = optimized(source, [str(scour), "-i", "{input}", "-o", "{output}", "--indent=none"])
    svgo = ROOT / "node_modules/.bin/svgo"
    if svgo.exists():
        results["svgo"] = optimized(source, [str(svgo), "{input}", "-o", "{output}"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("svg", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.svg.read_bytes()
    key = read_key(args.key)
    manifest = json.loads(args.manifest.read_text())
    metadata = metadata_embed(source, args.asset_id)
    numeric = numeric_lsb_embed(source, key, args.asset_id, chips=64)
    assisted_results = {}
    for name, candidate in variants(source).items():
        checked = verify_generation(candidate, manifest, key)
        assisted_results[name] = {
            "detected": checked["detected"],
            "conditional_detected": checked["conditional_detected"],
            "conditional_p_value": checked["conditional_p_value"],
            "conservative_p_value": checked["conservative_p_value"],
        }
    metadata_results = {name: {"detected": metadata_detect(candidate, args.asset_id)} for name, candidate in variants(metadata).items()}
    numeric_results = {name: numeric_lsb_detect(candidate, key, args.asset_id, chips=64) for name, candidate in variants(numeric).items()}
    report = {
        "asset": str(args.svg.resolve().relative_to(ROOT)),
        "methods": {
            "contourmark_assisted": assisted_results,
            "xml_comment_metadata": metadata_results,
            "numeric_coordinate_lsb": numeric_results,
        },
        "scope": "One generated asset and benign rewrites; adaptive and diverse-asset experiments remain required.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
