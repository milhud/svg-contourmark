"""Command-line interface for ContourMark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .core import Parameters, WatermarkError, embed, generate_key, read_key, verify
from .blind import BlindGenerationSession, detect_blind
from .capacity import assess_svg
from .geometry import GeometryError
from .inference import Candidate, GenerationSession, verify_generation
from .spectral import SpectralParameters
from .spectral import detect as spectral_detect
from .spectral import embed as spectral_embed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="contourmark", description="Keyed geometric watermarking for SVG contours")
    commands = parser.add_subparsers(dest="command", required=True)
    keygen = commands.add_parser("keygen", help="create a 256-bit secret key")
    keygen.add_argument("output", type=Path)
    mark = commands.add_parser("embed", help="embed a watermark into visible contour geometry")
    mark.add_argument("original", type=Path)
    mark.add_argument("output", type=Path)
    mark.add_argument("--key", type=Path, required=True)
    mark.add_argument("--manifest", type=Path, required=True)
    mark.add_argument("--asset-id")
    mark.add_argument("--amplitude", type=float, default=Parameters.amplitude)
    mark.add_argument("--samples", type=int, default=Parameters.samples)
    mark.add_argument("--harmonics", type=int, default=Parameters.harmonics)
    mark.add_argument("--min-perimeter", type=float, default=Parameters.min_perimeter)
    mark.add_argument("--precision", type=int, default=Parameters.precision)
    check = commands.add_parser("verify", help="verify a candidate against the owner's original and key")
    check.add_argument("original", type=Path)
    check.add_argument("candidate", type=Path)
    check.add_argument("--key", type=Path, required=True)
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--null-trials", type=int, default=999)
    sample = commands.add_parser("sample", help="select proposed SVG contours during generation using keyed randomness")
    sample.add_argument("proposals", type=Path, help="JSON document with viewBox and sequential candidate groups")
    sample.add_argument("output", type=Path)
    sample.add_argument("--key", type=Path, required=True)
    sample.add_argument("--manifest", type=Path, required=True)
    sample.add_argument("--asset-id")
    stream = commands.add_parser("sample-stream", help="live JSONL protocol: one candidate group per line")
    stream.add_argument("output", type=Path)
    stream.add_argument("--key", type=Path, required=True)
    stream.add_argument("--manifest", type=Path, required=True)
    stream.add_argument("--view-box", required=True)
    stream.add_argument("--asset-id")
    check_sample = commands.add_parser("verify-sample", help="verify generation-time geometric choices")
    check_sample.add_argument("candidate", type=Path)
    check_sample.add_argument("--key", type=Path, required=True)
    check_sample.add_argument("--manifest", type=Path, required=True)
    blind_sample = commands.add_parser("sample-blind", help="generate with keyed geometry choices and no private manifest")
    blind_sample.add_argument("proposals", type=Path)
    blind_sample.add_argument("output", type=Path)
    blind_sample.add_argument("--key", type=Path, required=True)
    blind_sample.add_argument("--asset-id", required=True)
    blind_check = commands.add_parser("detect-blind", help="detect a geometric mark without original or candidate manifest")
    blind_check.add_argument("candidate", type=Path)
    blind_check.add_argument("--key", type=Path, required=True)
    blind_check.add_argument("--asset-id", required=True)
    mark = commands.add_parser("mark", help="blind geometric watermark for any path-based SVG (recommended)")
    mark.add_argument("input", type=Path)
    mark.add_argument("output", type=Path)
    mark.add_argument("--key", type=Path, required=True)
    mark.add_argument("--report", type=Path, help="optional public embedding report (JSON)")
    mark.add_argument("--delta", type=float, default=SpectralParameters.delta)
    detect_cmd = commands.add_parser("detect", help="blind detection: needs only the SVG and the key")
    detect_cmd.add_argument("svg", type=Path)
    detect_cmd.add_argument("--key", type=Path, required=True)
    detect_cmd.add_argument("--delta", type=float, default=SpectralParameters.delta)
    detect_cmd.add_argument("--threshold", type=float, default=SpectralParameters.threshold)
    assess = commands.add_parser("assess", help="report fail-closed watermark coverage for an SVG")
    assess.add_argument("svg", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "keygen":
            if args.output.exists():
                raise WatermarkError(f"refusing to overwrite {args.output}")
            descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w") as stream:
                stream.write(generate_key().hex() + "\n")
            print(f"Created {args.output} (mode 0600)")
            return 0
        if args.command == "assess":
            result = assess_svg(args.svg.read_bytes())
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["whole_document_supported"] else 1
        key = read_key(args.key)
        if args.command == "mark":
            if args.output.exists():
                raise WatermarkError(f"refusing to overwrite {args.output}")
            svg, report = spectral_embed(args.input.read_bytes(), key, SpectralParameters(delta=args.delta))
            args.output.write_bytes(svg)
            if args.report:
                args.report.write_text(json.dumps(report, indent=2) + "\n")
            capacity = spectral_detect(svg, key, SpectralParameters(delta=args.delta))
            print(json.dumps({"output": str(args.output), "marked_contours": report["marked_contours"], "log10_p_value": capacity["log10_p_value"], "detectable_at_threshold": capacity["detected"]}))
            return 0
        if args.command == "detect":
            result = spectral_detect(args.svg.read_bytes(), key, SpectralParameters(delta=args.delta, threshold=args.threshold))
            result.pop("per_contour", None)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["detected"] else 1
        if args.command == "sample-blind":
            if args.output.exists():
                raise WatermarkError(f"refusing to overwrite {args.output}")
            proposals = json.loads(args.proposals.read_text())
            session = BlindGenerationSession(key, proposals["viewBox"], args.asset_id)
            for step in proposals["steps"]:
                weights = [item.get("weight", 1.0) for item in step["candidates"]]
                if len(set(weights)) != 1:
                    raise WatermarkError("blind sampling currently requires equal candidate weights")
                session.add_step([item["d"] for item in step["candidates"]], step.get("attributes", {}))
            args.output.write_bytes(session.finish())
            print(json.dumps({"output": str(args.output), "steps": len(session.emitted), "asset_id": args.asset_id}))
            return 0
        if args.command == "detect-blind":
            result = detect_blind(args.candidate.read_bytes(), key, args.asset_id)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["detected"] else 1
        if args.command == "sample-stream":
            for path in (args.output, args.manifest):
                if path.exists():
                    raise WatermarkError(f"refusing to overwrite {path}")
            session = GenerationSession(key, args.view_box, args.asset_id)
            for line in sys.stdin:
                if not line.strip():
                    continue
                step = json.loads(line)
                candidates = [Candidate(item["d"], item.get("weight", 1.0)) for item in step["candidates"]]
                selected = session.add_step(candidates, step.get("attributes", {}))
                print(json.dumps({"step": len(session.steps) - 1, "selected": selected, "d": candidates[selected].d}), flush=True)
            svg, manifest = session.finish()
            args.output.write_bytes(svg)
            args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            return 0
        if args.command == "sample":
            for path in (args.output, args.manifest):
                if path.exists():
                    raise WatermarkError(f"refusing to overwrite {path}")
            proposals = json.loads(args.proposals.read_text())
            session = GenerationSession(key, proposals["viewBox"], args.asset_id)
            for step in proposals["steps"]:
                candidates = [Candidate(item["d"], item.get("weight", 1.0)) for item in step["candidates"]]
                session.add_step(candidates, step.get("attributes", {}))
            svg, manifest = session.finish()
            args.output.write_bytes(svg)
            args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"output": str(args.output), "manifest": str(args.manifest), "steps": len(manifest["steps"])}))
            return 0
        if args.command == "verify-sample":
            result = verify_generation(args.candidate.read_bytes(), json.loads(args.manifest.read_text()), key)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["detected"] else 1
        if args.command == "embed":
            for path in (args.output, args.manifest):
                if path.exists():
                    raise WatermarkError(f"refusing to overwrite {path}")
            params = Parameters(args.amplitude, args.samples, args.harmonics, args.min_perimeter, args.precision)
            marked, manifest = embed(args.original.read_bytes(), key, args.asset_id, params)
            args.output.write_bytes(marked)
            args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"output": str(args.output), "manifest": str(args.manifest), "contours": len(manifest["contour_indices"])}))
            return 0
        result = verify(args.original.read_bytes(), args.candidate.read_bytes(), json.loads(args.manifest.read_text()), key, args.null_trials)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["detected"] else 1
    except (WatermarkError, GeometryError, OSError, json.JSONDecodeError) as exc:
        print(f"contourmark: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
