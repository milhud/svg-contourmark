"""Run a real local SVG proposal generator through ContourMark during inference.

Requires an already-running local Ollama service and an installed model. Each
model call proposes alternatives for one new path. The keyed sampler selects
one before the next model call, and that selected path is fed back as context.
No model training or model download occurs in this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path

from svgpathtools import CubicBezier, parse_path

from contourmark.blind import BlindGenerationSession
from contourmark.core import WatermarkError, read_key
from contourmark.inference import Candidate, GenerationSession


API = "http://127.0.0.1:11434/api/generate"
SCHEMA = {
    "type": "object",
    "properties": {"candidates": {
        "type": "array", "minItems": 2, "maxItems": 2,
        "items": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
    }},
    "required": ["candidates"],
}


def ask_model(model: str, prompt: str, seed: int, start: tuple[int, int], target: tuple[int, int]) -> tuple[list[str], dict]:
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "format": SCHEMA,
        "options": {"temperature": 0.8, "seed": seed, "num_predict": 220},
    }
    request = urllib.request.Request(API, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    parsed = json.loads(result["response"])
    controls = parsed["candidates"]
    if not isinstance(controls, list) or len(controls) != 2 or any(
        not isinstance(item, list) or len(item) != 4 or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not -64 <= value <= 320
            for value in item
        ) for item in controls
    ):
        raise WatermarkError("model did not return two finite four-number control-point sets")
    candidates = [f"M{start[0]},{start[1]} C{item[0]:g},{item[1]:g} {item[2]:g},{item[3]:g} {target[0]},{target[1]}" for item in controls]
    return candidates, {"eval_count": result.get("eval_count"), "eval_duration": result.get("eval_duration"), "total_duration": result.get("total_duration")}


def endpoints_for(motif: str, step: int, total: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if motif == "wave":
        y = round(18 + step * 220 / max(total - 1, 1))
        return (16, y), (240, y)
    if motif == "leaf":
        angle = 2 * math.pi * step / total
        return (128, 222), (round(128 + 66 * math.cos(angle)), round(120 + 92 * math.sin(angle)))
    angle = 2 * math.pi * step / total
    x0, y0 = 128, 128
    radius = 80 if motif == "flower" else 92
    return (x0, y0), (round(x0 + radius * math.cos(angle)), round(y0 + radius * math.sin(angle)))


def target_for(motif: str, step: int, total: int) -> tuple[int, int]:
    return endpoints_for(motif, step, total)[1]


def validate_proposals(paths: list[str], start: tuple[int, int], target: tuple[int, int]) -> None:
    """Reject malformed or off-target paths before the keyed choice is made."""
    for index, data in enumerate(paths):
        try:
            path = parse_path(data)
        except Exception as exc:
            raise WatermarkError(f"candidate {index} is not valid SVG path data: {exc}") from exc
        if len(path) != 1 or not isinstance(path[0], CubicBezier):
            raise WatermarkError(f"candidate {index} must be one cubic segment")
        if abs(path.start - complex(*start)) > 0.5 or abs(path.end - complex(*target)) > 0.5:
            raise WatermarkError(f"candidate {index} must run from {start} to {target}")


def prompt_for(motif: str, step: int, total: int, selected: list[str], error: str | None) -> str:
    (x0, y0), (x1, y1) = endpoints_for(motif, step, total)
    prior = "\n".join(selected[-3:]) or "(none yet)"
    repair = f"Previous proposal was rejected: {error}. Fix that in this response.\n" if error else ""
    return (
        f"You are drawing a 256 by 256 SVG line-art {motif}, one visible curved stroke at a time. "
        f"This is stroke {step + 1} of {total}. Generate exactly TWO different but visually comparable "
        f"sets of four cubic Bezier control-point coordinates for a stroke that starts at ({x0},{y0}) and ends at ({x1},{y1}). "
        f"Return JSON {{\"candidates\": [[c1x,c1y,c2x,c2y], [c1x,c1y,c2x,c2y]]}}. "
        f"The adapter supplies the fixed endpoints. Choose smooth control points for a coherent {motif}; "
        "the alternatives should differ in curvature by about 4 to 8 canvas units. "
        "Both alternatives must look comparable. Return only the JSON object.\n"
        f"Recent selected strokes:\n{prior}\n{repair}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--motif", choices=["flower", "sunburst", "wave", "leaf"], default="flower")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--mode", choices=["assisted", "blind"], default="assisted")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 4 <= args.steps <= 64:
        raise SystemExit("--steps must be in [4, 64]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    key = read_key(args.key)
    session = BlindGenerationSession(key, "0 0 256 256", args.asset_id) if args.mode == "blind" else GenerationSession(key, "0 0 256 256", args.asset_id)
    selected: list[str] = []
    records = []
    start = time.monotonic()
    for step in range(args.steps):
        error = None
        for attempt in range(4):
            prompt = prompt_for(args.motif, step, args.steps, selected, error)
            try:
                start_point, target_point = endpoints_for(args.motif, step, args.steps)
                candidates, timing = ask_model(args.model, prompt, args.seed + step * 10 + attempt, start_point, target_point)
                validate_proposals(candidates, start_point, target_point)
                if args.mode == "blind":
                    chosen = session.add_step(candidates, {"fill": "none", "stroke": "#303030", "stroke-width": "1.5"})
                else:
                    chosen = session.add_step([Candidate(path) for path in candidates], {"fill": "none", "stroke": "#303030", "stroke-width": "1.5"})
            except (ValueError, KeyError, WatermarkError) as exc:
                error = str(exc)
                records.append({"step": step, "attempt": attempt, "accepted": False, "error": error})
                continue
            selected.append(candidates[chosen])
            records.append({"step": step, "attempt": attempt, "accepted": True, "selected": chosen, "candidates": candidates, **timing})
            print(f"step {step + 1}/{args.steps}: selected {chosen} after {attempt + 1} proposal(s)", flush=True)
            break
        else:
            args.report.write_text(json.dumps({
                "model": args.model,
                "motif": args.motif,
                "mode": args.mode,
                "asset_id": args.asset_id,
                "steps_requested": args.steps,
                "steps_completed": len(selected),
                "status": "failed",
                "error": error,
                "records": records,
            }, indent=2) + "\n")
            raise SystemExit(f"step {step} failed after four model proposals: {error}")
    result = session.finish()
    if args.mode == "blind":
        svg = result
        manifest = None
    else:
        svg, manifest = result
    args.output.write_bytes(svg)
    if manifest is not None:
        args.output.with_suffix(".wm.json").write_text(json.dumps(manifest, indent=2) + "\n")
    args.report.write_text(json.dumps({
        "model": args.model,
        "motif": args.motif,
        "mode": args.mode,
        "asset_id": args.asset_id,
        "steps": args.steps,
        "seed": args.seed,
        "status": "completed",
        "elapsed_seconds": time.monotonic() - start,
        "svg_sha256": hashlib.sha256(svg).hexdigest(),
        "records": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
