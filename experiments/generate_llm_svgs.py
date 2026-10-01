"""Generate an AI-produced SVG corpus with local LLMs (Ollama).

This mirrors the deployment the watermark targets: a chat model writes SVG
code, and the provider marks the output before release.  Each response is
validated as XML with at least one drawable contour; invalid outputs are
recorded but not used.  No model is trained or downloaded here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

from contourmark.geometry import GeometryError, load_document

API = "http://127.0.0.1:11434/api/generate"
CONCEPTS = [
    "coffee cup", "rocket", "house", "tree", "cat face", "bicycle", "umbrella", "camera", "light bulb", "heart",
    "sun with rays", "cloud with rain", "anchor", "key", "lock", "bell", "leaf", "flower", "fish", "bird",
    "mountain landscape", "sailboat", "car", "gear", "trophy", "gift box", "shopping cart", "envelope", "phone", "laptop",
    "book", "pencil", "paint brush", "music note", "headphones", "microphone", "map pin", "compass", "globe", "star",
    "moon and stars", "snowflake", "fire flame", "water drop", "apple", "pizza slice", "ice cream cone", "cupcake", "robot face", "ghost",
    "owl", "turtle", "butterfly", "mushroom", "cactus", "castle", "lighthouse", "hot air balloon", "airplane", "train",
]
PROMPT = (
    "Write one complete, self-contained SVG document for a flat, colorful icon of a {concept}. "
    "Use viewBox=\"0 0 64 64\". Use only svg, g, path, circle, ellipse, rect, polygon elements with fill colors; "
    "prefer curved <path> elements with cubic Bezier commands for organic shapes. "
    "Output only the SVG code, starting with <svg and ending with </svg>."
)


def ask(model: str, concept: str, seed: int) -> tuple[str, dict]:
    body = {
        "model": model, "prompt": PROMPT.format(concept=concept), "stream": False, "think": False,
        "options": {"temperature": 0.7, "seed": seed, "num_predict": 2500},
    }
    request = urllib.request.Request(API, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    return result["response"], {"eval_count": result.get("eval_count"), "total_duration": result.get("total_duration")}


def extract(text: str) -> bytes | None:
    match = re.search(r"<svg\b.*?</svg>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    svg = match.group(0)
    if "xmlns=" not in svg[:200]:
        svg = svg.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    return svg.encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen3.5:4b", "gemma4:e2b"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "generation_log.jsonl"
    done = set()
    if log_path.exists():
        done = {(r["model"], r["concept"]) for r in map(json.loads, log_path.read_text().splitlines())}
    with log_path.open("a") as log:
        for model in args.models:
            for index, concept in enumerate(CONCEPTS):
                if (model, concept) in done:
                    continue
                started = time.time()
                record = {"model": model, "concept": concept, "seed": args.seed + index}
                try:
                    text, timing = ask(model, concept, args.seed + index)
                    record.update(timing)
                    svg = extract(text)
                    if svg is None:
                        raise GeometryError("no <svg> element in response")
                    contours = len(load_document(svg).contours)
                    if contours == 0:
                        raise GeometryError("no drawable contour")
                    name = f"{model.replace(':', '_').replace('.', '_')}_{index:02d}_{concept.replace(' ', '_')}.svg"
                    (args.output_dir / name).write_bytes(svg)
                    record.update({"status": "ok", "file": name, "contours": contours, "sha256": hashlib.sha256(svg).hexdigest()})
                except Exception as exc:
                    record.update({"status": "invalid", "error": f"{type(exc).__name__}: {exc}"[:300]})
                record["seconds"] = time.time() - started
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(model, concept, record["status"], f"{record['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
