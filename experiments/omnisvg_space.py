"""Generate an SVG with the official public OmniSVG Hugging Face Space.

This is an evaluation adapter, not a watermark adapter: the public Gradio API
returns only the completed SVG, so it cannot expose the token logits needed by
ContourMark's inference-time sampler.  The saved artifact is used to measure
coverage of specialist-model output and to make that limitation reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from contourmark.capacity import assess_svg


ROOT = "https://omnisvg-omnisvg-3b.hf.space"
API_NAME = "gradio_text_to_svg"
SVG_RE = re.compile(r"<svg\b.*?</svg>", re.IGNORECASE | re.DOTALL)


def request_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "contourmark-research/0.1"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def receive_result(url: str) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "text/event-stream", "User-Agent": "contourmark-research/0.1"})
    event = None
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
                if event == "error":
                    raise RuntimeError(f"OmniSVG Space error: {data}")
                if event in {"complete", "generating"}:
                    parsed = json.loads(data)
                    if event == "complete":
                        return parsed
    raise RuntimeError("OmniSVG Space stream ended without a complete event")


def extract_svg(result: Any) -> tuple[str, str | None]:
    """Extract the SVG and status from the three Gradio outputs."""
    outputs = result if isinstance(result, list) else [result]
    status = str(outputs[2]) if len(outputs) > 2 else None
    candidates = []
    if len(outputs) > 1:
        candidates.append(outputs[1])
    candidates.extend(outputs)
    for candidate in candidates:
        if isinstance(candidate, dict):
            candidate = candidate.get("value") or candidate.get("code") or json.dumps(candidate)
        if not isinstance(candidate, str):
            continue
        decoded = html.unescape(candidate)
        match = SVG_RE.search(decoded)
        if match:
            return match.group(0).strip() + "\n", status
    raise RuntimeError("OmniSVG completed but no <svg> document was present in its outputs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-size", choices=["4B", "8B"], default="4B")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.88)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    payload = {"data": [
        args.prompt,
        args.model_size,
        1,
        args.temperature,
        args.top_p,
        args.top_k,
        args.repetition_penalty,
        args.max_length,
    ]}
    submit_url = f"{ROOT}/gradio_api/call/{API_NAME}"
    try:
        queued = request_json(submit_url, payload)
        event_id = queued["event_id"]
        result = receive_result(f"{submit_url}/{event_id}")
        svg, status = extract_svg(result)
    except (KeyError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "status": "failed",
            "endpoint": ROOT,
            "api_name": API_NAME,
            "request": payload,
            "error": str(exc),
        }, indent=2) + "\n")
        raise SystemExit(str(exc)) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)
    assessment = assess_svg(svg.encode())
    args.report.write_text(json.dumps({
        "status": "completed",
        "endpoint": ROOT,
        "api_name": API_NAME,
        "request": payload,
        "space_status": status,
        "svg_sha256": hashlib.sha256(svg.encode()).hexdigest(),
        "svg_bytes": len(svg.encode()),
        "assessment": assessment,
        "watermark_injected": False,
        "limitation": "The public endpoint returns a completed SVG and exposes no logits or pre-sampling candidate hook.",
    }, indent=2) + "\n")
    print(f"saved {args.output} ({len(svg.encode())} bytes)")


if __name__ == "__main__":
    main()
