"""Build a corpus manifest for the generated LLM SVGs (same schema as build_corpus.py)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "experiments/corpus/llm_svgs"


def main() -> None:
    items = []
    for line in (DIRECTORY / "generation_log.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record.get("status") != "ok":
            continue
        path = DIRECTORY / record["file"]
        data = path.read_bytes()
        items.append({
            "source": "llm-" + record["model"].split(":")[0], "license": "generated", "style": "LLM-written SVG",
            "path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "concept": record["concept"],
        })
    output = ROOT / "experiments/corpus/llm.json"
    output.write_text(json.dumps({"seed": None, "split": "llm", "items": items}, indent=1) + "\n")
    print(f"wrote {len(items)} items to {output}")


if __name__ == "__main__":
    main()
