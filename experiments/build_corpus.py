"""Sample a reproducible, license-documented SVG evaluation corpus.

Sources are open-license icon, logo, and emoji sets installed as npm
development dependencies (node_modules is not committed).  The manifest
records each file's path, source, license, and SHA-256 so results can be
tied to exact inputs.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
from pathlib import Path

from contourmark.geometry import GeometryError, load_document

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "lucide": ("node_modules/lucide-static/icons/*.svg", "ISC", "stroke"),
    "tabler": ("node_modules/@tabler/icons/icons/outline/*.svg", "MIT", "stroke"),
    "bootstrap": ("node_modules/bootstrap-icons/icons/*.svg", "MIT", "fill"),
    "material": ("node_modules/@material-design-icons/svg/filled/*.svg", "Apache-2.0", "fill"),
    "fontawesome": ("node_modules/@fortawesome/fontawesome-free/svgs/solid/*.svg", "CC-BY-4.0", "fill"),
    "remix": ("node_modules/remixicon/icons/*/*.svg", "Apache-2.0", "fill"),
    "octicons": ("node_modules/@primer/octicons/build/svg/*.svg", "MIT", "fill"),
    "simple-icons": ("node_modules/simple-icons/icons/*.svg", "CC0-1.0", "logo"),
    "openmoji": ("node_modules/openmoji/color/svg/*.svg", "CC-BY-SA-4.0", "color illustration"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-source", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(f"{args.seed}|{args.split}")
    items = []
    for name, (pattern, license_id, style) in SOURCES.items():
        files = sorted(glob.glob(str(ROOT / pattern)))
        # Disjoint dev/test splits: even hash -> dev, odd -> test.
        files = [f for f in files if (int(hashlib.sha256(Path(f).name.encode()).hexdigest(), 16) % 2 == 0) == (args.split == "dev")]
        rng.shuffle(files)
        chosen = 0
        for path in files:
            if chosen >= args.per_source:
                break
            data = Path(path).read_bytes()
            try:
                if not load_document(data).contours:
                    continue
            except (GeometryError, ValueError):
                continue
            items.append({"source": name, "license": license_id, "style": style, "path": str(Path(path).relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
            chosen += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"seed": args.seed, "split": args.split, "items": items}, indent=1) + "\n")
    print(f"wrote {len(items)} items to {args.output}")


if __name__ == "__main__":
    main()
