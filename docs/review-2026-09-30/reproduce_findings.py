"""Small CPU reproductions for the independent review; no model downloads.

Run from the repository root:
  .venv/bin/python docs/review-2026-09-30/reproduce_findings.py

The visibility probe also requires the existing rsvg-convert executable.
This script prints evidence; it does not modify implementation or experiments.
"""
import hashlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from contourmark.attacks import render_png, subdivide
from contourmark.geosample import GeoParameters, GeoWatermark, detect, document_descriptors
from contourmark.spectral import detect as spectral_detect, embed as spectral_embed

ROOT = Path(__file__).resolve().parents[2]


def empty_clip(source):
    root = ET.fromstring(source)
    ns = "{http://www.w3.org/2000/svg}"
    children = list(root)
    for child in children:
        root.remove(child)
    defs = ET.SubElement(root, ns + "defs")
    ET.SubElement(defs, ns + "clipPath", {"id": "empty"})
    group = ET.SubElement(root, ns + "g", {"clip-path": "url(#empty)"})
    for child in children:
        group.append(child)
    return ET.tostring(root)


def main():
    n = 10_000
    repeated = adaptive_a = first_a = 0
    labels = [("v", 1, 1, 1), ("v", 2, 2, 2)]
    for i in range(n):
        wm = GeoWatermark(hashlib.sha256(str(i).encode()).digest())
        first = wm._gumbel(labels, [.5, .5]).index
        repeated += wm._gumbel(labels, [.5, .5]).index == first
        adaptive_a += wm._gumbel(labels, [.9, .1] if first == 0 else [.5, .5]).index == 0
        first_a += first == 0

    fixtures = runpy.run_path(str(ROOT / "tests/test_geosample.py"))
    key = fixtures["KEY"]
    svg = fixtures["toy_drawing"](GeoWatermark(key, seed=1), seed=1).svg()
    clean = detect(svg, key)["log10_p_value"]
    subdivisions = []
    attacked = svg
    for _ in range(4):
        attacked = subdivide(attacked)
        subdivisions.append(detect(attacked, key)["log10_p_value"])

    hidden = empty_clip(svg)
    pixels = np.asarray(Image.open(io.BytesIO(render_png(hidden, 128))).convert("RGB"))
    spectral_fixture = runpy.run_path(str(ROOT / "tests/test_spectral.py"))
    spectral_key = spectral_fixture["KEY"]
    spectral_svg, _ = spectral_embed(spectral_fixture["ICON"], spectral_key)
    spectral_hidden = empty_clip(spectral_svg)
    spectral_pixels = np.asarray(Image.open(io.BytesIO(render_png(spectral_hidden, 128))).convert("RGB"))

    def descriptors(path):
        source = f'<svg xmlns="http://www.w3.org/2000/svg"><path d="{path}"/></svg>'.encode()
        return document_descriptors(source, GeoParameters())

    result = {
        "reviewed_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sampler": {"trials": n, "first_A": first_a / n,
                    "repeat_same_choice": repeated / n, "independent_expected": .5,
                    "adaptive_second_A": adaptive_a / n, "unmarked_expected": .7},
        "visibility": {"clean_log10_p": clean, "empty_clip_log10_p": detect(hidden, key)["log10_p_value"],
                       "clipped_render_all_white": bool(np.all(pixels == 255))},
        "spectral_visibility": {
            "clean_log10_p": spectral_detect(spectral_svg, spectral_key)["log10_p_value"],
            "empty_clip_log10_p": spectral_detect(spectral_hidden, spectral_key)["log10_p_value"],
            "clipped_render_all_white": bool(np.all(spectral_pixels == 255)),
        },
        "subdivision_log10_p": subdivisions,
        "equivalent_quadratic_cubic": {
            "quadratic": descriptors("M0 0 Q6 12 12 0 C15 3 24 -6 30 0"),
            "elevated_cubic": descriptors("M0 0 C4 8 8 8 12 0 C15 3 24 -6 30 0"),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
