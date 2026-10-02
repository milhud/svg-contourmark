"""Review probe: invisible / mostly hidden marked geometry still "detected" in strict mode.

    PYTHONPATH=<snapshot>/src OMP_NUM_THREADS=1 .venv/bin/python docs/review-2026-10-02/probe_visibility.py

For each wrapper the marked paths are placed in a document that a renderer
(librsvg, white background, 256 px) draws with zero or almost zero ink, and the
strict-mode status of both detectors is printed.  ``ink`` is the number of
non-white pixels; the unwrapped baseline is the first row.
"""
from __future__ import annotations

import hashlib
import io
import math
import re
import subprocess

import numpy as np
from PIL import Image

from contourmark import spectral
from contourmark.geosample import GeoWatermark, detect as geo_detect
from contourmark.toygen import toy_drawing

KEY = hashlib.sha256(b"review-visibility").digest()


def ink(svg: bytes) -> int:
    try:
        png = subprocess.run(["rsvg-convert", "--width", "256", "--height", "256", "--background-color", "white"],
                             input=svg, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as exc:
        return -1
    return int(np.any(np.asarray(Image.open(io.BytesIO(png)).convert("RGB")) < 250, axis=2).sum())


def inner(svg: bytes) -> str:
    text = svg.decode()
    return text[text.index(">") + 1: text.rindex("</")]


# geosample-marked drawing (toy generator)
state, _ = toy_drawing(GeoWatermark(KEY, seed=3), 3)
geo_inner = re.sub(r"ns0:|:ns0", "", inner(state.svg()))
# spectral-marked drawing: three smooth blobs
blobs = []
for j, (cx, cy) in enumerate([(60, 60), (140, 70), (100, 140)]):
    pts = [complex(cx + (28 + 9 * math.sin(3 * t + j) + 5 * math.cos(5 * t)) * math.cos(t), cy + (28 + 9 * math.sin(3 * t + j) + 5 * math.cos(5 * t)) * math.sin(t))
           for t in np.linspace(0, 2 * math.pi, 13)[:-1]]
    d = f"M{pts[0].real:.3f} {pts[0].imag:.3f}"
    n = len(pts)
    for i in range(n):  # Catmull-Rom to cubic
        p0, p1, p2, p3 = pts[(i - 1) % n], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        c1, c2 = p1 + (p2 - p0) / 6, p2 - (p3 - p1) / 6
        d += f"C{c1.real:.3f} {c1.imag:.3f} {c2.real:.3f} {c2.imag:.3f} {p2.real:.3f} {p2.imag:.3f}"
    blobs.append(f'<path d="{d}Z"/>')
plain = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">{"".join(blobs)}</svg>'.encode()
marked, report = spectral.embed(plain, KEY)
spec_inner = inner(marked)

NS = 'xmlns="http://www.w3.org/2000/svg"'
VB = 'viewBox="0 0 200 200"'
WRAPPERS = {
    "baseline (visible)": lambda x: f'<svg {NS} {VB}>{x}</svg>',
    "fill=transparent": lambda x: f'<svg {NS} {VB}><g fill="transparent">{x.replace(chr(32) + "fill=" + chr(34) + "black" + chr(34), "")}</g></svg>',
    "fill=rgba(0,0,0,0)": lambda x: f'<svg {NS} {VB}><g fill="rgba(0,0,0,0)">{x.replace(" fill=" + chr(34) + "black" + chr(34), "")}</g></svg>',
    "fill=white (background colour)": lambda x: f'<svg {NS} {VB}><g fill="#fff">{x.replace(" fill=" + chr(34) + "black" + chr(34), "")}</g></svg>',
    "fill=url(#missing)": lambda x: f'<svg {NS} {VB}><g fill="url(#missing)">{x.replace(" fill=" + chr(34) + "black" + chr(34), "")}</g></svg>',
    "ancestor opacity=0.001": lambda x: f'<svg {NS} {VB}><g opacity="0.001">{x}</g></svg>',
    "ancestor style opacity:0.002": lambda x: f'<svg {NS} {VB}><g style="opacity:0.002">{x}</g></svg>',
    "ancestor opacity=-1 (clamps to 0)": lambda x: f'<svg {NS} {VB}><g opacity="-1">{x}</g></svg>',
    "style display:none !important": lambda x: f'<svg {NS} {VB}><g style="display:none !important">{x}</g></svg>',
    "style DISPLAY:NONE (case)": lambda x: f'<svg {NS} {VB}><g style="DISPLAY:NONE">{x}</g></svg>',
    "parent visibility=hidden, child visibility=inherit": lambda x: f'<svg {NS} {VB}><g visibility="hidden"><g visibility="inherit">{x}</g></g></svg>',
    "parent display=none attr, child style display:inline": lambda x: f'<svg {NS} {VB}><g display="none"><g style="display:inline">{x}</g></g></svg>',
    "zero-size viewBox": lambda x: f'<svg {NS} viewBox="0 0 0 0">{x}</svg>',
    "root width=0 height=0": lambda x: f'<svg {NS} {VB} width="0" height="0">{x}</svg>',
    "no viewBox, width/height=200, translate far off-canvas": lambda x: f'<svg {NS} width="200" height="200"><g transform="translate(100000 100000)">{x}</g></svg>',
    "transform scale(0.0005)": lambda x: f'<svg {NS} {VB}><g transform="scale(0.0005)">{x}</g></svg>',
    "transform matrix(1e-4 0 0 1e-4 100 100)": lambda x: f'<svg {NS} {VB}><g transform="matrix(0.0001 0 0 0.0001 100 100)">{x}</g></svg>',
    "viewport shows one corner only (viewBox 0 0 200 200, translate 150 150)": lambda x: f'<svg {NS} {VB}><g transform="translate(150 150)">{x}</g></svg>',
    "occluded by opaque white rect drawn later": lambda x: f'<svg {NS} {VB}>{x}<rect width="200" height="200" fill="#fff"/></svg>',
    "<switch>, second (never rendered) child": lambda x: f'<svg {NS} {VB}><switch><g/><g>{x}</g></switch></svg>',
    "systemLanguage that never matches": lambda x: f'<svg {NS} {VB}><g systemLanguage="x-none">{x}</g></svg>',
    "requiredExtensions unsupported": lambda x: f'<svg {NS} {VB}><g requiredExtensions="http://example.invalid/ext">{x}</g></svg>',
    "wrapper element in a foreign namespace": lambda x: f'<svg {NS} {VB}><x:g xmlns:x="urn:not-svg">{x}</x:g></svg>',
    "<use> of a zero-size <symbol> viewport": lambda x: f'<svg {NS} {VB}><symbol id="s" viewBox="0 0 200 200">{x}</symbol><use href="#s" width="0" height="0"/></svg>',
    "fill=none, stroke-width=0em": lambda x: f'<svg {NS} {VB}><g fill="none" stroke="black" stroke-width="0em">{x.replace(" fill=" + chr(34) + "black" + chr(34), "")}</g></svg>',
    "fill=none, stroke=transparent": lambda x: f'<svg {NS} {VB}><g fill="none" stroke="transparent">{x.replace(" fill=" + chr(34) + "black" + chr(34), "")}</g></svg>',
    "fill=none, stroke-width=1e-7": lambda x: f'<svg {NS} {VB}><g fill="none" stroke="black" stroke-width="1e-7">{x.replace(" fill=" + chr(34) + "black" + chr(34), "")}</g></svg>',
    "fill=none, stroke-dasharray 0 9999": lambda x: f'<svg {NS} {VB}><g fill="none" stroke="black" stroke-dasharray="0 9999">{x.replace(" fill=" + chr(34) + "black" + chr(34), "")}</g></svg>',
    # controls that the parser is documented to handle (expected: not detected / indeterminate)
    "CONTROL display=none": lambda x: f'<svg {NS} {VB}><g display="none">{x}</g></svg>',
    "CONTROL style opacity:0": lambda x: f'<svg {NS} {VB}><g style="opacity:0">{x}</g></svg>',
    "CONTROL fill=none stroke-width=0": lambda x: f'<svg {NS} {VB}><g fill="none" stroke="black" stroke-width="0">{x.replace(" fill=" + chr(34) + "black" + chr(34), "")}</g></svg>',
    "CONTROL <use> of defs path, use display=none": lambda x: f'<svg {NS} {VB}><defs><g id="d">{x}</g></defs><use href="#d" display="none"/></svg>',
}

print(f"{'wrapper':58s} | ink(px) | geosample strict        | spectral strict       | geosample render-mode")
for name, wrap in WRAPPERS.items():
    row = []
    inks = []
    for content, det in ((geo_inner, lambda s: geo_detect(s, KEY)), (spec_inner, lambda s: spectral.detect(s, KEY))):
        svg = wrap(content).encode()
        inks.append(ink(svg))
        try:
            r = det(svg)
            row.append(f"{r['status']:13s} {r['log10_p_value']:7.1f}")
        except Exception as exc:  # noqa: BLE001
            row.append(f"ERROR {type(exc).__name__}: {exc}"[:22])
    try:
        rendered = geo_detect(wrap(geo_inner).encode(), KEY, visibility="render")["status"]
    except Exception as exc:  # noqa: BLE001
        rendered = f"ERROR {type(exc).__name__}"
    print(f"{name[:58]:58s} | {inks[0]:5d}/{inks[1]:5d} | {row[0]} | {row[1]} | {rendered}")
