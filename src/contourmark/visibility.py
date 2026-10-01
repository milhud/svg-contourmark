"""Renderer-assisted visibility for contours the parser cannot resolve.

``geometry.load_document`` flags contours under ``clip-path``, ``mask``,
``filter`` or a nested viewport as *unresolved*, and documents with CSS rules
as indeterminate, because it does not implement those parts of SVG.  The
strict detectors exclude such contours.

This module asks a real renderer instead.  For each unresolved element it
renders the document with and without that element and compares the changed
pixels with the element's own unclipped footprint:

* contribution >= ``min_fraction`` of the footprint: resolved, visible;
* no changed pixel: resolved, invisible (fully clipped, masked, occluded,
  or painted in the background colour);
* otherwise: left unresolved (only a fragment is visible).

It is a measurement with one renderer (librsvg) at one resolution, not a
proof.  Scripts and animation are never resolved.
"""

from __future__ import annotations

import io
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from .geometry import SVG_NS, Contour, Document, GeometryError, Subpath, Segment, apply_affine, load_document, parse_svg, serialize_subpaths


def _render(root: ET.Element, size: int) -> np.ndarray:
    from PIL import Image

    data = ET.tostring(root)
    try:
        png = subprocess.run(
            ["rsvg-convert", "--width", str(size), "--height", str(size), "--keep-aspect-ratio", "--background-color", "white"],
            input=data, capture_output=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GeometryError(f"renderer unavailable or failed: {exc}") from exc
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.int16)


def _footprint(document: Document, contours: list[Contour], size: int) -> int:
    """Pixels the contours would cover if drawn alone, unclipped, in black."""
    root = ET.Element(f"{{{SVG_NS}}}svg")
    for name in ("viewBox", "width", "height"):
        if document.root.get(name):
            root.set(name, document.root.get(name))  # type: ignore[arg-type]
    for contour in contours:
        segments = []
        for segment in contour.subpath.segments:
            arc = segment.arc
            if arc is not None:  # keep it simple: arcs are drawn by their chord-preserving points
                arc = None
            segments.append(Segment(segment.kind if segment.arc is None else "L", apply_affine(contour.ctm, segment.points), arc))
        d = serialize_subpaths([Subpath(segments, contour.subpath.closed)], 4)
        scale = float(np.sqrt(abs(np.linalg.det(contour.ctm[:, :2])))) or 1.0
        element = contour.element
        stroke = (element.get("stroke") if element is not None else None) or "none"
        width = (element.get("stroke-width") if element is not None else None) or "1"
        try:
            stroke_width = float(width.rstrip("px")) * scale
        except ValueError:
            stroke_width = scale
        fill = (element.get("fill") if element is not None else None) or "black"
        ET.SubElement(root, f"{{{SVG_NS}}}path", {
            "d": d, "fill": "none" if fill == "none" else "black",
            "stroke": "black" if stroke != "none" or fill == "none" else "none", "stroke-width": f"{stroke_width:g}",
        })
    return int(np.any(_render(root, size) < 250, axis=2).sum())


def resolve_by_rendering(source: bytes | ET.Element, size: int = 256, min_fraction: float = 0.5) -> Document:
    """Load a document and resolve unresolved contours with librsvg."""
    root = parse_svg(source) if isinstance(source, bytes) else source
    document = load_document(root)
    css = "css" in document.unresolved_features
    pending: dict[int, list[Contour]] = {}
    elements: dict[int, ET.Element] = {}
    for contour in document.contours:
        if contour.element is None or not contour.visible:
            continue
        if contour.unresolved or css:
            pending.setdefault(id(contour.element), []).append(contour)
            elements[id(contour.element)] = contour.element
    if not pending:
        return document
    base = _render(root, size)
    verified_all = True
    for key, contours in pending.items():
        element = elements[key]
        previous = element.get("display")
        element.set("display", "none")
        try:
            without = _render(root, size)
        finally:
            if previous is None:
                element.attrib.pop("display", None)
            else:
                element.set("display", previous)
        changed = int(np.any(np.abs(base - without) > 8, axis=2).sum())
        footprint = max(_footprint(document, contours, size), 1)
        if changed == 0:
            for contour in contours:
                contour.visible = False
                contour.unresolved = ()
        elif changed >= min_fraction * footprint:
            for contour in contours:
                contour.unresolved = ()
        else:
            verified_all = False
            for contour in contours:
                contour.unresolved = contour.unresolved or ("partially-visible",)
    if css and verified_all:
        document.unresolved_features = tuple(f for f in document.unresolved_features if f != "css")
    return document
