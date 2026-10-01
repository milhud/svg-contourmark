"""Graphics-facing fidelity and editability metrics for marked SVGs.

Image PSNR punishes sub-pixel edge shifts on line art and says nothing about
what a designer cares about.  These metrics describe a marked file in the
terms of the drawing itself:

* boundary distance (symmetric, in document units and in pixels at chosen
  render sizes);
* silhouette overlap and colour error on several backgrounds (transparency
  and colour changes are invisible on white alone);
* curve smoothness (soft kinks introduced at joins);
* structure and editability (segments, primitives kept, bytes).

Rendering uses librsvg (``rsvg-convert``); the geometric metrics need no
renderer.
"""

from __future__ import annotations

import io
import math
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

from .geometry import Document, GeometryError, apply_affine, dense_points, load_document, tag

ICON_SIZES = (16, 24, 32, 64, 128)


def _boundary_points(document: Document, spacing_fraction: float = 0.002) -> np.ndarray:
    """Dense points on every scorable contour, in document coordinates."""
    scale = document.scale() or 1.0
    spacing = spacing_fraction * scale
    clouds = []
    for contour in document.contours:
        if not contour.visible:
            continue
        points = apply_affine(contour.ctm, dense_points(contour.subpath))
        if contour.closed and abs(points[-1] - points[0]) > 0:
            points = np.append(points, points[0])
        steps = np.abs(np.diff(points))
        length = float(steps.sum())
        if length <= 0:
            continue
        cumulative = np.concatenate([[0.0], np.cumsum(steps)])
        targets = np.linspace(0.0, length, max(8, int(length / spacing) + 1))
        keep = np.concatenate([[True], steps > 0])
        clouds.append(np.interp(targets, cumulative[keep], points.real[keep]) + 1j * np.interp(targets, cumulative[keep], points.imag[keep]))
    if not clouds:
        return np.zeros(0, dtype=complex)
    return np.concatenate(clouds)


def boundary_distance(original: bytes, marked: bytes, sizes: tuple[int, ...] = ICON_SIZES + (256, 1024)) -> dict:
    """Symmetric boundary distance between two drawings.

    Each direction measures, for dense points on one drawing's outlines, the
    distance to the nearest outline point of the other; the symmetric value
    takes the larger direction, so neither missing nor added geometry hides.
    The reference outline is sampled at 0.02% of the diagonal, which bounds
    the measurement floor at about 0.01%.
    """
    a_doc, b_doc = load_document(original), load_document(marked)
    a, b = _boundary_points(a_doc), _boundary_points(b_doc)
    if not len(a) or not len(b):
        raise GeometryError("no visible contours to compare")
    # Query points at 0.2% spacing against reference clouds ten times denser,
    # so the nearest-point floor is about 0.01% of the diagonal.
    fine_a, fine_b = _boundary_points(a_doc, 0.0002), _boundary_points(b_doc, 0.0002)
    tree_a = cKDTree(np.column_stack([fine_a.real, fine_a.imag]))
    tree_b = cKDTree(np.column_stack([fine_b.real, fine_b.imag]))
    a_to_b = tree_b.query(np.column_stack([a.real, a.imag]))[0]
    b_to_a = tree_a.query(np.column_stack([b.real, b.imag]))[0]
    both = np.concatenate([a_to_b, b_to_a])
    diagonal = a_doc.scale() or 1.0
    box = a_doc.view_box
    extent = max(box[2], box[3]) if box else diagonal / math.sqrt(2)
    summary = {
        "mean": float(both.mean()), "p95": float(np.percentile(both, 95)),
        "hausdorff": float(max(a_to_b.max(), b_to_a.max())),
    }
    return {
        "units": summary,
        "fraction_of_diagonal": {k: v / diagonal for k, v in summary.items()},
        # Pixels when the viewBox's longer side is rendered at each size.
        "pixels": {str(size): {k: v * size / extent for k, v in summary.items()} for size in sizes},
    }


def _render_rgba(source: bytes, size: int, background: str | None) -> np.ndarray:
    from PIL import Image

    command = ["rsvg-convert", "--width", str(size), "--height", str(size), "--keep-aspect-ratio"]
    if background:
        command += ["--background-color", background]
    try:
        png = subprocess.run(command, input=source, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GeometryError(f"renderer unavailable or failed: {exc}") from exc
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"), dtype=np.float64) / 255.0


def render_fidelity(original: bytes, marked: bytes, sizes: tuple[int, ...] = (32, 64, 256)) -> dict:
    """Colour error on white and dark backgrounds, and silhouette overlap."""
    out: dict = {}
    for size in sizes:
        entry: dict = {}
        for name, colour in (("white", "white"), ("dark", "#101010")):
            a, b = _render_rgba(original, size, colour), _render_rgba(marked, size, colour)
            diff = np.abs(a[..., :3] - b[..., :3])
            mse = float((diff ** 2).mean())
            entry[name] = {"mae": float(diff.mean()), "psnr": 99.0 if mse == 0 else float(10 * math.log10(1 / mse)), "max": float(diff.max())}
        # Coverage from the alpha channel of a transparent render.
        alpha_a, alpha_b = _render_rgba(original, size, None)[..., 3] > 0.5, _render_rgba(marked, size, None)[..., 3] > 0.5
        union = np.logical_or(alpha_a, alpha_b).sum()
        entry["silhouette_iou"] = float(np.logical_and(alpha_a, alpha_b).sum() / union) if union else 1.0
        out[str(size)] = entry
    return out


def _join_angles(document: Document) -> np.ndarray:
    """Tangent discontinuity (degrees) at every interior join of every contour."""
    angles = []
    for contour in document.contours:
        if not contour.visible:
            continue
        segments = contour.subpath.segments
        for before, after in zip(segments[:-1], segments[1:]):
            arriving = apply_affine(contour.ctm, before.points)
            departing = apply_affine(contour.ctm, after.points)
            a = arriving[-1] - arriving[-2]
            b = departing[1] - departing[0]
            if abs(a) > 1e-12 and abs(b) > 1e-12:
                angles.append(abs(math.degrees(np.angle(b / a))))
    return np.array(angles)


def smoothness(original: bytes, marked: bytes, soft: tuple[float, float] = (0.5, 12.0)) -> dict:
    """Soft kinks: joins that are neither smooth nor a deliberate corner.

    A join under 0.5 degrees reads as smooth and one over 12 degrees as a
    corner; in between is a visible, unintended kink at high zoom.  Arc
    segments are approximated by their chords here, so treat counts on
    arc-heavy files as upper bounds.
    """
    a, b = _join_angles(load_document(original)), _join_angles(load_document(marked))
    count = lambda x: int(((x > soft[0]) & (x < soft[1])).sum())  # noqa: E731
    return {
        "original_joins": int(len(a)), "marked_joins": int(len(b)),
        "original_soft_kinks": count(a), "marked_soft_kinks": count(b),
        "marked_soft_kink_fraction": float(count(b) / len(b)) if len(b) else 0.0,
    }


def structure(source: bytes) -> dict:
    """Editability-relevant structure of one SVG."""
    root = ET.fromstring(source)
    tags = Counter(tag(element) for element in root.iter())
    document = load_document(source)
    kinds = Counter(segment.kind for contour in document.contours for segment in contour.subpath.segments)
    return {
        "bytes": len(source),
        "elements": dict(tags),
        "primitives": {name: tags.get(name, 0) for name in ("rect", "circle", "ellipse", "line", "polyline", "polygon")},
        "paths": tags.get("path", 0),
        "contours": len(document.contours),
        "segments": int(sum(kinds.values())),
        "segments_by_kind": dict(kinds),
    }


def editability(original: bytes, marked: bytes) -> dict:
    a, b = structure(original), structure(marked)
    return {
        "byte_ratio": b["bytes"] / a["bytes"],
        "segment_ratio": b["segments"] / a["segments"] if a["segments"] else None,
        "primitives_lost": {k: a["primitives"][k] - b["primitives"][k] for k in a["primitives"] if a["primitives"][k] != b["primitives"][k]},
        "arcs_lost": a["segments_by_kind"].get("A", 0) - b["segments_by_kind"].get("A", 0),
        "contours_before": a["contours"], "contours_after": b["contours"],
    }


def fidelity_report(original: bytes, marked: bytes, render: bool = True) -> dict:
    """All metrics for one (original, marked) pair."""
    report = {
        "boundary": boundary_distance(original, marked),
        "smoothness": smoothness(original, marked),
        "editability": editability(original, marked),
    }
    if render:
        report["render"] = render_fidelity(original, marked)
    return report
