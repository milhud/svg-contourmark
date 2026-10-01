"""Keyless SVG transformations used to evaluate watermark robustness.

Every attack maps SVG bytes to SVG bytes without access to any watermark key.
Geometric attacks operate on a flattened copy of the document so that element
transforms, basic shapes, and <use> instances are handled uniformly.  External
optimizers (SVGO, Scour, picosvg) and raster round trips (librsvg + vtracer)
are wrapped when installed.
"""

from __future__ import annotations

import io
import json
import math
import random
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

import numpy as np

from .geometry import (
    SVG_NS,
    Segment,
    Subpath,
    _style,
    apply_affine,
    arc_to_cubics,
    dense_points,
    load_document,
    parse_path_data,
    resample,
    serialize_subpaths,
)

ROOT = Path(__file__).resolve().parents[2]
Attack = Callable[[bytes], bytes]


# ---------------------------------------------------------------------------
# Flattening


_INHERITED = ("fill", "stroke", "stroke-width", "fill-rule", "opacity", "fill-opacity", "stroke-opacity", "stroke-linecap", "stroke-linejoin")


def flatten(source: bytes, decimals: int = 6) -> tuple[ET.Element, list[tuple[list[Subpath], dict[str, str]]]]:
    """Return a new root plus per-element global subpaths and computed style."""
    root = ET.fromstring(source)
    document = load_document(root)
    parents = {child: parent for parent in root.iter() for child in parent}
    grouped: dict[int, tuple[list[Subpath], dict[str, str]]] = {}
    order: list[int] = []
    for contour in document.contours:
        if not contour.visible:
            continue
        key = id(contour.element) if contour.editable else id(contour)
        if key not in grouped:
            style: dict[str, str] = {}
            chain = []
            node = contour.element
            while node is not None:
                chain.append(node)
                node = parents.get(node)
            for node in reversed(chain):
                for name, value in _style(node).items():
                    style[name] = value
                for name in ("fill-rule", "stroke-linecap", "stroke-linejoin"):
                    if node.get(name):
                        style[name] = node.get(name)  # type: ignore[assignment]
            grouped[key] = ([], {k: v for k, v in style.items() if k in _INHERITED})
            order.append(key)
        grouped[key][0].append(_transform_subpath(contour.subpath, contour.ctm))
    new_root = ET.Element(f"{{{SVG_NS}}}svg")
    for name in ("viewBox", "width", "height"):
        if root.get(name):
            new_root.set(name, root.get(name))  # type: ignore[arg-type]
    return new_root, [grouped[key] for key in order]


def _transform_subpath(subpath: Subpath, matrix: np.ndarray) -> Subpath:
    linear = matrix[:, :2]
    determinant = float(np.linalg.det(linear))
    similarity = abs(linear[0, 0] - linear[1, 1] * np.sign(determinant)) < 1e-9 and abs(linear[0, 1] + linear[1, 0] * np.sign(determinant)) < 1e-9
    segments: list[Segment] = []
    for segment in subpath.segments:
        if segment.kind == "A" and not similarity:
            pieces = arc_to_cubics(segment)
        else:
            pieces = [segment]
        for piece in pieces:
            arc = piece.arc
            if arc is not None:
                scale = math.sqrt(abs(determinant))
                angle = math.degrees(math.atan2(linear[1, 0], linear[0, 0]))
                arc = (arc[0] * scale, arc[1] * scale, arc[2] + angle, arc[3], arc[4] if determinant > 0 else 1 - arc[4])
            segments.append(Segment(piece.kind, apply_affine(matrix, piece.points), arc))
    return Subpath(segments, subpath.closed)


def assemble(root: ET.Element, elements: list[tuple[list[Subpath], dict[str, str]]], decimals: int = 6) -> bytes:
    for subpaths, style in elements:
        if subpaths:
            ET.SubElement(root, f"{{{SVG_NS}}}path", {"d": serialize_subpaths(subpaths, decimals), **style})
    ET.register_namespace("", SVG_NS)
    return ET.tostring(root, encoding="unicode").encode()


def map_geometry(source: bytes, function: Callable[[Subpath], Subpath | None], decimals: int = 6) -> bytes:
    root, elements = flatten(source)
    mapped = [([s for s in (function(sub) for sub in subpaths) if s is not None], style) for subpaths, style in elements]
    return assemble(root, mapped, decimals)


def _view_box(root: ET.Element) -> tuple[float, float, float, float]:
    values = [float(v) for v in (root.get("viewBox") or "0 0 100 100").replace(",", " ").split()]
    return tuple(values)  # type: ignore[return-value]


def _diagonal(source: bytes) -> float:
    return load_document(source).scale() or 1.0


# ---------------------------------------------------------------------------
# Geometric attacks


def round_absolute(decimals: int) -> Attack:
    return lambda source: map_geometry(source, lambda subpath: subpath, decimals)


def affine(matrix: np.ndarray, fit_view: bool = True) -> Attack:
    def attack(source: bytes) -> bytes:
        root, elements = flatten(source)
        moved = [([_transform_subpath(s, matrix) for s in subpaths], style) for subpaths, style in elements]
        if fit_view:
            points = np.concatenate([dense_points(s) for subpaths, _ in moved for s in subpaths])
            pad = 0.02 * float(abs(complex(np.ptp(points.real), np.ptp(points.imag))))
            root.set("viewBox", f"{points.real.min() - pad:.4f} {points.imag.min() - pad:.4f} {np.ptp(points.real) + 2 * pad:.4f} {np.ptp(points.imag) + 2 * pad:.4f}")
            root.attrib.pop("width", None)
            root.attrib.pop("height", None)
        return assemble(root, moved)
    return attack


def rotation(degrees: float) -> Attack:
    radians = math.radians(degrees)
    return affine(np.array([[math.cos(radians), -math.sin(radians), 0.0], [math.sin(radians), math.cos(radians), 0.0]]))


def scaling(sx: float, sy: float | None = None) -> Attack:
    return affine(np.array([[sx, 0.0, 0.0], [0.0, sx if sy is None else sy, 0.0]]))


def translation(fraction: float) -> Attack:
    def attack(source: bytes) -> bytes:
        shift = fraction * _diagonal(source)
        return affine(np.array([[1.0, 0.0, shift], [0.0, 1.0, -shift]]), fit_view=False)(source)
    return attack


def group_transform(transform: str) -> Attack:
    def attack(source: bytes) -> bytes:
        root = ET.fromstring(source)
        group = ET.Element(f"{{{SVG_NS}}}g", {"transform": transform})
        for child in list(root):
            root.remove(child)
            group.append(child)
        root.append(group)
        ET.register_namespace("", SVG_NS)
        return ET.tostring(root, encoding="unicode").encode()
    return attack


def group_transform_centered(degrees: float = 15.0, scale: float = 0.7) -> Attack:
    """Wrap the drawing in a rotate+scale group about the viewBox centre.

    Keeps the content inside the viewport (geometry pushed off-canvas is not
    drawn and is not scored), while exercising transform flattening.
    """
    def attack(source: bytes) -> bytes:
        x, y, w, h = _view_box(ET.fromstring(source))
        cx, cy = x + w / 2, y + h / 2
        return group_transform(f"translate({cx:g} {cy:g}) rotate({degrees:g}) scale({scale:g}) translate({-cx:g} {-cy:g})")(source)
    return attack


def reverse_direction(source: bytes) -> bytes:
    def reverse(subpath: Subpath) -> Subpath:
        segments = []
        for segment in reversed(subpath.segments):
            arc = segment.arc
            if arc is not None:
                arc = (arc[0], arc[1], arc[2], arc[3], 1 - arc[4])
            segments.append(Segment(segment.kind, segment.points[::-1].copy(), arc))
        return Subpath(segments, subpath.closed)
    return map_geometry(source, reverse)


def rotate_start(source: bytes) -> bytes:
    """Start each closed contour at a different node (common after editing)."""
    def restart(subpath: Subpath) -> Subpath:
        segments = subpath.segments
        closed = subpath.closed or abs(segments[-1].points[-1] - segments[0].points[0]) < 1e-9
        if not closed or len(segments) < 2:
            return subpath
        if abs(segments[-1].points[-1] - segments[0].points[0]) > 1e-9:
            segments = segments + [Segment("L", np.array([segments[-1].points[-1], segments[0].points[0]]))]
        shift = len(segments) // 2
        return Subpath(segments[shift:] + segments[:shift], True)
    return map_geometry(source, restart)


def subdivide(source: bytes) -> bytes:
    """Split every segment at its parameter midpoint (exact geometry)."""
    def split(subpath: Subpath) -> Subpath:
        out: list[Segment] = []
        for segment in subpath.segments:
            if segment.kind == "A":
                out.extend(arc_to_cubics(segment))
                continue
            p = segment.points
            if segment.kind == "L":
                mid = (p[0] + p[1]) / 2
                out += [Segment("L", np.array([p[0], mid])), Segment("L", np.array([mid, p[1]]))]
            elif segment.kind == "Q":
                a, b = (p[0] + p[1]) / 2, (p[1] + p[2]) / 2
                m = (a + b) / 2
                out += [Segment("Q", np.array([p[0], a, m])), Segment("Q", np.array([m, b, p[2]]))]
            else:
                a, b, c = (p[0] + p[1]) / 2, (p[1] + p[2]) / 2, (p[2] + p[3]) / 2
                d, e = (a + b) / 2, (b + c) / 2
                m = (d + e) / 2
                out += [Segment("C", np.array([p[0], a, d, m])), Segment("C", np.array([m, e, c, p[3]]))]
        return Subpath(out, subpath.closed)
    return map_geometry(source, split)


def to_polyline(tolerance_fraction: float) -> Attack:
    """Flatten curves to polylines, then Douglas-Peucker simplify."""
    def attack(source: bytes) -> bytes:
        tolerance = tolerance_fraction * _diagonal(source)

        def flatten_one(subpath: Subpath) -> Subpath | None:
            points = dense_points(subpath)
            keep = _douglas_peucker(points, tolerance)
            if len(keep) < 2:
                return None
            segments = [Segment("L", np.array([a, b])) for a, b in zip(keep[:-1], keep[1:]) if a != b]
            return Subpath(segments, subpath.closed) if segments else None
        return map_geometry(source, flatten_one)
    return attack


def _douglas_peucker(points: np.ndarray, tolerance: float) -> list[complex]:
    unique = [points[0]] + [p for a, p in zip(points[:-1], points[1:]) if p != a]
    if len(unique) < 3 or tolerance <= 0:
        return list(unique)
    array = np.array(unique)
    keep = np.zeros(len(array), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(array) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = array[i], array[j]
        segment = b - a
        middle = array[i + 1:j]
        if abs(segment) == 0:
            distances = np.abs(middle - a)
        else:
            distances = np.abs(np.imag(np.conj(segment) * (middle - a))) / abs(segment)
        k = int(np.argmax(distances))
        if distances[k] > tolerance:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return list(array[keep])


def gaussian_noise(fraction: float, seed: int = 7) -> Attack:
    """Independent noise on every path handle (control points included)."""
    def attack(source: bytes) -> bytes:
        rng = np.random.default_rng(seed)
        sigma = fraction * _diagonal(source)

        def noisy(subpath: Subpath) -> Subpath:
            shared: dict[complex, complex] = {}

            def move(point: complex) -> complex:
                if point not in shared:
                    shared[point] = point + sigma * complex(rng.normal(), rng.normal())
                return shared[point]
            return Subpath([Segment(s.kind, np.array([move(complex(p)) for p in s.points]), s.arc) for s in subpath.segments], subpath.closed)
        return map_geometry(source, noisy)
    return attack


def delete_fraction(fraction: float, seed: int = 11) -> Attack:
    def attack(source: bytes) -> bytes:
        root, elements = flatten(source)
        rng = random.Random(seed)
        items = [(i, j) for i, (subpaths, _) in enumerate(elements) for j in range(len(subpaths))]
        doomed = set(rng.sample(items, int(round(fraction * len(items))))) if items else set()
        kept = [([s for j, s in enumerate(subpaths) if (i, j) not in doomed], style) for i, (subpaths, style) in enumerate(elements)]
        return assemble(root, kept)
    return attack


def crop_half(source: bytes) -> bytes:
    """Keep contours whose centroid lies in the left half of the drawing."""
    root, elements = flatten(source)
    centers = [(i, j, dense_points(s).mean()) for i, (subpaths, _) in enumerate(elements) for j, s in enumerate(subpaths)]
    if not centers:
        return source
    xs = sorted(c.real for _, _, c in centers)
    cut = (xs[0] + xs[-1]) / 2
    keep = {(i, j) for i, j, c in centers if c.real <= cut}
    return assemble(root, [([s for j, s in enumerate(subpaths) if (i, j) in keep], style) for i, (subpaths, style) in enumerate(elements)])


def merge_paths(source: bytes) -> bytes:
    """Concatenate all visible contours into one path element."""
    root, elements = flatten(source)
    if not elements:
        return source
    subpaths = [s for group, _ in elements for s in group]
    return assemble(root, [(subpaths, elements[0][1])])


def split_subpaths(source: bytes) -> bytes:
    root, elements = flatten(source)
    return assemble(root, [([s], style) for subpaths, style in elements for s in subpaths])


def reorder(source: bytes) -> bytes:
    root, elements = flatten(source)
    return assemble(root, elements[::-1])


def compose_with(others: list[bytes]) -> Attack:
    """Place the asset in a 2x2 grid next to unrelated drawings."""
    def attack(source: bytes) -> bytes:
        tiles = [source] + others[:3]
        out = ET.Element(f"{{{SVG_NS}}}svg", {"viewBox": "0 0 200 200"})
        for index, tile in enumerate(tiles):
            root, elements = flatten(tile)
            points = [dense_points(s) for group, _ in elements for s in group]
            if not points:
                continue
            stacked = np.concatenate(points)
            size = max(np.ptp(stacked.real), np.ptp(stacked.imag), 1e-9)
            factor = 90 / size
            dx = 5 + 100 * (index % 2) - stacked.real.min() * factor
            dy = 5 + 100 * (index // 2) - stacked.imag.min() * factor
            matrix = np.array([[factor, 0, dx], [0, factor, dy]])
            group = ET.SubElement(out, f"{{{SVG_NS}}}g")
            for subpaths, style in elements:
                if "stroke-width" in style:
                    style = {**style, "stroke-width": str(float(style["stroke-width"].rstrip("px") or 1) * factor)}
                ET.SubElement(group, f"{{{SVG_NS}}}path", {"d": serialize_subpaths([_transform_subpath(s, matrix) for s in subpaths], 4), **style})
        ET.register_namespace("", SVG_NS)
        return ET.tostring(out, encoding="unicode").encode()
    return attack


# ---------------------------------------------------------------------------
# External tools


class SVGO:
    """Persistent SVGO worker (node) to avoid per-call startup."""

    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None

    def __call__(self, source: bytes, precision: int | None = None, multipass: bool = False) -> bytes:
        if self.process is None or self.process.poll() is not None:
            self.process = subprocess.Popen(
                ["node", str(ROOT / "experiments/svgo_server.mjs")], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
            )
        assert self.process.stdin and self.process.stdout
        self.process.stdin.write(json.dumps({"svg": source.decode("utf-8"), "precision": precision, "multipass": multipass}) + "\n")
        self.process.stdin.flush()
        reply = json.loads(self.process.stdout.readline())
        if "error" in reply:
            raise RuntimeError(f"svgo: {reply['error']}")
        return reply["svg"].encode()


_SVGO = SVGO()


def svgo(precision: int | None = None, multipass: bool = False) -> Attack:
    return lambda source: _SVGO(source, precision, multipass)


def scour(*options: str) -> Attack:
    executable = shutil.which("scour") or str(ROOT / ".venv/bin/scour")

    def attack(source: bytes) -> bytes:
        return subprocess.run([executable, *options], input=source, capture_output=True, check=True).stdout
    return attack


def picosvg(source: bytes) -> bytes:
    from picosvg.svg import SVG

    return SVG.fromstring(source.decode("utf-8")).topicosvg(allow_text=True).tostring().encode()


def render_png(source: bytes, size: int) -> bytes:
    return subprocess.run(
        ["rsvg-convert", "--width", str(size), "--height", str(size), "--keep-aspect-ratio", "--background-color", "white"],
        input=source, capture_output=True, check=True,
    ).stdout


def revectorize(size: int = 512) -> Attack:
    """Rasterize with librsvg and trace the bitmap back to SVG with potrace."""
    def attack(source: bytes) -> bytes:
        from PIL import Image

        image = Image.open(io.BytesIO(render_png(source, size))).convert("L")
        buffer = io.BytesIO()
        image.save(buffer, format="PPM")  # 8-bit PGM; potrace thresholds at 0.5
        traced = subprocess.run(["potrace", "--svg", "--output", "-", "-"], input=buffer.getvalue(), capture_output=True, check=True).stdout
        return re.sub(rb"<!DOCTYPE[^>]*>", b"", traced)
    return attack


def to_cubics(source: bytes) -> bytes:
    """Exact representation change: every line and quadratic becomes a cubic."""
    def elevate(subpath: Subpath) -> Subpath:
        out: list[Segment] = []
        for segment in subpath.segments:
            p = segment.points
            if segment.kind == "L":
                out.append(Segment("C", np.array([p[0], p[0] + (p[1] - p[0]) / 3, p[0] + 2 * (p[1] - p[0]) / 3, p[1]])))
            elif segment.kind == "Q":
                out.append(Segment("C", np.array([p[0], p[0] + 2 * (p[1] - p[0]) / 3, p[2] + 2 * (p[1] - p[2]) / 3, p[2]])))
            elif segment.kind == "A":
                out.extend(arc_to_cubics(segment))
            else:
                out.append(segment)
        return Subpath(out, subpath.closed)
    return map_geometry(source, elevate)


def _split_at(segment: Segment, t: float) -> list[Segment]:
    p = segment.points
    mix = lambda a, b: a + t * (b - a)  # noqa: E731
    if segment.kind == "L":
        m = mix(p[0], p[1])
        return [Segment("L", np.array([p[0], m])), Segment("L", np.array([m, p[1]]))]
    if segment.kind == "Q":
        a, b = mix(p[0], p[1]), mix(p[1], p[2])
        m = mix(a, b)
        return [Segment("Q", np.array([p[0], a, m])), Segment("Q", np.array([m, b, p[2]]))]
    a, b, c = mix(p[0], p[1]), mix(p[1], p[2]), mix(p[2], p[3])
    d, e = mix(a, b), mix(b, c)
    m = mix(d, e)
    return [Segment("C", np.array([p[0], a, d, m])), Segment("C", np.array([m, e, c, p[3]]))]


def subdivide_uneven(t: float = 0.3) -> Attack:
    """Exact de Casteljau split of every segment at an off-centre parameter."""
    def attack(source: bytes) -> bytes:
        def split(subpath: Subpath) -> Subpath:
            out: list[Segment] = []
            for segment in subpath.segments:
                for piece in (arc_to_cubics(segment) if segment.kind == "A" else [segment]):
                    out.extend(_split_at(piece, t))
            return Subpath(out, subpath.closed)
        return map_geometry(source, split)
    return attack


def clip_half(source: bytes) -> bytes:
    """Clip every contour against the left half of the drawing (cuts curves).

    Unlike ``crop_half`` (which keeps or drops whole contours), this cuts
    through shapes: closed contours are clipped as polygons, open ones as
    polylines.  Curves are flattened first, so the result is polylines.
    """
    root, elements = flatten(source)
    clouds = [dense_points(s) for subpaths, _ in elements for s in subpaths]
    if not clouds:
        return source
    xs = np.concatenate(clouds).real
    cut = (xs.min() + xs.max()) / 2

    def cross(a: complex, b: complex) -> complex:
        return a + (cut - a.real) / (b.real - a.real) * (b - a)

    out = []
    for subpaths, style in elements:
        kept: list[Subpath] = []
        for subpath in subpaths:
            points = list(dense_points(subpath))
            closed = subpath.closed or abs(points[-1] - points[0]) < 1e-9
            if closed:  # Sutherland-Hodgman against x <= cut
                result: list[complex] = []
                for a, b in zip(points, points[1:] + points[:1]):
                    if a.real <= cut:
                        result.append(a)
                        if b.real > cut:
                            result.append(cross(a, b))
                    elif b.real <= cut:
                        result.append(cross(a, b))
                runs = [result] if len(result) >= 3 else []
            else:
                runs, current = [], []
                for a, b in zip(points[:-1], points[1:]):
                    if a.real <= cut:
                        current.append(a)
                        if b.real > cut:
                            current.append(cross(a, b))
                            runs.append(current)
                            current = []
                    elif b.real <= cut:
                        current = [cross(a, b)]
                if points[-1].real <= cut:
                    current.append(points[-1])
                if len(current) >= 2:
                    runs.append(current)
            for run in runs:
                segments = [Segment("L", np.array([a, b])) for a, b in zip(run[:-1], run[1:]) if a != b]
                if segments:
                    kept.append(Subpath(segments, closed))
        out.append((kept, style))
    return assemble(root, out, 4)


def chain(*steps: Attack) -> Attack:
    def attack(source: bytes) -> bytes:
        for step in steps:
            source = step(source)
        return source
    return attack


def extended_suite(corpus_others: list[bytes] | None = None) -> dict[str, Attack]:
    """The standard suite plus exact rewrites, real clipping and a pipeline.

    Kept separate so results from the standard suite stay comparable.
    """
    suite = standard_suite(corpus_others)
    suite.update({
        "to_cubics": to_cubics,
        "subdivide_uneven": subdivide_uneven(0.3),
        "clip_half": clip_half,
        "pipeline_svgo_scour_round": chain(svgo(), scour(), round_absolute(2)),
        "pipeline_picosvg_svgo": chain(picosvg, svgo()),
    })
    return suite


def standard_suite(corpus_others: list[bytes] | None = None) -> dict[str, Attack]:
    suite: dict[str, Attack] = {
        "identity": lambda source: source,
        "svgo_default": svgo(),
        "svgo_p2": svgo(2),
        "svgo_p1": svgo(1),
        "svgo_multipass": svgo(None, True),
        "scour_default": scour(),
        "scour_p3": scour("--set-precision=3"),
        "picosvg": picosvg,
        "round_2dp": round_absolute(2),
        "round_1dp": round_absolute(1),
        "round_0dp": round_absolute(0),
        "translate_5pct": translation(0.05),
        "scale_0.37": scaling(0.37),
        "scale_3": scaling(3.0),
        "rotate_30": rotation(30),
        "rotate_90": rotation(90),
        "mirror": affine(np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])),
        "aspect_1.2": scaling(1.2, 1.0),
        "group_transform": group_transform_centered(15.0, 0.7),
        "reorder": reorder,
        "reverse": reverse_direction,
        "restart": rotate_start,
        "subdivide": subdivide,
        "merge_paths": merge_paths,
        "split_subpaths": split_subpaths,
        "delete_25pct": delete_fraction(0.25),
        "delete_50pct": delete_fraction(0.5),
        "crop_half": crop_half,
        "noise_0.1pct": gaussian_noise(0.001),
        "noise_0.3pct": gaussian_noise(0.003),
        "noise_1pct": gaussian_noise(0.01),
        "polyline_0.05pct": to_polyline(0.0005),
        "polyline_0.2pct": to_polyline(0.002),
        "revectorize_512": revectorize(512),
        "revectorize_1024": revectorize(1024),
    }
    if corpus_others:
        suite["compose_grid"] = compose_with(corpus_others)
    return suite
