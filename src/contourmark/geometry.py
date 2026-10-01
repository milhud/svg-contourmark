"""Representation-independent SVG contour geometry.

This module parses path data and basic shapes into explicit segments, flattens
the element transform stack, and samples each continuous contour uniformly by
arc length with vectorized NumPy.  Watermark features computed from these
samples depend on the drawn curve, not on its serialization: absolute versus
relative commands, shorthand curves, segment subdivision, numeric formatting,
and path reordering all produce the same samples up to numerical tolerance.

The parser keeps each segment's type so an embedder can move the existing
control points and serialize the result without flattening curves to
polylines.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np


class GeometryError(ValueError):
    pass


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_TOKEN = re.compile(rf"([MmLlHhVvCcSsQqTtAaZz])|({_NUMBER})")
_FLAG = re.compile(r"\s*,?\s*([01])")
_NUM = re.compile(rf"\s*,?\s*({_NUMBER})")
_NON_RENDERED = {
    "defs", "clipPath", "mask", "symbol", "pattern", "marker", "metadata", "title", "desc", "style",
    "script", "linearGradient", "radialGradient", "filter", "foreignObject", "text", "image",
}
_ARITY = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}


@dataclass
class Segment:
    """One drawn segment in element-local coordinates.

    ``points`` holds complex control points including both endpoints:
    2 for lines, 3 for quadratics, 4 for cubics, 2 for arcs.  Arcs keep the
    SVG endpoint parameters in ``arc`` = (rx, ry, rotation_degrees, large, sweep).
    """

    kind: str
    points: np.ndarray
    arc: tuple[float, float, float, int, int] | None = None


@dataclass
class Subpath:
    segments: list[Segment]
    closed: bool  # an explicit Z command was present

    @property
    def start(self) -> complex:
        return complex(self.segments[0].points[0])


@dataclass
class Contour:
    """A continuous drawn contour with its element and global transform."""

    element: ET.Element | None
    subpath_index: int
    subpath: Subpath
    ctm: np.ndarray  # 2x3 affine matrix, local -> document coordinates
    visible: bool
    editable: bool  # False for shapes and <use> instances
    closed: bool = field(init=False)

    def __post_init__(self) -> None:
        self.closed = self.subpath.closed or closes(self.subpath)


CLOSURE_TOLERANCE = 5e-3  # fraction of control-polygon length


def closes(subpath: Subpath) -> bool:
    """Whether a subpath ends where it starts, up to optimizer rounding.

    Optimizers may drop a redundant closing segment or perturb the final point
    of an implicitly closed outline; a strict equality test would flip such a
    contour between the closed and open feature families.
    """
    polygon = sum(float(np.abs(np.diff(segment.points)).sum()) for segment in subpath.segments)
    gap = abs(complex(subpath.segments[-1].points[-1]) - subpath.start)
    return gap <= 1e-9 + CLOSURE_TOLERANCE * polygon


def _near(a: complex, b: complex, subpath: Subpath) -> bool:
    extent = max((abs(seg.points[-1] - seg.points[0]) for seg in subpath.segments), default=0.0)
    return abs(complex(a) - complex(b)) <= 1e-9 + 1e-7 * extent


def tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1] if isinstance(element.tag, str) else ""


# ---------------------------------------------------------------------------
# Path data parsing


def _read_args(text: str, position: int, command: str) -> tuple[list[float], int] | None:
    values: list[float] = []
    upper = command.upper()
    for index in range(_ARITY[upper]):
        pattern = _FLAG if upper == "A" and index in (3, 4) else _NUM
        match = pattern.match(text, position)
        if not match:
            return None if index == 0 else _raise(f"incomplete arguments for {command}")
        values.append(float(match.group(1)))
        position = match.end()
    return values, position


def _raise(message: str):
    raise GeometryError(message)


def parse_path_data(d: str) -> list[Subpath]:
    """Parse SVG path data into explicit absolute segments."""
    subpaths: list[Subpath] = []
    segments: list[Segment] = []
    current = 0j
    start = 0j
    last_control: complex | None = None
    last_kind = ""
    position = 0
    command = ""
    text = d.strip()

    def finish(closed: bool) -> None:
        nonlocal segments
        if segments:
            subpaths.append(Subpath(segments, closed))
        segments = []

    while position < len(text):
        while position < len(text) and text[position] in " \t\r\n,":
            position += 1
        if position >= len(text):
            break
        if text[position].isalpha():
            command = text[position]
            position += 1
            if command not in "MmLlHhVvCcSsQqTtAaZz":
                raise GeometryError(f"unknown path command {command!r}")
            if command in "Zz":
                if segments and not _near(current, start, Subpath(segments, True)):
                    segments.append(Segment("L", np.array([current, start], dtype=complex)))
                finish(True)
                current = start
                last_kind = "Z"
                continue
        elif not command or command in "Zz":
            raise GeometryError("path data must begin with a command")
        parsed = _read_args(text, position, command)
        if parsed is None:
            raise GeometryError(f"missing arguments for {command}")
        args, position = parsed
        relative = command.islower()
        upper = command.upper()
        base = current if relative else 0j

        def pt(x: float, y: float) -> complex:
            return complex(x, y) + base

        if upper == "M":
            finish(False)
            current = start = pt(args[0], args[1])
            command = "l" if relative else "L"  # implicit lineto for extra pairs
            last_kind = "M"
            continue
        if upper in "LHV":
            if upper == "L":
                end = pt(args[0], args[1])
            elif upper == "H":
                end = complex(args[0] + (current.real if relative else 0), current.imag)
            else:
                end = complex(current.real, args[0] + (current.imag if relative else 0))
            segments.append(Segment("L", np.array([current, end], dtype=complex)))
            current, last_control = end, None
        elif upper in "CS":
            if upper == "C":
                c1, c2, end = pt(args[0], args[1]), pt(args[2], args[3]), pt(args[4], args[5])
            else:
                c1 = 2 * current - last_control if last_kind in "CS" and last_control is not None else current
                c2, end = pt(args[0], args[1]), pt(args[2], args[3])
            segments.append(Segment("C", np.array([current, c1, c2, end], dtype=complex)))
            current, last_control = end, c2
        elif upper in "QT":
            if upper == "Q":
                c1, end = pt(args[0], args[1]), pt(args[2], args[3])
            else:
                c1 = 2 * current - last_control if last_kind in "QT" and last_control is not None else current
                end = pt(args[0], args[1])
            segments.append(Segment("Q", np.array([current, c1, end], dtype=complex)))
            current, last_control = end, c1
        else:  # arc
            end = pt(args[5], args[6])
            if abs(end - current) > 0:
                if args[0] == 0 or args[1] == 0:
                    segments.append(Segment("L", np.array([current, end], dtype=complex)))
                else:
                    segments.append(Segment("A", np.array([current, end], dtype=complex), (abs(args[0]), abs(args[1]), args[2], int(args[3]), int(args[4]))))
            current, last_control = end, None
        last_kind = upper
    finish(False)
    return subpaths


# ---------------------------------------------------------------------------
# Transforms and shapes

_IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    a = np.vstack([outer, [0, 0, 1]])
    b = np.vstack([inner, [0, 0, 1]])
    return (a @ b)[:2]


def parse_transform(text: str | None) -> np.ndarray:
    matrix = _IDENTITY.copy()
    if not text:
        return matrix
    for name, body in re.findall(r"([a-zA-Z]+)\s*\(([^)]*)\)", text):
        values = [float(v) for v in re.findall(_NUMBER, body)]
        if name == "matrix" and len(values) == 6:
            a, b, c, d, e, f = values
            step = np.array([[a, c, e], [b, d, f]])
        elif name == "translate" and values:
            step = np.array([[1, 0, values[0]], [0, 1, values[1] if len(values) > 1 else 0.0]])
        elif name == "scale" and values:
            sx = values[0]
            sy = values[1] if len(values) > 1 else sx
            step = np.array([[sx, 0, 0], [0, sy, 0]])
        elif name == "rotate" and values:
            angle = math.radians(values[0])
            cos, sin = math.cos(angle), math.sin(angle)
            step = np.array([[cos, -sin, 0], [sin, cos, 0]])
            if len(values) == 3:
                cx, cy = values[1], values[2]
                step = _compose(_compose(np.array([[1, 0, cx], [0, 1, cy]]), step), np.array([[1, 0, -cx], [0, 1, -cy]]))
        elif name == "skewX" and values:
            step = np.array([[1, math.tan(math.radians(values[0])), 0], [0, 1, 0]])
        elif name == "skewY" and values:
            step = np.array([[1, 0, 0], [math.tan(math.radians(values[0])), 1, 0]])
        else:
            raise GeometryError(f"unsupported transform {name}")
        matrix = _compose(matrix, step)
    return matrix


def apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    x, y = points.real, points.imag
    return (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) + 1j * (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2])


def apply_linear_inverse(matrix: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    linear = matrix[:, :2]
    inverse = np.linalg.inv(linear)
    x, y = vectors.real, vectors.imag
    return (inverse[0, 0] * x + inverse[0, 1] * y) + 1j * (inverse[1, 0] * x + inverse[1, 1] * y)


def _length(value: str | None, default: float = 0.0) -> float:
    if value is None or not value.strip():
        return default
    match = re.match(rf"\s*({_NUMBER})", value)
    if not match:
        raise GeometryError(f"unsupported length {value!r}")
    return float(match.group(1))


_KAPPA = 0.5522847498307936


def _ellipse_d(cx: float, cy: float, rx: float, ry: float) -> str:
    k = _KAPPA
    return (
        f"M{cx + rx} {cy}C{cx + rx} {cy + k * ry} {cx + k * rx} {cy + ry} {cx} {cy + ry}"
        f"C{cx - k * rx} {cy + ry} {cx - rx} {cy + k * ry} {cx - rx} {cy}"
        f"C{cx - rx} {cy - k * ry} {cx - k * rx} {cy - ry} {cx} {cy - ry}"
        f"C{cx + k * rx} {cy - ry} {cx + rx} {cy - k * ry} {cx + rx} {cy}Z"
    )


def shape_to_d(element: ET.Element) -> str | None:
    """Return equivalent path data for a basic shape, or None."""
    name = tag(element)
    get = element.get
    if name == "path":
        return get("d", "")
    if name == "rect":
        x, y, w, h = _length(get("x")), _length(get("y")), _length(get("width")), _length(get("height"))
        if w <= 0 or h <= 0:
            return None
        rx_text, ry_text = get("rx"), get("ry")
        rx = _length(rx_text) if rx_text not in (None, "auto") else None
        ry = _length(ry_text) if ry_text not in (None, "auto") else None
        rx = ry if rx is None else rx
        ry = rx if ry is None else ry
        rx, ry = min(max(rx or 0.0, 0.0), w / 2), min(max(ry or 0.0, 0.0), h / 2)
        if rx == 0 or ry == 0:
            return f"M{x} {y}H{x + w}V{y + h}H{x}Z"
        k = _KAPPA
        return (
            f"M{x + rx} {y}H{x + w - rx}C{x + w - rx + k * rx} {y} {x + w} {y + ry - k * ry} {x + w} {y + ry}"
            f"V{y + h - ry}C{x + w} {y + h - ry + k * ry} {x + w - rx + k * rx} {y + h} {x + w - rx} {y + h}"
            f"H{x + rx}C{x + rx - k * rx} {y + h} {x} {y + h - ry + k * ry} {x} {y + h - ry}"
            f"V{y + ry}C{x} {y + ry - k * ry} {x + rx - k * rx} {y} {x + rx} {y}Z"
        )
    if name == "circle":
        r = _length(get("r"))
        return _ellipse_d(_length(get("cx")), _length(get("cy")), r, r) if r > 0 else None
    if name == "ellipse":
        rx, ry = _length(get("rx")), _length(get("ry"))
        return _ellipse_d(_length(get("cx")), _length(get("cy")), rx, ry) if rx > 0 and ry > 0 else None
    if name == "line":
        return f"M{_length(get('x1'))} {_length(get('y1'))}L{_length(get('x2'))} {_length(get('y2'))}"
    if name in {"polyline", "polygon"}:
        numbers = [float(v) for v in re.findall(_NUMBER, get("points", ""))]
        if len(numbers) < 4:
            return None
        pairs = list(zip(numbers[::2], numbers[1::2]))
        d = f"M{pairs[0][0]} {pairs[0][1]}" + "".join(f"L{x} {y}" for x, y in pairs[1:])
        return d + ("Z" if name == "polygon" else "")
    return None


# ---------------------------------------------------------------------------
# Document traversal


def _style(element: ET.Element) -> dict[str, str]:
    values = {k: v.strip() for k, v in element.attrib.items() if k in _PAINT}
    for part in element.get("style", "").split(";"):
        if ":" in part:
            name, value = part.split(":", 1)
            if name.strip() in _PAINT:
                values[name.strip()] = value.strip()
    return values


_PAINT = {"display", "visibility", "opacity", "fill", "stroke", "fill-opacity", "stroke-opacity", "stroke-width"}


def _number(value: str, default: float) -> float:
    try:
        return float(value.strip().removesuffix("%").removesuffix("px"))
    except ValueError:
        return default


def _visible(chain: list[dict[str, str]]) -> bool:
    paint: dict[str, str] = {}
    for declarations in chain:  # outermost first; inner overrides
        if declarations.get("display") == "none" or _number(declarations.get("opacity", "1"), 1.0) == 0:
            return False
        paint.update(declarations)
    if paint.get("visibility") in {"hidden", "collapse"}:
        return False
    fill_visible = paint.get("fill", "black") != "none" and _number(paint.get("fill-opacity", "1"), 1.0) != 0
    stroke_visible = (
        paint.get("stroke", "none") != "none"
        and _number(paint.get("stroke-opacity", "1"), 1.0) != 0
        and _number(paint.get("stroke-width", "1"), 1.0) != 0
    )
    return fill_visible or stroke_visible


@dataclass
class Document:
    root: ET.Element
    contours: list[Contour]
    view_box: tuple[float, float, float, float] | None

    def scale(self) -> float:
        """Diagonal of the drawn geometry's bounding box."""
        points = [sample_contour(c, 64) for c in self.contours]
        if not points:
            return 0.0
        stacked = np.concatenate(points)
        return float(abs(complex(np.ptp(stacked.real), np.ptp(stacked.imag))))


def parse_svg(source: bytes) -> ET.Element:
    upper = source.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise GeometryError("DTD/entity declarations are not accepted")
    try:
        return ET.fromstring(source)
    except ET.ParseError as exc:
        raise GeometryError(f"invalid SVG XML: {exc}") from exc


def load_document(source: bytes | ET.Element, include_shapes: bool = True) -> Document:
    root = parse_svg(source) if isinstance(source, bytes) else source
    if tag(root) != "svg":
        raise GeometryError("root element must be svg")
    ids = {element.get("id"): element for element in root.iter() if element.get("id")}
    contours: list[Contour] = []

    def visit(element: ET.Element, ctm: np.ndarray, styles: list[dict[str, str]], editable: bool, depth: int) -> None:
        name = tag(element)
        if name in _NON_RENDERED or depth > 32:
            return
        local = parse_transform(element.get("transform"))
        matrix = _compose(ctm, local)
        chain = styles + [_style(element)]
        if name == "use":
            href = element.get("href") or element.get(f"{{{XLINK_NS}}}href") or ""
            target = ids.get(href[1:]) if href.startswith("#") else None
            if target is not None and target is not element:
                shift = np.array([[1, 0, _length(element.get("x"))], [0, 1, _length(element.get("y"))]])
                if tag(target) == "symbol":
                    for child in target:
                        visit(child, _compose(matrix, shift), chain, False, depth + 1)
                else:
                    visit(target, _compose(matrix, shift), chain, False, depth + 1)
            return
        if name in {"svg", "g", "a", "switch"}:
            if name == "svg" and element is not root:
                matrix = _compose(matrix, np.array([[1, 0, _length(element.get("x"))], [0, 1, _length(element.get("y"))]]))
            for child in element:
                visit(child, matrix, chain, editable, depth + 1)
            return
        if name != "path" and not include_shapes:
            return
        d = shape_to_d(element)
        if not d:
            return
        try:
            subpaths = parse_path_data(d)
        except GeometryError:
            return
        visible = _visible(chain)
        for index, subpath in enumerate(subpaths):
            contours.append(Contour(element, index, subpath, matrix, visible, editable and name == "path"))

    visit(root, _IDENTITY.copy(), [], True, 0)
    view_box = None
    if root.get("viewBox"):
        values = [float(v) for v in re.findall(_NUMBER, root.get("viewBox", ""))]
        if len(values) == 4:
            view_box = tuple(values)  # type: ignore[assignment]
    return Document(root, contours, view_box)


# ---------------------------------------------------------------------------
# Sampling

_GRID = 48  # minimum samples per curved segment
_BUDGET = 1024  # dense samples per contour, distributed by control-polygon length


@lru_cache(maxsize=256)
def _bernstein(order: int, count: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, count)
    if order == 2:
        return np.stack([1 - t, t], axis=1)
    if order == 3:
        return np.stack([(1 - t) ** 2, 2 * (1 - t) * t, t ** 2], axis=1)
    return np.stack([(1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3], axis=1)


def arc_center(start: complex, end: complex, arc: tuple[float, float, float, int, int]) -> tuple[complex, float, float, float, float, float]:
    """Center parameterization (SVG 1.1 F.6.5): center, rx, ry, phi, theta, delta."""
    rx, ry, rotation, large, sweep = arc
    phi = math.radians(rotation)
    cos, sin = math.cos(phi), math.sin(phi)
    dx, dy = (start.real - end.real) / 2, (start.imag - end.imag) / 2
    x1 = cos * dx + sin * dy
    y1 = -sin * dx + cos * dy
    rx, ry = abs(rx), abs(ry)
    lam = (x1 * x1) / (rx * rx) + (y1 * y1) / (ry * ry)
    if lam > 1:
        rx, ry = rx * math.sqrt(lam), ry * math.sqrt(lam)
    numerator = rx * rx * ry * ry - rx * rx * y1 * y1 - ry * ry * x1 * x1
    denominator = rx * rx * y1 * y1 + ry * ry * x1 * x1
    factor = math.sqrt(max(0.0, numerator / denominator)) if denominator else 0.0
    if large == sweep:
        factor = -factor
    cxp, cyp = factor * rx * y1 / ry, -factor * ry * x1 / rx
    center = complex(cos * cxp - sin * cyp + (start.real + end.real) / 2, sin * cxp + cos * cyp + (start.imag + end.imag) / 2)

    def angle(ux: float, uy: float, vx: float, vy: float) -> float:
        return math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)

    theta = angle(1, 0, (x1 - cxp) / rx, (y1 - cyp) / ry)
    delta = angle((x1 - cxp) / rx, (y1 - cyp) / ry, (-x1 - cxp) / rx, (-y1 - cyp) / ry)
    if not sweep and delta > 0:
        delta -= 2 * math.pi
    elif sweep and delta < 0:
        delta += 2 * math.pi
    return center, rx, ry, phi, theta, delta


def _arc_points(start: complex, end: complex, arc: tuple[float, float, float, int, int], count: int = _GRID) -> np.ndarray:
    center, rx, ry, phi, theta, delta = arc_center(start, end, arc)
    angles = theta + delta * np.linspace(0.0, 1.0, count)
    points = center + complex(math.cos(phi), math.sin(phi)) * (rx * np.cos(angles) + 1j * ry * np.sin(angles))
    points[0], points[-1] = start, end
    return points


def split_segment(segment: Segment) -> list[Segment]:
    """Split a segment at its parameter midpoint without changing geometry."""
    p = segment.points
    if segment.kind == "L":
        mid = (p[0] + p[1]) / 2
        return [Segment("L", np.array([p[0], mid])), Segment("L", np.array([mid, p[1]]))]
    if segment.kind == "Q":
        a, b = (p[0] + p[1]) / 2, (p[1] + p[2]) / 2
        m = (a + b) / 2
        return [Segment("Q", np.array([p[0], a, m])), Segment("Q", np.array([m, b, p[2]]))]
    if segment.kind == "C":
        a, b, c = (p[0] + p[1]) / 2, (p[1] + p[2]) / 2, (p[2] + p[3]) / 2
        d, e = (a + b) / 2, (b + c) / 2
        m = (d + e) / 2
        return [Segment("C", np.array([p[0], a, d, m])), Segment("C", np.array([m, e, c, p[3]]))]
    center, rx, ry, phi, theta, delta = arc_center(complex(p[0]), complex(p[1]), segment.arc)  # type: ignore[arg-type]
    middle_angle = theta + delta / 2
    mid = center + complex(math.cos(phi), math.sin(phi)) * complex(rx * math.cos(middle_angle), ry * math.sin(middle_angle))
    arc = (rx, ry, segment.arc[2], 0, segment.arc[4])  # type: ignore[index]
    return [Segment("A", np.array([p[0], mid]), arc), Segment("A", np.array([mid, p[1]]), arc)]


def segment_counts(subpath: Subpath) -> list[int]:
    """Dense sample count per segment: lines need 2; curves share a budget.

    Allocating by length keeps the chord-approximation error small and nearly
    independent of how a curve is split into segments.
    """
    polygons = [float(np.abs(np.diff(segment.points)).sum()) if segment.kind != "A" else 2.0 * abs(segment.points[1] - segment.points[0]) for segment in subpath.segments]
    curved = sum(length for length, segment in zip(polygons, subpath.segments) if segment.kind != "L") or 1.0
    counts = []
    for length, segment in zip(polygons, subpath.segments):
        if segment.kind == "L":
            counts.append(2)
        elif segment.kind == "A":
            _, _, _, _, _, delta = arc_center(complex(segment.points[0]), complex(segment.points[1]), segment.arc)  # type: ignore[arg-type]
            counts.append(int(min(1024, max(_GRID, math.ceil(_BUDGET * length / curved), math.ceil(abs(delta) * 64)))))
        else:
            counts.append(int(min(1024, max(_GRID, math.ceil(_BUDGET * length / curved)))))
    return counts


def dense_points(subpath: Subpath, counts: list[int] | None = None) -> np.ndarray:
    """Dense polyline through a subpath in local coordinates."""
    counts = counts or segment_counts(subpath)
    pieces = []
    for segment, count in zip(subpath.segments, counts):
        if segment.kind == "L":
            pieces.append(segment.points)
        elif segment.kind == "A":
            pieces.append(_arc_points(complex(segment.points[0]), complex(segment.points[1]), segment.arc, count))  # type: ignore[arg-type]
        else:
            pieces.append(_bernstein(len(segment.points), count) @ segment.points)
    return np.concatenate(pieces)


def resample(polyline: np.ndarray, count: int, closed: bool) -> tuple[np.ndarray, float]:
    """Uniform arc-length samples of a polyline; returns (samples, length)."""
    if closed and abs(polyline[-1] - polyline[0]) > 0:
        polyline = np.append(polyline, polyline[0])
    steps = np.abs(np.diff(polyline))
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    length = float(cumulative[-1])
    if length <= 0:
        return np.full(count, polyline[0]), 0.0
    targets = np.arange(count) * (length / count) if closed else np.linspace(0.0, length, count)
    keep = np.concatenate([[True], steps > 0])
    x = np.interp(targets, cumulative[keep], polyline.real[keep])
    y = np.interp(targets, cumulative[keep], polyline.imag[keep])
    return x + 1j * y, length


def sample_contour(contour: Contour, count: int) -> np.ndarray:
    points = apply_affine(contour.ctm, dense_points(contour.subpath))
    return resample(points, count, contour.closed)[0]


# ---------------------------------------------------------------------------
# Serialization


def _fmt(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"-0", ""}:
        text = "0"
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    return text


def _pair(point: complex, decimals: int) -> str:
    y = _fmt(point.imag, decimals)
    return _fmt(point.real, decimals) + ("" if y.startswith("-") else " ") + y


def serialize_subpaths(subpaths: list[Subpath], decimals: int) -> str:
    """Absolute-coordinate path data preserving segment types."""
    out: list[str] = []
    for subpath in subpaths:
        out.append("M" + _pair(subpath.start, decimals))
        segments = subpath.segments
        if subpath.closed and len(segments) > 1 and segments[-1].kind == "L":
            # The closing line is implied by Z.
            segments = segments[:-1]
        for segment in segments:
            if segment.kind == "L":
                out.append("L" + _pair(complex(segment.points[1]), decimals))
            elif segment.kind == "Q":
                out.append("Q" + " ".join(_pair(complex(p), decimals) for p in segment.points[1:]))
            elif segment.kind == "C":
                out.append("C" + " ".join(_pair(complex(p), decimals) for p in segment.points[1:]))
            else:
                rx, ry, rotation, large, sweep = segment.arc  # type: ignore[misc]
                out.append(f"A{_fmt(rx, decimals)} {_fmt(ry, decimals)} {_fmt(rotation, 3)} {large} {sweep} " + _pair(complex(segment.points[1]), decimals))
        if subpath.closed:
            out.append("Z")
    return "".join(out)


def arc_to_cubics(segment: Segment) -> list[Segment]:
    """Approximate an arc by cubic segments (at most 90 degrees each)."""
    start, end = complex(segment.points[0]), complex(segment.points[1])
    center, rx, ry, phi, theta, delta = arc_center(start, end, segment.arc)  # type: ignore[arg-type]
    pieces = max(1, math.ceil(abs(delta) / (math.pi / 2) - 1e-9))
    step = delta / pieces
    alpha = 4 / 3 * math.tan(step / 4)
    rotate = complex(math.cos(phi), math.sin(phi))

    def point(t: float) -> complex:
        return center + rotate * complex(rx * math.cos(t), ry * math.sin(t))

    def derivative(t: float) -> complex:
        return rotate * complex(-rx * math.sin(t), ry * math.cos(t))

    out = []
    for i in range(pieces):
        a, b = theta + i * step, theta + (i + 1) * step
        p0 = start if i == 0 else point(a)
        p3 = end if i == pieces - 1 else point(b)
        out.append(Segment("C", np.array([p0, p0 + alpha * derivative(a), p3 - alpha * derivative(b), p3], dtype=complex)))
    return out


def _compact_numbers(values: list[float], decimals: int) -> str:
    """Join numbers SVGO-style: omit separators before '-' and after a '.'-number."""
    out = ""
    previous = ""
    for value in values:
        text = _fmt(value, decimals)
        if out and not text.startswith("-") and not (text.startswith(".") and "." in previous and "e" not in previous):
            out += " "
        out += text
        previous = text
    return out


def serialize_compact(subpaths: list[Subpath], decimals: int) -> str:
    """Short path data: per segment, the shorter of absolute and relative forms.

    Relative offsets are computed from the *rounded* current point so rounding
    errors do not accumulate along the path.
    """
    quantum = 10.0 ** -decimals

    def q(value: float) -> float:
        return round(value / quantum) * quantum

    out: list[str] = []
    current = 0j
    last_command = ""
    for index, subpath in enumerate(subpaths):
        start = complex(q(subpath.start.real), q(subpath.start.imag))
        absolute = "M" + _compact_numbers([start.real, start.imag], decimals)
        relative = "m" + _compact_numbers([start.real - current.real, start.imag - current.imag], decimals)
        out.append(absolute if index == 0 or len(absolute) <= len(relative) else relative)
        last_command = "M"
        current = start
        segments = subpath.segments
        if subpath.closed and len(segments) > 1 and segments[-1].kind == "L":
            segments = segments[:-1]
        for segment in segments:
            points = [complex(q(p.real), q(p.imag)) for p in segment.points[1:]]
            end = points[-1]
            if segment.kind == "L":
                if abs(end.imag - current.imag) < quantum / 2 and abs(end.real - current.real) >= quantum / 2:
                    options = [("H", [end.real]), ("h", [end.real - current.real])]
                elif abs(end.real - current.real) < quantum / 2 and abs(end.imag - current.imag) >= quantum / 2:
                    options = [("V", [end.imag]), ("v", [end.imag - current.imag])]
                else:
                    options = [("L", [end.real, end.imag]), ("l", [end.real - current.real, end.imag - current.imag])]
            elif segment.kind in "QC":
                letter = segment.kind
                options = [
                    (letter, [v for p in points for v in (p.real, p.imag)]),
                    (letter.lower(), [v for p in points for v in (p.real - current.real, p.imag - current.imag)]),
                ]
            else:
                rx, ry, rotation, large, sweep = segment.arc  # type: ignore[misc]
                head = [rx, ry, rotation]
                options = [("A", head), ("a", head)]
                texts = []
                for letter, numbers in options:
                    offset = (end.real, end.imag) if letter == "A" else (end.real - current.real, end.imag - current.imag)
                    texts.append((letter, _compact_numbers(numbers, decimals) + f" {large} {sweep} " + _compact_numbers(list(offset), decimals)))
                letter, body = min(texts, key=lambda item: len(item[1]))
                out.append(("" if letter == last_command else letter) + body if False else letter + body)
                last_command = letter
                current = end
                continue
            best = min(((letter, _compact_numbers(numbers, decimals)) for letter, numbers in options), key=lambda item: len(item[1]))
            letter, body = best
            if letter == last_command and letter not in "Mm":
                out.append(("" if body.startswith("-") else " ") + body)
            else:
                out.append(letter + body)
            last_command = letter
            current = end
        if subpath.closed:
            out.append("z")
            last_command = "z"
            current = start
    return "".join(out)
