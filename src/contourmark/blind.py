"""Manifest-free statistical watermarking over geometric contour choices.

The generator samples between equal-weight alternatives using a keyed PRF of a
coarse, serialization-invariant contour centroid. The verifier requires only
the key and a precommitted asset ID. It never sees the candidate alternatives.
"""

from __future__ import annotations

import hmac
import math
import secrets
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache

from scipy.special import gammainc

from .core import WatermarkError, _contours, _parse_svg, _sample, _tag
from .inference import SVG_NS, _distance, _path_points, _validate_render_subset, _visible


@dataclass(frozen=True)
class BlindParameters:
    cell_fraction: float = 0.005
    guard_fraction: float = 0.10
    max_variation: float = 0.03
    threshold: float = 0.001

    def validate(self) -> None:
        if not 0.001 <= self.cell_fraction <= 0.05:
            raise WatermarkError("cell_fraction must be in [0.001, 0.05]")
        if not 0 <= self.guard_fraction < 0.5:
            raise WatermarkError("guard_fraction must be in [0, 0.5)")
        if not 0 < self.max_variation <= 0.1:
            raise WatermarkError("max_variation must be in (0, 0.1]")
        if not 0 < self.threshold < 1:
            raise WatermarkError("threshold must be in (0, 1)")


def _box(view_box: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(value) for value in view_box.replace(",", " ").split())
    except ValueError as exc:
        raise WatermarkError("viewBox must contain four numbers") from exc
    if len(values) != 4 or not all(math.isfinite(v) for v in values) or values[2] <= 0 or values[3] <= 0:
        raise WatermarkError("viewBox must contain four finite numbers and positive dimensions")
    return values  # type: ignore[return-value]


def _symbol(points: tuple[complex, ...] | list[complex], view_box: str, params: BlindParameters) -> str | None:
    x0, y0, width, height = _box(view_box)
    mean_x = sum(point.real for point in points) / len(points)
    mean_y = sum(point.imag for point in points) / len(points)
    bins: list[int] = []
    for value, origin, extent in ((mean_x, x0, width), (mean_y, y0, height)):
        coordinate = (value - origin) / (extent * params.cell_fraction) + 0.5
        fraction = coordinate - math.floor(coordinate)
        if min(fraction, 1 - fraction) < params.guard_fraction:
            return None
        bins.append(math.floor(coordinate))
    return f"{bins[0]}:{bins[1]}"


def _uniform(key: bytes, asset_id: str, symbol: str) -> float:
    digest = hmac.digest(key, f"contourmark-blind-v1|{asset_id}|{symbol}".encode(), "sha256")
    integer = int.from_bytes(digest[:8], "big") >> 11
    return min(max((integer + 0.5) / (1 << 53), 2 ** -53), 1 - 2 ** -53)


class BlindGenerationSession:
    """Choose equal-quality geometry alternatives during a live drawing run."""

    def __init__(self, key: bytes, view_box: str, asset_id: str | None = None, params: BlindParameters | None = None):
        if len(key) < 16:
            raise WatermarkError("key must contain at least 16 bytes")
        self.key = key
        self.view_box = view_box
        self.asset_id = asset_id or secrets.token_hex(16)
        if not 0 < len(self.asset_id) <= 256:
            raise WatermarkError("asset ID must contain 1 to 256 characters")
        self.params = params or BlindParameters()
        self.params.validate()
        self.box = _box(view_box)
        self.scale = math.hypot(self.box[2], self.box[3])
        self.emitted: list[tuple[str, dict[str, str]]] = []
        self.symbols: set[str] = set()

    def add_step(self, paths: list[str], attributes: dict[str, str] | None = None) -> int:
        if not 2 <= len(paths) <= 16:
            raise WatermarkError("each step requires 2 to 16 alternatives")
        points = [_path_points(path) for path in paths]
        symbols = [_symbol(item, self.view_box, self.params) for item in points]
        if any(symbol is None for symbol in symbols):
            raise WatermarkError("candidate lies near a geometric quantization boundary")
        if len(set(symbols)) != len(symbols):
            raise WatermarkError("candidate alternatives have the same geometric symbol")
        if any(symbol in self.symbols for symbol in symbols):
            raise WatermarkError("candidate geometric symbol repeats an earlier step")
        for i in range(len(points)):
            for j in range(i):
                if _distance(points[i], points[j], self.scale) > self.params.max_variation:
                    raise WatermarkError("candidate geometries differ beyond max_variation")
        attributes = attributes or {}
        allowed = {"fill", "stroke", "stroke-width", "fill-rule", "opacity", "stroke-linecap", "stroke-linejoin"}
        if any(name not in allowed for name in attributes):
            raise WatermarkError("unsupported path attribute")
        chosen = max(range(len(paths)), key=lambda index: _uniform(self.key, self.asset_id, symbols[index]))  # type: ignore[arg-type]
        self.symbols.update(symbols)  # type: ignore[arg-type]
        self.emitted.append((paths[chosen], attributes))
        return chosen

    def finish(self) -> bytes:
        if not self.emitted:
            raise WatermarkError("generation contains no drawing steps")
        root = ET.Element(f"{{{SVG_NS}}}svg", {"viewBox": self.view_box})
        for d, attributes in self.emitted:
            ET.SubElement(root, f"{{{SVG_NS}}}path", {"d": d, **attributes})
        return ET.tostring(root, encoding="utf-8", xml_declaration=True)


@lru_cache(maxsize=512)
def _extract_symbols(svg: bytes, params: BlindParameters) -> tuple[tuple[str, ...], int]:
    tree = _parse_svg(svg)
    root = tree.getroot()
    _validate_render_subset(root)
    view_box = root.get("viewBox", "")
    _box(view_box)
    parents = {child: parent for parent in root.iter() for child in parent}
    symbols = []
    for element, _, contour in _contours(tree):
        if not _visible(element, parents):
            continue
        symbol = _symbol(_sample(contour, 96), view_box, params)
        if symbol is not None:
            symbols.append(symbol)
    unique = tuple(dict.fromkeys(symbols))
    return unique, len(symbols) - len(unique)


def detect_blind(svg: bytes, key: bytes, asset_id: str, params: BlindParameters | None = None) -> dict:
    if len(key) < 16:
        raise WatermarkError("key must contain at least 16 bytes")
    params = params or BlindParameters()
    params.validate()
    unique, duplicates = _extract_symbols(svg, params)
    if not unique:
        raise WatermarkError("no usable visible contour symbols")
    values = [_uniform(key, asset_id, symbol) for symbol in unique]
    statistic = sum(-math.log(value) for value in values)
    # Under a fixed key and unmarked geometry independent of that key, each
    # distinct symbol's PRF score is uniform; the sum is Gamma(n, 1).
    p_value = float(gammainc(len(unique), statistic))
    return {
        "detected": p_value <= params.threshold,
        "p_value": p_value,
        "usable_contours": len(unique),
        "skipped_contours": duplicates,
        "statistic": statistic,
        "assumption": "The asset ID and key were committed before generation; unmarked geometry is independent of the key; distinct geometric symbols behave as independent PRF inputs.",
    }
