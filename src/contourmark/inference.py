"""Generation-time, distribution-preserving SVG watermark sampler.

An upstream generator proposes multiple visually interchangeable contours at
each drawing step. This module replaces only its random choice: keyed Gumbel
sampling selects the contour before it enters the SVG. No generated SVG is
modified afterwards. The selected geometry itself carries the evidence.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from svgpathtools import parse_path
from scipy.optimize import linear_sum_assignment

from .core import WatermarkError, _contours, _manifest_mac, _parse_svg, _sample, _tag


SVG_NS = "http://www.w3.org/2000/svg"
_PAINT_PROPERTIES = {
    "fill", "stroke", "stroke-width", "fill-rule", "opacity", "fill-opacity",
    "stroke-opacity", "stroke-linecap", "stroke-linejoin", "display", "visibility",
}
_ROOT_ATTRIBUTES = {"viewBox", "width", "height", "version", "xmlns", "preserveAspectRatio"} | _PAINT_PROPERTIES
# ``filling`` is inert metadata emitted by OmniSVG's tokenizer/decoder.  It is
# not an SVG presentation attribute and has no browser rendering semantics.
_NODE_ATTRIBUTES = {"id", "style", "filling"} | _PAINT_PROPERTIES


def _declarations(element: ET.Element) -> dict[str, str]:
    declarations = {name: value.strip().lower() for name, value in element.attrib.items() if name in _PAINT_PROPERTIES}
    for part in element.get("style", "").split(";"):
        if not part.strip():
            continue
        if ":" not in part:
            raise WatermarkError("invalid inline style")
        name, value = part.split(":", 1)
        name = name.strip().lower()
        if name not in _PAINT_PROPERTIES:
            raise WatermarkError(f"unsupported rendering property: {name}")
        declarations[name] = value.strip().lower()
    return declarations


def _numeric_paint(value: str, name: str) -> float:
    token = value.removesuffix("px").removesuffix("%")
    try:
        number = float(token)
    except ValueError as exc:
        raise WatermarkError(f"unsupported numeric rendering value for {name}") from exc
    if not math.isfinite(number) or number < 0:
        raise WatermarkError(f"invalid numeric rendering value for {name}")
    return number


def _validate_render_subset(root: ET.Element) -> None:
    """Fail closed on SVG features whose rendered geometry we do not evaluate."""
    if _tag(root) != "svg":
        raise WatermarkError("root element must be svg")
    for element in root.iter():
        tag = _tag(element)
        if tag not in {"svg", "g", "path"}:
            raise WatermarkError(f"unsupported rendering element: {tag}")
        allowed = _ROOT_ATTRIBUTES if element is root else (_NODE_ATTRIBUTES | ({"d"} if tag == "path" else set()))
        for name in element.attrib:
            if name not in allowed:
                raise WatermarkError(f"unsupported rendering attribute: {name}")
        for name, value in _declarations(element).items():
            if "url(" in value or "!important" in value or value in {"inherit", "currentcolor"}:
                raise WatermarkError(f"unsupported rendering value for {name}")
            if name in {"opacity", "fill-opacity", "stroke-opacity", "stroke-width"}:
                _numeric_paint(value, name)


@dataclass(frozen=True)
class Candidate:
    d: str
    weight: float = 1.0


@lru_cache(maxsize=8192)
def _path_points(d: str, count: int = 96) -> tuple[complex, ...]:
    try:
        path = parse_path(d)
    except Exception as exc:
        raise WatermarkError(f"invalid candidate path: {exc}") from exc
    subpaths = path.continuous_subpaths()
    if len(subpaths) != 1 or not len(subpaths[0]) or subpaths[0].length() <= 0:
        raise WatermarkError("each candidate must be one nonempty contour")
    return tuple(_sample(subpaths[0], count))


def _fingerprint(d: str) -> str:
    points = _path_points(d)
    x = [p.real for p in points]
    y = [p.imag for p in points]
    scale = max(math.hypot(max(x) - min(x), max(y) - min(y)), 1e-12)
    center = complex((max(x) + min(x)) / 2, (max(y) + min(y)) / 2)
    shape = {
        "center": [round(center.real, 5), round(center.imag, 5)],
        "scale": round(scale, 5),
        "points": [(round((p.real - center.real) / scale, 5), round((p.imag - center.imag) / scale, 5)) for p in points],
    }
    return hashlib.sha256(json.dumps(shape, separators=(",", ":")).encode()).hexdigest()


def _distance(a: list[complex], b: list[complex], scale: float) -> float:
    return math.sqrt(sum(abs(x - y) ** 2 for x, y in zip(a, b)) / len(a)) / scale


def _mean(points: list[complex] | tuple[complex, ...]) -> complex:
    return sum(points) / len(points)


def _shape_distance(a: list[complex] | tuple[complex, ...], b: list[complex] | tuple[complex, ...], scale: float) -> float:
    """Contour distance invariant to independent path translations."""
    offset = _mean(a) - _mean(b)
    return math.sqrt(sum(abs(x - y - offset) ** 2 for x, y in zip(a, b)) / len(a)) / scale


def _matching_distance(a: list[complex] | tuple[complex, ...], b: list[complex] | tuple[complex, ...], scale: float, position_tolerance: float) -> float:
    drift = abs(_mean(a) - _mean(b)) / scale
    if drift > position_tolerance:
        return math.inf
    # Position disambiguates similar contours at different drawing steps,
    # without turning small translations into a failed shape comparison.
    return _shape_distance(a, b, scale) + drift * 1e-6


def _acceptance_radius(candidates: list[tuple[complex, ...]], scale: float, base_tolerance: float) -> float:
    """Largest conservative radius that keeps candidate regions disjoint."""
    separation = min(
        _shape_distance(candidates[i], candidates[j], scale)
        for i in range(len(candidates))
        for j in range(i)
    )
    # Candidate creation requires separation >= 2 * base_tolerance.  Expanding
    # to 45% of the closest pair distance preserves a 10% rejection gap around
    # each decision boundary while adapting robustness to proposal diversity.
    return max(base_tolerance, 0.45 * separation)


def _winner(key: bytes, asset_id: str, step: int, candidates: list[dict[str, Any]]) -> int:
    scores = []
    for candidate in candidates:
        seed = f"contourmark-sampler-v1|{asset_id}|{step}|{candidate['fingerprint']}".encode()
        integer = int.from_bytes(hmac.digest(key, seed, "sha256")[:8], "big") >> 11
        u = min(max((integer + 0.5) / (1 << 53), 2 ** -53), 1 - 2 ** -53)
        scores.append(math.log(candidate["probability"]) - math.log(-math.log(u)))
    return max(range(len(scores)), key=scores.__getitem__)


def _binomial_tail(probabilities: list[float], successes: int) -> float:
    """P[X >= successes] for independent, nonidentical Bernoulli variables."""
    distribution = [1.0]
    for probability in probabilities:
        next_distribution = [0.0] * (len(distribution) + 1)
        for count, mass in enumerate(distribution):
            next_distribution[count] += mass * (1 - probability)
            next_distribution[count + 1] += mass * probability
        distribution = next_distribution
    return sum(distribution[successes:])


def _visible(element: ET.Element, parents: dict[ET.Element, ET.Element]) -> bool:
    """Determine visibility for the strictly supported paint/style subset."""
    current: ET.Element | None = element
    paint: dict[str, str] = {}
    while current is not None:
        declarations = _declarations(current)
        if declarations.get("display") == "none" or declarations.get("visibility") in {"hidden", "collapse"}:
            return False
        if "opacity" in declarations and _numeric_paint(declarations["opacity"], "opacity") == 0:
            return False
        for name in ("fill", "stroke", "stroke-width", "fill-opacity", "stroke-opacity"):
            if name not in paint and name in declarations:
                paint[name] = declarations[name]
        current = parents.get(current)
    fill = paint.get("fill", "black")
    stroke = paint.get("stroke", "none")
    fill_visible = fill != "none" and _numeric_paint(paint.get("fill-opacity", "1"), "fill-opacity") != 0
    stroke_visible = stroke != "none" and _numeric_paint(paint.get("stroke-opacity", "1"), "stroke-opacity") != 0 and _numeric_paint(paint.get("stroke-width", "1"), "stroke-width") != 0
    return fill_visible or stroke_visible


class GenerationSession:
    """Call ``add_step`` while a provider is generating; ``finish`` emits the SVG.

    Proposal groups must be independent of the secret key. The caller is
    responsible for ensuring candidate contours have comparable visual quality.
    """

    def __init__(self, key: bytes, view_box: str, asset_id: str | None = None, match_tolerance: float = 0.0005, position_tolerance: float = 0.01):
        if len(key) < 16:
            raise WatermarkError("key must contain at least 16 bytes")
        self.key = key
        self.asset_id = asset_id or secrets.token_hex(16)
        if not 0 < len(self.asset_id) <= 256:
            raise WatermarkError("asset ID must contain 1 to 256 characters")
        self.view_box = view_box
        try:
            x, y, w, h = [float(v) for v in view_box.replace(",", " ").split()]
        except ValueError as exc:
            raise WatermarkError("viewBox must contain four numbers") from exc
        if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
            raise WatermarkError("viewBox dimensions must be positive finite numbers")
        self.scale = math.hypot(w, h)
        if not 0 < match_tolerance <= 0.01:
            raise WatermarkError("match_tolerance must be in (0, 0.01]")
        if not 0 < position_tolerance <= 0.05:
            raise WatermarkError("position_tolerance must be in (0, 0.05]")
        self.match_tolerance = match_tolerance
        self.position_tolerance = position_tolerance
        self.steps: list[dict[str, Any]] = []
        self.emitted: list[tuple[str, dict[str, str]]] = []

    def add_step(self, candidates: list[Candidate], attributes: dict[str, str] | None = None) -> int:
        """Choose one proposed path before it becomes part of the output SVG."""
        if not 2 <= len(candidates) <= 16:
            raise WatermarkError("each step requires 2 to 16 candidate contours")
        if any(not math.isfinite(c.weight) or c.weight <= 0 for c in candidates):
            raise WatermarkError("candidate weights must be positive and finite")
        total = sum(c.weight for c in candidates)
        points = [_path_points(c.d) for c in candidates]
        fingerprints = [_fingerprint(c.d) for c in candidates]
        if len(set(fingerprints)) != len(fingerprints):
            raise WatermarkError("candidates must have distinct geometry")
        for i in range(len(points)):
            for j in range(i):
                distance = _distance(points[i], points[j], self.scale)
                if _shape_distance(points[i], points[j], self.scale) < 2 * self.match_tolerance:
                    raise WatermarkError("candidate shapes are too close for reliable verification")
                if distance > 0.03:
                    raise WatermarkError("candidate geometries differ by more than 3% of the canvas diagonal")
        attributes = attributes or {}
        allowed = {"fill", "stroke", "stroke-width", "fill-rule", "opacity", "stroke-linecap", "stroke-linejoin"}
        if any(name not in allowed for name in attributes):
            raise WatermarkError("unsupported path attribute")
        recorded = [
            {"d": c.d, "probability": c.weight / total, "fingerprint": fingerprint}
            for c, fingerprint in zip(candidates, fingerprints)
        ]
        step_index = len(self.steps)
        chosen = _winner(self.key, self.asset_id, step_index, recorded)
        self.steps.append({"candidates": recorded, "attributes": attributes})
        self.emitted.append((candidates[chosen].d, attributes))
        return chosen

    def finish(self) -> tuple[bytes, dict[str, Any]]:
        if not self.steps:
            raise WatermarkError("generation contains no drawing steps")
        root = ET.Element(f"{{{SVG_NS}}}svg", {"viewBox": self.view_box})
        for d, attributes in self.emitted:
            ET.SubElement(root, f"{{{SVG_NS}}}path", {"d": d, **attributes})
        output = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        unsigned = {
            "schema": "contourmark-inference-v1",
            "asset_id": self.asset_id,
            "view_box": self.view_box,
            "match_tolerance": self.match_tolerance,
            "position_tolerance": self.position_tolerance,
            "steps": self.steps,
        }
        return output, {**unsigned, "mac": _manifest_mac(unsigned, self.key)}


def verify_generation(svg: bytes, manifest: dict[str, Any], key: bytes) -> dict[str, Any]:
    unsigned = {k: v for k, v in manifest.items() if k != "mac"}
    if not hmac.compare_digest(str(manifest.get("mac", "")), _manifest_mac(unsigned, key)):
        raise WatermarkError("manifest authentication failed")
    if unsigned.get("schema") != "contourmark-inference-v1":
        raise WatermarkError("unsupported generation manifest")
    tree = _parse_svg(svg)
    root = tree.getroot()
    _validate_render_subset(root)
    parents = {child: parent for parent in root.iter() for child in parent}
    raw_contours = _contours(tree)
    contours = [contour for element, _, contour in raw_contours if _visible(element, parents)]
    steps = unsigned["steps"]
    if not contours:
        raise WatermarkError("no visible contours remain")
    x, y, w, h = [float(v) for v in unsigned["view_box"].replace(",", " ").split()]
    scale = math.hypot(w, h)
    try:
        candidate_box = [float(v) for v in root.get("viewBox", "").replace(",", " ").split()]
    except ValueError as exc:
        raise WatermarkError("candidate viewBox is invalid") from exc
    if len(candidate_box) != 4 or any(abs(a - b) > scale * 1e-6 for a, b in zip(candidate_box, (x, y, w, h))):
        raise WatermarkError("candidate viewBox changed")
    tolerance = unsigned["match_tolerance"]
    position_tolerance = unsigned.get("position_tolerance", 0.01)
    observed_points = [_sample(contour, 96) for contour in contours]
    proposed_points = [[_path_points(candidate["d"]) for candidate in step["candidates"]] for step in steps]
    acceptance_radii = [_acceptance_radius(candidates, scale, tolerance) for candidates in proposed_points]
    # Dummy columns let the assignment treat removed/redrawn paths as erasures.
    cost = [
        [min(_matching_distance(observed, candidate, scale, position_tolerance) for candidate in candidates) for observed in observed_points]
        + [acceptance_radii[index]] * len(steps)
        for index, candidates in enumerate(proposed_points)
    ]
    row_indices, column_indices = linear_sum_assignment(cost)
    assignments = {
        int(row): int(column) if column < len(contours) and cost[row][column] <= acceptance_radii[row] else None
        for row, column in zip(row_indices, column_indices)
    }
    matched = 0
    recognized = 0
    null_probabilities = []
    all_null_probabilities = []
    details = []
    for index, step in enumerate(steps):
        observed_index = assignments[index]
        expected = _winner(key, unsigned["asset_id"], index, step["candidates"])
        all_null_probabilities.append(step["candidates"][expected]["probability"])
        if observed_index is None:
            details.append({"step": index, "observed_contour": None, "matched": False, "distance": None, "acceptance_radius": acceptance_radii[index], "expected_index": expected, "nearest_index": None})
            continue
        recognized += 1
        null_probabilities.append(step["candidates"][expected]["probability"])
        observed = observed_points[observed_index]
        distances = [_matching_distance(observed, candidate, scale, position_tolerance) for candidate in proposed_points[index]]
        nearest = min(range(len(distances)), key=distances.__getitem__)
        exact = nearest == expected and distances[nearest] <= acceptance_radii[index]
        if exact:
            matched += 1
        details.append({"step": index, "observed_contour": observed_index, "matched": exact, "distance": distances[nearest], "acceptance_radius": acceptance_radii[index], "expected_index": expected, "nearest_index": nearest})
    p_value = _binomial_tail(null_probabilities, matched) if null_probabilities else 1.0
    conservative_p_value = _binomial_tail(all_null_probabilities, matched)
    # Extra visible contours may be used to retain the watermark as a decoy
    # while replacing the actual drawing; reject them until render-aware
    # verification supports this case.
    detected = len(contours) <= len(steps) and p_value <= 0.01
    return {
        "detected": detected,
        "matched_steps": matched,
        "recognized_steps": recognized,
        "total_steps": len(steps),
        "conditional_p_value": p_value,
        "conservative_p_value": conservative_p_value,
        "visible_contours": len(contours),
        "details": details,
        "assumption": "Candidates and probabilities were fixed independently of the secret key; unmarked recognized choices follow their recorded categorical distributions; conditional detection additionally assumes erasure/recognition is independent of the secret keyed winner. The conservative p-value treats erasures as failures. Attack survival is empirical.",
    }
