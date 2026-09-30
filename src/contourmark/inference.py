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
from typing import Any

from svgpathtools import parse_path
from scipy.optimize import linear_sum_assignment

from .core import WatermarkError, _contours, _manifest_mac, _parse_svg, _sample


SVG_NS = "http://www.w3.org/2000/svg"


@dataclass(frozen=True)
class Candidate:
    d: str
    weight: float = 1.0


def _path_points(d: str, count: int = 96) -> list[complex]:
    try:
        path = parse_path(d)
    except Exception as exc:
        raise WatermarkError(f"invalid candidate path: {exc}") from exc
    subpaths = path.continuous_subpaths()
    if len(subpaths) != 1 or not len(subpaths[0]) or subpaths[0].length() <= 0:
        raise WatermarkError("each candidate must be one nonempty contour")
    return _sample(subpaths[0], count)


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


def _winner(key: bytes, asset_id: str, step: int, candidates: list[dict[str, Any]]) -> int:
    scores = []
    for candidate in candidates:
        seed = f"contourmark-sampler-v1|{asset_id}|{step}|{candidate['fingerprint']}".encode()
        integer = int.from_bytes(hmac.digest(key, seed, "sha256")[:8], "big") >> 11
        u = min(max((integer + 0.5) / (1 << 53), 2 ** -53), 1 - 2 ** -53)
        scores.append(math.log(candidate["probability"]) - math.log(-math.log(u)))
    return max(range(len(scores)), key=scores.__getitem__)


class GenerationSession:
    """Call ``add_step`` while a provider is generating; ``finish`` emits the SVG.

    Proposal groups must be independent of the secret key. The caller is
    responsible for ensuring candidate contours have comparable visual quality.
    """

    def __init__(self, key: bytes, view_box: str, asset_id: str | None = None, match_tolerance: float = 0.0005):
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
        self.match_tolerance = match_tolerance
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
                if distance < 2 * self.match_tolerance:
                    raise WatermarkError("candidate geometries are too close for reliable verification")
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
            "steps": self.steps,
        }
        return output, {**unsigned, "mac": _manifest_mac(unsigned, self.key)}


def verify_generation(svg: bytes, manifest: dict[str, Any], key: bytes) -> dict[str, Any]:
    unsigned = {k: v for k, v in manifest.items() if k != "mac"}
    if not hmac.compare_digest(str(manifest.get("mac", "")), _manifest_mac(unsigned, key)):
        raise WatermarkError("manifest authentication failed")
    if unsigned.get("schema") != "contourmark-inference-v1":
        raise WatermarkError("unsupported generation manifest")
    contours = [contour for _, _, contour in _contours(_parse_svg(svg))]
    steps = unsigned["steps"]
    if len(contours) != len(steps):
        raise WatermarkError("contour count changed; generation steps cannot be synchronized")
    x, y, w, h = [float(v) for v in unsigned["view_box"].replace(",", " ").split()]
    scale = math.hypot(w, h)
    tolerance = unsigned["match_tolerance"]
    observed_points = [_sample(contour, 96) for contour in contours]
    proposed_points = [[_path_points(candidate["d"]) for candidate in step["candidates"]] for step in steps]
    cost = [
        [min(_distance(observed, candidate, scale) for candidate in candidates) for observed in observed_points]
        for candidates in proposed_points
    ]
    row_indices, column_indices = linear_sum_assignment(cost)
    assignments = {int(row): int(column) for row, column in zip(row_indices, column_indices)}
    matched = 0
    probability = 1.0
    details = []
    for index, step in enumerate(steps):
        observed_index = assignments[index]
        observed = observed_points[observed_index]
        distances = [_distance(observed, candidate, scale) for candidate in proposed_points[index]]
        nearest = min(range(len(distances)), key=distances.__getitem__)
        expected = _winner(key, unsigned["asset_id"], index, step["candidates"])
        exact = nearest == expected and distances[nearest] <= tolerance
        if exact:
            matched += 1
            probability *= step["candidates"][expected]["probability"]
        details.append({"step": index, "observed_contour": observed_index, "matched": exact, "distance": distances[nearest], "expected_index": expected, "nearest_index": nearest})
    # This is the exact conditional null probability of *all* keyed choices
    # matching when unmarked choices are independently drawn from recorded q.
    detected = matched == len(steps) and probability <= 0.01
    return {
        "detected": detected,
        "matched_steps": matched,
        "total_steps": len(steps),
        "conditional_p_value": probability if matched == len(steps) else None,
        "details": details,
        "assumption": "Candidates and probabilities were fixed independently of the secret key; unmarked choices follow their recorded categorical distributions.",
    }
