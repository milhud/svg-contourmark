"""Geometry-domain watermarking of SVG path contours.

The manifest and original are retained by the owner. Only the modified contour
coordinates go into the released SVG. No watermark XML nodes or attributes are
added. A keyed normal-displacement field is detected with a matched filter.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

from svgpathtools import CubicBezier
from svgpathtools import Path as SVGPath
from svgpathtools import parse_path


class WatermarkError(ValueError):
    pass


@dataclass(frozen=True)
class Parameters:
    amplitude: float = 0.0015  # fraction of original drawing diagonal
    samples: int = 256
    harmonics: int = 16
    min_perimeter: float = 20.0  # fraction of drawing diagonal, in percent
    precision: int = 5

    def validate(self) -> None:
        if not 0 < self.amplitude <= 0.01:
            raise WatermarkError("amplitude must be in (0, 0.01]")
        if not 64 <= self.samples <= 2048 or self.samples % 8:
            raise WatermarkError("samples must be a multiple of 8 in [64, 2048]")
        if not 4 <= self.harmonics <= 32:
            raise WatermarkError("harmonics must be in [4, 32]")
        if not 0 <= self.min_perimeter <= 100:
            raise WatermarkError("min_perimeter must be in [0, 100]")
        if not 3 <= self.precision <= 10:
            raise WatermarkError("precision must be in [3, 10]")


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _points_to_d(text: str, closed: bool) -> str:
    import re

    numbers = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[Ee][-+]?\d+)?", text)]
    if len(numbers) < 4 or len(numbers) % 2:
        raise WatermarkError("invalid polyline/polygon points")
    pairs = list(zip(numbers[::2], numbers[1::2]))
    d = f"M {pairs[0][0]} {pairs[0][1]} " + " ".join(f"L {x} {y}" for x, y in pairs[1:])
    return d + (" Z" if closed else "")


def _element_path(element: ET.Element) -> SVGPath | None:
    name = _tag(element)
    if name == "path":
        data = element.get("d", "")
    elif name in {"polyline", "polygon"}:
        data = _points_to_d(element.get("points", ""), name == "polygon")
    else:
        return None
    try:
        return parse_path(data)
    except Exception as exc:
        raise WatermarkError(f"invalid {name} geometry: {exc}") from exc


def _contours(tree: ET.ElementTree) -> list[tuple[ET.Element, int, SVGPath]]:
    out: list[tuple[ET.Element, int, SVGPath]] = []
    for element in tree.iter():
        path = _element_path(element)
        if path is None:
            continue
        for index, contour in enumerate(path.continuous_subpaths()):
            if len(contour) and contour.length() > 0:
                out.append((element, index, contour))
    return out


def _sample(path: SVGPath, count: int) -> list[complex]:
    # Arc-length samples make the detector independent of path command syntax,
    # segment subdivision, and relative versus absolute coordinates.
    length = path.length(error=1e-5)
    closed = path.isclosed()
    divisor = count if closed else count - 1
    points: list[complex] = []
    def fallback_point(target: float) -> complex:
        """Invert arc length by monotone bisection when svgpathtools stalls."""
        remaining = target
        for segment in path:
            segment_length = segment.length(error=1e-8)
            if remaining > segment_length and segment is not path[-1]:
                remaining -= segment_length
                continue
            if remaining <= 0:
                return segment.start
            if remaining >= segment_length:
                return segment.end
            lower, upper = 0.0, 1.0
            for _ in range(60):
                middle = (lower + upper) / 2
                if segment.length(0, middle, error=1e-9) < remaining:
                    lower = middle
                else:
                    upper = middle
            return segment.point((lower + upper) / 2)
        return path.end

    for i in range(count):
        if not closed and i == 0:
            points.append(path.start)
        elif not closed and i == count - 1:
            points.append(path.end)
        else:
            target = length * i / divisor
            try:
                points.append(path.point(path.ilength(target, error=1e-5)))
            except Exception:
                try:
                    points.append(fallback_point(target))
                except Exception as exc:
                    raise WatermarkError(f"path arc length could not be inverted: {exc}") from exc
    return points


def _frame(points: list[complex], closed: bool) -> tuple[list[complex], list[complex]]:
    tangents: list[complex] = []
    normals: list[complex] = []
    n = len(points)
    for i in range(n):
        before = points[(i - 1) % n] if closed else points[max(0, i - 1)]
        after = points[(i + 1) % n] if closed else points[min(n - 1, i + 1)]
        tangent = after - before
        tangent = tangent / abs(tangent) if abs(tangent) > 1e-12 else 1 + 0j
        tangents.append(tangent)
        normals.append(1j * tangent)
    return tangents, normals


def _carrier(key: bytes, asset_id: str, contour_id: int, count: int, harmonics: int, closed: bool) -> list[float]:
    seed = f"contourmark-v1|{asset_id}|{contour_id}".encode()
    digest = hmac.digest(key, seed, "sha256")
    coefficients = [1 if (digest[i // 8] >> (i % 8)) & 1 else -1 for i in range(harmonics)]
    values: list[float] = []
    for i in range(count):
        u = i / (count if closed else count - 1)
        value = sum(
            coefficients[j] * (math.cos if j % 2 == 0 else math.sin)(2 * math.pi * (j // 2 + 2) * u)
            for j in range(harmonics)
        )
        if not closed:
            value *= math.sin(math.pi * u) ** 2
        values.append(value)
    rms = math.sqrt(sum(v * v for v in values) / len(values))
    return [v / rms for v in values]


def _format(value: float, precision: int) -> str:
    return f"{value:.{precision}f}".rstrip("0").rstrip(".") or "0"


def _polyline_d(points: list[complex], closed: bool, precision: int) -> str:
    text = "M" + _format(points[0].real, precision) + " " + _format(points[0].imag, precision)
    text += "".join("L" + _format(p.real, precision) + " " + _format(p.imag, precision) for p in points[1:])
    return text + ("Z" if closed else "")


def _cubic_d(contour: SVGPath, carrier: list[float], amplitude: float, precision: int) -> str:
    """Perturb a cubic contour without flattening it into a large polyline.

    Splitting a cubic is geometrically exact before displacement. The control
    offsets approximate a continuous normal field, keeping shared endpoints
    continuous and leaving the path's visible contour as the carrier.
    """
    length = contour.length(error=1e-5)
    closed = contour.isclosed()
    count = len(carrier)

    def field(s: float) -> complex:
        s = s % length if closed else min(max(s, 0.0), length)
        u = s / length
        position = u * (count if closed else count - 1)
        index = min(int(position), count - 1)
        fraction = position - index
        chip = carrier[index] * (1 - fraction) + carrier[(index + 1) % count] * fraction if closed else carrier[index] * (1 - fraction) + carrier[min(index + 1, count - 1)] * fraction
        epsilon = length * 1e-4
        low = (s - epsilon) % length if closed else max(0, s - epsilon)
        high = (s + epsilon) % length if closed else min(length, s + epsilon)
        tangent = contour.point(contour.ilength(high)) - contour.point(contour.ilength(low))
        if abs(tangent) < 1e-12:
            tangent = 1 + 0j
        return amplitude * chip * 1j * tangent / abs(tangent)

    pieces: list[CubicBezier] = []
    offset = 0.0
    target = max(32, count * 2)
    for segment in contour:
        segment_length = segment.length()
        n = max(1, round(target * segment_length / length))
        for j in range(n):
            a, b = j / n, (j + 1) / n
            part = segment.cropped(a, b)
            sa = offset + segment.length(0, a)
            sb = offset + segment.length(0, b)
            delta0 = field(sa)
            delta1 = field(sa + (sb - sa) / 3)
            delta2 = field(sa + 2 * (sb - sa) / 3)
            delta3 = field(sb)
            start = pieces[-1].end if pieces else part.start + delta0
            pieces.append(CubicBezier(start, part.control1 + delta1, part.control2 + delta2, part.end + delta3))
        offset += segment_length
    if closed:
        last = pieces[-1]
        pieces[-1] = CubicBezier(last.start, last.control1, last.control2, pieces[0].start)
    d = SVGPath(*pieces).d()
    if closed:
        d += " Z"
    # svgpathtools uses more digits than needed; keep the requested bound.
    import re

    return re.sub(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[Ee][-+]?\d+)?", lambda m: _format(float(m.group()), precision), d)


def _parse_svg(source: bytes) -> ET.ElementTree:
    if b"<!DOCTYPE" in source.upper() or b"<!ENTITY" in source.upper():
        raise WatermarkError("DTD/entity declarations are not accepted")
    try:
        return ET.ElementTree(ET.fromstring(source))
    except ET.ParseError as exc:
        raise WatermarkError(f"invalid SVG XML: {exc}") from exc


def _scale(contours: list[tuple[ET.Element, int, SVGPath]]) -> float:
    xs: list[float] = []
    ys: list[float] = []
    for _, _, contour in contours:
        left, right, bottom, top = contour.bbox()
        xs.extend((left, right))
        ys.extend((bottom, top))
    if not xs:
        raise WatermarkError("no supported paths or polygonal contours")
    diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    if diagonal <= 0:
        raise WatermarkError("drawing has zero-size bounding box")
    return diagonal


def _eligible(contours: list[tuple[ET.Element, int, SVGPath]], scale: float, params: Parameters) -> list[int]:
    return [i for i, (_, _, path) in enumerate(contours) if path.length() >= scale * params.min_perimeter / 100]


def _manifest_mac(manifest: dict, key: bytes) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hmac.digest(key, b"manifest-v1|" + payload, "sha256").hex()


def generate_key() -> bytes:
    return secrets.token_bytes(32)


def embed(source: bytes, key: bytes, asset_id: str | None = None, params: Parameters | None = None) -> tuple[bytes, dict]:
    params = params or Parameters()
    params.validate()
    if len(key) < 16:
        raise WatermarkError("key must contain at least 16 bytes")
    asset_id = asset_id or secrets.token_hex(16)
    if len(asset_id) > 256 or not asset_id:
        raise WatermarkError("asset ID must contain 1 to 256 characters")
    tree = _parse_svg(source)
    contours = _contours(tree)
    scale = _scale(contours)
    selected = _eligible(contours, scale, params)
    if not selected:
        raise WatermarkError("no contour meets the minimum perimeter; lower --min-perimeter")
    modified: dict[ET.Element, dict[int, str]] = {}
    for contour_id in selected:
        element, subpath_id, contour = contours[contour_id]
        carrier = _carrier(key, asset_id, contour_id, params.samples, params.harmonics, contour.isclosed())
        if all(isinstance(segment, CubicBezier) for segment in contour):
            d = _cubic_d(contour, carrier, params.amplitude * scale, params.precision)
        else:
            original = _sample(contour, params.samples)
            _, normals = _frame(original, contour.isclosed())
            marked = [point + params.amplitude * scale * chip * normal for point, chip, normal in zip(original, carrier, normals)]
            if not contour.isclosed():
                marked[0], marked[-1] = original[0], original[-1]
            d = _polyline_d(marked, contour.isclosed(), params.precision)
        modified.setdefault(element, {})[subpath_id] = d
    # Preserve unsupported/unselected subpaths and every original SVG attribute.
    for element, pieces in modified.items():
        original_path = _element_path(element)
        assert original_path is not None
        subpaths = original_path.continuous_subpaths()
        d = " ".join(pieces.get(i, subpath.d()) for i, subpath in enumerate(subpaths))
        element.set("d", d)
        if _tag(element) != "path":
            element.tag = element.tag.rsplit("}", 1)[0] + "}path" if "}" in element.tag else "path"
            element.attrib.pop("points", None)
    output = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
    unsigned = {
        "schema": "contourmark-v1",
        "asset_id": asset_id,
        "original_sha256": hashlib.sha256(source).hexdigest(),
        "parameters": asdict(params),
        "contour_indices": selected,
    }
    return output, {**unsigned, "mac": _manifest_mac(unsigned, key)}


def verify(original: bytes, candidate: bytes, manifest: dict, key: bytes, null_trials: int = 999) -> dict:
    if not 0 <= null_trials <= 9999:
        raise WatermarkError("null_trials must be in [0, 9999]")
    unsigned = {k: v for k, v in manifest.items() if k != "mac"}
    if not hmac.compare_digest(str(manifest.get("mac", "")), _manifest_mac(unsigned, key)):
        raise WatermarkError("manifest authentication failed")
    if unsigned.get("schema") != "contourmark-v1":
        raise WatermarkError("unsupported manifest schema")
    if hashlib.sha256(original).hexdigest() != unsigned["original_sha256"]:
        raise WatermarkError("original SVG does not match manifest")
    params = Parameters(**unsigned["parameters"])
    params.validate()
    before = _contours(_parse_svg(original))
    after = _contours(_parse_svg(candidate))
    scale = _scale(before)
    selected: list[int] = unsigned["contour_indices"]
    if not selected or max(selected) >= len(after) or len(before) != len(after):
        raise WatermarkError("contour count changed; geometric synchronization failed")
    observations: list[tuple[int, list[float], bool]] = []
    for index in selected:
        a = before[index][2]
        b = after[index][2]
        if a.isclosed() != b.isclosed():
            raise WatermarkError("contour open/closed status changed")
        base = _sample(a, params.samples)
        test = _sample(b, params.samples)
        _, normals = _frame(base, a.isclosed())
        residual = [((q - p) / scale * n.conjugate()).real for p, q, n in zip(base, test, normals)]
        observations.append((index, residual, a.isclosed()))
    def score(probe_key: bytes) -> float:
        total = 0.0
        energy = 0.0
        for index, residual, closed in observations:
            carrier = _carrier(probe_key, unsigned["asset_id"], index, params.samples, params.harmonics, closed)
            total += sum(x * c for x, c in zip(residual, carrier))
            energy += sum(c * c for c in carrier)
        return total / energy
    observed = score(key)
    null_scores = [score(hmac.digest(key, f"null-probe-{i}".encode(), "sha256")) for i in range(null_trials)]
    exceed = sum(s >= observed for s in null_scores)
    p_value = (exceed + 1) / (null_trials + 1)
    detected = observed >= params.amplitude * 0.35 and (not null_trials or p_value <= 0.01)
    return {
        "detected": detected,
        "estimated_amplitude": observed,
        "expected_amplitude": params.amplitude,
        "normalized_strength": observed / params.amplitude,
        "empirical_p_value": p_value,
        "null_trials": null_trials,
        "contours": len(selected),
        "limitation": "Empirical null probes are not a formal false-positive guarantee.",
    }


def read_key(path: str | Path) -> bytes:
    try:
        key = bytes.fromhex(Path(path).read_text().strip())
    except (OSError, ValueError) as exc:
        raise WatermarkError(f"cannot read hex key: {exc}") from exc
    if len(key) < 16:
        raise WatermarkError("key must contain at least 16 bytes")
    return key
