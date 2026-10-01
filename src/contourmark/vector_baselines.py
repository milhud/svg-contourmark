"""Faithful re-implementations of prior vector-graphics marking schemes.

These are comparison baselines, evaluated under the same corpus and attacks
as the spectral watermark:

* ``fd_vertex``: zero-bit watermark in Fourier-descriptor magnitudes of the
  vertex sequence (Solachidis and Pitas, ICASSP 2000; Doncel, Nikolaidis and
  Pitas, IEEE TVCG 2007).  As in the TVCG paper, curves are handled by
  treating their control points as polygon vertices.  Detection is blind and
  uses the normalized correlation with the keyed sign sequence, calibrated by
  a keyed null (the correlation under other keys).
* ``bezier_split``: data hiding by splitting cubic Beziers at keyed ratios
  (after Blinova and Urbanovich's StegoSVG).  The split parameter is read
  back from the collinear handles at each smooth join.
* ``metadata_element``: a signed-manifest stand-in stored in ``<metadata>``,
  as C2PA-style content credentials are stored in SVG.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import binom, norm

from .geometry import SVG_NS, Segment, Subpath, load_document, parse_path_data, serialize_compact, tag
from .spectral import _shapes_to_paths


def _parse(source: bytes) -> ET.Element:
    return ET.fromstring(source, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))


def _dump(root: ET.Element) -> bytes:
    ET.register_namespace("", SVG_NS)
    return ET.tostring(root, encoding="unicode").encode()


def _bits(key: bytes, label: str, count: int) -> np.ndarray:
    stream = b""
    block = 0
    while len(stream) * 8 < count:
        stream += hmac.digest(key, f"{label}|{block}".encode(), "sha256")
        block += 1
    return np.unpackbits(np.frombuffer(stream, dtype=np.uint8))[:count].astype(int)


def _paths(root: ET.Element) -> list[ET.Element]:
    skip = {"defs", "clipPath", "mask", "symbol", "pattern", "marker"}
    out = []

    def visit(element: ET.Element) -> None:
        for child in element:
            if tag(child) in skip:
                continue
            if tag(child) == "path" and child.get("d"):
                out.append(child)
            visit(child)

    visit(root)
    return out


# ---------------------------------------------------------------------------
# Fourier-descriptor magnitude watermark on vertex sequences


FD_BAND = (0.1, 0.4)  # fraction of the positive half spectrum
FD_MIN_VERTICES = 16


def _vertex_sequence(subpath: Subpath) -> tuple[np.ndarray, list[tuple[int, int]]]:
    positions: list[complex] = []
    owners: list[tuple[int, int]] = []
    for s, segment in enumerate(subpath.segments):
        for p in range(0 if s == 0 else 1, len(segment.points)):
            positions.append(complex(segment.points[p]))
            owners.append((s, p))
    if subpath.closed and len(positions) > 1 and abs(positions[-1] - positions[0]) < 1e-9:
        positions.pop()
        owners.pop()
    return np.array(positions), owners


def _band(n: int) -> np.ndarray:
    half = n // 2
    low, high = max(2, int(FD_BAND[0] * half)), max(3, int(FD_BAND[1] * half))
    return np.arange(low, min(high, half))


def fd_vertex_embed(source: bytes, key: bytes, strength: float = 0.02) -> bytes:
    root = _parse(source)
    _shapes_to_paths(root)
    for element in _paths(root):
        subpaths = parse_path_data(element.get("d", ""))
        changed = False
        for subpath in subpaths:
            vertices, owners = _vertex_sequence(subpath)
            n = len(vertices)
            if n < FD_MIN_VERTICES:
                continue
            spectrum = np.fft.fft(vertices)
            band = _band(n)
            signs = 2 * _bits(key, f"fd-vertex|{n}", len(band)) - 1
            for k, sign in zip(band, signs):
                # Modify |Z_k| and |Z_-k| equally so traversal reversal and
                # reflection (which swap k and -k) preserve the mark.
                spectrum[k] *= 1 + strength * sign
                spectrum[-k] *= 1 + strength * sign
            moved = np.fft.ifft(spectrum)
            for (s, p), point in zip(owners, moved):
                subpath.segments[s].points[p] = point
                if p == len(subpath.segments[s].points) - 1 and s + 1 < len(subpath.segments):
                    subpath.segments[s + 1].points[0] = point
            if subpath.closed:
                subpath.segments[-1].points[-1] = subpath.segments[0].points[0]
            changed = True
        if changed:
            element.set("d", serialize_compact(subpaths, 4))
    return _dump(root)


def _fd_statistic(document_contours: list[tuple[np.ndarray, int]], key: bytes) -> float:
    total = 0.0
    count = 0
    for magnitudes, n in document_contours:
        band = _band(n)
        signs = 2 * _bits(key, f"fd-vertex|{n}", len(band)) - 1
        # Multiplicative marks are additive in log magnitude.  Estimate the
        # host spectrum by a running median of neighbouring coefficients (the
        # keyed signs are zero-mean, so the median is nearly mark-free) and
        # correlate the residual with the signs.
        values = np.log(magnitudes[band] + 1e-12)
        values = values - median_filter(values, size=min(9, len(values) | 1), mode="nearest")
        values = values / (values.std() + 1e-12)
        total += float(values @ signs) / math.sqrt(len(band))
        count += 1
    return total / math.sqrt(count) if count else 0.0


def fd_vertex_detect(source: bytes, key: bytes, null_keys: int = 200) -> dict:
    document = load_document(source)
    contours = []
    for contour in document.contours:
        if not contour.visible:
            continue
        vertices, _ = _vertex_sequence(contour.subpath)
        n = len(vertices)
        if n < FD_MIN_VERTICES:
            continue
        spectrum = np.fft.fft(vertices)
        contours.append((np.abs(spectrum) + np.abs(np.roll(spectrum[::-1], 1)), n))
    if not contours:
        return {"detected": False, "p_value": 1.0, "log10_p_value": 0.0, "contours": 0}
    observed = _fd_statistic(contours, key)
    null = np.array([_fd_statistic(contours, hashlib.sha256(key + i.to_bytes(4, "big")).digest()) for i in range(null_keys)])
    z = (observed - null.mean()) / (null.std() + 1e-12)
    p = float(norm.sf(z))
    return {"detected": p <= 1e-6, "p_value": p, "log10_p_value": math.log10(max(p, 1e-300)), "z": float(z), "contours": len(contours)}


# ---------------------------------------------------------------------------
# Bezier split steganography


SPLIT_OFFSET = 0.15


def bezier_split_embed(source: bytes, key: bytes) -> bytes:
    root = _parse(source)
    _shapes_to_paths(root)
    elements = _paths(root)
    total = sum(1 for element in elements for sub in parse_path_data(element.get("d", "")) for s in sub.segments if s.kind == "C")
    bits = _bits(key, "bezier-split", max(total, 1))
    index = 0
    for element in elements:
        subpaths = parse_path_data(element.get("d", ""))
        rebuilt = []
        for subpath in subpaths:
            segments: list[Segment] = []
            for segment in subpath.segments:
                if segment.kind != "C":
                    segments.append(segment)
                    continue
                t = 0.5 + SPLIT_OFFSET * (2 * bits[index] - 1)
                index += 1
                p0, p1, p2, p3 = segment.points
                a, b, c = p0 + t * (p1 - p0), p1 + t * (p2 - p1), p2 + t * (p3 - p2)
                d, e = a + t * (b - a), b + t * (c - b)
                m = d + t * (e - d)
                segments += [Segment("C", np.array([p0, a, d, m])), Segment("C", np.array([m, e, c, p3]))]
            rebuilt.append(Subpath(segments, subpath.closed))
        element.set("d", serialize_compact(rebuilt, 4))
    return _dump(root)


def bezier_split_detect(source: bytes, key: bytes) -> dict:
    document = load_document(source)
    estimates = []
    for contour in document.contours:
        segments = contour.subpath.segments
        for first, second in zip(segments[:-1], segments[1:]):
            if first.kind != "C" or second.kind != "C":
                continue
            join = first.points[3]
            before, after = join - first.points[2], second.points[1] - join
            if abs(before) < 1e-9 or abs(after) < 1e-9 or abs(np.angle(after / before)) > 0.02:
                continue
            t = abs(before) / (abs(before) + abs(after))
            # A split point must satisfy de Casteljau exactly: rebuild the
            # parent cubic and re-split it; natural smooth joins fail this.
            p0, a, d, m = first.points
            _, e, c, p3 = second.points
            parent1 = p0 + (a - p0) / t
            parent2 = p3 + (c - p3) / (1 - t)
            rebuilt_d = (1 - t) * ((1 - t) * p0 + t * parent1) + t * ((1 - t) * parent1 + t * parent2)
            scale = abs(p3 - p0) + 1e-12
            if abs(rebuilt_d - d) > 2e-3 * scale:
                continue
            estimates.append(t)
    if not estimates:
        return {"detected": False, "p_value": 1.0, "log10_p_value": 0.0, "splits": 0}
    # Each embedded split contributes one bit in document order.
    bits = _bits(key, "bezier-split", len(estimates))
    decoded = (np.array(estimates) > 0.5).astype(int)
    matches = int((decoded == bits).sum())
    p = float(binom.sf(matches - 1, len(estimates), 0.5))
    return {"detected": p <= 1e-6, "p_value": p, "log10_p_value": math.log10(max(p, 1e-300)), "splits": len(estimates), "matches": matches}


# ---------------------------------------------------------------------------
# Metadata element (content-credential stand-in)


def metadata_element_embed(source: bytes, key: bytes) -> bytes:
    root = _parse(source)
    tag_value = hmac.digest(key, b"metadata-credential", "sha256").hex()
    metadata = ET.Element(f"{{{SVG_NS}}}metadata")
    metadata.text = f"credential:{tag_value}"
    root.insert(0, metadata)
    return _dump(root)


def metadata_element_detect(source: bytes, key: bytes) -> dict:
    found = hmac.digest(key, b"metadata-credential", "sha256").hex().encode() in source
    return {"detected": found, "p_value": 0.0 if found else 1.0, "log10_p_value": -300.0 if found else 0.0}


# ---------------------------------------------------------------------------
# Coordinate-parity (numeric LSB) watermark, geometry-aware


LSB_QUANTUM = 1e-3


def _coordinates(subpaths: list[Subpath]) -> list[tuple[Subpath, int, int, int]]:
    """(subpath, segment, point, axis) for every explicit coordinate, in order."""
    out = []
    for subpath in subpaths:
        for s, segment in enumerate(subpath.segments):
            for p in range(0 if s == 0 else 1, len(segment.points)):
                out += [(subpath, s, p, 0), (subpath, s, p, 1)]
    return out


def numeric_lsb_embed(source: bytes, key: bytes, chips: int = 64, quantum: float = LSB_QUANTUM) -> bytes:
    """Encode keyed bits in the parity of absolute coordinates (not arc flags)."""
    root = _parse(source)
    _shapes_to_paths(root)
    elements = _paths(root)
    parsed = [(element, parse_path_data(element.get("d", ""))) for element in elements]
    slots = [slot for _, subpaths in parsed for slot in _coordinates(subpaths)]
    bits = _bits(key, "numeric-lsb", len(slots))
    for (subpath, s, p, axis), bit in zip(slots[:chips], bits):
        value = subpath.segments[s].points[p]
        coordinate = value.real if axis == 0 else value.imag
        q = round(coordinate / quantum)
        if q % 2 != bit:
            q += 1 if coordinate / quantum > q else -1
        new = q * quantum
        point = complex(new, value.imag) if axis == 0 else complex(value.real, new)
        subpath.segments[s].points[p] = point
        if p == len(subpath.segments[s].points) - 1 and s + 1 < len(subpath.segments):
            subpath.segments[s + 1].points[0] = point
    for element, subpaths in parsed:
        element.set("d", serialize_compact(subpaths, max(0, int(round(-math.log10(quantum))))))
    return _dump(root)


def numeric_lsb_detect(source: bytes, key: bytes, chips: int = 64, quantum: float = LSB_QUANTUM) -> dict:
    document = load_document(source, include_shapes=True)
    slots = [slot for contour in document.contours if contour.editable for slot in _coordinates([contour.subpath])]
    count = min(chips, len(slots))
    if count == 0:
        return {"detected": False, "p_value": 1.0, "log10_p_value": 0.0}
    bits = _bits(key, "numeric-lsb", count)
    values = [(sub.segments[s].points[p].real if axis == 0 else sub.segments[s].points[p].imag) for sub, s, p, axis in slots[:count]]
    matches = int(sum(round(v / quantum) % 2 == b for v, b in zip(values, bits)))
    p = float(binom.sf(matches - 1, count, 0.5))
    return {"detected": p <= 1e-6, "p_value": p, "log10_p_value": math.log10(max(p, 1e-300)), "matches": matches, "chips": count}
