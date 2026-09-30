"""Small direct baselines for identical-attack comparisons.

These are intentionally simple reference methods, not presented as new work.
They make it possible to quantify how metadata and numeric serialization marks
fail under the same optimizers and coordinate edits used for ContourMark.
"""

from __future__ import annotations

import hmac
import math
import re

from .core import WatermarkError


_NUMBER = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[Ee][-+]?\d+)?")


def metadata_embed(svg: bytes, payload: str) -> bytes:
    if "--" in payload or payload.endswith("-"):
        raise WatermarkError("payload cannot be represented in an XML comment")
    marker = f"<!-- contourmark-metadata:{payload} -->".encode()
    declaration_end = svg.find(b"?>")
    position = declaration_end + 2 if declaration_end >= 0 else 0
    return svg[:position] + marker + svg[position:]


def metadata_detect(svg: bytes, payload: str) -> bool:
    return f"<!-- contourmark-metadata:{payload} -->".encode() in svg


def _bits(key: bytes, asset_id: str, count: int) -> list[int]:
    output: list[int] = []
    block = 0
    while len(output) < count:
        digest = hmac.digest(key, f"numeric-lsb-v1|{asset_id}|{block}".encode(), "sha256")
        output.extend((byte >> shift) & 1 for byte in digest for shift in range(8))
        block += 1
    return output[:count]


def numeric_lsb_embed(svg: bytes, key: bytes, asset_id: str, *, chips: int = 64, quantum: float = 0.001) -> bytes:
    """Encode keyed bits in numeric path-coordinate parity.

    This baseline assumes path numbers are coordinates; it is suitable for the
    cubic-only evaluation fixtures and is not a general SVG rewriter.
    """
    if len(key) < 16 or not 8 <= chips <= 1024 or not 0 < quantum <= 1:
        raise WatermarkError("invalid numeric-LSB parameters")
    text = svg.decode("utf-8")
    matches = []
    for attribute in re.finditer(r"\bd\s*=\s*(['\"])(.*?)\1", text, re.DOTALL):
        start = attribute.start(2)
        matches.extend((start + number.start(), start + number.end(), number.group()) for number in _NUMBER.finditer(attribute.group(2)))
    if len(matches) < chips:
        raise WatermarkError(f"numeric-LSB baseline needs {chips} path numbers; found {len(matches)}")
    bits = _bits(key, asset_id, chips)
    replacements = []
    for (start, end, token), bit in zip(matches[:chips], bits):
        value = float(token)
        quantized = round(value / quantum)
        if quantized % 2 != bit:
            lower, upper = quantized - 1, quantized + 1
            quantized = lower if abs(lower * quantum - value) <= abs(upper * quantum - value) else upper
        replacement = f"{quantized * quantum:.9f}".rstrip("0").rstrip(".")
        replacements.append((start, end, replacement or "0"))
    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]
    return text.encode()


def numeric_lsb_detect(svg: bytes, key: bytes, asset_id: str, *, chips: int = 64, quantum: float = 0.001) -> dict:
    if len(key) < 16:
        raise WatermarkError("key must contain at least 16 bytes")
    text = svg.decode("utf-8")
    values = []
    for attribute in re.finditer(r"\bd\s*=\s*(['\"])(.*?)\1", text, re.DOTALL):
        values.extend(float(number.group()) for number in _NUMBER.finditer(attribute.group(2)))
    if len(values) < chips:
        raise WatermarkError(f"numeric-LSB baseline needs {chips} path numbers; found {len(values)}")
    expected = _bits(key, asset_id, chips)
    matches = sum(round(value / quantum) % 2 == bit for value, bit in zip(values[:chips], expected))
    p_value = sum(math.comb(chips, count) for count in range(matches, chips + 1)) / 2**chips
    return {"detected": p_value <= 0.01, "matches": matches, "chips": chips, "p_value": p_value}
