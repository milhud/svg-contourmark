"""Fail-closed coverage assessment for arbitrary SVG documents."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .core import WatermarkError, _contours, _parse_svg, _tag
from .inference import _validate_render_subset


_FEATURE_CLASS = {
    "text": "text",
    "tspan": "text",
    "textPath": "text",
    "image": "embedded_raster",
    "filter": "filter",
    "feBlend": "filter",
    "feColorMatrix": "filter",
    "feGaussianBlur": "filter",
    "feOffset": "filter",
    "rect": "rigid_primitive",
    "circle": "rigid_primitive",
    "ellipse": "rigid_primitive",
    "line": "rigid_primitive",
    "polyline": "polygonal_path",
    "polygon": "polygonal_path",
    "use": "indirection",
    "symbol": "indirection",
    "foreignObject": "foreign_content",
    "mask": "compositing",
    "clipPath": "compositing",
    "pattern": "paint_server",
    "linearGradient": "paint_server",
    "radialGradient": "paint_server",
}


def assess_svg(source: bytes) -> dict[str, Any]:
    tree = _parse_svg(source)
    root = tree.getroot()
    tags = Counter(_tag(element) for element in root.iter())
    feature_counts: Counter[str] = Counter()
    for tag, count in tags.items():
        feature_counts[_FEATURE_CLASS.get(tag, "path_geometry" if tag in {"path", "g", "svg"} else "unsupported_other")] += count
    contours = _contours(tree)
    rendering_error = None
    try:
        _validate_render_subset(root)
    except WatermarkError as exc:
        rendering_error = str(exc)
    unsupported = {
        name: count for name, count in feature_counts.items()
        if name not in {"path_geometry"} and count
    }
    whole_document_supported = rendering_error is None and bool(contours)
    if whole_document_supported:
        status = "eligible_path_subset"
        reasons = ["The document is inside the verifier's strict path/group rendering subset."]
    elif not contours:
        status = "no_supported_geometric_carrier"
        reasons = ["No nonempty path or polygonal contour is available as a geometric carrier."]
    else:
        status = "mixed_or_unsupported_rendering"
        reasons = [rendering_error or "The document contains unsupported rendering features."]
    if unsupported:
        reasons.append("Whole-document attribution is unavailable while unsupported visible feature classes remain.")
    return {
        "status": status,
        "whole_document_supported": whole_document_supported,
        "supported_contours": len(contours),
        "element_counts": dict(sorted(tags.items())),
        "unsupported_feature_counts": dict(sorted(unsupported.items())),
        "choice_capacity_bits": None,
        "capacity_reason": "Capacity requires generator-provided admissible alternatives; syntax alone cannot establish nonzero choice capacity.",
        "universal_watermark_guarantee": False,
        "reasons": reasons,
    }
