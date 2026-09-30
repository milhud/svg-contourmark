"""Toy equal-weight contour proposals for the manifest-free sampler."""

import json


steps = []
for i in range(32):
    y = 5 + 6 * i
    steps.append({
        "attributes": {"fill": "none", "stroke": "#333", "stroke-width": "1"},
        "candidates": [
            {"d": f"M 10 {y} C 60 {y-9} 140 {y+9} 190 {y}"},
            {"d": f"M 10 {y} C 60 {y-5} 140 {y+13} 190 {y}"},
        ],
    })
print(json.dumps({"viewBox": "0 0 200 200", "steps": steps}, indent=2))
