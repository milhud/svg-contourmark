"""Toy proposal stream; replace this with a real model's contour proposals."""

import json


steps = []
for i in range(9):
    y = 15 + 19 * i
    steps.append({
        "attributes": {"fill": "none", "stroke": "#333", "stroke-width": "1.5"},
        "candidates": [
            {"d": f"M 10 {y} C 60 {y-9} 140 {y+9} 190 {y}", "weight": 1},
            {"d": f"M 10 {y} C 60 {y-7} 140 {y+11} 190 {y}", "weight": 1},
        ],
    })
print(json.dumps({"viewBox": "0 0 200 200", "steps": steps}, indent=2))

