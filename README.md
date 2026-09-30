# ContourMark

Research prototype for **generation-time SVG watermarking**. A drawing model proposes two or more visually comparable contour candidates at each step. ContourMark uses a secret key to choose one *before* that contour enters the output SVG. The mark is a statistical pattern of **geometric choices**, so XML whitespace, path command notation, and ordinary numeric cleanup do not carry the evidence.

The project includes a separate reference-assisted contour-perturbation baseline (`embed`/`verify`). The generation-time sampler (`sample`/`verify-sample` or `GenerationSession`) is the primary method.

## Install and run

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/contourmark keygen owner.key
.venv/bin/python examples/make_proposals.py > proposals.json
.venv/bin/contourmark sample proposals.json drawing.svg --key owner.key --manifest drawing.wm.json
.venv/bin/contourmark verify-sample drawing.svg --key owner.key --manifest drawing.wm.json
.venv/bin/python -m pytest -q
```

The key and manifest are private verification material. `.gitignore` excludes `*.key` and `*.wm.json`. The released SVG contains only ordinary path geometry and styling.

## Live generator integration

Use `GenerationSession` in a provider-neutral drawing loop. The generator supplies candidate contours and their categorical sampling weights; ContourMark makes each choice before SVG assembly:

```python
from contourmark.inference import Candidate, GenerationSession

session = GenerationSession(secret_key, "0 0 200 200", asset_id="asset-123")
for proposed_step in generator:
    choices = [Candidate(item["d"], item["weight"]) for item in proposed_step]
    selected_index = session.add_step(choices, {"fill": "none", "stroke": "black"})
    generator.observe_selection(selected_index)  # optional feedback for next step
svg_bytes, private_manifest = session.finish()
```

`sample-stream` is a JSONL stdin/stdout version for process integration. Each input line is `{"candidates":[{"d":"M...","weight":1}, ...],"attributes":{"stroke":"black","fill":"none"}}`; each output line immediately returns the chosen index and path. At EOF it writes the SVG and private manifest.

With a closed model API, this controls **selection between completed contour proposals**, not the model's internal token sampler. With an open model, the same selection can be placed inside a structured SVG decoder. No training or model download is required for the current prototype.

## Detection and current scope

Verification authenticates the private manifest, parses candidate paths, samples contours by arc length, and tests whether the released geometry matches the keyed choices. It reports the conditional probability of all matches under the declared unwatermarked categorical sampler. The inference-time method does not alter any chosen candidate after selection.

Current prototype supports one contour per proposed path and 2–16 alternatives per step. Candidates must differ enough to remain distinguishable after minification but by at most 3% of the canvas diagonal. Verification matches contours geometrically across path reordering and merging. Tests exercise real Scour minification. This is a **research prototype**, not a validated provenance service: other optimizers, transforms, deliberate redrawing, and visual-quality judgments still need systematic evaluation.

See [research design](docs/research.md) for the mathematical argument, related work, threat model, and experimental plan.
