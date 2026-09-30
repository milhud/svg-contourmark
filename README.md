# ContourMark

Research prototype for **generation-time SVG watermarking**. A drawing model proposes two or more visually comparable contour candidates at each step. ContourMark uses a secret key to choose one *before* that contour enters the output SVG. The mark is a statistical pattern of **geometric choices**, so XML whitespace and path command notation do not carry the evidence.

Two inference-time detectors are implemented:

* **Manifest-free (`sample-blind` / `detect-blind`):** equal-weight alternatives are chosen with a keyed score of a coarse contour centroid. The verifier needs the key and a precommitted asset ID, but no original SVG or candidate list. The centroid quantizer has guard bands to reduce rounding failures.
* **Candidate-assisted (`sample` / `verify-sample`):** weighted alternatives use keyed Gumbel-max sampling. The verifier uses a private authenticated candidate manifest and geometry matching. It supports partial path deletion through a Poisson-binomial score.

A separate reference-assisted contour-perturbation baseline (`embed`/`verify`) is retained for comparison.

## Install and run

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/contourmark keygen owner.key
.venv/bin/python examples/make_proposals.py > proposals.json
.venv/bin/contourmark sample proposals.json drawing.svg --key owner.key --manifest drawing.wm.json
.venv/bin/contourmark verify-sample drawing.svg --key owner.key --manifest drawing.wm.json
.venv/bin/python examples/make_blind_proposals.py > blind-proposals.json
.venv/bin/contourmark sample-blind blind-proposals.json blind.svg --key owner.key --asset-id demo-2026-001
.venv/bin/contourmark detect-blind blind.svg --key owner.key --asset-id demo-2026-001
.venv/bin/contourmark assess drawing.svg
.venv/bin/python -m pytest -q
```

The key and assisted-mode manifest are private verification material. `.gitignore` excludes `*.key` and `*.wm.json`. The released SVG contains only ordinary path geometry and styling. Commit the asset ID and key identity with a trusted timestamp **before** generation if provenance evidence is needed; otherwise a claimant could search for a key after seeing an SVG.

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

With a closed model API, this controls **selection between completed contour proposals**, not the model's internal token sampler. With an open model, the tested framework-neutral sampler in `contourmark.token_sampling` can replace categorical draws at grammar-confirmed geometry tokens. No training is required.

## Detection and current scope

Candidate-assisted verification authenticates the private manifest, parses candidate paths, samples contours by arc length, and tests whether released geometry matches keyed choices. It reports a Poisson-binomial tail probability for the number of matches. Manifest-free verification computes a coarse centroid symbol for each visible contour and tests whether its keyed scores are unusually high under an unmarked null. Neither method alters a chosen candidate after selection.

Current prototype supports one contour per proposed path and 2–16 alternatives per step. Candidates must differ enough to remain distinguishable after minification but by at most 3% of the canvas diagonal. Assisted verification matches contours geometrically across path reordering, merging, and bounded independent translations. Tests exercise Scour and SVGO. Three 16-stroke Qwen-generated trials verify before and after both optimizers, rounding, reorder, and translation. A keyless control-point attack defeats all three with small but measurable render changes, so these trials are evidence of execution and limits rather than a model-wide robustness claim.

The official hosted OmniSVG 4B option also produced a real one-path artifact that passes the strict carrier assessor. Its public API returns only the final SVG, so that run validates output compatibility but cannot exercise inference-time insertion; the model runtime must expose logits or a pre-sampling candidate hook.

`contourmark assess` inventories an arbitrary SVG and fails closed for text, embedded images, filters, rigid primitives, CSS/transform indirection, and mixed content. It reports path eligibility but never infers nonzero watermark capacity from syntax alone; capacity requires actual generator alternatives.

See [research design](docs/research.md) for the mathematical argument, related work, threat model, and experimental plan.

See [model adapters](docs/model-adapters.md) for the OmniSVG 1.1 reference integration and the adapter contract for icon-focused autoregressive generators.
