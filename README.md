# ContourMark

Research prototype for **blind, representation-invariant watermarking of SVG geometry**. The target use is marking AI-generated SVG before release and detecting it later, after the file has passed through real toolchains.

**Primary method (`mark` / `detect`, `src/contourmark/spectral.py`).**
- **Works after generation.** It needs no model access, so it applies to output from an LLM writing SVG code, OmniSVG, StarVector, a vectorizer, or a human designer. It needs path geometry with curved contours: text, raster images and straight-line drawings carry nothing.
- **Embedding:**
  1. Resample each visible contour by arc length.
  2. Compute its normalized radial function (distance to centroid over mean distance).
  3. Move 8 mid-frequency Fourier/DCT magnitudes onto a secret dithered lattice (keyed QIM).
- **Invariant by construction to:** SVGO/Scour rewriting, absolute/relative commands, segment subdivision, path merge/split/reorder, start point, traversal direction, translation, uniform scale, rotation, and mirroring.
- **Detection:** needs only the SVG and the key. It returns a p-value that is a valid upper bound over the key's randomness for any SVG chosen independently of the key, so no corpus-fitted threshold is needed.
- **Cost:** the mark moves each contour by about 0.4% of its radius (RMS) and is not visible at normal sizes.

Known limits, measured in `experiments/results/blind/`:
- Heavy rounding or noise comparable to the mark amplitude removes it.
- Aspect-ratio stretch, stroke-to-outline conversion, and rasterize-and-retrace mostly remove it.
- Icons made only of straight lines have little or no capacity.

**Zero-distortion generation-time variant (`sample` / `verify-sample`).** If a generator exposes several candidate contours per step, a keyed Gumbel-max choice marks the output without moving any geometry. Verification needs a private candidate manifest. See [research design](docs/research.md).

Legacy modes (`sample-blind`, `embed`/`verify`) are kept for comparison. See [literature review](docs/literature.md) and [HANDOFF.md](HANDOFF.md) for the current state of the project.

## Inference-time watermark (`geosample`)

A second, distortion-free mode for generators whose decoder you control. It works like text watermarking (SynthID / Gumbel sampling), but keys a **geometric** quantity instead of token IDs, so detection survives SVGO, Scour, rounding, reordering, reversal, and similarity transforms.

- **Keyed choices.** At each line/curve endpoint, the model's candidate tokens are grouped by the corner (vertex) descriptor they create: interior angle, arm-length ratio, and summed bulge. At each curve's first control point, they are grouped by the tangent descriptor at that corner.
- **Distribution preserved, at the sequence level.** A keyed Gumbel-max draw over descriptor groups, then ordinary sampling within the group. A descriptor's keyed score is used only the first time it is looked at in a drawing; later looks use fresh randomness. That rule is what makes the joint output law equal the model's (random-PRF idealization); naive reuse does not, and costs less power.
- **Detection** needs only the final SVG and the key. It uses an exact Gamma null over distinct descriptors (`contourmark.geosample.detect`).
- **Models.** Any point-token SVG model via a small grammar adapter (`contourmark.point_token_models`):
  - IconShop: run here with the public checkpoint.
  - OmniSVG: adapter implemented and tested on synthetic streams.
  - Anything decoded through Hugging Face `generate`, via `GeoWatermarkLogitsProcessor`.

```python
from contourmark.geosample import GeoWatermark, detect
from contourmark.point_token_models import DecodeState, IconShopGrammar, watermarked_step

state, wm = DecodeState(IconShopGrammar()), GeoWatermark(key)
while True:                                   # your decoder loop
    logits = model_next_token_logits(...)     # over the point-token vocabulary
    token, _ = watermarked_step(state, logits, wm, rng, top_p=0.9)
    if token == 0: break
    state.feed(token)
svg = state.svg()
detect(svg, key)["p_value"]
```

For `model.generate(...)`, pass `GeoWatermarkLogitsProcessor(grammar, GeoWatermark(key), prompt_length, top_p=...)` with `do_sample=True`. Truncation is configured on the processor. See [docs/inference-watermark.md](docs/inference-watermark.md) for definitions, the distribution-preservation proof, the adapter contract, and limits: non-uniform scaling, retracing, low-entropy decoding, and text LLMs needing a number-level adapter.

## Install and run

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/contourmark keygen owner.key
.venv/bin/contourmark mark icon.svg icon.marked.svg --key owner.key      # blind mark, path-based SVG
.venv/bin/contourmark detect icon.marked.svg --key owner.key             # needs only SVG + key
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
