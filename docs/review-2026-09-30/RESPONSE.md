# Response to the independent review (1 October 2026)

The review in this directory was assessed against the code and found correct
on every reproduced finding. This file records what was changed in response,
what was done differently from `proposed-fixes.patch`, and what is still open.

Verification: `reproduce_findings.py` output after the fixes is in
`findings-after-fixes.json`; regression tests are in
`tests/test_review_regressions.py` (10 tests). The full suite is 100 passing.

## Implemented

### P0: the sequence-level distribution claim (finding accepted, cause fixed)

The review showed that reusing `PRF(key, descriptor)` at every step changes
the joint output law (0.50 observed against 0.70 expected in a two-step
example). The patch proposed retracting the claim. Instead the sampler was
changed so the claim holds:

* Within one drawing, a descriptor's keyed uniform is used only the first
  time that descriptor is examined, whether or not it wins. Later looks use a
  fresh unkeyed uniform (`GeoWatermark(reuse="mask")`, now the default).
* A uniform at a never-examined index is independent of the generated prefix,
  so the grouped Gumbel-max lemma applies conditionally at every step and the
  joint law is preserved by induction (proof in `docs/inference-watermark.md`
  §3.2).
* `reuse="allow"` keeps the old behaviour as an ablation only.
* The Hugging Face processor now keeps one sampler state and RNG per batch row.

Measured after the fix: repeat-choice rate 0.504 (expected 0.5); adaptive
second-step probability 0.703 (expected 0.7).

Cost: on the toy generator the median clean log10 p goes from −49 to −18.7
(100% still below 1e-6). The earlier IconShop pilot numbers used the biased
sampler and are withdrawn.

### P0: visible-content binding (finding accepted, different remedy)

Hidden marked geometry under an empty `clipPath` kept its full score in both
detectors. The patch proposed rejecting any document with `<style>`, `<use>`,
text, images, nested SVG, or clip/mask/filter. That would refuse 24 of the 40
LLM-written SVGs in the corpus. The implemented behaviour is:

* **Strict mode (default).** Contours under `clip-path`, `mask`, `filter` or a
  nested viewport are excluded from evidence. Geometry entirely outside the
  root viewBox is treated as not drawn.
* **Status field.** Results carry `status`: `detected`, `not_detected`, or
  `indeterminate`. Documents with CSS rules, scripts or animation, or with
  excluded contours and no remaining evidence, are `indeterminate`, which is
  explicitly not "unmarked".
* **Render mode (optional).** `visibility="render"` (new module
  `visibility.py`) asks librsvg whether each unresolved contour contributes
  pixels, relative to its unclipped footprint. It applies to both embedding
  and detection.

Measured after the fix: the empty-clip decoy scores log10 p = 0 in both
detectors. A full-canvas (benign) clip is still detected in render mode.

Consequence for results: the LLM-SVG detection rate was 100% only because
clipped contours were counted. It is now 78% (strict) and 90% (render). The
icon corpora contain no clip, mask or CSS and are unchanged.

### P1: handle descriptors depend on representation (fixed for degree)

Tangent descriptors now use end derivatives (degree × control offset). A
quadratic and its degree-elevated cubic give the same descriptor, and a
straight cubic matches a line. The detector uses the same 32-sample bulge
estimate as the sampler for cubic and quadratic segments.

Not fixed: subdivision still rescales chords and derivatives. With the
corrected sampler, midpoint subdivision reduces the toy example to log10 p of
about −2. The descriptors are documented as similarity-, reversal- and
degree-invariant, not intrinsic.

### P1: GPU harness (patch applied as proposed)

Applied from `proposed-fixes.patch`:

* an independent watermark RNG per sample;
* a run identity (configuration, code and checkpoint hashes, package versions)
  with refusal to resume a mismatched run;
* strict checkpoint-compatibility check;
* shard PIDs waited individually, with no merge after a failed shard;
* `completed` / `truncated` recorded per sample.

Added on top:

* summaries report attempted, attack-error and detector-error counts next to
  every rate;
* truncated generations and marked samples without descriptors are kept in
  the denominators and reported.

### Spectral embedder (review §2A)

* `seed_stable` was recorded but not enforced. The embedder now re-embeds with
  the seed class the marked contour lands in (up to three attempts) and
  otherwise leaves the contour untouched (`status: skipped_unstable_seed`).
  On the 900-asset test split 52 of about 3,930 contours are skipped this way.
* A displacement cap (`max_relative_displacement`, 8% of the contour's mean
  radius) rolls back a contour whose curve moves further
  (`status: skipped_distortion`; 1 contour on the test split).

### Manuscript and documentation corrections

* "million-test" null replaced by the actual count (268,200).
* "Every generator" / "any SVG" scoped to path geometry with enough curved
  contours.
* Invisible-decoy claim replaced by the stated visibility subset and its
  limits.
* Baselines described as adaptations at fixed strengths, **not**
  distortion-matched (54–83 dB against 25 dB). The vertex-FD result is
  described as "no operating point under our settings", not a reproduction
  failure.
* Coordinate-LSB claim corrected: it partly survives ×3 scaling (70%),
  mirroring (70%), translation (34%), and an unflattened group transform
  (80%).
* "Exact" tail bound qualified: no formal floating-point error analysis.
* `<metadata>` baseline labelled a placeholder, not C2PA.
* The inference sampler is now a separate section with its own (toy) evidence
  and caveats; the 900-asset results are stated not to validate it.
* `crop_half` described as removing contours in one half (it does not clip
  curves).

## Re-run after the fixes (`experiments/results/blind/v2/`)

| Quantity | Before | After |
|---|---|---|
| Clean detection, 900 icons | 747 | 748 |
| SVGO survival (conditional) | 94.2% | 94.3% |
| 2-decimal rounding survival | 77.2% | 77.4% |
| LLM-SVG clean detection | 100% | 78% strict / 90% render |
| Random-key re-embedding removal | 100% | 98.3% |
| Spectral null (268,200 tests) | valid | unchanged |
| Geosample null (267,000 tests) | valid | valid (0.063 / 0.0068 / 0.00072 / 7.9e-5 / 3.7e-6) |

## Not done (still open, as the review says)

* Distortion-matched baseline frontiers, positive controls for the vertex-FD
  adaptation, and arc-length spread-spectrum / QIM ablations.
* Real IconShop results with the corrected sampler; a real OmniSVG
  integration; multi-key and repeated-prompt diversity measurements.
* An adaptive-removal study for the inference sampler.
* Intrinsic (subdivision-stable) descriptors.
* One production key shared across many assets (seed-class leakage).
* Occlusion in strict mode; multiple renderers in render mode; real curve
  clipping in the crop attack.
* Floating-point error analysis of the tilted convolution.
* Perceptual study, RGB/alpha-aware fidelity, editability metrics.
* CGF template and the workflow-centred framing proposed in review §5.
