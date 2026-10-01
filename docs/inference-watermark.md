# Inference-time geometric watermark (`geosample`)

Status: 30 September 2026. This document describes what is implemented in:

* `src/contourmark/geosample.py`: the model-agnostic core and the blind
  detector.
* `src/contourmark/point_token_models.py`: grammar adapters, the decoding
  step, and the Hugging Face hook.

Tests are in `tests/test_geosample.py`.

## 1. Motivation

Text watermarks such as SynthID-Text, Kirchenbauer green lists, and
Aaronson/Kuditipudi Gumbel sampling key the choice of the next *token*. The
detector later re-tokenizes the text and recomputes the keyed scores.

For SVG this fails at the first optimizer. SVGO, Scour, picosvg, and editors
rewrite almost every token while preserving the drawing:

* absolute ↔ relative commands
* shorthand curves (`S`/`T`, `H`/`V`)
* number precision and formatting
* path merging and reordering
* transform flattening

The token sequence a detector sees has little to do with the one the model
sampled. This is the extreme case of the known fragility of code watermarks
to semantics-preserving rewrites.

`geosample` keeps the text-watermark *mechanism*: a keyed,
distribution-preserving choice among the model's own candidates. It changes
the keyed *quantity*. The key acts on a quantized **geometric descriptor that
the candidate creates in the drawing**, not on the token ID. A detector can
recompute that descriptor from any rendering-equivalent SVG.

## 2. The two keyed decisions

Point-token generators emit a command, then one token per point. A cubic
`C c1 c2 end` therefore involves three point decisions. Two of them are keyed.

### 2.1 Segment endpoint → vertex descriptor

When the decoder is about to emit the endpoint of a line or curve, a vertex
already exists in the current subpath at the current point `V`. Each candidate
endpoint `E` completes that vertex. Define:

* the arriving chord arm `a = P_prev − V`;
* the departing chord arm `b = E − V`.

The descriptor is

```
("v", ⌊θ/Δθ⌋, ⌊|log(|a|/|b|)|/Δr⌋, ⌊(β_before + β_after)/Δβ⌋)
```

where:

* `θ = |arg(b/a)|` ∈ [0°, 180°] is the interior angle.
* `β` is a segment's *bulge*: the maximum distance of the drawn segment from
  its chord, divided by the chord length (0 for lines).
* Defaults: `Δθ = 6°`, `Δr = 0.12`, `Δβ = 0.05`.

A vertex is skipped (no keyed group) in two cases:

* `θ ≥ 180° − 1°`: the segment continues straight, so there is no corner.
* `|log ratio| > 2.5`: the arm lengths are too different for the descriptor
  to be stable.

### 2.2 First curve handle → tangent descriptor

When the decoder is about to emit `c1` of a cubic, each candidate `c1` fixes
the departing handle `h_out = c1 − V`. The arriving handle `h_in` is already
known:

* `end − c2` if the previous segment is a curve;
* the chord direction if it is a line.

The descriptor is

```
("t", ⌊|arg(h_out/h_in)| / Δθ⌋, ρ)
```

where `ρ = ⌊|log(|h_out|/|h_in|)| / Δr⌋` when both incident segments are
curves, and `ρ = −1` otherwise. A line's handle length is its chord, which is
not a free design choice. The detector computes tangent descriptors at every
vertex that has at least one incident curve, so the definition is symmetric.

`c2` is not keyed. When `c2` is chosen the endpoint is unknown, and every
descriptor of `(P0, c1, c2)` alone changes under traversal reversal.

### 2.3 Invariances

| Transformation | Vertex `v` | Tangent `t` | Why |
|---|---|---|---|
| Translation, rotation, uniform scale | yes | yes | Angles and length ratios only |
| Mirror | yes | yes | Absolute angles |
| Reversal of traversal | yes | yes | Arms swap; handles swap and negate; `|log|` ratio |
| Restart of a closed contour | yes | yes | The detector includes the closure vertex |
| Command syntax, abs/rel, shorthand | yes | yes | Computed from drawn geometry |
| Precision rounding | mostly | mostly | Fails only when a value crosses a bin edge |
| Merging collinear lines (SVGO) | yes | yes | Straight runs are merged on both sides before descriptors |
| Path merge / split / reorder | yes | yes | Per-contour, order-free set |
| Non-uniform scale / skew | **no** | **no** | Angles change |
| Exact subdivision | **degraded** | partly | New smooth vertices; chords and handles shrink; tangent angles at original vertices survive |
| Rasterize and retrace, polyline flattening | **no** | **no** | Vertices are re-placed |

The sampler and the detector share one implementation of descriptors and
straight-run merging (`vertex_descriptor`, `tangent_descriptor`,
`_merge_straight`). This keeps the values the generator keyed and the values
the detector reads consistent.

## 3. Distribution preservation

**Lemma (grouped Gumbel-max).** Setup:

* Let a decoding step have truncated candidates `i` with probabilities `p_i`
  (after the decoder's own temperature, top-k, and top-p).
* Partition the candidates into groups `b`:
  * all candidates with the same descriptor form one keyed group;
  * each candidate that creates no descriptor (straight continuation,
    degenerate arm, non-point token) forms its own free group.
* Let `P_b = Σ_{i∈b} p_i`.
* Draw `u_b` independent Uniform(0,1) for distinct groups. For keyed groups
  `u_b = PRF_K(b)`; free groups use fresh unkeyed randomness.
* Choose `b* = argmax_b [log P_b − log(−log u_b)]`, then `i*` inside `b*` with
  probability `p_i / P_{b*}`.

Then `Pr[i* = i] = p_i` for every `i`.

*Proof.* `−log(−log u_b)` are i.i.d. standard Gumbel variables, so by the
Gumbel-max identity `Pr[b* = b] = P_b`. Conditioned on `b* = b`, the
within-group draw gives `i` with probability `p_i / P_b`. Multiplying gives
`p_i`. ∎

Under the random-PRF idealization the watermarked decoder therefore has
exactly the unmarked next-token distribution at every step, with no logit
bias and no quality loss in expectation. Grouping is what makes this hold
even though many tokens share a descriptor. Assigning a keyed uniform per
*token* and then reading only per-descriptor scores would not be
distribution-preserving at the descriptor level.

**Caveats** (the same as for text watermarks):

1. The same descriptor recurring at a later step reuses the same `u_b`. Steps
   are therefore not independent across the drawing, although each step's
   marginal is exact. Repeated motifs are pushed toward the same keyed
   choices.
2. Given the key, the prompt, and the unkeyed RNG, the output is
   deterministic.

## 4. Blind detection

The detector (`geosample.detect`):

1. Parses the SVG with the shared geometry layer. That flattens transforms,
   converts shapes, resolves `<use>`, and drops invisible contours.
2. Builds segments with chords, bulges, and handles, then merges straight
   runs.
3. Collects all `v` and `t` descriptors per contour.
4. Keeps only **distinct** descriptors, `n` of them.
5. Computes `S = Σ_d −log(1 − PRF_K(d))`.

**Null.** For an SVG chosen independently of `K`, distinct descriptors give
independent uniforms, so `S ~ Gamma(n, 1)` *exactly*. The global p-value is
`Pr[Gamma(n,1) ≥ S]`. No corpus-fitted threshold is involved.

**Deduplication** is required for this. A repeated descriptor has one `u`,
and counting it twice would inflate `S` under the null.

**Composition.** A marked icon pasted among unmarked art dilutes the global
sum. The detector therefore also computes a per-contour Gamma p-value and
reports

```
p = min(1, 2 · min(p_global, C · min_c p_c))
```

This is valid by the union bound under any dependence between contours. The
default decision threshold is `p ≤ 10⁻⁶`.

**Power.** At a step with `m` keyed groups of equal mass, the chosen `u` is
the maximum of `m` uniforms. `E[−log(1 − max)] = H_m`, the m-th harmonic
number (1.5 for m = 2, 1.83 for m = 3), against 1 under the null. Evidence
per keyed decision therefore grows with the number of descriptor groups
spanned by the truncated candidate set. That number grows with:

* per-step entropy (temperature, top-p);
* a finer quantization (smaller `Δθ`, `Δr`), at a cost in rounding robustness.

Vertices the model emits with no choice add null terms that dilute `S`.
Low-entropy decoding (for example IconShop's default top-p 0.5) is the main
limit on power, exactly as in text watermarking.

## 5. Adapter contract: making a model "generic"

The core never sees tokens. It sees a `PathTracker` (the current subpath's
segments) and, at a keyed slot, a list of candidate points with
probabilities. A model adapter is a small `PointTokenGrammar`:

| Member | Meaning |
|---|---|
| `command(token) → "M"/"L"/"C"/"Z" or None` | Command tokens |
| `point(token) → complex or None` | Grid point of a coordinate token |
| `arity[cmd]` | Number of point tokens after each command |
| `is_end(token)` | End-of-SVG token |
| `grid` | Grid size, used by `DecodeState.svg()` |

`DecodeState` replays tokens into the tracker. It exposes `handle_slot()`
(next point is a curve's `c1`) and `endpoint_slot()` (next point is a line or
curve end, with `c1, c2` already known). `watermarked_step(state, logits,
watermark, rng, top_p, top_k, temperature, token_offset)` then:

1. truncates exactly as the decoder would (`truncate` drops zero-probability
   tokens);
2. at a keyed slot calls `GeoWatermark.choose_handle` or
   `GeoWatermark.choose`, with non-point candidates passed as free groups;
3. otherwise samples normally.

**Hugging Face `generate`.** For any model that decodes through `generate`:

```python
from transformers import LogitsProcessorList
from contourmark.geosample import GeoWatermark
from contourmark.point_token_models import GeoWatermarkLogitsProcessor, OmniSVGGrammar

grammar = OmniSVGGrammar(move=..., line=..., curve=..., close=..., end=..., coordinate_start=151_944)
processor = GeoWatermarkLogitsProcessor(grammar, GeoWatermark(key), prompt_length=inputs.input_ids.shape[1],
                                        top_p=0.9, temperature=1.0)
out = model.generate(**inputs, do_sample=True, top_p=1.0, top_k=0,  # truncation happens inside the processor
                     logits_processor=LogitsProcessorList([processor]))
```

The processor returns a one-hot distribution on its choice at every step, so
`generate`'s own sampling is a no-op. Truncation must be configured on the
processor, not on `generate`, so the keyed choice sees the real truncated
distribution.

### IconShop (`IconShopGrammar`, run on this host)

The public checkpoint `m1357l/iconshop-svg-generator` (CC BY-NC-SA 4.0) is
downloaded to the git-ignored `models/iconshop`.

* Token values are taken after subtracting the BERT text vocabulary. `0` ends
  the SVG; `3`/`4`/`5` are Move/Line/Curve.
* Coordinates are `v ≥ 6`: `q = v − 6` on a 200×200 grid, `x = q mod 200`,
  `y = ⌊q/200⌋`.
* **Move carries two points (from, to)**; the subpath starts at "to".
* The decoder recomputes the full forward pass each step (no KV cache), so
  `experiments/iconshop_geosample.py` drives its own batched loop with
  `watermarked_step`.

### OmniSVG (`OmniSVGGrammar`, implemented and unit-tested, not run here)

* One token per point on a 200×200 grid, row-major.
* `M` and `L` take one point, `C` takes three, `Z` takes none.
* Coordinate tokens start at 151 944 (4B) or 152 072 (8B), per the public
  `config.yaml`. Command, close, and end token IDs must be passed from the
  released config.
* The 4B model needs about 17 GB of CUDA memory, which this host lacks, and
  the hosted Space exposes no logits. The adapter is therefore verified only
  on synthetic token streams.
* Colour tokens are ignored by the grammar and sampled normally.

### Text LLMs writing SVG code (future work)

A model that writes `d="M10.5 20…"` emits numbers digit by digit across BPE
tokens. A keyed choice needs the complete point, so an adapter must lift
decisions to the *number* level. Two options:

* enumerate the model's distribution over complete numbers for the endpoint
  (a short constrained beam over digit tokens) and apply the same grouped
  Gumbel choice;
* key only the final, value-determining digit of the endpoint's last
  coordinate.

The core is unchanged; only the grammar/state layer differs. This is not
implemented. For text LLMs the post-hoc spectral watermark (`contourmark
mark`) is the deployable option today.

## 6. Relation to other approaches

* **SynthID-Text / Aaronson / Kuditipudi (Gumbel family).** Same
  distribution-preserving principle. Ours keys *geometric descriptor groups*
  rather than token IDs, which is what makes detection survive SVG rewriting.
* **Kirchenbauer green lists.** A biased-logit alternative with more power at
  low entropy, at a quality cost. It could be layered on the same groups
  (boost groups with `u_b < γ`) but is not implemented.
* **Unigram watermark (Zhao et al.).** Like our context-free PRF on the
  descriptor alone, it trades some security for robustness to edits.
  Contextual keys (previous descriptor) would break under reversal and
  reordering, so they are not used.
* **Code watermarks** (SWEET, ACW, STONE) are fragile to semantics-preserving
  rewriting. Moving to geometry removes the dominant rewrite family for SVG.
* **This repo's candidate-manifest sampler** (`inference.py`, `sample` /
  `verify-sample`) keys a choice among whole proposed contours. It verifies
  against a private manifest and needs a generator that proposes
  alternatives. `geosample` needs no manifest, no proposals, and no model
  change beyond a sampling hook.
* **Post-hoc spectral watermark** (`spectral.py`, `contourmark mark/detect`)
  modifies geometry after generation. Use it for any SVG, any generator, or
  existing art. It moves contours by about 0.3% of the diagonal and has
  stronger per-icon capacity. Use `geosample` when you control the decoder of
  a point-token model and want zero distortion: the output is a genuine model
  sample. The two can be combined, with separate keys and both detectors.

## 7. Results

### Toy point-token generator (`tests/test_geosample.py`)

* **Setup:** Gaussian logits with σ = 3 grid units inside a 10-unit window on
  the IconShop vocabulary. Five noisy 14-gons per drawing, each edge randomly
  a line or a cubic, top-p 1.0. Seeds 0–2 give 68, 74, and 65 distinct
  descriptors.
* **Clean:** marked log10 p = −50.0, −49.5, −39.1. Unmarked drawings with the
  key give −0.5, −1.4, −0.6; marked drawings with a wrong key give −0.2, 0.0,
  −1.4.
* **Unchanged** (identical log10 p on all three seeds): reorder, reverse,
  restart, mirror, translate 5%, SVGO default, Scour default, round to 2 dp,
  round to 1 dp, merge paths, split subpaths.
* **Essentially unchanged:** rotate 30° (−50.0 / −47.7 / −39.1), scale 0.37
  (−50.0 / −49.5 / −38.9), nested group transform (−50.0 / −48.9 / −39.1).
* **Degraded but detected:** subdivision −9.6 / −8.8 / −7.9; Gaussian handle
  noise at 0.1% of the diagonal −17.6 / −15.3 / −11.4.
* **Fails:** aspect 1.2:1 stretch (−2.3 / −3.1 / −2.0); polyline flattening
  at 0.05% (−0.4 / −0.7 / −2.1); rasterize-and-retrace at 1024 px (about 0).
* **Distribution preservation:** across 6,000 keys on a 10-candidate step with
  a shared group, a straight-continuation candidate, and a non-point
  candidate, empirical frequencies match `p_i` (χ² test). Detector null over
  random keys on unmarked drawings is super-uniform.

### IconShop (real checkpoint)

TODO (parent): fill from `experiments/results/geosample/iconshop_p05` and
`iconshop_p09`, using `evaluate_geosample.py` and `summarize_geosample.py`.
Report:

* detection rates at 1e-6 and 1e-3;
* survival under attacks;
* the false-positive rate for plain samples and wrong keys;
* plain vs marked statistics;
* log10 p vs distinct descriptors, at top-p 0.5 and 0.9.

Pilot (4 prompts, top-p 0.5, after tangent keying): log10 p of −3.5 to −4.7
per icon. Every keyed descriptor was recovered exactly by the detector. Power
is limited by low per-step entropy and by unkeyed low-entropy vertices.

### Null on human icons

`experiments/results/geosample/null_corpus.json` covers 900 test-corpus
icons × 300 keys (266,700 tests). Observed rates were 0.064 at 1e-1, 0.0071
at 1e-2, 8.3e-4 at 1e-3, 5.6e-5 at 1e-4, and 0 at 1e-5. This run used the
vertex-only descriptor version; re-run `geosample_null.py` for the current
`v`+`t` version.
