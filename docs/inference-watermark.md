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
the departing **end derivative** `h_out = 3 (c1 − V)`. The arriving derivative
`h_in` is already known:

* `3 (end − c2)` for a cubic, or `2 (end − c)` for a quadratic;
* the chord direction if it is a line.

Derivatives (degree × control offset) are used, not raw handles. A quadratic
and its exactly degree-elevated cubic then give the same descriptor; raw
handles differ by the degree factor (found in review, now a regression test).

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
| Degree elevation (Q → C), straight cubic ↔ line | yes | yes | Derivative-scaled handles; shared 32-sample bulge estimate |
| Exact subdivision | **degraded** | partly | New smooth vertices; chords and derivatives rescale; tangent angles at original vertices survive. Not an intrinsic invariant |
| Rasterize and retrace, polyline flattening | **no** | **no** | Vertices are re-placed |

**Visibility.** Detection scores only contours whose visibility the parser
fully determines. The rules are:

* Contours under `clip-path`, `mask`, `filter` or a nested `<svg>` viewport
  are excluded.
* Geometry entirely outside the root viewBox is treated as not drawn.
* Documents with CSS rules, scripts or animation yield `status:
  "indeterminate"` rather than a detection.
* `detect(..., visibility="render")` asks librsvg whether each such contour
  actually contributes pixels.
* An `indeterminate` result is not a statement that the file is unmarked.

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

The lemma is a **one-step** statement: it needs the group uniforms to be
independent of the probabilities and of the grouping. Grouping is what makes
it hold even though many tokens share a descriptor.

### 3.1 The sequence law, and why naive reuse breaks it

An earlier version of this sampler used `u_b = PRF_K(b)` every time
descriptor `b` appeared. That is **not** distribution-preserving for the
sequence. After a keyed choice, the generated prefix carries information
about the uniforms that were compared. Later probabilities depend on that
prefix, so the "fresh uniforms" premise fails at later steps.

Counterexample, found in the independent review (`docs/review-2026-09-30/`):

* Two groups A and B. Step 1 has probabilities (0.5, 0.5).
* Step 2 has (0.9, 0.1) if A was chosen and (0.5, 0.5) if B was chosen.
* Ordinary sampling gives `Pr[step 2 = A] = 0.7`.
* Reusing the two scores gives 0.5, and two identical contests repeat the
  same choice 100% of the time instead of 50%.

### 3.2 The rule that restores it

Within one drawing, a keyed uniform is used only the **first** time its
descriptor is looked at, meaning the first time it appears among the
candidate groups, whether or not it wins. Every later look at that
descriptor uses a fresh unkeyed uniform (`GeoWatermark(reuse="mask")`, the
default; one instance, or `reset()`, per drawing).

**Theorem (joint law).** In the random-PRF idealization, with the rule above,
the watermarked decoder's output sequence has exactly the model's own
distribution.

*Proof.* Induct over steps.

1. Let `H_t` be everything generated before step `t`, and `Q_t` the set of
   descriptors looked at before `t`. `H_t` is a function of the uniforms
   `{u_b : b ∈ Q_t}` and of unkeyed randomness.
2. At step `t` each candidate group uses either a keyed `u_b` with
   `b ∉ Q_t`, or a fresh unkeyed uniform.
3. A random function's values at indices outside `Q_t` are independent of its
   values inside `Q_t`. This holds even though *which* indices are examined
   at step `t` depends on `H_t`: choosing where to look, based on other
   independent values, does not bias the values found there.
4. So, conditioned on `H_t`, the step's uniforms are i.i.d. Uniform(0,1) and
   independent of the step's probabilities and grouping. The lemma gives
   `Pr[token | H_t] = p(token | H_t)`.
5. Multiplying over steps gives the joint law. ∎

Regression tests (`tests/test_review_regressions.py`) check the
counterexample: 0.50 and 0.70 with masking; 1.00 and 0.50 with
`reuse="allow"`, which is kept only as an ablation.

### 3.3 What the rule costs

* **Power.** Descriptors looked at earlier carry no new signal later. A
  descriptor that lost an earlier contest and is later chosen by unkeyed
  randomness appears in the drawing with a below-average score. On the toy
  generator (5 polygons, about 100 distinct descriptors), the median clean
  log10 p goes from −49 (naive reuse) to −18 (masked), with 100% still below
  1e-6. Real IconShop numbers must be re-measured on the cluster.
* **Diversity.** Masked sampling repeats fewer motifs: median distinct
  descriptors 100 versus 72 under naive reuse.

### 3.4 The bias (green-list) alternative

`GeoWatermark(mode="bias", delta=δ, gamma=γ)` is a second sampler for
decoders with little entropy. It adds δ nats to the log mass of descriptor
groups whose keyed uniform exceeds 1 − γ, then samples. It **changes the
distribution** by design; the sampler reports the per-step KL divergence
(`last_kl`). Detect with `detect(..., statistic="green")`, an exact binomial
test on distinct descriptors. Choose the statistic before looking at a
document: taking the better of the two tests needs a correction.

Measured on the toy generator (`experiments/sampler_lab.py`, 20 drawings per
cell, detection at p ≤ 1e-6):

| Entropy per keyed step | Descriptors | Preserving | Naive reuse | Bias δ=4 |
|---|---|---|---|---|
| about 1 nat | about 25 | 0% | 0% | 0% |
| about 1 nat | about 100 | 15% | 35% | 65% (0.61 nats shift per step) |
| about 1.7 nats | about 100 | 70% | 100% | 100% (0.68) |
| about 2.9 nats | about 100 | 100% | 100% | 100% (0.68) |

IconShop at its default top-p measures about 1 nat per keyed step. At that
entropy a small icon cannot be marked detectably by any variant here. The
bias sampler buys power only on larger drawings, and at a distribution shift
that must be weighed against output quality on a real model.

**Remaining caveats:**

1. This is an idealization. HMAC is a PRF, not a random function, and the
   claim is about the sampler, not about a model's quality at a given top-p.
2. One fixed deployment key gives the same keyed preference to every drawing's
   first look at a descriptor. Across many outputs this can show up as a
   key-specific style bias, and gives an observer of many outputs information
   about the key. This has not been measured.
3. Given the key, the prompt, and the unkeyed RNG, the output is
   deterministic.
4. Batch rows must own their sampler state. `GeoWatermarkLogitsProcessor`
   keeps one sampler per row and requires sampling (not beam search).

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

### Toy point-token generator (corrected, masked sampler)

Source: `experiments/results/geosample/toy_masked_robustness.json`.

**Setup:** 30 drawings. Gaussian logits with σ = 3 grid units inside a 10-unit
window on the IconShop vocabulary. Five noisy 14-gons per drawing, each edge
randomly a line or a cubic, top-p 1.0. Median 100 distinct descriptors.

| Transformation | median log10 p | detected at 1e-6 |
|---|---|---|
| none | −18.7 | 100% |
| SVGO default / precision 2, Scour, picosvg | −18.6 to −18.7 | 100% |
| round to 2 dp / 1 dp | −18.7 | 100% |
| translate, mirror, reorder, reverse, restart, merge, split | −18.7 | 100% |
| rotate 30°, scale 0.37, nested group transform | −17.8 to −18.1 | 100% |
| delete 50% of contours | −16.5 | 100% |
| handle noise 0.1% of diagonal | −3.7 | 27% |
| handle noise 0.3% | −1.0 | 0% |
| exact midpoint subdivision | −1.9 | 0% |
| aspect stretch 1.2:1 | −1.4 | 0% |
| polyline flattening 0.05% | 0.0 | 0% |
| rasterize + retrace 1024 px | −0.2 | 0% |

* **Nulls:** unmarked drawings with the key and marked drawings with a wrong
  key never go below log10 p = −1.9.
* **Comparison with naive reuse** (`reuse="allow"`): about −49 clean and about
  −8 after subdivision. That sampler does not preserve the output
  distribution (§3.1), so the numbers above are the ones to quote.

### IconShop (real checkpoint)

**To be measured on the cluster** (`hpc/README.md`). The 4-prompt pilot
reported earlier (log10 p −3.5 to −4.7 at top-p 0.5) used the naive-reuse
sampler. It is superseded and its raw outputs were not archived. Expect lower
power with the corrected sampler. The run should report:

* detection rates at 1e-6 and 1e-3;
* survival under attacks, with attack-error counts;
* false-positive rates for plain samples and wrong keys;
* plain vs marked statistics;
* log10 p against the number of distinct descriptors, at top-p 0.5, 0.9 and
  1.0.

### Null on human icons

`experiments/results/geosample/null_corpus.json`: 900 test-corpus icons × 300
keys, current `v`+`t` descriptors with derivative-scaled handles. 890
documents have descriptors (median 11 distinct); 267,000 tests.

| Level α | Observed rate | Hits |
|---|---|---|
| 1e-1 | 0.063 | 16,910 |
| 1e-2 | 0.0068 | 1,824 |
| 1e-3 | 0.00072 | 191 |
| 1e-4 | 7.9e-5 | 21 |
| 1e-5 | 3.7e-6 | 1 |

The smallest log10 p was −5.34. This many trials cannot confirm a rate near
1e-6 empirically (with zero hits the 95% upper bound is about 1.1e-5); that
level rests on the Gamma null and the random-PRF idealization.
