# ContourMark: independent review and research directions for Computer Graphics Forum

**Reviewed snapshot:** `da841d1` on `main`, 30 September 2026.  
**Purpose:** feedback for the implementation/research agent before a larger GPU run or training decision.  
**Scope:** Git history; core watermark architectures; geometry/parser layer; baselines; attack and evaluation code; saved aggregate results; tests; manuscript; HPC scripts; targeted primary-source literature checks. This is an assessment of that snapshot, not of another agent's subsequent work.

## 1. Recommendation before spending compute

The project has a substantial working prototype and a useful evaluation foundation. It has **not yet demonstrated the original goal of universal, removal-resistant, superior SVG watermarking**, and the current manuscript is not ready for CGF submission. The most promising graphics contribution is the connection between **inference decisions, intrinsic curve features, and survival through vector editing and optimization**. That connection still needs mathematical repair and stronger experiments.

Proceed with small diagnostic inference runs after fixing the experiment harness. Defer expensive training until the carrier and evaluation failures below have been measured. The committed HPC workflow currently **runs inference with an existing IconShop checkpoint; it does not train a new model**. `generate` uses `torch.no_grad()` and the loader calls `eval()`. If a separate training plan exists, it is outside this reviewed snapshot.

Five priorities:

1. Correct the inference sampler's distribution-preservation claim. A one-step Gumbel identity does not establish the claimed autoregressive guarantee when descriptor randomness is reused.
2. Define which visible content a detection authenticates. Fully clipped, invisible marked geometry currently retains its detection score.
3. Make representation invariance measurable and precise. An exactly equivalent quadratic-to-cubic rewrite changes the new tangent descriptor.
4. Repair reproducibility and failure accounting before scaling IconShop experiments; then test a real OmniSVG checkpoint.
5. Compare methods at matched visual distortion, capacity, false-positive operating points, and resource budgets. Current baseline tables do not establish superiority.

The existing **90 tests passed in 28.37 seconds** in this review. The additional CPU probes in [reproduce_findings.py](reproduce_findings.py) reveal gaps outside those tests. No model training, generation campaign, or corpus-scale experiment was rerun. Existing implementation and experiment files were left unchanged.

## 2. What the Git history actually establishes

| Commit | Main contribution | Interpretation |
|---|---|---|
| `91588fe` | Initial generation-time prototype and research design | Early architecture and hypothesis |
| `1721535` | Real-generator evaluation and model-neutral sampling | Candidate-selection evidence, with assisted verification |
| `4dccdeb` | Matching hardening and OmniSVG output validation | Useful compatibility evidence; not real OmniSVG logit watermarking |
| `95eb539` | Blind spectral method, corpus evaluation, baselines and attacks | The main body of large-scale evidence |
| `6081a99` | Geometric inference sampler, adapters and HPC workflow | New architecture with much less real-model evidence |
| `da841d1` | Runbook/handoff pointed at main | Documentation checkpoint |

Keep these architectures separate in every figure and claim:

| Architecture | Where marking happens | Detection inputs | Evidence available |
|---|---|---|---|
| Candidate-manifest sampler, `inference.py` | Selects generator-proposed contours | SVG, key, private candidate manifest | Small Qwen experiments; assisted matching |
| Spectral QIM, `spectral.py` | Changes geometry after generation | SVG and key | 900 real assets, 40 Qwen assets, baseline and informed-removal results |
| Geometric sampler, `geosample.py` | Selects coordinate/handle tokens during inference | SVG and key | Toy generator, human-icon null tests, documented four-prompt IconShop pilot |

The paper still centers on the spectral method and describes the older assisted sampler as its generation-time variant. It does not yet present the new `geosample` architecture. The 900-asset robustness results must not be presented as real-generator validation of `geosample`.

The saved spectral summary reports 83% clean detection at `p <= 1e-6`, 78% after default SVGO, and 65% after picosvg. Conditional SVGO survival is 94.2% among the 747 clean detections. Both denominators matter. The informed random-key re-embedding experiment removes detection from **all 228 initially detectable assets** in its 270-asset subset. Those are valuable failure results, not evidence of resistance to informed removal.

The documented IconShop pilot has `log10(p)` between about −3.5 and −4.7: those examples **do not meet the advertised 1e-6 threshold**. I did not find the pilot's per-sample logs under the reviewed `experiments/results/geosample/`; the committed file there is the human-icon null report. Treat the pilot as a documented observation until its raw outputs are archived.

## 2A. Separate assessment of the two active approaches

### Approach A: spectral geometric watermarking after generation

**What is technically promising.** Arc-length resampling moves the carrier away from XML syntax and control-point enumeration. Normalized radial spectra have a sound similarity-invariance motivation. Recomputing coefficient eligibility from geometry, pooling repeated dither seeds, and supporting a per-contour test are thoughtful responses to actual SVG problems. The existing corpus and optimizer results make this the more mature experimental track.

For a graphics paper, its strongest potential contribution is a **constrained curve embedding method with a measured capacity–fidelity–robustness frontier**. QIM and Fourier magnitudes themselves are established ideas. The research burden is showing why the geometry-aware realization, coefficient selection and SVG workflow handling create a meaningful improvement over strong adapted vector baselines.

**What the evidence currently supports.** It works on many path-based icons and survives several common rewrites. It can be applied to outputs from different generators without touching their decoders. It also works on pre-existing human artwork. This makes it a useful provider-independent deployment option and an important comparator for inference watermarking.

**What needs more work.** The random-key re-embedding attack is a demonstrated removal failure, and even 0.1%-diagonal handle noise nearly eliminates detection in the saved results. Nonuniform stretching changes the carrier. The 17% clean misses and almost 1.8× optimized byte cost matter for practical icon workflows. Text/image/filter-only documents are not covered. The corpus builder explicitly skips files for which its parser finds no contours, so the 900-asset result is conditional on that selection and cannot measure universal coverage.

There are additional implementation details to investigate before presenting it as a constrained graphics optimizer:

* `_solve` minimizes watermark coefficient error. It does not enforce a hard rendered-error bound, topology preservation, symmetry or minimum clearance between curves. “Small distortion” is currently an observed statistic/test, not a universal embedding constraint.
* `_curve_distance` measures one-way distance from sampled marked points to sampled reference vertices. It is neither a certified curve Hausdorff bound nor a symmetric boundary metric. It can miss damage between samples and does not establish topology preservation.
* `_embed_contour` records `seed_stable`, but the embedding routine does not reject or retry an unstable seed. Seed flips can change every dither for a contour. Measure the fraction of failures caused by seed changes versus capacity changes versus coefficient drift.
* A coarse class based on compactness and elongation is not a unique robust content hash. Many unrelated icons can share it. Pooling solves within-document statistical dependence; it does not establish resistance to learning a lattice from many marked files under one production key. The main evaluation derives a different owner key per asset, so it does not test that deployment scenario.
* Shape-to-path conversion changes editable primitive type. Uneditable `<use>` instances and their referenced definitions need explicit coverage accounting. Higher-level SVG semantics may also change when a circle or rectangle becomes a path.

**Next experiments for A.** Add a distortion-constrained solve with rollback/abstention and report achieved coverage. Evaluate under one key shared across many assets as well as independent keys. Sweep QIM strength and spread-transform variants against random re-embedding, smoothing, refitting and class-boundary attacks, all at matched appearance error. Add scalar-QIM, spread-spectrum and resampling ablations. Finish with real editor round-trips and a perception study.

**Training decision for A.** A new generator is unnecessary. The algorithm already operates on existing SVG geometry. A learned perceptual metric or learned decoder could be a later extension, but first test whether deterministic constraints and a stronger carrier solve the measured failures. If only the geometric embedder is changed, preserve it as a separately evaluated post-generation method.

### Approach B: geometric watermarking during inference

**What is technically promising.** Grouping token alternatives by a geometric feature is a sensible way to seek evidence that outlives token rewriting. The blind detector avoids the private candidate manifest required by the older approach. Grammar adapters separate the watermark from a model's vocabulary. This is the track most directly aligned with the original request to influence inference.

For a graphics paper, the strongest potential contribution is **causal selection of geometrically stable alternatives**. The geometry must be inferable both when sampling and after editing. That synchronization challenge is more distinctive than applying a familiar sampler to a new vocabulary.

**What the evidence currently supports.** There is a working sampler/detector, extensive toy tests, a human-icon null evaluation and a documented real IconShop pilot. The toy examples provide useful evidence of optimizer survival. They have far more usable evidence than many real icons, however. The real pilot's scores miss the default 1e-6 operating point, and a real OmniSVG sampling integration has not yet been evaluated.

**What needs more work.** Reused key scores invalidate the full autoregressive distribution argument. Handle and chord features retain representation dependence, and quantized descriptors are sensitive to boundaries. Distinct-descriptor deduplication is necessary for the null but caps the evidence available from repeated motifs. The fixed-key sampler can also encourage repeated preferences across many outputs; measure that as a quality and security issue. There is no comparable adaptive-removal campaign for this architecture yet.

**Next experiments for B.** First repair the probability claim and run the descriptor conformance tests. Log candidate-group mass, descriptor collisions, prefix-dependent reuse and recovered evidence. Compare ordinary sampling, the current method, a controlled-bias alternative and token-level watermarking at the same decoding settings. Evaluate several keys and repeated prompts, then a real OmniSVG model. Include fine curve refitting and quantization-boundary attacks, not just large generic coordinate noise.

**Training decision for B.** The current mechanism changes inference only. Training a whole generator should follow evidence that usable geometric entropy is the bottleneck and that a smaller sampler/feature change cannot resolve it. Training must not merely teach the model to add extra corners, paths or high-frequency detail to make the detector happy. If learned features are used, test their synchronization and key independence before a generator fine-tune.

### Choosing a paper direction, or testing a hybrid

| Question | Spectral approach A | Inference approach B |
|---|---|---|
| Matches the original inference-only preference? | No; modifies a completed SVG | Yes |
| Requires model access? | No | Yes, structured candidate/logit access |
| Broad evidence today? | Substantially more | Early real-model evidence |
| Primary visible cost | Geometric distortion and extra representation complexity | Potential distribution/style/diversity change; requires measurement |
| Primary known robustness gap | Cheap re-embedding/noise and affine changes | Descriptor synchronization and low evidence; adaptive robustness not established |
| Strongest graphics research direction | Constrained intrinsic curve marking | Intrinsic geometric choices during generation |
| Need a new model now? | No | No, diagnose capacity and synchronization first |

My recommendation is to maintain both tracks, with **A as the mature comparator and possible standalone graphics contribution, and B as the main inference-time research direction**. A narrower A-only CGF paper could be credible if it contributes a substantially better constrained embedder/workflow and honestly reports security limits. It would still not satisfy the user's inference-only objective. B offers the closer conceptual fit, but its claims and real-model evidence need more development.

A hybrid deserves a controlled ablation, not an assumption of improvement: test neither, A only, B only, and B followed by A. Use separate domain-separated keys. Measure whether A's geometric changes erase B's quantized descriptors, whether attacks removing one channel also remove the other, and the combined quality/byte cost. Correct the combined detector's multiple-testing rule. The hybrid should be described as inference plus postprocessing; it does not retain an inference-only claim.

## 3. Findings that should be addressed first

### P0 — The autoregressive distribution claim is too strong

**Locations:** `geosample.py`, `GeoWatermark._gumbel` (line 294); `docs/inference-watermark.md`, section 3; module docstring.

The grouped Gumbel-max lemma is correct when the probability vector and grouping are fixed independently of fresh group uniforms. The implementation instead uses `PRF(key, descriptor)` repeatedly. After a keyed choice, the generated history carries information about those same random values. Future model probabilities and geometric candidate groups depend on that history. Freshness/independence is therefore unavailable at later steps.

A minimal counterexample uses two descriptor groups A and B:

* At step 1, the probabilities are `(0.5, 0.5)`.
* At step 2, they are `(0.9, 0.1)` if A was selected, and `(0.5, 0.5)` if B was selected.
* Ordinary sampling gives `P(step 2 = A) = 0.5*0.9 + 0.5*0.5 = 0.7`.
* Reusing the two keyed Gumbels gives 0.5. If A won step 1, increasing its mass preserves its win; if B won, the unchanged equal-mass contest repeats B.

The implementation probe over 10,000 keys produced **0.5048**, versus the ordinary-sampling expectation **0.7**. Even two identical equal-mass contests repeat the same group 100% of the time, versus 50% for independent ordinary draws. Thus the joint law changes; in the adaptive example even a later marginal changes. The current caveat that “each step's marginal is exact” is insufficient.

**Required response:** retain the valid fixed-context one-step lemma; retract the full-sequence/no-quality-loss guarantee until proved. Add repeated-descriptor and adaptive-history tests. Measure motif repetition and diversity under one fixed deployment key and across independent keys.

Possible research paths are occurrence/context-separated randomness with recoverable synchronization, or an explicitly biased sampler with quantified quality cost. Tracking previously queried descriptor labels can help diagnose reuse, but simply skipping repeats is not automatically a proof when the candidate construction is itself adaptive. A timestamp, token index, or nonce that the detector cannot recover after rewriting is not a complete solution.

Google's SynthID documentation explicitly handles repeated contexts and describes an inference hook without generator retraining. That is a useful design comparison, not a proof that this different descriptor scheme inherits SynthID's guarantees. [SynthID documentation](https://ai.google.dev/responsible/docs/safeguards/synthid)

### P0 — Visibility filtering does not match SVG rendering

**Locations:** `geometry.py`, `_NON_RENDERED`, `_visible`, `load_document` (line 407); `geosample.py`, `document_descriptors`; spectral detection shares this geometry layer.

The parser handles some inline paint and visibility attributes, but it is not a full SVG renderer. It does not compute clipping, masking, CSS cascade, filter output, viewport visibility, or occlusion. It skips text and image elements. Nested SVG/symbol viewport handling is also incomplete.

**Reproduced in both approaches:** wrap marked paths in a group referencing an empty `clipPath`. librsvg renders a completely white image. On the geometric sampler's toy fixture, `geosample` returns exactly the same **log10 p = −49.5088** as for the visible original. On the spectral test fixture, the spectral detector likewise retains **log10 p = −19.4853**. An unrelated visible drawing could be added while retaining hidden evidence.

This is a **content-binding/attribution failure**, not a contradiction of the independent-document null theorem: the attacker reused marked content, so the SVG depends on the key through that content. The paper's statement that invisible decoys are excluded is nevertheless false for the implemented SVG feature set.

**Required response:** specify a supported SVG profile; return an explicit unsupported/indeterminate status when visibility cannot be resolved; test empty clips, black masks, stylesheet hiding, transparent paint, occluded layers, off-canvas geometry, and nested viewports. For full SVG support, investigate renderer-assisted contribution masks. Even perfect visibility filtering does not stop a tiny visible marked fragment from triggering a document-level “contains evidence” test; report localization and marked visible area, and avoid attributing the whole image to that fragment.

### P1 — Handle descriptors are not intrinsic to the curve

**Locations:** `geosample.py`, `tangent_descriptor` (line 147) and extraction of raw handles (line 342).

These two paths draw exactly the same geometry, using exact degree elevation:

```text
M0 0 Q6 12 12 0 C15 3 24 -6 30 0
M0 0 C4 8 8 8 12 0 C15 3 24 -6 30 0
```

The extracted descriptor changes from `('t', 18, 9)` to `('t', 18, 6)`. Quadratic and cubic handles encode derivatives with different degree factors. Raw handle-length ratios therefore depend on representation. Multiplying handles by the polynomial degree fixes this particular mismatch, but arbitrary subdivision still rescales parameter derivatives. It does not establish intrinsic invariance.

The sampler also estimates cubic bulge from 32 parameter samples while the detector uses adaptive dense sampling. Values near quantization boundaries can disagree without any external attack. Add boundary-focused conformance tests instead of relying on four pilot outputs.

**Required response:** distinguish syntax robustness, exact representation changes, and approximate geometric edits. Test quadratic/cubic equivalence, straight cubic/line equivalence, arc conversion, nonuniform de Casteljau subdivision, degree reduction, and curve refitting. Consider intrinsic arc-length neighborhoods around stable geometric landmarks. This is a central graphics research problem, including how to compute a stable descriptor causally during generation.

In this review, four successive midpoint subdivisions of the high-capacity toy example still detected (`log10 p` approximately −8.82, −7.60, −8.31, −9.06). That is useful positive evidence for that example; it neither establishes exact invariance nor establishes survival for low-capacity real icons.

### P1 — Baselines are not yet a fair superiority experiment

**Locations:** `vector_baselines.py`, `fd_vertex_embed` / `fd_vertex_detect`; `evaluate_blind.py`; `test_main.summary.md`.

The vertex-FD baseline has median PSNR **83.1 dB**, compared with **25.3 dB** for the spectral method. Coordinate LSB is **54.4 dB**. A common numerical strength parameter would not imply equal perceptual distortion, and these reported results plainly do not demonstrate matched visual budgets.

The FD detector estimates a mean and variance from 200 randomized keys and extrapolates a Gaussian tail to `1e-6`. That tail requires justification beyond 200 calibration samples. Its statistic detrends log magnitudes and correlates signs; a faithful reproduction of the cited optimal detector has not been established by the present code or three round-trip tests. A 0% clean baseline demands a positive control on data/parameters where the reference method should work.

**Required response:** sweep each method on the development split, compare robustness–distortion–size frontiers, validate false-positive calibration, and include long/dense-curve positive controls. Call adaptations adaptations, with a mapping from reference equations to code. Separate tiny-icon capacity limits from algorithm failure. Add an arc-length FD/spread-spectrum baseline and an arc-length QIM baseline to isolate the contributions of resampling, QIM, capacity selection, and content seeding.

The `<metadata>` baseline is a placeholder for metadata survival, not a complete C2PA implementation or evaluation of cryptographic credentials. Label it accordingly.

### P1 — The GPU harness can produce misleading or irreproducible runs

**Locations:** `experiments/iconshop_geosample.py`, lines 125–145; `hpc/generate_iconshop.sbatch`, lines 24–30; evaluation resume logic.

| Issue found by inspection | Why it matters | Before the full run |
|---|---|---|
| One `GeoWatermark` RNG is shared across a batch, seeded from its first sample | Free-group/within-group draws depend on batching, shard layout and completion of other samples | Use an independent RNG stream per sample; verify batch/shard/resume equivalence |
| Resume identifies only prompt, seed and marked flag | Changed top-p, parameters, model or code can silently reuse old outputs | Immutable run configuration and hash; reject mismatched resume |
| Bare shell `wait` after background shards | Does not reliably surface every failed shard as a failed generation job | Save PIDs, wait on each, propagate failures; validate expected unique row count |
| Model loads with `strict=False` and ignores returned missing/unexpected keys | Partial checkpoint loading can go unnoticed | Assert expected compatibility or explicitly list approved exceptions |
| No locked Python environment or checkpoint revision in run records | Future reruns can execute different models/libraries | Record exact revisions, package versions and configuration |
| Summaries exclude attacks with `attack_error` | Missing tools can make robustness look better than coverage warrants | Report attempted/completed/error counts and make required attack failures fail the run |
| Truncated generations lack an explicit completion/validity outcome in the log | A partial path may be serialized and counted as a finished icon | Record EOS, truncation, parse/render validity and include failures in end-to-end success |

The default generation campaign is 30 prompts × 16 seeds × 2 marking conditions × 3 top-p values = **2,880 samples**, before adding keys or another model. Make a tiny end-to-end job exercise generation, attack tools, summarization, resume and counts first. No training is required for this gate.

## 4. A defensible mathematical scope

### Null validity, authenticity and removal resistance are separate

The geometric detector has a useful idealized null: for a document fixed independently of the secret key, distinct descriptors produce independent uniform PRF outputs; summing `-log(1-u)` gives a Gamma law. Deduplication and Bonferroni combination are appropriate parts of that argument. The spectral detector also correctly recognizes that repeated seeds create dependence and pools their contributions.

State the qualifications precisely: ideal random function versus HMAC pseudorandomness; finite precision; fixed document independent of key; a predeclared descriptor/parameter set. A p-value is not the probability that a provider authored the artwork. The theorem does not establish resistance to detector queries, transplantation from marked documents, key estimation from many outputs, or selective reporting across many keys.

The spectral convolution is mathematically conservative when upward discretization is exact. The implementation rounds weights to 12 decimals, uses FFTs and discards nonpositive numerical masses. These operations deserve an error analysis before calling the machine-computed value an “exact upper bound.” Validate adversarial weights and near-boundary statistics against high-precision calculations; use a rigorously conservative fallback when necessary. This review did not reproduce a numerical anti-conservatism failure.

Searching multiple keys, windows, scales or descriptor variants requires a declared combined test. If both spectral and geometric detectors are combined, account for that selection too. A 300-key null study is a useful diagnostic, but 267,000 trials cannot tightly validate a `1e-6` tail empirically. With zero hits, the rough 95% upper bound is `3/N`, about `1.1e-5` at that sample count; with a hit it is larger. Shared keys/assets also require attention to dependence in reported uncertainty.

### The universal requirement needs an explicit capacity statement

“Works with every generator” and “can mark every SVG” are different claims. Text-only, image-only, filter-generated, empty, single-line, and highly constrained geometric files are not covered by the current carriers. Parsing a file successfully does not mean it has usable evidence.

There is also a basic limit: if an asset has only one allowed visible output under the stated fidelity/editability constraints, an inference sampler has no choice in which to encode a key. If all allowed alternatives collapse to the same output under the attack family, no detector can recover a mark from that channel. This simple zero-capacity case rules out the conjunction of universal coverage, unchanged constrained output and unrestricted removal resistance.

A useful formal target is a robust packing number. For an original asset `x`, fidelity budget `epsilon`, and declared transformation family `A`, let `M(x,epsilon,A)` count allowed alternatives whose attacked observations remain distinguishable. Then `log2 M` is an intuitive ceiling on robust payload; `M=1` gives zero capacity. Define the rendering/editability metric and the admissible attacker before trying to make this a theorem.

Maintain the universal requirement as an explicit unmet research item. To approach broader coverage, add a carrier-routing layer with supported/unsupported/insufficient-capacity outcomes. Text-outline marking, raster-image marking, and appearance-field marking each require their own constraints and tests. Outlining live text sacrifices searchability, accessibility and font editability; rasterizing everything sacrifices vector structure. Those costs cannot be silently counted as universal success.

## 5. How I would position the work for CGF

Lead with a graphics workflow: an illustrator generates an icon, edits its curves, exports through an editor, minifies it for a website, and composites it into a larger asset. Ask which provenance evidence remains attached to the **visible, editable design** throughout that workflow.

My recommended central question is:

> Can inference-time watermarking operate on intrinsic geometric choices that remain detectable after ordinary SVG optimization, while preserving visual quality, design diversity and editability?

This is a research question, not an established claim. Make the spectral method a serious comparator or companion, with separately reported results. A coherent paper could contribute a stable geometric carrier, an inference adapter, a calibrated detector, and a graphics-workflow benchmark. Combining many incomplete methods into one universal claim will weaken it.

Three application demonstrations would help:

* **Icon publishing:** generator → local curve editing → SVG export → SVGO → browser rendering at 16–128 px. Show recognizable style and editable handles after each step.
* **Asset reuse:** marked icon inserted into a human-authored layout. Localize the evidence and quantify its visible area without claiming the entire composition is generated.
* **Mixed vector artwork:** colored fills, open strokes, gradients and clipping, with a clearly scoped text/image extension or explicit unsupported outcomes.

The current two-column article is a working draft, not the CGF template. Verify the actual submission route and use its current template; regular journal and conference-special-issue processes should not be conflated. CGF's editorial guidance lists incorrect formatting/category among initial submission problems. [CGF review guidance](https://www.eg.org/wp/eurographics-publications/cgf/computer-graphics-forum-faq/review-and-decision/)

## 6. Research directions worth testing

### A. Intrinsic local descriptors with explicit stability margins — highest priority

Explore tangent *directions*, turning-angle integrals, curvature integrated over normalized arc-length windows, and stable corner/junction neighborhoods. Estimate each descriptor's distance to a quantization boundary and its response to realistic coordinate rounding. Define a margin `m` and a descriptor sensitivity bound `L`; a perturbation budget `epsilon` preserves a bin when `L*epsilon < m`, under the bound's assumptions.

The challenge is synchronization: stable anchors may disappear under simplification, and a complete neighborhood may be unavailable when choosing the next token. Test bounded lookahead or complete-segment proposals, while explicitly deriving what distribution they sample. Extra candidate search cannot inherit the original one-step preservation proof automatically.

Use optimizer outputs as paired equivalence examples. A descriptor that is invariant to syntax but unstable to exact curve subdivision needs a different claim than an intrinsic contour descriptor.

### B. Stability-aware sampling and capacity prediction

Measure group probability mass and entropy, not only the number of keyed steps. Two dozen candidate tokens may fall into one descriptor group and provide no useful choice. Record descriptor reuse, distinct recovered evidence, bin margins, competing free-group mass, and model probability of the selected token.

Compare the current Gumbel method with a tunable descriptor green-list bias, an unwatermarked sampler and a token-watermark control. Plot clean and attacked detection against prompt fidelity, diversity and inference cost. A model can expose logits yet have too little entropy to watermark a short icon at the chosen threshold.

### C. Optimizer-aware learned descriptors — only after a fixed-carrier baseline

If learning is justified, train a small feature extractor before considering a whole new SVG generator. Use paired rewrites and bounded edits for consistency; include different shapes as negatives to avoid a collapsed representation. Preserve discriminative capacity and evaluate on unseen optimizer configurations, renderers, sources and generator families.

A possible objective combines rewrite consistency, discrimination/capacity, rendered distortion, topology and complexity penalties. DiffVG provides a route to differentiable rendering losses, but real optimizer operations are discrete and must also be evaluated directly. [DiffVG project](https://people.csail.mit.edu/tzumao/diffvg/)

Freeze learned features before key-randomized calibration. A feature extractor trained using the evaluation key or selected by owner-detector scores changes the independence assumptions. If learned features are quantized and keyed, prove the corresponding null for that construction rather than reusing the Gamma assertion by analogy.

### D. Spectral hardening through secret projections

Spread-transform dither modulation is a plausible ablation against simple public-coordinate re-quantization. It is not a guaranteed fix. Measure removal distortion with attacker-known algorithms, multiple marked observations, and detector-query budgets. Low-dimensional bands and repeated shape classes may still leak useful structure.

Compare against unkeyed smoothing, curve refitting and random perturbation at the same rendered error. A security improvement exists only if the best allowed attack pays more visible or structural damage, rather than merely needing a different implementation.

### E. Affine normalization and a second appearance channel

Area-moment whitening or local affine invariants could address aspect stretch, but near-symmetric shapes create ambiguous frames and cropping changes normalization. Evaluate instability explicitly. Do not claim affine invariance from a canonicalizer tested only on generic shapes.

A raster-decoded channel could help screenshots, outlined strokes and retracing. Treat it as an additional mode with new capacity, quality and calibration results. DeepMorph is particularly relevant because it embeds information through vector primitive perturbations and decodes from images; FontCode is relevant to glyph-specific carriers. Neither should be dismissed solely because it trains a decoder. [DeepMorph](https://arxiv.org/abs/2011.09783), [FontCode](https://arxiv.org/abs/1707.09418)

## 7. Graphics metrics that are currently missing

The current fidelity code renders on white and converts to grayscale. This is useful for shape shifts but misses color errors and background-dependent appearance. It does not measure editability or topology.

| Dimension | Recommended measurement | Why it matters |
|---|---|---|
| Visible fidelity | RGB/alpha-aware error on white, dark and checkerboard backgrounds; multiple renderers | Gradients, transparency, color and renderer effects |
| Icon-size appearance | 16, 24, 32, 64, 128 px, plus 256/1024 diagnostics | Tiny icons expose alignment and stroke-weight changes |
| Boundary accuracy | Symmetric boundary distance, 95th percentile and maximum in pixels; silhouette IoU | More interpretable than image PSNR alone |
| Curve quality | Tangent/curvature discontinuities, self-intersections, winding/hole changes, endpoint/junction drift | Prevents subtle geometry damage |
| Design constraints | Symmetry, spacing, stroke-width consistency, corner radius, alignment | Captures what designers notice |
| Editability | Nodes/segments, editable primitives retained, text retained, layers/groups, export round-trip | A vector result should remain usable as vector art |
| Size and latency | Raw/SVGO/gzip bytes; median/p95 generation overhead and detector time; memory | Rendering and serving costs |
| Generative quality | Prompt adherence, invalid/empty/truncated rate, human preference, novelty/uniqueness and motif repetition | Detection strength alone can reward bad drawings |
| Evidence capacity | Distinct descriptors, group entropy, recovered fraction, detection versus complexity | Explains low-capacity failures |
| Local provenance | Visible marked area and localization precision/recall after composition | Separates fragment evidence from whole-document attribution |

For post-hoc marking, compare each source with its marked version. For inference-time methods, plain and marked generations can be different valid drawings; pixel distance between that pair is not an embedding-distortion metric. Use matched prompt/seed blocks for quality comparisons and distributional statistics with uncertainty. A nonsignificant Mann–Whitney test on path counts does not prove distribution equality or visual equivalence. Define equivalence margins and study power in advance. FID on a few hundred icons is noisy and should be supplementary; CLIP alone can miss damaged geometry.

A blinded perception study should separate “which version differs?” from “which design is better?”. Show normal-size and enlarged views, randomize presentation, include easy catch trials, and report confidence intervals. Avoid claiming imperceptibility from a few side-by-side examples.

## 8. Attack matrix for the next evaluation

Evaluate each attack on marked and unmarked inputs. Record validity and rendered/structural damage. An invalid SVG or destroyed illustration is not a successful quality-preserving removal attack.

| Family | Additions | Essential reporting |
|---|---|---|
| Optimizer | Pinned SVGO versions/configurations, individual geometry plugins, multipass, Scour, picosvg, chained pipelines | Version, exact configuration, size reduction and visual error |
| Exact representation | Nonuniform subdivision; Q↔C; straight curves↔lines; relative/absolute and shorthand; transform flattening; shape↔path | Render equivalence and descriptor equality before detection |
| Editor export | Inkscape/Illustrator or another available editor round-trip; PDF→SVG; stroke outlining; boolean union; simplify/refit | Application/version and editability change |
| Precision | ViewBoxes at 16/24/100/1000 and normalized units; rounding to a matched screen-space error | Absolute decimals alone are not comparable across assets |
| Layout | CSS/symbol/use; nested viewBox; clipping/masks; crop through curves; mixed marked/unmarked composition | Visible retained area and evidence localization |
| Geometric | Skew/aspect sweeps, local handle edits, smoothing, curve fitting, repeated irregular subdivision | Removal versus pixel/curve distortion |
| Raster round-trip | Multicolor vectorization; multiple renderers/resolutions/antialiasing; thresholds; screenshot/compression | Preserve color and topology, not just binary potrace output |
| Adaptive removal | Bin-boundary crossing, seed-class crossing, public-parameter sweep, surrogate descriptor destruction | Attacker knowledge, query count, distortion budget |
| Provenance attacks | Mark transplantation, tiny marked fragment, hidden/occluded evidence, wrong-provider key, multiple-key searches | False attribution and visible content coverage |
| Repeated-output attacks | Same prompt with many seeds; marked/unmarked pairs if available; collusion across keys | Data budget, key reuse, descriptor-frequency leakage |

Current `crop_half` removes contours based on their centroids; it does not geometrically clip curves against a half-plane. Rename the existing result and add real clipping. Binary potrace is an appropriate monochrome channel but cannot establish robustness for colored illustrations, gradients or filters.

For adaptive testing, distinguish no detector access, a binary detector, a numerical score oracle, and white-box access. A practical target is an attack curve

`minimum visible/structural distortion needed to make detection fail`,

as a function of compute/query budget. Compare adaptive search against random search with the same budget. Keyless attacks can target quantization stability without recovering the key. Give the inference method its own adaptive evaluation; the spectral re-embedding result does not cover it.

SVGO's official default preset lists the actual transformations to test. Pin the repository's installed version/configuration rather than describing all optimizers as one generic minification operation. [SVGO preset](https://svgo.dev/docs/preset-default/)

## 9. Generator transfer and a staged experiment plan

### Stage 0 — correctness and data integrity

Fix/restrict the claims identified in section 3. Add independent parser/renderer conformance fixtures and the exact-rewrite cases. Make batch, shard and resume outputs reproducible. Archive model/checkpoint revisions, tokenizer/grammar configuration, keys or key identifiers, code commit, package lock and attack configuration in each run. Public deterministic benchmark keys are acceptable for reproduction; they do not model secrecy against a key-aware attacker.

### Stage 1 — pilot without training

Use a small disjoint development prompt set across several independent keys. Compare plain sampling, geometric Gumbel sampling and a biased descriptor alternative at the model's normal top-p and at exploratory higher top-p. Measure group entropy, reuse, valid outputs, clean detectability, optimizer survival and quality. Keep the default-decoder result prominent: increasing top-p may improve evidence by changing the generation regime.

Decide from these measurements whether the bottleneck is capacity, feature instability, key reuse, or generator quality. Each needs a different intervention; training a larger model does not automatically solve any of them.

### Stage 2 — locked test on at least two model families

Run IconShop and an actual OmniSVG checkpoint with logits access. Validate native token decoding against the adapter SVG: colors, close commands, multiple paths, special tokens, malformed/truncated sequences and rendered equivalence. A synthetic grammar test or watermarking an already generated OmniSVG output is not a successful OmniSVG inference integration.

The native IconShop project provides the reference synthesis implementation, while OmniSVG's project provides a distinct multimodal SVG generation setting. Those support a meaningful transfer experiment. [IconShop](https://github.com/kingnobro/IconShop), [OmniSVG](https://omnisvg.github.io/)

A shared adapter interface establishes engineering extensibility, not universal empirical performance. Model size, tokenizer, coordinate grid, entropy and command vocabulary can change capacity. Report transfer without tuning on the second model, then a separately labeled adapted result if needed. For text-code generators, complete-number/BPE handling remains unimplemented in the new architecture. Also resolve whether the requested “iconsvg” refers to IconShop or a different model before labeling coverage.

### Stage 3 — diversity, uncertainty and matched baselines

Extend beyond library icons to illustrations, technical diagrams, charts, typography, gradients, clipping and mixed raster/vector documents. Keep low-capacity assets in end-to-end denominators and label unsupported files. Split by source/template/semantic family as appropriate; filename hashing alone does not prevent near-duplicate design leakage. Keep generated test prompts separate from tuning prompts.

For every configuration report: attempted generation, valid SVGs, markable outputs, clean detections, attacked detections, unsupported cases and errors. Present unconditional and conditional survival, ROC or TPR at fixed FPR, quality/size trade-offs and confidence intervals. Use prompt/asset/key-aware resampling rather than treating every attack and every seed as an independent artwork. Freeze attack budgets and configurations before opening the final test set.

### Stage 4 — training only if an ablation identifies its purpose

If a stable intrinsic descriptor plus fixed sampler is adequate, generator retraining is unnecessary. If robustness is the limitation, learn a rewrite-stable feature/decoder and compare against the fixed feature. If capacity is the limitation, study a constrained generator objective with quality regularization and explicit minimum evidence; be honest that this can change the output distribution. Hold out optimizer families and real edit pipelines from training. Stop a training direction if it gains detectability only by adding complexity, visible artifacts or style collapse.

The reviewed workspace was approximately **2.0 GB**. Keep the original 100 GB ceiling in run planning: include model and HF caches, temporary renders, optimizer intermediates and checkpoints; estimate storage before downloads and retain compact manifests/results instead of every intermediate image.

## 10. Literature, novelty and manuscript repairs

The literature review has useful coverage, but “combines existing ingredients” is not itself a demonstrated novel contribution. Establish a claim-by-claim comparison: carrier, insertion point, detector side information, supported primitive classes, invariance family, learned components, capacity, calibrated FPR, and editability. Citation searches cannot prove the absence of all prior work; report the searched scope and unresolved closest comparisons.

Priorities for the next literature pass:

* **Vector Fourier descriptors and GIS QIM:** obtain and check the original embedding/detector equations before calling the local baseline faithful. The Doncel/Nikolaidis/Pitas full PDF link was inaccessible in this review; its exact reproduction remains unverified.
* **DeepMorph and FontCode:** treat as direct graphics comparisons for geometric information embedding and raster recovery, with their different assumptions made explicit. A learned decoder is a trade-off, not disqualification.
* **SynthID-Text and other sampling watermarks:** compare randomness reuse, conditioning, entropy, detector independence and finite-sample guarantees. The novelty candidate is the geometric synchronization problem, not Gumbel-max itself. [SynthID-Text paper](https://www.nature.com/articles/s41586-024-08025-4)
* **Unigram/semantic/code watermarking and feature-space watermarks:** investigate whether robust feature groups, context-free keying, or transformation equivalence classes already address analogous synchronization problems. Do not claim priority until this search and equation-level comparison are complete.
* **Differentiable vector graphics and curve simplification:** connect fidelity constraints and optimization attacks to graphics practice, rather than relying exclusively on generic watermarking citations.

Use Anthropic's own explanation for the original motivation. It identifies its text mechanism as a version of SynthID-Text and describes separate metadata credentials for supported file outputs including SVG. It is not evidence that Anthropic has implemented this project's geometric inference carrier. [Anthropic explanation](https://www.anthropic.com/news/claude-text-watermark)

Specific manuscript corrections:

1. Replace “million-test null calibration” in the contributions with the actual experiment count; the saved spectral null reports 268,200 tests. Do not confuse separate Monte Carlo tail checks with corpus null trials.
2. Reconcile `docs/inference-watermark.md`'s older vertex-only null numbers with the current saved v+t report (267,000 tests, 890 documents with descriptors).
3. Remove “any SVG/every generator” where usable geometric capacity or decoder access is required.
4. Remove the blanket “no quality loss” inference claim and the unqualified invisible-decoy protection claim pending repairs.
5. Correct “matched distortion” for baselines. Show measured operating curves.
6. Reconcile narrative claims with tables: for example, coordinate LSB is not zero under every similarity transform in the saved results.
7. Replace claims of exact representation invariance with the implemented scope and measured numerical error; disclose the degree-elevation counterexample for the new method.
8. State that adaptive removal currently succeeds against the spectral method and remains insufficiently evaluated for the inference method.
9. Add the new architecture to the manuscript only with its own results and limitations. Avoid silently swapping methods behind one system name.
10. Regenerate all tables from versioned data, attach a failure gallery, publish supported-SVG conformance tests and provide a small reproducible graphics workflow.

## 11. Suggested handoff to the research agent

> The codebase now supports a credible research investigation, but the large spectral evaluation does not validate the new inference sampler. Before scaling compute, fix batch/resume/error accounting, correct the reused-randomness distribution claim, and address visible-content binding and exact-rewrite descriptor instability. Then run a small multi-key IconShop pilot to diagnose entropy and synchronization, followed by real OmniSVG transfer. For CGF, prioritize intrinsic geometry, editable visual quality, optimizer/editor workflows, fair distortion-matched baselines, and budgeted adaptive attacks. Train a learned component only when those ablations identify the missing capability. Keep universal SVG coverage and superiority explicitly unproven until the evidence actually supports them.

## Reproduction and evidence files

Run from the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q
.venv/bin/python docs/review-2026-09-30/reproduce_findings.py
```

The second command uses CPU and the already installed renderer; it does not download models. Its captured output is [findings.json](findings.json). Existing corpus results are in `experiments/results/blind/`; the saved new-method null is `experiments/results/geosample/null_corpus.json`. Statements labeled reproduced above come from this review's probes; large-corpus rates come from saved reports and were not independently rerun.
