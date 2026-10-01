# Code companion: implementing and testing the difficult parts

Companion to [the CGF research review](CGF_RESEARCH_REVIEW.md), for snapshot `da841d1`. Written 30 September 2026. Covers **both the spectral method and inference-time geometric sampler**. This is an implementation guide and set of demonstrations, not a claim that the research problems have been solved.

## What is runnable, what is proposed

* [reproduce_findings.py](reproduce_findings.py): executable counterexamples against the reviewed implementation, including both detectors' clipped-geometry failure, the reused-randomness counterexample, and exact curve rewriting. Captured evidence: [findings.json](findings.json).
* [companion_examples.py](companion_examples.py): CPU examples of detection statistics, conservative feature gating, geometric metrics, constraint rollback, budgeted attacks, confidence intervals and run identity. Its `self_check()` exercises these examples without downloads or model inference.
* [proposed-fixes.patch](proposed-fixes.patch): an **unapplied draft patch** prepared during this session. It demonstrates conservative unsupported-feature handling, per-sample inference RNGs, checkpoint compatibility checks, resume identities and shard exit handling. It requires integration tests and is not a complete renderer, synchronization fix or experiment runner.

The patch was moved out of the implementation when the request changed to a companion document. The production files remain at the reviewed snapshot. Do not apply the patch on top of another agent's changes without reviewing the diff.

```sh
.venv/bin/python docs/review-2026-09-30/reproduce_findings.py
.venv/bin/python docs/review-2026-09-30/companion_examples.py
git apply --check docs/review-2026-09-30/proposed-fixes.patch
```

The last command only checks applicability. This companion deliberately distinguishes a runnable toy demonstration from a validated production implementation.

## 1. Repair the inference claim before designing a replacement

The problem is not the grouped Gumbel identity for one fixed probability vector. It is reusing those same scores after the model's history becomes correlated with them.

```python
labels = [("v", 1, 1, 1), ("v", 2, 2, 2)]
wm = GeoWatermark(key)
first = wm._gumbel(labels, [0.5, 0.5]).index
probs = [0.9, 0.1] if first == 0 else [0.5, 0.5]
second = wm._gumbel(labels, probs).index
```

Over independent keys, ordinary autoregressive sampling would choose A at step 2 with probability 0.7; this construction gives 0.5. The reproduction uses the real implementation and measures 0.5048 over 10,000 keys.

**Immediate change:** document a fixed-context one-step lemma and make full-sequence quality an empirical question. The draft patch corrects the module description, but README and mathematical documentation must be updated consistently in an implementation pass.

**Research alternative, not a drop-in fix:** use fresh randomness conditioned on recoverable geometric context. A code sketch such as `PRF(key, context, occurrence, descriptor)` is easy; recovering the same context and occurrence after subdivision, deletion, reversal and composition is the hard part. Token offsets are fragile synchronization labels. A global nonce stored only in metadata disappears during minification. A nonce represented in geometry consumes capacity and needs its own robust decoder.

For an explicitly biased alternative, a descriptor-level green list can be written as:

```python
# Fixed public gamma and bias; this intentionally changes the distribution.
green = np.array([keyed_uniform(key, d) < gamma for d in descriptors])
adjusted_logits = logits + delta * green
probabilities = softmax(adjusted_logits)
```

This snippet requires handling free candidates, repeated descriptors and token groups consistently. It is a candidate comparison, not an established robust method. Record KL or log-probability cost where feasible, diversity and quality. Use the same base model/temperature/top-p when comparing samplers.

## 2. Visible-content binding: reject unsupported effects, then build a renderer contract

Both methods share the incomplete geometry parser. Until visibility is resolved, interpreting clipped paths as visible evidence is unsafe.

```python
issues = appearance_issues(svg)
if issues:
    return {"status": "unsupported", "reason": issues,
            "detected": None, "p_value": None}
result = detector(svg, key)
return {"status": "scored", **result}
```

The runnable `appearance_issues` demonstrates conservative checks for clip/mask/filter references, style blocks, nested viewports, instances, text/images, animation and external stylesheet instructions. The draft integration patch uses an exception and an explicit CLI status instead of the dictionary above. They are alternative API designs; choose one and propagate it through evaluators. Do not map unsupported to “unmarked” or fabricate a statistical p-value.

**This gate is incomplete.** Transparent color syntax, opaque overpainting, off-canvas geometry, markers, paint servers, resource dependencies and other rendering semantics need a complete supported-profile specification. Passing the gate does not certify visible contribution.

A renderer-assisted visibility experiment can begin with this pseudocode:

```python
full = render(svg, viewport, background)
without = render(remove_contour(svg, contour_id), viewport, background)
contribution = pixel_distance(full, without)
```

It measures an intervention's visual effect, not ownership. Identical overlaid contours can each have zero leave-one-out contribution, while blend/filter interactions can have nonlocal effects. Evaluate several backgrounds and sizes; retain attribution at the visible-fragment level. Validate any cached mask against real editor/export cases. Add a regression where marked paths are fully clipped and an unrelated visible icon remains: neither method should authenticate the whole artwork from the hidden paths.

## 3. Intrinsic geometry: demonstrate exact rewrites before claiming invariance

The real implementation counterexample is:

```text
M0 0 Q6 12 12 0 C15 3 24 -6 30 0
M0 0 C4 8 8 8 12 0 C15 3 24 -6 30 0
```

These paths are identical curves. The tangent ratio bins differ because a quadratic endpoint derivative is `2*(P2-P1)` and a cubic endpoint derivative is `3*(P3-P2)`.

```python
degree = {"Q": 2, "C": 3}[segment.kind]
incoming_derivative = degree * (points[-1] - points[-2])
```

This repairs **degree elevation only**. Subdivision changes local parameter speed, so derivative magnitudes still do not define an intrinsic length scale. Do not rename the result “representation invariant.” A versioned descriptor/schema is required if deployed descriptor values change.

A more intrinsic experiment uses neighborhoods sampled at normalized arc length:

```python
samples = equal_arclength(densely_flatten(curve), count=256)
before = samples[j] - samples[j - window]
after = samples[j + window] - samples[j]
turn = abs(np.angle(complex(*after) / complex(*before)))
```

`equal_arclength` is runnable for a polyline and checked against equivalent subdivision. `densely_flatten`, stable anchor selection, boundary handling, closed-curve registration and approximation bounds remain to be supplied. Adaptive flattening needs an error budget smaller than the descriptor's bin margin. The inference sampler must also know enough future curve geometry to compute this descriptor; bounded lookahead is a new algorithm requiring probability analysis.

For a descriptor coordinate `f`, bin width `h`, and bounded perturbation response `L*epsilon`, test stability using:

```python
r = f % h
margin = min(r, h - r)
stable = margin > L * epsilon
```

Estimating `L` from a few perturbations is a diagnostic, not a proof. Report actual bin flips and false rejections across optimizer precision levels.

## 4. Spectral embedding needs enforced constraints and rollback

The current solver optimizes coefficient alignment. A graphics objective should separate that objective from fidelity and design constraints:

```python
x, accepted = constrained_proposal(
    original_handles,
    objective=watermark_residual,
    distortion=rendered_and_curve_distance,
    epsilon=declared_budget,
    acceptable=topology_style_and_seed_checks,
)
```

The runnable helper shows constrained optimization followed by independent checks and rollback. Its self-check verifies rejection when the acceptance predicate fails. It does not implement the difficult callbacks: they must check self-intersections, winding, curve clearance, symmetry, stroke consistency, seed stability and actual rendered output after serialization.

Run acceptance **after** serialize→parse→render. Export rounding can break both a watermark and a fidelity constraint. Reject unstable seeds or explicitly retry with a bounded procedure and record its selection effects. Report inability to embed within budget as insufficient capacity, not a successful watermark.

For an STDM research branch, the basic feature-space idea is:

```python
# f: selected spectral features; a: secret unit projection vector
z = a @ f
target = step * (np.round(z / step - dither) + dither)
feature_target = f + (target - z) * a
```

Realizing `feature_target` as a valid curve still requires the constrained solver. Multiple projections can interfere, eligibility must survive rewriting, and the null detector must be rederived. This snippet is not evidence that STDM resists re-embedding. Evaluate removal cost against scalar QIM at matched quality and include many-output key-estimation attacks.

## 5. Statistical detection: nulls, repeated evidence and search correction

`gamma_evidence` demonstrates why distinct descriptors must be counted once. Its null assumption is an SVG chosen independently of the key with ideal independent PRF outputs. The existing detector additionally combines global and per-contour tests; do not replace it with this shorter example without accounting for that difference.

```python
p = combine_tests([p_global, corrected_p_contour])
```

Bonferroni is valid for a predeclared finite set regardless of dependence. Include all attempted windows, channels, keys or scales. Selecting the lowest p-value across training configurations and reporting it without correction is not valid evaluation.

The spectral statistic has terms `w*cos(U)`. A simple analytic fallback is:

```python
log_bound = -theta * t + sum(log(I0(theta * w_i)) for w_i in weights)
```

The runnable version uses scaled Bessel functions for numerical stability. This is mathematically conservative for any fixed nonnegative `theta`; floating-point certification is a separate problem. For a guaranteed computed bound, use directed rounding/interval arithmetic or a proved numerical error allowance. Neither the toy helper nor the draft patch provides that certificate. Validate FFT tails against high-precision reference cases before describing them as exact.

Report fixed-key across-assets nulls as well as random-key conditional nulls. Scores on documents adaptively selected with detector access fall outside the independent-document theorem. With zero hits, `3/N` is only a rough independent-trial 95% bound; it does not validate a 1e-6 tail with a few hundred examples.

## 6. Graphics metrics and fair comparisons

The executable examples provide two building blocks:

```python
appearance_metrics(original_rgba, marked_rgba)
boundary_distances(original_boundary_points, marked_boundary_points)
```

RGBA comparison composites on white and dark backgrounds and reports alpha error. Add checkerboard backgrounds, multiple rasterizers and color-managed comparisons for the study. Boundary distances are symmetric sampled distances. They do **not** certify continuous-curve Hausdorff distance; adaptively refine or use geometric bounds where that claim is needed.

Measure at small display sizes as well as large diagnostic sizes. Add symmetry, corner/junction displacement, curvature discontinuity, node counts, optimized bytes and primitive editability. These require domain-specific implementations and visual inspection; a single scalar image metric does not replace them.

For the spectral method, tune every baseline to comparable fidelity on development data:

```python
for method in methods:
    for setting in development_grid[method]:
        marked = method.embed(original, key, setting)
        quality = measure_quality(original, marked)
        robustness = evaluate_attacks(marked, key)
        save(method, setting, quality, robustness, size(marked))
# Freeze nondominated operating points before evaluating the test split.
```

Do not pick a separate strongest setting after observing each test attack. Confirm the FD baseline works on a dense-curve positive control and calibrate its detector. An 83 dB baseline versus a 25 dB proposed method is not a matched-distortion comparison.

For inference methods, ordinary and marked samples may be different valid drawings. Use prompt fidelity, preference, validity, diversity and structural statistics rather than interpreting their pixel distance as watermark distortion. The runnable paired cluster bootstrap illustrates prompt-level uncertainty. Extend it to asset/key hierarchy. A confidence interval excluding unacceptable quality loss is more informative than a nonsignificant difference test.

## 7. Attacks: budget quality, oracle access and computation separately

The runnable `budgeted_attack` demonstrates a score-oracle search with an explicit query cap:

```python
result = budgeted_attack(
    marked_handles,
    propose=bounded_handle_and_refit_proposals,
    quality_ok=within_visual_and_topology_budget,
    score=owner_detector_score,
    queries=100,
)
```

The caller must bound proposal/render cost too; candidates rejected by the quality predicate do not consume owner-detector queries. In deployment an attacker may have only a binary oracle, or none. Label each experiment correctly. In a no-oracle attack, use a public surrogate such as descriptor instability and query the owner detector only afterward for evaluation.

For spectral QIM, implement random-key re-embedding, smoothing, frequency suppression, seed-class boundary crossing and constrained refitting. For the inference method, implement descriptor-bin crossing, irregular subdivision, degree changes and simplification. Compare adaptive search to random search at the same budgets. Random-key re-embedding has already succeeded on the spectral subset; it remains a required baseline for hardening work.

Distinguish these channels:

```python
exact_rewrite -> assert_geometry_equivalent -> detect
approximate_edit -> measure_quality_cost -> detect
invalid_or_unsupported -> record_status
```

The `outcome_counts` helper keeps failure and unsupported categories in the denominator. Report attack errors separately from successful removal. An absent optimizer executable is neither robust survival nor a successful attack.

Add true clipping through curves: the current centroid-based contour deletion is a different operation. For raster round-trips, binary potrace is only a monochrome test. A multicolor vectorization experiment must assess palette, topology and editability costs.

## 8. Reproducible GPU experiments without new training

The draft patch demonstrates these changes:

```python
watermarks = [GeoWatermark(key, params, seed=s) for s in seeds]
# use watermarks[sample] in the batched loop
```

Each sample must own both ordinary-sampling and watermark RNG streams. Compare tokens for a single sample alone, in different batches, across shards, and after resume. Real GPU floating-point differences can remain; record hardware and deterministic settings, and distinguish numerical from RNG coupling.

The patch hashes model/config/code files and records package versions in a per-shard run manifest. Refuse resume when identity changes or artifacts are missing. The runnable `run_hash` demonstrates canonical configuration hashing; the draft `run_identity.py` adds filesystem checks. Production hardening still needs atomic manifests/log records, truncated-tail recovery, tokenizer revision capture, duplicate prompt validation, result checksums and a clean policy for intentional migrations.

```bash
pids=()
for shard in 0 1 2 3; do
    python generate.py --shard "$shard" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
test "$failed" -eq 0 || exit 1
```

After waiting, validate expected sample IDs and counts before merging. Avoid a wildcard that includes stale shards from another configuration. Propagate `run_id` into evaluation and summaries. The draft addresses per-child exits but not the full validated merge protocol.

Record EOS/truncation, empty drawing, parse validity, renderer validity, unsupported appearance and detector result independently. The draft records termination but does not implement every validity category. Checkpoint `missing_keys`/`unexpected_keys` should be asserted or explicitly allowlisted; do not silently ignore them.

## 9. Model adapters and transfer: conformance before scale

A minimal conformance contract is:

```python
native_svg = native_model_decoder(tokens)
adapter_svg = adapter_decoder(tokens)
assert same_commands_coordinates_colors_and_closure(native_svg, adapter_svg)
assert_render_equivalent(native_svg, adapter_svg)
```

Test move/line/cubic/close commands, colors, invalid and truncated sequences, multiple contours, coordinate-grid edges and special tokens. The callbacks above are **test requirements**, not implemented helpers. Archive small native token fixtures for each supported checkpoint.

Run an actual IconShop model and an actual OmniSVG checkpoint with logits. Synthetic grammar tests demonstrate parsing logic, not end-to-end model transfer. Keep parameters fixed for an initial transfer result, then separately report any model-specific tuning. A text-code/BPE adapter needs complete coordinate decisions; selecting a digit is not automatically selecting a geometric candidate with the assumed probability.

A generic API can make integrations easy while detection remains weak for low-entropy outputs. Record candidate-group entropy, descriptor reuse, bin margin, evidence recovered after serialization and detection versus path complexity. These diagnostics should determine whether training is justified.

## 10. If learning is needed: separate feature learning from generator training

For a learned descriptor, an objective sketch is:

```python
z = feature(svg)
z_rewrite = feature(rewrite(svg))
loss = consistency(z, z_rewrite) + separation(z, negative_shapes)
loss += capacity_penalty(z) + complexity_penalty(svg)
```

This is pseudocode. A graph/curve encoder, a noncollapsed representation, a definition of negatives, augmentation budgets and held-out optimizer families are unresolved design choices. If geometry is optimized, add differentiable raster losses and explicit topology constraints, then evaluate discrete real optimizers independently. Differentiating through a rendering surrogate does not establish robustness to an editor export.

If fine-tuning the generator, compare a frozen-generator sampler first. Then use a bounded quality objective and regularization toward the original model, and measure style/diversity changes. Rewarding raw watermark score can teach extra corners, repeated features or invisible geometry. Gate rewards on supported visible content and usable vector structure; deduplicate evidence.

Freeze feature models before randomized-key calibration. Training on the evaluation key or selecting test examples by detector score invalidates simple independence arguments. Use separate train/development/test designs, keys and attack families according to a declared protocol. No code snippet here resolves that statistical design automatically.

## 11. Capacity, universal support and hybrids

Create a routing interface with explicit outcomes:

```python
if unsupported_render_semantics(svg):
    return Unsupported(reason)
if estimated_robust_capacity(svg, fidelity_budget, attack_family) < required:
    return InsufficientCapacity(diagnostics)
return chosen_carrier.embed_or_sample(...)
```

These types and the capacity estimator are a design sketch. Empirical markability is not a proof of robust capacity. A deterministic single allowed output provides no inference choice; alternatives erased by the attack channel provide no recoverable information. The universal claim therefore needs declared fidelity, editability and attack restrictions.

For text, image and filter-only documents, outline/raster/appearance carriers require separate implementations. Preserve live text, accessibility and editability as explicit requirements instead of converting everything to paths and calling it universal.

A hybrid study needs a 2×2 design: neither method, spectral only, inference only, both. Use separate keys and test interference between channels. A spectral perturbation can change inference descriptor bins. Combine p-values with a declared rule and report additional visual/byte cost. A hybrid still includes postprocessing and must be labeled that way.

## 12. What should become regression tests next

| Hard part | Test to add in the implementation branch | Method |
|---|---|---|
| Reused randomness | Adaptive two-step counterexample; repeated fixed-key motifs | Inference |
| Hidden evidence | Empty clips/masks; CSS hiding; off-canvas and occluded paths | Both |
| Exact curve rewrites | Degree elevation, irregular subdivision, line/curve equivalence | Both, particularly inference |
| Sampling mismatch | Bulge values near bin boundaries after serialization | Inference |
| Fidelity constraints | Solver rollback; post-export constraint violation; seed flips | Spectral |
| Calibrated scores | Repeated descriptors/seeds, declared search correction, numerical tails | Both |
| Baseline fairness | Dense-curve positive controls; matched fidelity sweeps | Spectral comparison |
| Reproducibility | Batch/shard/resume equivalence; config mismatch; failed child job | Inference experiments |
| Failure accounting | Missing optimizer, unsupported SVG, truncation all retained | Both |
| Generator transfer | Native versus adapter fixtures plus real checkpoint smoke test | Inference |
| Learning | Held-out optimizer/asset/key splits and collapse diagnostics | Future learned extensions |
| Hybrid | Cross-channel interference and corrected combined decision | Both |

The companion supplies demonstrations and starting code for these tasks. It does not claim universal SVG support, solved adaptive robustness, certified floating-point tails, or demonstrated model-transfer superiority. Those remain research and engineering work documented in the review.
