# Generation-time geometric watermarking of SVG: research design

Status: prototype and research plan, 30 September 2026. This is not a claim of novelty or a completed USENIX evaluation.

> **Update (end of 30 September 2026).** The project's primary method is now the
> blind, post-hoc spectral watermark (`src/contourmark/spectral.py`). Its design,
> results, and limits are in `HANDOFF.md`, `docs/literature.md`, and
> `paper/contourmark.tex`. This document still describes the generation-time
> (candidate-sampling) variant, which remains valid as the zero-distortion option.

## Question and threat model

Can a generator create SVGs whose *chosen visible geometry* carries a keyed provenance signal, while preserving its normal output distribution and surviving syntax-level optimization? The generator emits a sequence of paths. At step (t), it supplies a set of geometrically distinct but visually comparable candidates (C_t=\{c_{t,1},\ldots,c_{t,m_t}\}), with categorical probabilities (q_{t,i}>0) that sum to one. The candidate set and weights must be fixed independently of the secret key. The key stays with the generator/verifier. The released artifact is an ordinary SVG; a private authenticated manifest records candidate sets and weights.

The attacker may read and minify the SVG and may perturb or redraw contours. The private key and manifest are outside the attacker model. The initial goal is robustness to syntax changes and bounded coordinate rounding. It does **not** promise survival of unrestricted quality-preserving redrawing, rasterization/revectorization, or disclosure of the candidate sets.

## Candidate-assisted sampler

Let (K) be a 256-bit key and (F(c)) a geometric fingerprint formed from uniformly spaced arc-length samples of a contour. Domain-separated HMAC produces (u_{t,i}=\operatorname{PRF}_K(\text{asset-id},t,F(c_{t,i}))\in(0,1)). The generator chooses

\[
J_t=\operatorname*{argmax}_i\left[\log q_{t,i}-\log(-\log u_{t,i})\right].
\]

This is the Gumbel-max trick. If the (u_{t,i}) are independent uniforms, then (\Pr[J_t=i]=q_{t,i}). A secure PRF approximates those draws for anyone without the key. Thus the *marginal distribution over proposed contours is unchanged*; only the otherwise random choice is keyed. Unlike selecting favorable text tokens, the choice concerns contour geometry that persists through path command rewriting. This is our design inference from the SynthID-Text sampling principle, not a claim made by that paper.

The SVG is assembled only after each (J_t) is chosen. There is no post-generation patch. The upstream generator may observe each choice and continue drawing. A closed provider API cannot expose or change its own internal token sampling; this wrapper controls a structured path-proposal boundary.

## Detector and statistical meaning

The verifier authenticates the private manifest, parses the released SVG, samples each path by arc length, and matches it to the recorded candidate geometries within a tolerance measured as a fraction of the canvas diagonal. It recomputes (J_t) from the key. For the event that **every** observed step equals its keyed winner, the conditional null probability is

\[
p_{\text{null}}=\prod_{t=1}^{T}q_{t,J_t}.
\]

For partial edits, let (R) be the steps whose released contour remains inside exactly one disjoint candidate acceptance region, and let (M) be the number classified as the keyed winner. The conditional statistic is the Poisson-binomial upper tail (\Pr[\sum_{t\in R} B_t\ge M]), where (B_t\sim\operatorname{Bernoulli}(q_{t,J_t})). It assumes an unmarked generator independently follows the recorded categorical distributions and, additionally, that erasure/recognition is independent of the keyed winner. The implementation also reports a conservative tail over all original steps, treating erasures as failures. Nine equiprobable binary decisions give (2^{-9}\approx0.00195) when all match. The statistic measures consistency with a specific key and manifest, not authorship, legal ownership, or which model drew the SVG.

The private manifest is authenticated with HMAC, but the current prototype has no independent timestamp or public commitment. A key holder could fabricate a manifest after seeing an SVG. A provenance deployment needs an append-only external commitment of the manifest hash, a trusted signing service, or an equivalent precommitment protocol.

## Manifest-free geometric sampler

For equal-probability alternatives, a second mode quantizes each candidate contour's arc-length centroid into a geometric symbol (z=Q_\Delta(C)). The generator rejects candidates near quantization boundaries and requires distinct symbols. It chooses the candidate with the largest keyed pseudorandom value (U_K(\text{asset-id},z)). Under a random key, each candidate wins with probability (1/m), preserving the generator's categorical distribution. The SVG contains the selected contour, not a hidden annotation.

The verifier needs only the SVG, key, and a **precommitted** asset ID. It recomputes each distinct visible contour symbol and value (U_i), then calculates (S=\sum_i-\log U_i). For unmarked geometry fixed independently of the key, the PRF idealization gives independent uniforms and (S\sim\mathrm{Gamma}(n,1)). The lower-tail (p=F_{\mathrm{Gamma}(n,1)}(S)) detects an excess of high keyed values. With two equal-weight alternatives, a marked choice has (U=\max(U_1,U_2)), so (E[-\log U]=1/2) instead of 1. Detection power grows with the number of independent contours and falls under path deletion.

The quantized centroid is a first prototype, not a proven robust invariant. It survives tested syntax rewrites and modest rounding, but an attacker can shift contours across quantization cells. Key or asset-ID search after observing an SVG can fabricate statistical evidence; both must be committed before generation for a provenance claim. Distinct symbols are required for the Gamma calibration.

### Exact idealized power and the coverage limit

For (m) equal-weight alternatives at each of (n) independent steps, the selected value is the maximum of (m) independent uniforms. Thus (U\sim\mathrm{Beta}(m,1)) and (-\log U\sim\mathrm{Exp}(m)). The marked statistic has (S\sim\mathrm{Gamma}(n,\text{rate}=m)), while the unmarked statistic has rate (1). At false-positive threshold (\alpha), idealized detection power is

\[
\Pr_{H_1}\!\left[S\le F^{-1}_{\mathrm{Gamma}(n,1)}(\alpha)\right]
=F_{\mathrm{Gamma}(n,m)}\!\left(F^{-1}_{\mathrm{Gamma}(n,1)}(\alpha)\right).
\]

At (\alpha=10^{-3}) with two alternatives, this is approximately 0.220, 0.696, and 0.990 for 16, 32, and 64 independent usable contours. These are power calculations under the model, not measured generator results. They explain why a 32-contour watermarked asset can legitimately fail detection. They also omit candidate rejection, dependence, geometric edits, and selection bias from using a public test key.

A universal watermark for **every** SVG cannot satisfy a nonzero payload and strict visual equivalence. Let (\mathcal A_\epsilon(x)) be the set of SVG renderings admissible under the fidelity policy for an asset (x). Any choice-based embedded signal has at most (\log_2|\mathcal A_\epsilon(x)|) bits of choice capacity. If an SVG has a unique admissible rendering, this bound is zero. Text, embedded images, filters, and rigid shapes may offer generator-specific choices, but this implementation has no verified encoder or detector for them. An honest system must measure usable capacity and decline to claim a mark where there is none; path-only evidence cannot authenticate a mixed-content drawing as a whole.

## Prior work and relationship

1. **SVG representation steganography.** Blinova and Urbanovich hide data by splitting cubic Bézier curves in SVG and provide StegoSVG. This is directly relevant. A path optimizer capable of merging or reparameterizing curves may erase split-based encodings; that vulnerability is an inference from where their signal resides, and must be tested rather than presumed. [Paper](https://www.mathnet.ru/eng/bgumi18)
2. **Geometric curve marking.** Kwon et al. use cross-ratios of roots in rational Bézier/B-spline representations; they prove affine and Möbius reparameterization robustness and explicitly identify degree reduction as a failure mode. [Paper](https://www.sciencedirect.com/science/article/pii/S0010-4485%2811%2900040-6)
3. **Vector drawing transforms.** Nakai and Kohara embed through Haar coefficients of sampled vector drawings and evaluate translation, rotation, and scaling. Their paper notes a file-size tradeoff from dense line approximation. [Paper](https://www.ijcaonline.org/archives/volume182/number49/nakai-2019-ijca-918752.pdf)
4. **Topology preservation.** Huber et al. define maximum perturbation regions guaranteeing a planar drawing's vertex/edge, containment, incidence, and planarity properties. Their framework suggests a quality constraint for candidate generation; the present prototype only caps geometric distance and does not yet certify topology. [Paper](https://research-explorer.ista.ac.at/download/1816/4704/IST-2016-443-v1%2B1_S0218195914500034.pdf)
5. **Keyed inference sampling.** SynthID-Text changes sampling during inference and supports detection without retraining the base model. Anthropic states Claude's text watermark is a version of this approach. Its SVG Content Credential is signed file metadata, which has different persistence properties from a geometry mark. [Nature paper](https://doi.org/10.1038/s41586-024-08025-4), [Anthropic explanation](https://www.anthropic.com/news/claude-text-watermark), [reference implementation](https://github.com/google-deepmind/synthid-text)
6. **Limits of robustness.** Zhang et al. show that strong watermarking is impossible under explicit quality and perturbation-oracle assumptions. The appropriate claim is robustness against a specified attack suite, not arbitrary semantics-preserving edits. [Paper](https://arxiv.org/abs/2311.04378)

The combination proposed here is *keyed, distribution-preserving choice among contour geometries during SVG generation, with post-minification geometric verification*. This preliminary review found adjacent methods, but does not establish that the combination is new. A systematic prior-art search, including patents and code, is required before a novelty claim.

## What a credible submission needs

* **Corpus:** real SVGs from multiple sources and drawing styles; document licenses, path counts, contour lengths, and how many usable candidate choices each asset provides. Separate train/dev/test even if no model training occurs, because thresholds and candidate policies can overfit.
* **Generator:** at least one actual text-to-SVG or image-to-SVG system that emits candidate paths during inference. The toy example currently validates only the sampler interface. Measure candidate quality and generation cost, including rejection rates when no visually equivalent alternative exists.
* **Baselines:** metadata/C2PA, numeric syntax encodings, Bézier split encodings, post-hoc geometric perturbation, and a contemporary vector-domain method. Compare under identical asset and attack sets.
* **Attacks:** Scour and SVGO at multiple precisions, path reorder/merge/split, coordinate transforms, scaling/rotation, simplification, rasterize/revectorize, cropping, and an adaptive attacker using a quality oracle. Report both benign processing and adversarial removal separately.
* **Metrics:** detection TPR at fixed FPR (at least (10^{-3}) and (10^{-4})), calibration of claimed (p)-values on unmarked outputs and wrong keys, survival versus optimizer settings, render differences at multiple resolutions, topology validity, byte overhead, latency, and key/manifest disclosure effects. Include confidence intervals and ablations over candidate count, entropy, geometric separation, and matching tolerance.
* **Security:** model key compromise, manifest fabrication, candidate-set leakage, multiple-key testing, false attribution to unrelated SVGs, and cross-asset substitution. Precommitment must be evaluated if the system is presented as provenance evidence.

## Current evidence

As of 30 September 2026, 38 unit tests cover both inference-time samplers, a 9-step and 32-step toy generator, negative controls, wrong-key probes, categorical distribution preservation, Scour/SVGO optimization, path reordering/merging, adaptive disjoint matching radii, bounded independent translation, rendering-indirection rejection, mixed-content coverage assessment, OmniSVG metadata handling, and partial path deletion. No model training is required.

Three completed local `qwen3.5:4b` generations (radial flower, parallel-wave, and leaf layouts) each produced 16 pairs of cubic-contour proposals and accepted one keyed alternative at every step before SVG assembly. All 16 released contours in each asset match the keyed winners, giving the conditional null probability (2^{-16}=1.5259\times10^{-5}). All three retain all 16 matches after Scour, SVGO, two-decimal path rounding, path reorder, a one-unit global translation, and alternating half-unit path translations. Deleting four of 16 paths leaves 12 recognized matches: the conditional score remains significant at (2^{-12}), while the conservative all-step score is (p=0.0384). The generator needed proposal retries because the validity and geometric-separation gate rejected some outputs. Files and machine-readable reports are under `experiments/results/`.

On all three assets, an XML-comment baseline survives Scour but is removed by default SVGO. A 64-chip numeric coordinate-LSB baseline fails after two-decimal path rounding on every asset and additionally fails after Scour on the wave and leaf assets. ContourMark assisted mode remains detected in all four benign variants. This is a direct same-asset comparison, but three assets cannot establish superiority.

An adaptive, keyless shape attack moves the two cubic control points of every contour in opposite directions while retaining endpoints. After switching to disjoint per-step acceptance radii and conditioning the primary score on recognized contours, the first tested failing amplitudes are 12 units for flower, 6 for wave, and 4 for leaf. Their sampled-geometry RMS values are 0.754%, 0.375%, and 0.266% of the canvas diagonal; 512-pixel librsvg normalized pixel MAE is 3.56%, 8.48%, and 2.31%, respectively. The immediately smaller tested amplitudes remain detected. The conservative statistic, which treats erasures as failures, rejects much earlier. These pixel metrics do not establish perceptual equivalence, and conditional calibration requires recognition to be independent of the keyed winner, but the revised detector forces substantially larger distortion than the original fixed-radius rule.

The manifest-free centroid prototype has contrary evidence: a marked 32-contour fixture with a fixed public test key produced (p=0.0558), so it was not detected, and translations of 0.5 to 2 canvas units did not restore the signal. This agrees with its idealized power calculation and shows that 32 binary choices are insufficient for reliable detection at stringent false-positive rates. It should not be presented as the primary robust method without a stronger invariant and at least 64 usable choices.

The provider-neutral categorical-token sampler has a 4,000-key Monte Carlo test against a 10/30/60 distribution and an OmniSVG 1.1 coordinate-token policy. A call to the official hosted 4B option produced a one-path rocket SVG; the untouched 3,381-byte document passes the strict geometric-carrier assessor. This executes the specialist generator and validates output-structure compatibility. It does not inject a watermark because the public API returns only the completed SVG and exposes no logits or candidate hook. See `docs/model-adapters.md` and `experiments/results/omnisvg_rocket.json`.

These experiments validate implementation properties and expose failures. They do not yet establish model-wide visual-quality parity, diverse-asset robustness, novelty, or superiority.
