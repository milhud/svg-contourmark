# Generation-time geometric watermarking of SVG: research design

Status: prototype and research plan, 30 September 2026. This is not a claim of novelty or a completed USENIX evaluation.

## Question and threat model

Can a generator create SVGs whose *chosen visible geometry* carries a keyed provenance signal, while preserving its normal output distribution and surviving syntax-level optimization? The generator emits a sequence of paths. At step (t), it supplies a set of geometrically distinct but visually comparable candidates (C_t=\{c_{t,1},\ldots,c_{t,m_t}\}), with categorical probabilities (q_{t,i}>0) that sum to one. The candidate set and weights must be fixed independently of the secret key. The key stays with the generator/verifier. The released artifact is an ordinary SVG; a private authenticated manifest records candidate sets and weights.

The attacker may read and minify the SVG and may perturb or redraw contours. The private key and manifest are outside the attacker model. The initial goal is robustness to syntax changes and bounded coordinate rounding. It does **not** promise survival of unrestricted quality-preserving redrawing, rasterization/revectorization, or disclosure of the candidate sets.

## Sampler

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

This exact probability assumes that an unmarked generator would independently sample from the recorded categorical distributions along the observed trajectory, that candidate sets were fixed without knowledge of the current keyed draw, and that geometry matching has negligible ambiguity. Nine equiprobable binary decisions give (2^{-9}\approx0.00195). This measures consistency with a specific key and manifest, not authorship, legal ownership, or whether a particular model wrote the SVG.

The private manifest is authenticated with HMAC, but the current prototype has no independent timestamp or public commitment. A key holder could fabricate a manifest after seeing an SVG. A provenance deployment needs an append-only external commitment of the manifest hash, a trusted signing service, or an equivalent precommitment protocol.

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

Unit tests cover keyed selection, a 9-step toy generator, negative controls, a distribution-preservation Monte Carlo check, and Scour optimization. These validate implementation properties, not the broad robustness or visual-quality claims above. No model training is required by this method.
