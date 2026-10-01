# Literature review: watermarking and provenance for vector graphics

Status: 30 September 2026. This review supports the blind spectral watermark in
`src/contourmark/spectral.py`. It covers what already exists, what each line of
work assumes, and where this system sits relative to it. A patent search has
not been done.

## 1. Why SVG needs its own treatment

An SVG is both a program and a picture. The same rendering can be written with
absolute or relative commands, shorthand curves, different numeric precision,
merged or split paths, group transforms, or basic shapes instead of paths.
Production pipelines rewrite all of these routinely:

* SVGO (the default optimizer in most web build chains) rounds to 3 decimals,
  converts commands, merges paths, and removes `<metadata>`, comments, and
  unknown attributes.
* Scour and picosvg normalize files in a similar way. picosvg also flattens
  transforms and converts strokes into filled outlines.

A provenance signal that lives in the syntax does not survive this. That
includes XML metadata, C2PA manifests embedded in the file, comments, number
formatting, and segment structure.

Regulation raises the stakes. Article 50 of the EU AI Act, applicable from
2 August 2026, requires AI-generated content to carry machine-readable marking
([overview](https://datanorth.ai/blog/ai-watermarks-text-images-and-the-push-for-provenance)).
Current practice for SVG is signed metadata. For example, Anthropic attaches
C2PA credentials to SVG output
([UNU C3 note](https://c3.unu.edu/blog/claude-ai-watermark-eu-ai-act-coverage)).
That metadata is exactly what SVGO removes by default.

## 2. Vector-graphics and GIS watermarking

### Fourier descriptors of polylines

Solachidis and Pitas (ICASSP 2000) modify mid-frequency Fourier-descriptor (FD)
magnitudes of polygonal lines. Doncel, Nikolaidis and Pitas (IEEE TVCG 2007)
add an optimal blind detector and handle multiple lines
([TVCG PDF](https://aiia.csd.auth.gr/wp-content/uploads/papers/PUBLISHED/JOURNAL/pdf/IEEE_tvcg-0007-0106_svg_watermarking.pdf)).

* **Invariances:** FD magnitudes are unchanged by rotation, translation,
  scaling, start point, and traversal direction. This is the closest prior art
  to our feature choice.
* **Differences from our method:**
  1. They operate on the **vertex sequence**. Curves are handled by treating
     control points as vertices. As a result the mark depends on the
     representation: splitting a segment or converting between curve types
     changes the vertex sequence.
  2. They use **additive/multiplicative spread spectrum with correlation
     detection**. This suffers host interference. The authors recommend lines
     with at least 300 vertices, while icons have tens.
  3. Thresholds come from Gaussian or empirical approximations, with no
     exact false-positive bound.
* **Our re-implementation** (`vector_baselines.fd_vertex_*`) reproduces this
  failure at icon scale.

The ICMAI 2024 paper *A Novel Frequency Domain Watermarking of 2D Vector
Graphics* uses a V-system orthogonal basis that represents Bézier curves
exactly. Its robustness claims are limited to transformation attacks
([ACM](https://dl.acm.org/doi/10.1145/3670085.3670095)).

### GIS vector maps

This literature is large and mostly QIM-based:

* Vertex or angle quantization index modulation (QIM), feature-point distance
  ratios, and virtual-triangle features with improved QIM (Wu et al., 2025,
  *Transactions in GIS*,
  [link](https://onlinelibrary.wiley.com/doi/10.1111/tgis.70036)).
* Hybrid DFT+SVD schemes
  ([Computers & Geosciences 2023](https://www.sciencedirect.com/science/article/abs/pii/S0098300423002194)).
* Schemes robust to Douglas–Peucker simplification
  ([street networks](https://scialert.net/fulltext/?doi=itj.2009.982.989)).

How these data differ from SVG artwork:

* Maps have many dense vertices, tolerance-bounded coordinates, and GIS-specific
  attacks (cropping, vertex editing, simplification).
* SVG artwork has few, curved segments and is dominated by optimizer and
  precision attacks.
* Most map schemes quantize per-vertex quantities, so they inherit the vertex
  representation dependence.

### SVG steganography

Blinova and Urbanovich split cubic Béziers at ratios that encode the message
(StegoSVG)
([ResearchGate](https://www.researchgate.net/publication/357474736_Steganographic_method_based_on_hidden_messages_embedding_into_Bezier_curves_of_SVG_images)).
This is lossless in rendering. However, it is fragile to any re-parameterization
and to rounding that breaks the de Casteljau relation. Our re-implementation is
`vector_baselines.bezier_split_*`.

Other schemes hide data in:

* numeric least-significant digits (coordinate LSB / parity);
* comments, attributes, and colour LSBs.

Survey-level tooling notes these survive SVGO only when SVGO leaves the
modified field alone ([stegotoolkit](https://stegotoolkit.com/steganography/svg-steganography-hider)).

### Topology-preserving perturbation

Huber et al. bound vertex perturbations that preserve planar-drawing topology.
This is relevant as a fidelity constraint, not as a watermark.

## 3. Glyph and drawing marks decoded from rasters

* **FontCode** (Xiao, Zhang, Zheng, ACM TOG 2018) perturbs glyph outlines and
  decodes from images with a CNN
  ([arXiv](https://arxiv.org/abs/1707.09418)).
* **AutoStegaFont** (AAAI 2023) and **GlyphShield** (AAAI 2026) learn
  vector-font perturbations with differentiable rasterization and noise layers,
  so they survive screenshots and print-scan
  ([GlyphShield](https://ojs.aaai.org/index.php/AAAI/article/view/37073)).
* **DeepMorph** (Rasmussen et al., 2020) trains an encoder/decoder over a soft
  rasterizer to hide bitstrings in SVG drawings and decodes from photographs
  ([arXiv](https://arxiv.org/abs/2011.09783)).

These systems target the *physical/raster* channel. They require trained
decoders, fixed glyph sets or drawing templates, and empirical thresholds. None
evaluates SVG optimizers or gives false-positive guarantees. They are
complementary: a raster-domain decoder is the natural extension for
rasterize-and-retrace attacks, which our geometry-domain detector only partly
survives.

## 4. Quantization-based watermarking and content-dependent keys

* **QIM / dither modulation.** Chen and Wornell (IEEE Trans. IT 2001) show QIM
  achieves near-capacity rate/robustness trade-offs and removes host
  interference, unlike spread spectrum
  ([PDF](https://sia.mit.edu/wp-content/uploads/2015/04/2001-chen-wornell-it.pdf)).
  Our embedder uses keyed dither modulation on invariant spectral magnitudes.
* **Spread-transform dither modulation (STDM)** raises the distortion an
  uninformed re-quantizer needs. It is our main candidate hardening against
  informed removal (not implemented).
* **Content-dependent keys.** Deriving randomness from a robust hash of content
  (Fridrich, 1999, "robust bit extraction") prevents pooling marked items to
  estimate a global key. We seed each contour's dither with a coarse
  similarity-invariant shape class.
* **3D mesh analogues.** Cho, Prost and Jung (IEEE TSP 2007) is an oblivious
  mesh watermark using vertex-norm distributions. Later work adds QIM and
  saliency.

## 5. Generative-model watermarking

* **Inference-time text marks.**
  * SynthID-Text (Dathathri et al., Nature 2024) uses tournament sampling.
  * Kirchenbauer et al. (2023) use a green list.
  * Gumbel-max / distribution-preserving samplers (Aaronson; Kuditipudi et al.)
    change token selection without retraining.
  * Anthropic's text watermark is described as a version of this approach.
* **Code watermarks break under semantics-preserving rewrites.** Examples:
  SWEET, ACW, and STONE; see "Is The Watermarking Of LLM-Generated Code
  Robust?" ([arXiv 2403.17983](https://arxiv.org/abs/2403.17983)). SVG code
  is the extreme case, because optimizers rewrite nearly every token while
  preserving the rendering. This is the argument for moving the mark from
  tokens to geometry.
* **Image watermarks.** Stable Signature, Tree-Ring, TrustMark, VINE (ICLR
  2025), and benchmarks such as WAVES and "What breaks local watermarks?"
  (2026) study raster robustness. Regeneration attacks provably remove
  invisible raster marks under assumptions (Zhao et al. 2023). The vector
  analogue in our threat model is rasterize-and-retrace.
* **Limits.** Zhang et al., *Watermarks in the Sand* (2023), prove that strong
  watermarking is impossible against attackers with quality and perturbation
  oracles ([arXiv](https://arxiv.org/abs/2311.04378)). Claims must be scoped to
  named transformation families. We follow that scoping.

## 6. SVG generators the system must cover

* **Code-writing LLMs**, for example Claude, GPT, Qwen, and Gemma writing SVG
  directly.
* **Specialist models:**
  * StarVector (CVPR 2025; image/text→SVG code)
  * OmniSVG (NeurIPS 2025; Qwen-VL-based, tokenized coordinates)
  * InternSVG, SVGen, Reason-SVG, LLM4SVG, IconShop
  * Chat2SVG
* **Optimization-based models:** VectorFusion, SVGDreamer, DiffVG-based.

Benchmarks: SVGenius, VectorGym, SVG-Bench, MMSVGBench.

These generators differ widely internally. They share only the output: path
geometry. A post-hoc geometric watermark therefore needs no model access. It
covers every generator with the same code, which is the generality argument
of this project. The repository's original generation-time sampler remains a
zero-distortion option for generators that expose candidate choices or logits.

## 7. Gap this project addresses

To our knowledge, no prior system offers all of the following at once:

1. **Blind and key-only:** no original, database, or manifest.
2. **Representation-invariant:** robust to the optimizer/rewriting pipeline
   that SVG actually passes through.
3. **Exact false-positive bound**, not a threshold fitted to a corpus.
4. **Evaluated on real icon/logo/emoji corpora and LLM-generated SVG** against
   syntax, geometric, structural, and re-drawing attacks, with direct
   baselines.

The closest pieces are:

* FD-magnitude invariance (Solachidis/Doncel/Pitas)
* QIM with dither (Chen and Wornell)
* content-seeded keys (Fridrich)
* keyed, distribution-preserving sampling from LLM watermarking

The contribution is to combine these for SVG, with:

* a representation-free capacity model;
* structure-preserving embedding;
* an exactly calibrated detector.

Novelty should still be confirmed against patents and non-indexed industry
work before submission.
