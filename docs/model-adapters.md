# Autoregressive model adapters

Status: implementable adapter contract, with the probability-preserving core tested locally and an official hosted OmniSVG 4B output assessed. OmniSVG weights have not been run on this Apple Silicon host, and the hosted API does not expose its decoding logits.

## Integration boundary

`KeyedCategoricalSampler` accepts the exact candidate token IDs and logits that an unwatermarked decoder would sample from after grammar masking, temperature, top-k, and top-p truncation. It replaces that one random draw with a keyed Gumbel-max draw. In the random-PRF idealization,

\[
J=\arg\max_i\{\log p_i-\log(-\log U_i)\},\qquad
\Pr[J=i]=p_i.
\]

The core has no dependency on PyTorch, Transformers, or a particular tokenizer. A model adapter is responsible for four facts:

1. determine that the decoder is currently expecting a visible-geometry token;
2. pass the **complete** truncated candidate set, rather than a hand-picked favorable subset;
3. map the selected tokens to canonical visible geometry for verification after serialization changes; and
4. decline to mark if the grammar, geometry map, or fidelity constraint is uncertain.

The fourth condition is essential. Calling all numeric-looking tokens “geometry” can corrupt command arguments, arc flags, colors, or unrelated XML.

## OmniSVG 1.1 reference adapter

OmniSVG 1.1 is the concrete reference because its released inference path uses a Qwen 2.5 VL decoder and its tokenizer publishes disjoint command, coordinate, color, and arc-parameter regions. Its 4B model uses coordinate token IDs 151944 through 191946; the 8B model uses 152072 through 192075. `OmniSVGTokenPolicy` encodes those ranges from the upstream configuration.

The upstream code currently calls `transformer.generate(...)` with stochastic decoding and can return multiple candidates. A complete integration should insert a logits processor before each categorical draw:

```python
policy = OmniSVGTokenPolicy("4B")
sampler = KeyedCategoricalSampler(key, asset_id)

# Inside the decoding callback, after the normal grammar and top-k/top-p mask:
if grammar.expects_coordinate and policy.eligible_step(candidate_token_ids):
    choice = sampler.choose(
        candidate_token_ids,
        candidate_logits,
        context=canonical_prefix,
        step=geometry_step,
    )
    force_next_token(choice.token_id)
else:
    use_normal_decoder_sample()
```

The published tokenizer maps each coordinate token to one point on a 200 by 200 grid, which makes this boundary unusually clean. The remaining research work is a grammar state machine and a geometry-synchronizing verifier that turns a minified SVG back into the same sequence of coordinate decisions. A token-only detector would fail after equivalent path command rewriting, so it is insufficient for this project. Upstream sources: [OmniSVG repository](https://github.com/OmniSVG/OmniSVG), [inference implementation](https://github.com/OmniSVG/OmniSVG/blob/main/inference.py), [tokenizer](https://github.com/OmniSVG/OmniSVG/blob/main/tokenizer.py), and [configuration](https://github.com/OmniSVG/OmniSVG/blob/main/config.yaml).

The released README reports about 17 GB of GPU memory for OmniSVG 1.1 4B. This machine exposes 16 GB of Metal GPU memory, while the official environment targets CUDA 12.1. Downloading the 7.69 GB checkpoint would remain below the project's 100 GB limit, but it would not establish runnable compatibility on this host.

We instead called the official public Hugging Face Space with its 4B option. It generated `experiments/results/omnisvg_rocket.svg`, a 3,381-byte document containing one path. The untouched output passes the strict path-subset assessor after recognizing OmniSVG's nonstandard `filling="0"` field as inert producer metadata. This is a genuine specialist-model output and establishes structural carrier coverage. It is not a watermarked OmniSVG run: the Gradio endpoint returns a completed SVG and exposes neither logits nor a pre-sampling callback. `experiments/omnisvg_space.py` and the adjacent JSON report make this boundary reproducible. [Official Space](https://huggingface.co/spaces/OmniSVG/OmniSVG-3B)

## Icon-focused generators

“IconSVG” does not uniquely identify a published generative model in the literature search. The closest established model name is **IconShop**, an autoregressive text-to-vector icon generator. Its path-command sequence is compatible with the same adapter architecture if its decoder exposes logits and a grammar-aware coordinate-token map. This is an architectural compatibility statement, not an execution result. [IconShop paper](https://arxiv.org/abs/2304.14400)

DeepSVG is a different case: it predicts a hierarchy of shapes and commands rather than XML tokens. Its adapter should apply keyed categorical sampling to discrete command choices and keyed reparameterization to continuous coordinates, subject to a visual-fidelity gate. [DeepSVG paper](https://arxiv.org/abs/2007.11301), [code](https://github.com/alexandre01/deepsvg)

## What generalizes and what does not

The probability-preserving sampler generalizes to any model exposing a categorical distribution over alternatives. The geometry policy and verifier do not automatically generalize: they are tied to the model's representation and the SVG rendering semantics. An adapter conformance suite must establish:

* distribution preservation on unmarked versus keyed draws;
* valid SVG rate and render-quality parity;
* at least 64 usable independent choices for high power at false-positive rate (10^{-3}) in the binary idealization;
* recovery after minification and equivalent path rewriting;
* failure under unsupported text, image, filter, CSS, and compositing constructs rather than false attribution; and
* adaptive-removal cost measured in rendered distortion.

This contract makes provider neutrality precise while avoiding a claim that one untested hook works unchanged in every generator.
