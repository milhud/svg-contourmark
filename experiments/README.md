# Experiments index

Every harness here runs on CPU except the IconShop generator. Each has a tiny mode, and `./smoke_all.sh` exercises all of them in about two minutes. Run it after any code change and before any long run.

Conventions:
- Run from `experiments/` with `OMP_NUM_THREADS=1`; worker pools oversubscribe BLAS otherwise.
- Tune on the **dev** split only (`corpus/dev.json`). Freeze settings before touching `corpus/test.json`.
- Result files are JSON or JSON lines under `results/`. Summaries report attack and detector **error counts** next to every rate.

## Which script answers which question

| Question | Script | Tiny mode | Output |
|---|---|---|---|
| Does the blind spectral mark survive attacks, against baselines? | `evaluate_blind.py` + `summarize_blind.py` | `--limit 1` | `results/blind/*.jsonl`, `.summary.md` |
| What are the confidence intervals, resampling assets? | `bootstrap_ci.py` | `--resamples 50` | `results/blind/v2/bootstrap_ci.json` |
| Is the comparison fair at equal distortion? | `frontier.py` | `--smoke` | `results/blind/frontier.json` |
| Are false positives calibrated? | `null_calibration.py` (spectral), `geosample_null.py` (inference) | `--keys 5` | `*_null*.json` |
| Can an informed attacker remove the mark? | `adaptive_attacks.py` | `--per-source 1` | `adaptive.json` |
| How does strength trade distortion for robustness? | `evaluate_blind.py --params '{"delta": ...}'` | `--limit 1` | `dev_delta_*.jsonl` |
| Which inference sampler variant has power, at what entropy and cost? | `sampler_lab.py` | `--smoke` | `results/geosample/sampler_lab.json` |
| Does the inference mark work on a real model? | `iconshop_geosample.py`, then `evaluate_geosample.py` + `summarize_geosample.py` | `--prompts star --samples-per-prompt 1 --max-tokens 40` | `results/geosample/iconshop_*` |
| Can people see the mark? | `make_study.py` (builds the study; people run it) | `--assets 2` | `<out>/index.html`, `trials.json` |
| Secondary statistics for the paper | `analyze_main.py` | any results file | `*.analysis.json` |
| Paper figures | `make_figures.py` | any results file | `../paper/figures/` |

GPU runs: `../hpc/README.md` (SLURM, 4 GPUs) or `../colab/README.md` (one GPU, Colab).

## Options worth knowing

`evaluate_blind.py`:
- `--key-mode shared` uses one key for every asset, as a deployment would. The default is a fresh key per asset, which hides anything an attacker could learn from many files marked with the same key.
- `--suite extended` adds exact rewrites (`to_cubics`, `subdivide_uneven`), real curve clipping (`clip_half`), and chained pipelines. The standard suite is kept unchanged so stored results stay comparable.
- `--visibility render` asks librsvg whether clipped, masked or CSS-styled shapes are actually drawn, for both marking and detection.
- `--params '{"scheme": "ss"}'` runs the spread-spectrum ablation on the same carrier.

Library functions used by the harnesses:
- `contourmark.spectral.capacity(svg)`: keyless check with an explicit outcome (`supported`, `insufficient_capacity`, `unsupported`). Use it for coverage denominators; a file that parses is not necessarily markable.
- `contourmark.metrics.fidelity_report(original, marked)`: boundary distance in document units and in pixels at icon sizes, silhouette overlap, colour error on white and dark backgrounds, soft kinks at joins, and editability (segments, primitives lost, bytes).
- `contourmark.geosample.GeoWatermark(..., mode="bias", delta=...)`: a green-list sampler that trades distribution shift for power; detect with `statistic="green"`. The default `mode="gumbel"` preserves the output distribution.
- `contourmark.toygen.toy_drawing(..., sigma=...)`: toy point-token generator with an entropy knob.

## Findings so far from the exploration harnesses

**Sampler lab** (`results/geosample/sampler_lab.json`, toy generator, 20 drawings per cell):

| Entropy per keyed step | Descriptors per drawing | Distribution-preserving | Naive reuse | Bias δ=4 (shift per step) |
|---|---|---|---|---|
| about 1 nat (IconShop at top-p 0.5) | about 25 | 0% | 0% | 0% (0.53 nats) |
| about 1 nat | about 100 | 15% | 35% | 65% (0.61 nats) |
| about 1.7 nats | about 100 | 70% | 100% | 100% (0.68 nats) |
| about 2.9 nats | about 100 | 100% | 100% | 100% (0.68 nats) |

Entries are detection rates at p ≤ 1e-6. At IconShop-like entropy a small icon cannot be marked detectably by any variant, and a large drawing only weakly. Power needs higher entropy per step or more decisions per drawing.

**Frontier** (`results/blind/frontier.json`, 54 dev icons plus 6 long polylines):

- At about 0.16% boundary distance, the spectral mark detects 91% clean, 85% after SVGO, and 91% after rotation and subdivision.
- Coordinate LSB at its closest setting detects 78% clean, 69% after SVGO, and 0% after rotation or subdivision.
- The vertex Fourier-descriptor adaptation detects 0% on icons at every strength. On 600-vertex polylines it detects 33% at its highest strength.
- The spread-spectrum ablation on our own carrier detects 0% everywhere.

**Metrics:** marking used to introduce soft kinks at smooth joins (up to half of the joins on some icons). The embedder now keeps smooth joins smooth.
