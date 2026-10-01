# HPC runbook: inference-time watermark (geosample) on IconShop

This runbook covers the GPU experiments for the inference-time SVG watermark (`src/contourmark/geosample.py`). Everything else in the repo (the blind spectral watermark and its 900-asset evaluation) is already done and committed. See `HANDOFF.md`.

**What the experiment shows.** The real IconShop checkpoint generates icons twice per prompt and seed: once with ordinary sampling, once with the keyed geometric choice. Both runs use the same model, prompt, truncation and RNG seed. We then measure:
- detection;
- robustness to SVG optimizers, rounding and similarity transforms;
- calibration of false positives;
- whether marked and plain outputs differ statistically.

## 0. What you need

- A cluster node with 4 × A100 (any CUDA GPU works), plus a CPU node for the evaluation.
- Python ≥ 3.10 (`module load python/3.11` or conda).
- Node.js ≥ 18 for SVGO (`module load nodejs` or `conda install -c conda-forge nodejs`).
- Optional:
  - `rsvg-convert` (librsvg) and `potrace` for the rasterize-and-retrace attacks (`conda install -c conda-forge librsvg potrace`). Without them those attacks are recorded as errors and skipped; everything else runs.
- About 2 GB of disk: about 540 MB checkpoint, about 400 MB of npm packages, plus results.
- Internet on the **login node** only. Compute jobs run offline from the cache.

## 1. Get the code (login node)

```bash
git clone https://github.com/milhud/svg-contourmark.git
cd svg-contourmark
```

## 2. One-time setup (login node, needs internet)

```bash
# optional: put the HF cache on scratch
export HF_HOME=/scratch/$USER/hf_cache

# pick the torch wheel for the cluster's CUDA
TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash hpc/setup.sh
```

`hpc/setup.sh` does the following:
1. Creates `.venv` and installs CUDA torch, transformers, the `contourmark` package and its test/eval extras.
2. Runs `npm install` for SVGO and the icon corpora.
3. Downloads the IconShop checkpoint (`m1357l/iconshop-svg-generator`, CC BY-NC-SA 4.0) into `models/iconshop/` and the BERT tokenizer into `$HF_HOME`.
4. Runs the CPU unit tests for the watermark.

`models/` and the cache are git-ignored. **Do not commit the weights.**

## 3. Smoke test (one GPU, about 2 minutes)

```bash
srun --gres=gpu:1 --time=00:20:00 bash -c '
  export HF_HOME=${HF_HOME:-$PWD/.hf_cache} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  cd experiments
  ../.venv/bin/python iconshop_geosample.py --output results/geosample/smoke \
      --prompts star heart house car --samples-per-prompt 2 --batch 8 --device cuda
  ../.venv/bin/python -c "
import json
for l in open(\"results/geosample/smoke/samples.jsonl\"):
    r = json.loads(l); print(r[\"marked\"], r[\"prompt\"], len(r[\"tokens\"]), round(r[\"log10_p\"], 2))"'
```

Expected:
- 16 lines.
- Marked rows mostly have log10 p below −2 at top-p 0.5.
- Plain rows sit near 0.

Delete `results/geosample/smoke` afterwards.

## 4. Generation (GPU job)

```bash
sbatch hpc/generate_iconshop.sbatch
# options (environment variables):
#   TOP_PS="0.5 0.9 1.0"  top-p values to run (default all three)
#   SEEDS=16              samples per prompt; 30 prompts -> 480 plain + 480 marked per top-p
#   BATCH=32              per-GPU batch
#   GPUS=4                shards; shard k runs on GPU k
```

- **Sharding:** each top-p value is split across the 4 GPUs by prompt and seed (`--shard k --num-shards 4`). Each GPU writes its own `samples.shardK.jsonl`; the script then merges them into `samples.jsonl`.
- **Resumable:** rerunning skips finished (prompt, seed, marked) triples per shard.
- **Expected time:** IconShop has no KV cache, so each step recomputes the full prefix. On a Mac it was about 2 min per batch of 8. On an A100 expect seconds per batch, so about 30–90 min for all three top-p values.
- **Logs:** `hpc/logs/`.

Outputs go to `experiments/results/geosample/iconshop_p05/`, `iconshop_p09/` and `iconshop_p10/`:
- `plain_<prompt>_<seed>.svg` and `wm_<prompt>_<seed>.svg`: generated icons, 200×200 viewBox.
- `samples.jsonl`: one record per sample, with:
  - tokens, keyed steps, timing;
  - `log10_p` (detection with the evaluation key) and `wrong_key_log10_p`;
  - number of distinct vertex/tangent descriptors.

## 5. Evaluation (CPU job, after generation)

```bash
sbatch hpc/evaluate_geosample.sbatch      # WORKERS defaults to the allocated CPUs
```

For every `results/geosample/iconshop_p*/` run, the job:
1. Runs the keyless attack suite and detection (`evaluate_geosample.py`), writing `results/geosample/<run>_attacks.jsonl`. The attacks are SVGO default/multipass/precision 2/precision 1, Scour, picosvg, rounding, translate, scale, rotate, mirror, group transform, aspect stretch, reorder, reverse, restart, subdivide, merge, split, deletion, crop, composition, noise, polyline simplification, and rasterize-and-retrace.
2. Writes summaries to `results/geosample/<run>_attacks.summary.{md,json}`:
   - detection rates at 1e-6 and 1e-3 per attack, overall and conditional on clean detection;
   - false-positive rates;
   - log10 p against the number of descriptors;
   - Mann-Whitney comparisons of plain vs marked token, vertex and path counts (a distribution-preservation sanity check).
3. Runs a key-randomized null over the plain samples (300 keys each), writing `results/geosample/<run>_null.json`.
4. Reruns the null over the 900 human icons, if the npm corpora are present, writing `results/geosample/null_corpus.json`.

Expected time: tens of minutes on 32 cores.

## 6. What to bring back

Either commit the results on the cluster and push to main:

```bash
git add -f experiments/results/geosample hpc/logs
git commit -m "IconShop geosample results (HPC)"
git push origin main
```

Or copy the directory back:

```bash
rsync -av cluster:svg-contourmark/experiments/results/geosample/ experiments/results/geosample/
```

The essential files are:
- `experiments/results/geosample/iconshop_p*/samples.jsonl` and the SVGs;
- `experiments/results/geosample/*_attacks.jsonl`, `*_attacks.summary.{md,json}`, `*_null.json`, `null_corpus.json`.

Then ask Claude to fill the Results section of `docs/inference-watermark.md`, `HANDOFF.md` and the paper from these files.

## 7. Notes and pitfalls

- **Distribution preservation is exact only at matched truncation.** The keyed choice is made on the decoder's own truncated distribution. Compare plain and marked runs only at the same top-p and temperature, as the scripts do.
- **Power depends on entropy, and is unknown for the corrected sampler.** IconShop's default top-p 0.5 gives about 1 nat per keyed step; top-p 0.9 gives about 1.9 (measured on a few prompts). The earlier pilot numbers (log10 p −3.5 to −4.7 at top-p 0.5) came from a sampler that reused keyed scores and did not preserve the output distribution. The corrected sampler is weaker (toy generator: −49 → −18). Expect many icons to miss 1e-6 at top-p 0.5; that is a result to report, not a failure of the run.
- **Failures propagate.** If any shard fails, the job exits non-zero and does not merge partial results. Samples record `completed` and `truncated`; summaries report attack and detector error counts next to every rate.
- **Determinism and resume:** every sample owns its random streams (model sampling and watermark), so results do not depend on batch size or shard layout. Each output directory records a run identity (`samples*.run.json`: configuration, code and checkpoint hashes, package versions). Resuming with a different configuration is refused; use a new output directory. Different GPUs and drivers can change logits slightly, so exact token streams are not portable between machines; detection does not depend on that.
- **The evaluation key** is fixed in `iconshop_geosample.py`, derived from a public string. It is fine for experiments; a real deployment uses `contourmark keygen`.
- **OmniSVG:** the adapter (`OmniSVGGrammar` + `GeoWatermarkLogitsProcessor`) is ready but untested on real weights. To try it on the A100s:
  1. Install the OmniSVG repo.
  2. Read the command and coordinate token ids from its `config.yaml`, set them in `OmniSVGGrammar(...)`, and pass `logits_processor=[GeoWatermarkLogitsProcessor(grammar, GeoWatermark(key), prompt_length=..., top_p=..., temperature=...)]` to `model.generate(do_sample=True, top_p=1.0, top_k=0)`. Truncation is applied inside the processor so the keyed choice sees the real distribution.
  3. Decode with `decode_tokens(grammar, ids).svg()` or OmniSVG's own renderer, then run `contourmark.geosample.detect`.
