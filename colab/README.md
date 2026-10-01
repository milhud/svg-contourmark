# Running the IconShop experiment in Google Colab

These instructions are written for an AI assistant (for example Gemini in Colab) or a person running the notebook. They cover one experiment: generating icons with the real IconShop model, with and without the inference-time watermark, and evaluating detection.

**Fastest path:** open `colab/iconshop_geosample.ipynb` in Colab, choose a GPU runtime, and run the cells top to bottom. The rest of this file explains each step, what a correct result looks like, and what to do when something fails.

## Rules for the assistant running this

1. **Do not edit any file in the repository**, and do not change thresholds, keys, or sampler parameters. The point is to measure the code as committed. If something fails, report it; do not patch it.
2. **Stop and report on the first error.** Give the exact command and the last 30 lines of output. Do not retry with altered settings, except where a section below says to.
3. **Do not commit or upload model weights.** `models/` is git-ignored; leave it that way.
4. **Weak results are valid results.** Many marked icons may not reach the detection threshold at top-p 0.5. Report the numbers as they are.
5. **Always report:** the commit hash, GPU type, package versions, how many samples were generated, and any error counts in the summaries.

## 0. Runtime

- Runtime → Change runtime type → **GPU** (T4 is enough; L4 or A100 is faster).
- The run needs about 2 GB of disk and internet access for setup.

## 1. Persistent storage (recommended)

Colab sessions disconnect. Put results on Google Drive so a rerun continues where it stopped.

```python
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/contourmark_runs/run1'
```

Without Drive, use `OUT = '/content/runs/run1'` and download the zip at the end before the session ends.

## 2. Clone and install

```bash
cd /content
git clone https://github.com/milhud/svg-contourmark.git
cd svg-contourmark
git rev-parse HEAD
# Colab already has a CUDA build of torch. Do NOT reinstall torch.
pip install -q -e '.[test,eval]' transformers safetensors huggingface_hub einops matplotlib
apt-get -qq install -y librsvg2-bin potrace > /dev/null
npm install --silent
node --input-type=module -e "import('svgo').then(() => console.log('svgo ok'))"
```

- `librsvg2-bin` and `potrace` are used by the rasterize-and-retrace attacks.
- `npm install` provides `svgo`, the optimizer attack (it also downloads the icon sets used by other experiments, about 340 MB). The last line must print `svgo ok`. In run 1 SVGO was not installed and every SVGO row was lost.
- If `apt-get` or `npm` is unavailable, continue. Those attacks will be counted as errors in the summary, which is acceptable, but say so in the report.

## 3. Download the model

```python
from huggingface_hub import snapshot_download
snapshot_download("m1357l/iconshop-svg-generator", local_dir="models/iconshop",
                  allow_patterns=["*.json", "*.py", "*.safetensors", "*.pt", "deepsvg/**/*.py", "model/**/*.py"])
```

This is about 540 MB (licence CC BY-NC-SA 4.0, research use). The text tokenizer (`google/bert_uncased_L-12_H-512_A-8`) downloads automatically on first use.

## 4. Check the installation (CPU, under 1 minute)

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/test_geosample.py tests/test_review_regressions.py tests/test_hf_generate.py
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

- **Expected:** all tests pass, and the second line prints `True` and a GPU name.
- If tests fail, stop and report.
- If CUDA is `False`, the runtime is not a GPU runtime: fix that first.

## 5. Smoke test (a few minutes)

```bash
cd experiments
python iconshop_geosample.py --output /content/smoke --prompts star heart house car --samples-per-prompt 2 --batch 8
python - <<'EOF'
import json
for line in open('/content/smoke/samples.jsonl'):
    r = json.loads(line)
    print(r['marked'], r['prompt'], len(r['tokens']), 'truncated' if r['truncated'] else 'complete', round(r['log10_p'], 2))
EOF
cd ..
```

**Expected:**
- 16 lines: 8 with `False` (plain) and 8 with `True` (marked).
- Plain rows have `log10_p` near 0, normally above −2.
- Marked rows are usually more negative, but at top-p 0.5 many will be only −1 to −4. That is expected for this sampler.

Note the time this took; it predicts the full run. If the loader raises `checkpoint mismatch`, stop and report.

## 6. Full run

```bash
OUT=/content/drive/MyDrive/contourmark_runs/run1 TOP_PS="0.5 0.9" SEEDS=4 BATCH=16 bash colab/run_single_gpu.sh
```

- **Size:** 30 prompts × 4 seeds × (plain + marked) × 2 top-p values = 480 icons.
- **Stages:** the script generates, then runs the attack suite, the summaries, and a key-randomized null.
- **If time allows,** increase `SEEDS` to 8 or 16 and add `1.0` to `TOP_PS`, **in a new `OUT` directory**. A run directory is locked to its configuration.
- **If the session disconnects,** run the same command again with the same `OUT`. It skips finished samples.
- **If you see `run configuration changed`,** the directory belongs to a different configuration or code version. Use a new `OUT`.
- **To run in two steps,** use `STAGE=generate` and later `STAGE=evaluate`.

## 7. What to report

Print the summaries:

```bash
cat $OUT/commit.txt $OUT/environment.txt
cat $OUT/iconshop_p05_attacks.summary.md
cat $OUT/iconshop_p09_attacks.summary.md
python -c "import json,sys; [print(p, json.load(open(p))['levels']) for p in sys.argv[1:]]" $OUT/iconshop_p05_null.json $OUT/iconshop_p09_null.json
```

Report these, per top-p value:

1. Marked samples detected at p ≤ 1e-6 and at p ≤ 1e-3, with no attack.
2. The same after SVGO, Scour, rounding, rotation, and subdivision (the table rows), including the `attempted` and `errors` columns.
3. False positives: plain samples with the key, and marked samples with the wrong key.
4. How many generations were truncated at the token limit.
5. Median number of distinct descriptors, and the detection rate against that number (the last table).
6. The plain-vs-marked comparison lines (token, vertex, path counts).

## 8. Save the results

With Drive mounted they are already saved. Otherwise:

```bash
cd /content && zip -qr contourmark_results.zip runs && ls -lh contourmark_results.zip
```

Then download the zip from the Colab file browser. The files that matter are:

- `iconshop_p*/samples.jsonl`, `iconshop_p*/samples.run.json`, and the `.svg` files;
- `*_attacks.jsonl`, `*_attacks.summary.md`, `*_attacks.summary.json`, `*_null.json`;
- `commit.txt` and `environment.txt`.

To return them to the project, copy the `OUT` directory into `experiments/results/geosample/` of a local clone.

## Run 2 (after run 1 has succeeded)

Run 2 answers the questions run 1 raised. It needs the setup from sections 0–4 done in this session (clone, pip install, model download).

**Paste this as one new Colab cell and run it.** Do not use the "Full run" cell from section 6: that repeats run 1. An already-open notebook does not update itself when the repository changes, so paste the cell below even if the notebook has no "section 9".

```
%cd /content/svg-contourmark
!git pull -q && git rev-parse HEAD
!OUT=/content/drive/MyDrive/contourmark_runs/run2_variants SEEDS=8 BATCH=16 bash colab/run2.sh
```

Without Drive, use `OUT=/content/runs/run2_variants`. The script installs SVGO itself if it is missing and stops with a clear message if that fails. A correct start prints `svgo: ok` and then `mask_p05: ... samples`. When it ends, `OUT` contains twelve configuration folders (`mask_p05` … `mask_p09_key2`) and `RUN_SUMMARY.md`; if you only see `iconshop_p05` and `iconshop_p09`, the wrong cell was run.

What it generates (30 prompts × `SEEDS` each):

| Group | Configurations | Purpose |
|---|---|---|
| A | distribution-preserving sampler at top-p 0.5, 0.7, 0.8, 0.9, 1.0 (plain + marked) | Detection and quality against top-p |
| B | at top-p 0.5 and 0.9, marked only: naive reuse; bias δ=2; bias δ=4 | Does a non-preserving sampler buy power on a real model, and at what quality cost? |
| C | second key at top-p 0.9, marked only | How much do results depend on the key? |

Then, for every configuration, it runs the attack suite (SVGO must be installed; the script stops if it is not), a key-randomized null where plain samples exist, and a CLIP prompt-retrieval quality score. It finishes with one table, `RUN_SUMMARY.md`.

- **Time:** about 2.5 hours on an A100 at `SEEDS=8`; use `SEEDS=4` for about half. It resumes if interrupted.
- **Downloads:** the CLIP model `openai/clip-vit-base-patch32` (about 600 MB) on first use.
- **Report:** print `cat $OUT/RUN_SUMMARY.md`, `cat $OUT/environment.txt $OUT/commit.txt`, and the `levels` of each `*_null.json`. Then zip and download `OUT` as in section 8.
- Group B results may show stronger detection with lower CLIP accuracy. That trade-off is the measurement; report both numbers.

## Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `CUDA out of memory` | Batch too large for the GPU | Rerun with `BATCH=8` in a **new** `OUT` |
| Generation is very slow | The model has no key-value cache; long icons cost more | Lower `SEEDS`; do not change `--max-tokens` |
| `run configuration changed` | Reusing a directory with different settings or a different commit | Use a new `OUT` |
| `checkpoint mismatch` | Model files incomplete | Re-run the download step, then report if it persists |
| SVGO rows show errors | `npm install svgo` failed | Report it; other rows remain valid |
| `revectorize_*` rows show errors | `potrace` or `rsvg-convert` missing | Report it; other rows remain valid |
| Drive not mounted after reconnect | Session reset | Re-run the mount cell, then the same run command |
