#!/bin/bash
# Variant experiment: the polygon scheme, sampler variants, top-p grid, a
# second key, and CLIP quality.  Each configuration is evaluated as soon as it
# is generated, so a session that stops early still leaves usable results.
# One GPU.  From the repo root:
#   OUT=/content/drive/MyDrive/contourmark_runs/run2 bash colab/run2.sh
# Environment variables:
#   OUT      results directory (new, on Drive)
#   SEEDS    samples per prompt (30 prompts), default 8
#   BATCH    default 16
#   WORKERS  CPU workers for evaluation, default 2
#   STAGE    generate | evaluate | all (default)
#   EXTRA    extra generator arguments, for testing only
# Re-running continues where it stopped.  16 configurations: roughly 3 hours
# on an A100 at SEEDS=8, half that at SEEDS=4.  The first four are the ones
# that matter most.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python}"
OUT="${OUT:?set OUT to a results directory}"
SEEDS="${SEEDS:-8}"; BATCH="${BATCH:-16}"; WORKERS="${WORKERS:-2}"; STAGE="${STAGE:-all}"; EXTRA="${EXTRA:-}"
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"
cd "$REPO/experiments"
git -C "$REPO" rev-parse HEAD > "$OUT/commit.txt"
$PY -c "import torch, transformers, numpy, scipy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('transformers', transformers.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)" | tee "$OUT/environment.txt"
# SVGO must work before evaluation: earlier Colab runs silently lost every SVGO row.
svgo_ok() { (cd "$REPO" && node --input-type=module -e "import('svgo').then(() => console.log('svgo ok'))" 2>/dev/null | grep -q "svgo ok"); }
if ! svgo_ok; then
  echo "svgo missing: installing"
  (cd "$REPO" && (npm install --silent --no-audit --no-fund || npm install --silent --no-audit --no-fund --no-save svgo)) || true
fi
if svgo_ok; then
  echo "svgo: ok" | tee -a "$OUT/environment.txt"
else
  echo "svgo: MISSING after install attempt (node: $(command -v node || echo none), $(node --version 2>/dev/null || true))" | tee -a "$OUT/environment.txt"
  echo "Fix: cd $REPO && npm install ; or set ALLOW_NO_SVGO=1 to continue without SVGO attacks."
  [ "${ALLOW_NO_SVGO:-0}" = "1" ] || exit 1
fi

evaluate() {  # evaluate NAME: attacks, summary, null; then refresh the run table
  local name="$1" run="$OUT/$1"
  [ -f "$run/samples.jsonl" ] || return 0
  $PY evaluate_geosample.py --samples-dir "$run" --output "$OUT/${name}_attacks.jsonl" --workers "$WORKERS"
  $PY summarize_geosample.py "$OUT/${name}_attacks.jsonl" --markdown "$OUT/${name}_attacks.summary.md" --json "$OUT/${name}_attacks.summary.json" > /dev/null
  $PY geosample_null.py --samples-dir "$run" --keys 300 --workers "$WORKERS" --output "$OUT/${name}_null.json" > /dev/null
  $PY summarize_run.py "$OUT" --markdown "$OUT/RUN_SUMMARY.md" --json "$OUT/run_summary.json" > /dev/null
  echo "---- $name evaluated ----"; grep -E "^\| (identity|svgo_default|rotate_30) " "$OUT/${name}_attacks.summary.md" || true
}

run() {  # run NAME top_p [generator args...]: generate (resumable), then evaluate at once
  local name="$1" top_p="$2"; shift 2
  if [ "$STAGE" = "generate" ] || [ "$STAGE" = "all" ]; then
    $PY iconshop_geosample.py --output "$OUT/$name" --top-p "$top_p" --samples-per-prompt "$SEEDS" --batch "$BATCH" "$@" $EXTRA
    echo "$name: $(wc -l < "$OUT/$name/samples.jsonl") samples"
  fi
  if [ "$STAGE" = "evaluate" ] || [ "$STAGE" = "all" ]; then
    evaluate "$name"
  fi
}

# Ordered by value, so an interrupted session still answers the main question:
# does the polygon scheme lift detection at the model's default top-p?
# "mask" = distribution-preserving vertex scheme (plain + marked; the plain
# samples are shared by every marked-only configuration at the same top-p).
run mask_p05 0.5
run poly_p05 0.5 --scheme polygon --marked-only
run mask_p09 0.9
run poly_p09 0.9 --scheme polygon --marked-only
run mask_p07 0.7
run poly_p07 0.7 --scheme polygon --marked-only
# Non-preserving samplers at the default top-p: what does extra power cost?
run bias4_p05 0.5 --mode bias --delta 4 --marked-only
run allow_p05 0.5 --reuse allow --marked-only
run bias2_p05 0.5 --mode bias --delta 2 --marked-only
# A second key, on the scheme that matters.
run poly_p05_key2 0.5 --scheme polygon --key-label k2 --marked-only
# Remaining grid.
run mask_p08 0.8
run mask_p10 1.0
run poly_p10 1.0 --scheme polygon --marked-only
run allow_p09 0.9 --reuse allow --marked-only
run bias2_p09 0.9 --mode bias --delta 2 --marked-only
run bias4_p09 0.9 --mode bias --delta 4 --marked-only

if [ "$STAGE" = "evaluate" ] || [ "$STAGE" = "all" ]; then
  DIRS=()
  for RUN in "$OUT"/*/; do [ -f "${RUN}samples.jsonl" ] && DIRS+=("${RUN%/}"); done
  if [ "${#DIRS[@]}" -gt 0 ]; then
    $PY clip_quality.py --samples-dir "${DIRS[@]}" --output "$OUT/clip_quality.json"
    $PY summarize_run.py "$OUT" --markdown "$OUT/RUN_SUMMARY.md" --json "$OUT/run_summary.json"
  else
    echo "no generated samples found in $OUT; nothing to evaluate"
  fi
fi
echo "DONE. Results in $OUT"
