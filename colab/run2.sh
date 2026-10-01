#!/bin/bash
# Run 2: sampler variants, more top-p values, a second key, and CLIP quality.
# One GPU.  From the repo root:
#   OUT=/content/drive/MyDrive/contourmark_runs/run2 bash colab/run2.sh
# Environment variables:
#   OUT      results directory (new, on Drive)
#   SEEDS    samples per prompt (30 prompts), default 8
#   BATCH    default 16
#   WORKERS  CPU workers for evaluation, default 2
#   STAGE    generate | evaluate | all (default)
#   EXTRA    extra generator arguments, for testing only
# Re-running continues where it stopped.  About 2.5 hours on an A100 at SEEDS=8;
# use SEEDS=4 for about half that.
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

gen() {  # gen NAME top_p [generator args...]
  local name="$1" top_p="$2"; shift 2
  $PY iconshop_geosample.py --output "$OUT/$name" --top-p "$top_p" --samples-per-prompt "$SEEDS" --batch "$BATCH" "$@" $EXTRA
  echo "$name: $(wc -l < "$OUT/$name/samples.jsonl") samples"
}

if [ "$STAGE" = "generate" ] || [ "$STAGE" = "all" ]; then
  # A. Distribution-preserving sampler across top-p (plain + marked).
  for TOP_P in 0.5 0.7 0.8 0.9 1.0; do gen "mask_p${TOP_P/./}" "$TOP_P"; done
  # B. Variants at the default and at a high top-p (marked only; plain is shared with A).
  for TOP_P in 0.5 0.9; do
    gen "allow_p${TOP_P/./}" "$TOP_P" --reuse allow --marked-only
    gen "bias2_p${TOP_P/./}" "$TOP_P" --mode bias --delta 2 --marked-only
    gen "bias4_p${TOP_P/./}" "$TOP_P" --mode bias --delta 4 --marked-only
  done
  # C. A second key, to see how much results depend on the key.
  gen "mask_p09_key2" 0.9 --key-label k2 --marked-only
fi

if [ "$STAGE" = "evaluate" ] || [ "$STAGE" = "all" ]; then
  DIRS=()
  for RUN in "$OUT"/*/; do
    RUN="${RUN%/}"; NAME="$(basename "$RUN")"
    [ -f "$RUN/samples.jsonl" ] || continue
    DIRS+=("$RUN")
    $PY evaluate_geosample.py --samples-dir "$RUN" --output "$OUT/${NAME}_attacks.jsonl" --workers "$WORKERS"
    $PY summarize_geosample.py "$OUT/${NAME}_attacks.jsonl" --markdown "$OUT/${NAME}_attacks.summary.md" --json "$OUT/${NAME}_attacks.summary.json" > /dev/null
    $PY geosample_null.py --samples-dir "$RUN" --keys 300 --workers "$WORKERS" --output "$OUT/${NAME}_null.json" > /dev/null
    echo "---- $NAME ----"; sed -n '1,12p' "$OUT/${NAME}_attacks.summary.md"
  done
  if [ "${#DIRS[@]}" -gt 0 ]; then
    $PY clip_quality.py --samples-dir "${DIRS[@]}" --output "$OUT/clip_quality.json"
    $PY summarize_run.py "$OUT" --markdown "$OUT/RUN_SUMMARY.md" --json "$OUT/run_summary.json"
  else
    echo "no generated samples found in $OUT; nothing to evaluate"
  fi
fi
echo "DONE. Results in $OUT"
