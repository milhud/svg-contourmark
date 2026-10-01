#!/bin/bash
# Generate and evaluate IconShop geosample runs on ONE GPU (Colab or any box).
# Run from the repo root:
#   OUT=/content/drive/MyDrive/contourmark_runs/run1 bash colab/run_single_gpu.sh
# Environment variables:
#   OUT      results directory (put it on Google Drive so it survives disconnects)
#   TOP_PS   top-p values, default "0.5 0.9"
#   SEEDS    samples per prompt (30 prompts), default 4 -> 120 plain + 120 marked per top-p
#   BATCH    generation batch size, default 16
#   WORKERS  CPU workers for the evaluation, default 2
#   STAGE    "generate", "evaluate", or "all" (default)
#   EXTRA    extra arguments for the generator, for testing only
#            (e.g. EXTRA="--prompts star heart --max-tokens 40")
# Re-running continues where it stopped.  A run directory is tied to its
# configuration and code version; to change anything, use a new OUT.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python}"
OUT="${OUT:?set OUT to a results directory}"
TOP_PS="${TOP_PS:-0.5 0.9}"
SEEDS="${SEEDS:-4}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-2}"
STAGE="${STAGE:-all}"
EXTRA="${EXTRA:-}"
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"
cd "$REPO/experiments"
git -C "$REPO" rev-parse HEAD > "$OUT/commit.txt"
$PY -c "import torch, transformers, numpy, scipy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('transformers', transformers.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)" | tee "$OUT/environment.txt"

if [ "$STAGE" = "generate" ] || [ "$STAGE" = "all" ]; then
  for TOP_P in $TOP_PS; do
    RUN="$OUT/iconshop_p${TOP_P/./}"
    $PY iconshop_geosample.py --output "$RUN" --top-p "$TOP_P" --samples-per-prompt "$SEEDS" --batch "$BATCH" $EXTRA
    echo "top_p=$TOP_P: $(wc -l < "$RUN/samples.jsonl") samples"
  done
fi

if [ "$STAGE" = "evaluate" ] || [ "$STAGE" = "all" ]; then
  for RUN in "$OUT"/iconshop_p*/; do
    RUN="${RUN%/}"; NAME="$(basename "$RUN")"
    [ -f "$RUN/samples.jsonl" ] || continue
    $PY evaluate_geosample.py --samples-dir "$RUN" --output "$OUT/${NAME}_attacks.jsonl" --workers "$WORKERS"
    $PY summarize_geosample.py "$OUT/${NAME}_attacks.jsonl" --markdown "$OUT/${NAME}_attacks.summary.md" --json "$OUT/${NAME}_attacks.summary.json" > /dev/null
    $PY geosample_null.py --samples-dir "$RUN" --keys 300 --workers "$WORKERS" --output "$OUT/${NAME}_null.json" > /dev/null
    echo "---- $NAME ----"; head -40 "$OUT/${NAME}_attacks.summary.md"
  done
fi
echo "DONE. Results in $OUT"
