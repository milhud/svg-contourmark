#!/bin/bash
#SBATCH --job-name=geosample-gen
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=hpc/logs/%x-%j.out
# Generate plain and watermarked IconShop samples on 4 GPUs (one shard per GPU).
# Submit from the repo root:  sbatch hpc/generate_iconshop.sbatch
# Override defaults:          TOP_PS="0.5 0.9" SEEDS=16 sbatch hpc/generate_iconshop.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export HF_HOME="${HF_HOME:-$(pwd)/.hf_cache}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
TOP_PS="${TOP_PS:-0.5 0.9 1.0}"
SEEDS="${SEEDS:-16}"        # samples per prompt (30 prompts) -> 480 plain + 480 marked per top-p
BATCH="${BATCH:-32}"
GPUS="${GPUS:-4}"
mkdir -p hpc/logs
cd experiments
for TOP_P in $TOP_PS; do
  OUT="results/geosample/iconshop_p${TOP_P/./}"
  mkdir -p "$OUT"
  PIDS=()
  for SHARD in $(seq 0 $((GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$SHARD ../.venv/bin/python iconshop_geosample.py \
      --output "$OUT" --top-p "$TOP_P" --samples-per-prompt "$SEEDS" --batch "$BATCH" \
      --device cuda --shard "$SHARD" --num-shards "$GPUS" > "../hpc/logs/gen_p${TOP_P}_shard${SHARD}.log" 2>&1 &
    PIDS+=("$!")
  done
  FAILED=0
  for PID in "${PIDS[@]}"; do
    wait "$PID" || FAILED=1
  done
  if [ "$FAILED" -ne 0 ]; then
    echo "generation shard failed; refusing to merge partial results" >&2
    exit 1
  fi
  cat "$OUT"/samples.shard*.jsonl > "$OUT/samples.jsonl"
  echo "top_p=$TOP_P: $(wc -l < "$OUT/samples.jsonl") samples in $OUT"
done
