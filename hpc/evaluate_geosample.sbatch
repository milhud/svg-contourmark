#!/bin/bash
#SBATCH --job-name=geosample-eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=hpc/logs/%x-%j.out
# CPU-only: attacks + detection, key-randomized nulls, summaries.
# Submit after generation finishes:  sbatch hpc/evaluate_geosample.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}/experiments"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
PY=../.venv/bin/python
for RUN in results/geosample/iconshop_p*/; do
  RUN="${RUN%/}"; NAME="$(basename "$RUN")"
  [ -f "$RUN/samples.jsonl" ] || { echo "skip $RUN (no samples.jsonl)"; continue; }
  $PY evaluate_geosample.py --samples-dir "$RUN" --output "results/geosample/${NAME}_attacks.jsonl" --workers "$WORKERS"
  $PY summarize_geosample.py "results/geosample/${NAME}_attacks.jsonl" \
      --markdown "results/geosample/${NAME}_attacks.summary.md" --json "results/geosample/${NAME}_attacks.summary.json"
  $PY geosample_null.py --samples-dir "$RUN" --keys 300 --workers "$WORKERS" --output "results/geosample/${NAME}_null.json"
done
# Null on the 900 human icons (needs `npm install` corpora); skipped if absent.
if [ -f corpus/test.json ] && [ -d ../node_modules/lucide-static ]; then
  $PY geosample_null.py --corpus corpus/test.json --keys 300 --workers "$WORKERS" --output results/geosample/null_corpus.json
fi
echo "evaluation complete"
