#!/bin/bash
# Stable entry point for Colab: runs whichever experiment is currently queued.
#
# The notebook's "Full run" cell calls this script.  An open notebook does not
# update when the repository changes, so the queue lives here: after
# `git clone` / `git pull`, the same cell runs the newest experiment.
#
# Currently queued: the VARIANT experiment (colab/run2.sh): five top-p values,
# naive reuse and bias samplers, a second key, attacks, nulls and CLIP quality.
# The original two-configuration run is colab/run1.sh.
#
# Uses OUT, SEEDS, BATCH, WORKERS, STAGE from the environment (TOP_PS is
# ignored; run2.sh fixes its own grid).  Results go directly into OUT, so the
# notebook's results and download cells keep working; use a new OUT (a new
# RUN_NAME in the notebook) for each run.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
: "${OUT:?set OUT to a results directory}"
echo "=================================================================="
echo " Queued experiment: sampler variants (colab/run2.sh)"
echo " Results: $OUT   Seeds per prompt: ${SEEDS:-8}"
echo " Expect 12 configuration folders and RUN_SUMMARY.md when finished."
echo "=================================================================="
exec bash "$REPO/colab/run2.sh"
