#!/bin/sh
# Follow-up experiments after the main test run. Run from experiments/.
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=../.venv/bin/python
$PY evaluate_blind.py --corpus corpus/test.json --output results/blind/test_numeric_lsb.jsonl --methods numeric_lsb --workers 9
for DELTA in 0.002 0.004 0.008 0.016; do
  $PY evaluate_blind.py --corpus corpus/dev.json --output results/blind/dev_delta_$DELTA.jsonl --methods spectral --workers 9 \
    --attacks identity svgo_default svgo_p2 round_2dp round_1dp noise_0.1pct noise_0.3pct polyline_0.2pct revectorize_1024 aspect_1.2 \
    --params "{\"delta\": $DELTA}"
done
$PY null_calibration.py --corpus corpus/test.json --keys 300 --workers 9 --output results/blind/null_calibration.json
$PY adaptive_attacks.py --corpus corpus/test.json --per-source 30 --workers 9 --output results/blind/adaptive.json
echo FOLLOWUPS-DONE
