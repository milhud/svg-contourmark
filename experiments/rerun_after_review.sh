#!/bin/sh
# Re-run experiments affected by the 2026-10-01 review fixes. Run from experiments/.
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=../.venv/bin/python
V=results/blind/v2
mkdir -p $V results/geosample
$PY evaluate_blind.py --corpus corpus/test.json --output $V/test_spectral.jsonl --methods spectral --workers 9
$PY evaluate_blind.py --corpus corpus/llm.json --output $V/llm_strict.jsonl --methods spectral --workers 9 --visibility strict
$PY evaluate_blind.py --corpus corpus/llm.json --output $V/llm_render.jsonl --methods spectral --workers 9 --visibility render
for DELTA in 0.002 0.004 0.008 0.016; do
  $PY evaluate_blind.py --corpus corpus/dev.json --output $V/dev_delta_$DELTA.jsonl --methods spectral --workers 9 \
    --attacks identity svgo_default svgo_p2 round_2dp round_1dp noise_0.1pct noise_0.3pct polyline_0.2pct revectorize_1024 aspect_1.2 \
    --params "{\"delta\": $DELTA}"
done
$PY null_calibration.py --corpus corpus/test.json --keys 300 --workers 9 --output $V/null_calibration.json
$PY adaptive_attacks.py --corpus corpus/test.json --per-source 30 --workers 9 --output $V/adaptive.json
$PY geosample_null.py --corpus corpus/test.json --keys 300 --workers 9 --output results/geosample/null_corpus.json
echo RERUN-DONE
