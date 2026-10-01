#!/bin/sh
# Exercise every CPU experiment harness on tiny inputs (about 2-3 minutes).
# Run from experiments/:  ./smoke_all.sh
# Nothing here loads a model or needs a GPU.  Outputs go to a temp directory.
# Use this after any code change and before any long run (cluster or Colab).
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY="${PY:-$(cd .. && pwd)/.venv/bin/python}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
step() { printf '\n== %s ==\n' "$1"; }

[ -d ../node_modules/lucide-static ] || { echo "icon corpora missing: run 'npm install' in the repo root"; exit 1; }

step "unit tests"
(cd .. && $PY -m pytest -q)

step "corpus"
$PY build_corpus.py --per-source 2 --split dev --output "$OUT/corpus.json"

step "blind evaluation (spectral, extended suite, shared key)"
$PY evaluate_blind.py --corpus "$OUT/corpus.json" --output "$OUT/blind.jsonl" --methods spectral numeric_lsb --suite extended --key-mode shared --workers 4
$PY summarize_blind.py "$OUT/blind.jsonl" --markdown "$OUT/blind.md" --json "$OUT/blind.json" > /dev/null
$PY analyze_main.py "$OUT/blind.jsonl" "$OUT/analysis.json" > /dev/null
$PY bootstrap_ci.py "$OUT/blind.jsonl" --resamples 50 --output "$OUT/ci.json" > /dev/null

step "null calibration"
$PY null_calibration.py --corpus "$OUT/corpus.json" --keys 5 --workers 4 --output "$OUT/null.json" > /dev/null
$PY geosample_null.py --corpus "$OUT/corpus.json" --keys 5 --workers 4 --output "$OUT/geonull.json" > /dev/null

step "informed attacker"
$PY adaptive_attacks.py --corpus "$OUT/corpus.json" --per-source 1 --workers 4 --output "$OUT/adaptive.json" > /dev/null

step "robustness-distortion frontier"
$PY frontier.py --smoke --corpus "$OUT/corpus.json" --workers 4 --output "$OUT/frontier.json" > /dev/null

step "inference sampler lab"
$PY sampler_lab.py --smoke --output "$OUT/lab.json" > /dev/null

step "perceptual study kit"
$PY make_study.py --corpus "$OUT/corpus.json" --assets 2 --output "$OUT/study" > /dev/null
[ -f "$OUT/study/index.html" ]

step "figures"
$PY make_figures.py "$OUT/blind.jsonl" --out-dir "$OUT/figures" > /dev/null

step "command line"
$PY -m contourmark.cli capacity ../node_modules/simple-icons/icons/github.svg > /dev/null

printf '\nALL SMOKE CHECKS PASSED\n'
