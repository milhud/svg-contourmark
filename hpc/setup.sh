#!/bin/bash
# One-time setup on the HPC LOGIN node (needs internet).  Run from the repo root:
#   bash hpc/setup.sh
# Creates .venv, installs Python deps (CUDA torch), installs svgo, and downloads
# the IconShop checkpoint + BERT tokenizer into a shared Hugging Face cache so
# compute nodes can run offline.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
: "${HF_HOME:=$REPO/.hf_cache}"   # override with a shared/scratch path if you like
export HF_HOME

PYTHON="${PYTHON:-python3}"       # needs Python >= 3.10 (e.g. `module load python/3.11`)
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# CUDA build of torch. Pick the index matching the cluster's CUDA (cu121 / cu124 ...).
.venv/bin/python -m pip install torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
.venv/bin/python -m pip install -e '.[test,eval]' transformers safetensors huggingface_hub einops matplotlib

# SVGO (Node >= 18) for the optimizer attacks.  `npm install` also pulls the
# icon corpora used by the blind-spectral experiments (~340 MB); for the
# IconShop experiments alone, `npm install svgo` is enough.
if command -v npm >/dev/null; then npm install; else echo "WARNING: npm not found; SVGO attacks will be recorded as errors"; fi

# Model + tokenizer into the HF cache / models dir (not committed).
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
snapshot_download("m1357l/iconshop-svg-generator", local_dir="models/iconshop",
                  allow_patterns=["*.json", "*.py", "*.safetensors", "*.pt", "deepsvg/**/*.py", "model/**/*.py"])
AutoTokenizer.from_pretrained("google/bert_uncased_L-12_H-512_A-8")
print("downloaded IconShop checkpoint and tokenizer")
PY

# Optional system tools for render/retrace attacks (librsvg + potrace).
# If your cluster has conda:  conda install -c conda-forge librsvg potrace nodejs
command -v rsvg-convert >/dev/null || echo "NOTE: rsvg-convert missing -> revectorize attacks will be skipped"
command -v potrace >/dev/null || echo "NOTE: potrace missing -> revectorize attacks will be skipped"

# Quick sanity check (CPU, ~30 s).
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_geosample.py tests/test_hf_generate.py
echo "setup complete; export HF_HOME=$HF_HOME in your job scripts"
