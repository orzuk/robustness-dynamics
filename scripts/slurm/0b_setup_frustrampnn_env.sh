#!/bin/bash
# ============================================================================
# 0b_setup_frustrampnn_env.sh — one-time setup for the FrustraMPNN scorer.
#
# FrustraMPNN (Beining et al. 2026) ships as a pip package `frustrampnn` that
# pulls torch + torch-geometric. These can conflict with the ThermoMPNN env,
# so we build a DEDICATED venv (FRUSTRAMPNN_ENV) instead of reusing VENV_DIR.
#
# Run once on the cluster (login node or an interactive GPU session):
#   bash scripts/slurm/0b_setup_frustrampnn_env.sh
#
# After it finishes, put the model checkpoint at FRUSTRAMPNN_CHECKPOINT
# (download from the FrustraMPNN Zenodo, doi 10.5281/zenodo.17978321).
# ============================================================================
set -euo pipefail

if [[ -z "${FRUSTRAMPNN_ENV:-}" ]]; then
    _cfg="$(dirname "${BASH_SOURCE[0]}")/config.sh"
    source "$_cfg"
fi

echo "=== FrustraMPNN env setup ==="
echo "Env:        ${FRUSTRAMPNN_ENV}"
echo "Checkpoint: ${FRUSTRAMPNN_CHECKPOINT}"

# --- 1. Create the venv (use the cluster's Python 3.9+; load a module if needed) ---
# module load python/3.10   # <-- uncomment / adjust for your cluster if required
python3 -m venv "${FRUSTRAMPNN_ENV}"
source "${FRUSTRAMPNN_ENV}/bin/activate"
python -m pip install --upgrade pip wheel

# --- 2. Install FrustraMPNN FROM SOURCE (not on PyPI as of 2026-07) ---
# If the cluster needs a CUDA-specific torch wheel, install torch FIRST, e.g.:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
FRUST_SRC="$(dirname "${REPO_DIR}")/frustraMPNN"   # clone next to robustness-dynamics
if [[ ! -d "${FRUST_SRC}/.git" ]]; then
    git clone https://github.com/schoederlab/frustraMPNN.git "${FRUST_SRC}"
fi
pip install -e "${FRUST_SRC}[all]" || pip install -e "${FRUST_SRC}"
pip install pandas   # compute_frustration.py reads model.predict() DataFrame output

# --- 3. Model checkpoint ---
mkdir -p "$(dirname "${FRUSTRAMPNN_CHECKPOINT}")"
if [[ -f "${FRUSTRAMPNN_CHECKPOINT}" ]]; then
    echo "Checkpoint already present: ${FRUSTRAMPNN_CHECKPOINT}"
else
    cat <<EOF

  >>> ACTION REQUIRED: download the FrustraMPNN checkpoint (8.1 GB zip) <<<
  Weights live inside full_training_runs_and_weights.zip on Zenodo
  (doi 10.5281/zenodo.17978321). Download, unzip, and symlink the BEST model
  (paper's best = balanced FireProt + ESM + light attention, highest
   val_frustration_spearman) to \${FRUSTRAMPNN_CHECKPOINT}:
      mkdir -p "\$(dirname "${FRUSTRAMPNN_CHECKPOINT}")"
      wget -O /tmp/frust_weights.zip \\
        "https://zenodo.org/records/17978321/files/full_training_runs_and_weights.zip?download=1"
      unzip -o /tmp/frust_weights.zip -d "\$(dirname "${FRUSTRAMPNN_CHECKPOINT}")/"
      find "\$(dirname "${FRUSTRAMPNN_CHECKPOINT}")" -name '*.ckpt' | sort
      ln -sf "<best_fireprot_ESM_LT_*.ckpt>" "${FRUSTRAMPNN_CHECKPOINT}"

EOF
fi

# --- 4. Import smoke test ---
python - <<'PY'
try:
    import frustrampnn, torch, torch_geometric, pandas  # noqa
    print("  OK: frustrampnn + torch + torch_geometric + pandas import cleanly")
    print("  torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
except Exception as e:
    print("  IMPORT PROBLEM:", e)
    raise
PY

echo "=== Done. Next: smoke-test on one PDB (see run plan step 1). ==="
