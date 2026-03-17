#!/bin/bash
#SBATCH --job-name=bbflow_pipe
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l4:1
#
# Full BBFlow de novo designed protein analysis pipeline:
#   1. Extract per-residue RMSF from MD trajectories (CPU, ~5 min)
#   2. Compute robustness with ESM-1v + ThermoMPNN (GPU, ~10 min)
#   3. Correlate robustness vs. RMSF (CPU, ~1 min)
#
# Usage:
#   sbatch scripts/slurm/4_bbflow_pipeline.sh
#   sbatch scripts/slurm/4_bbflow_pipeline.sh --skip-rmsf    # if already extracted
#
# Requires: mdtraj (pip install mdtraj) in the venv
# ============================================================================

set -euo pipefail
if [[ -z "${REPO_DIR:-}" ]]; then
    _cfg="$(dirname "${BASH_SOURCE[0]}")/config.sh"
    [[ ! -f "$_cfg" ]] && _cfg="${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/config.sh"
    source "$_cfg"
fi

source "${VENV_DIR}/bin/activate"

# Parse args
SKIP_RMSF=false
for arg in "$@"; do
    case "$arg" in
        --skip-rmsf) SKIP_RMSF=true ;;
    esac
done

echo "============================================"
echo "BBFlow De Novo Protein Analysis Pipeline"
echo "Date:       $(date)"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"
echo ""

# ---- Step 1: Extract RMSF from trajectories ----
if [ "$SKIP_RMSF" = false ]; then
    echo "==== Step 1/3: Extracting RMSF from MD trajectories ===="
    python "${REPO_DIR}/scripts/prepare_bbflow.py" \
        --bbflow_dir "${BBFLOW_RAW}" \
        --output_dir "${BBFLOW_PROCESSED}"
    echo ""
else
    echo "==== Step 1/3: Skipping RMSF extraction (--skip-rmsf) ===="
    echo ""
fi

# ---- Step 2a: Compute ESM-1v robustness ----
echo "==== Step 2a/3: Computing ESM-1v robustness ===="
python "${REPO_DIR}/scripts/compute_robustness.py" \
    --scorer esm1v \
    --atlas_dir "${BBFLOW_PROCESSED}" \
    --output_dir "${BBFLOW_ROBUSTNESS}" \
    --batch \
    --batch_start 0 \
    --batch_end 200 \
    --device cuda \
    --skip_existing
echo ""

# ---- Step 2b: Compute ThermoMPNN robustness ----
echo "==== Step 2b/3: Computing ThermoMPNN robustness ===="
python "${REPO_DIR}/scripts/compute_robustness.py" \
    --scorer thermompnn \
    --atlas_dir "${BBFLOW_PROCESSED}" \
    --output_dir "${BBFLOW_ROBUSTNESS}" \
    --batch \
    --batch_start 0 \
    --batch_end 200 \
    --device cuda \
    --skip_existing \
    --thermompnn_dir "${THERMOMPNN_DIR}"
echo ""

# ---- Step 3: Correlate robustness vs RMSF ----
echo "==== Step 3/3: Correlating robustness vs dynamics ===="
python "${REPO_DIR}/scripts/correlate_robustness_dynamics.py" \
    --atlas_dir "${BBFLOW_PROCESSED}" \
    --robustness_dir "${BBFLOW_ROBUSTNESS}" \
    --scorer esm1v thermompnn \
    --output_dir "${BBFLOW_ANALYSIS}" \
    --no_dssp
echo ""

echo "============================================"
echo "BBFlow pipeline finished at $(date)"
echo "Results in: ${BBFLOW_ANALYSIS}"
echo "============================================"
