#!/bin/bash
#SBATCH --job-name=esmfold_plddt
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:l4:1
#
# Compute per-residue pLDDT using ESMFold for designed proteins (BBFlow).
# GPU job — ESMFold inference requires ~4 GB VRAM.
#
# Usage:
#   sbatch scripts/slurm/5_compute_plddt_esmfold.sh
#   sbatch scripts/slurm/5_compute_plddt_esmfold.sh --max 10   # test on 10
#
# Requires: transformers (pip install transformers) in the venv
# ============================================================================

set -euo pipefail
if [[ -z "${REPO_DIR:-}" ]]; then
    _cfg="$(dirname "${BASH_SOURCE[0]}")/config.sh"
    [[ ! -f "$_cfg" ]] && _cfg="${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/config.sh"
    source "$_cfg"
fi

source "${VENV_DIR}/bin/activate"

# Parse args
EXTRA_FLAGS="--skip_existing"
for arg in "$@"; do
    case "$arg" in
        --max) shift; EXTRA_FLAGS="${EXTRA_FLAGS} --max_proteins $1" ;;
        [0-9]*) EXTRA_FLAGS="${EXTRA_FLAGS} --max_proteins $arg" ;;
    esac
done

echo "============================================"
echo "ESMFold pLDDT Computation (BBFlow)"
echo "Date:       $(date)"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Proteins:   ${BBFLOW_PROCESSED}/proteins"
echo "============================================"
echo ""

python "${REPO_DIR}/scripts/compute_plddt_esmfold.py" \
    --proteins_dir "${BBFLOW_PROCESSED}/proteins" \
    --device cuda \
    ${EXTRA_FLAGS}

echo ""
echo "ESMFold pLDDT finished at $(date)"
