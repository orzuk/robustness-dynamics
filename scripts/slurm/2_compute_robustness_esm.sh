#!/bin/bash
#SBATCH --job-name=robustness_esm
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:l4:1
#
# Compute ESM-1v mutational robustness for ATLAS proteins.
# Designed as a SLURM array job: each task processes a chunk of proteins.
#
# Usage:
#   sbatch --array=0-27 scripts/slurm/2_compute_robustness_esm.sh
#   sbatch --array=0 scripts/slurm/2_compute_robustness_esm.sh         # test
#   sbatch --array=5,12,20 scripts/slurm/2_compute_robustness_esm.sh   # resume
#
# Skips already-computed proteins, so re-submitting is safe.
# ============================================================================

set -euo pipefail
if [[ -z "${REPO_DIR:-}" ]]; then
    _cfg="$(dirname "${BASH_SOURCE[0]}")/config.sh"
    [[ ! -f "$_cfg" ]] && _cfg="${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/config.sh"
    source "$_cfg"
fi

source "${VENV_DIR}/bin/activate"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
BATCH_START=$((TASK_ID * CHUNK_SIZE))
BATCH_END=$((BATCH_START + CHUNK_SIZE))

echo "============================================"
echo "Robustness: ESM-1v (masked marginals)"
echo "Date:       $(date)"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Array task: ${TASK_ID}"
echo "Proteins:   ${BATCH_START} to ${BATCH_END}"
echo "============================================"

python "${REPO_DIR}/scripts/compute_robustness.py" \
    --scorer esm1v \
    --atlas_dir "${ATLAS_DIR}" \
    --output_dir "${ROBUSTNESS_DIR}" \
    --batch \
    --batch_start "${BATCH_START}" \
    --batch_end "${BATCH_END}" \
    --device cuda \
    --skip_existing

echo ""
echo "Task ${TASK_ID} finished at $(date)"
