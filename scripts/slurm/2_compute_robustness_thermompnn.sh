#!/bin/bash
#SBATCH --job-name=robustness_tmpnn
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:l4:1
#
# Compute ThermoMPNN mutational robustness for ATLAS proteins.
# Designed as a SLURM array job: each task processes a chunk of proteins.
#
# Usage:
#   sbatch --array=0-27 scripts/slurm/2_compute_robustness_thermompnn.sh
#   sbatch --array=0 scripts/slurm/2_compute_robustness_thermompnn.sh   # test
#
# Requires ThermoMPNN to be cloned with model weights available.
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
echo "Robustness: ThermoMPNN"
echo "Date:       $(date)"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Array task: ${TASK_ID}"
echo "Proteins:   ${BATCH_START} to ${BATCH_END}"
echo "============================================"

python "${REPO_DIR}/scripts/compute_robustness.py" \
    --scorer thermompnn \
    --atlas_dir "${ATLAS_DIR}" \
    --output_dir "${ROBUSTNESS_DIR}" \
    --batch \
    --batch_start "${BATCH_START}" \
    --batch_end "${BATCH_END}" \
    --device cuda \
    --skip_existing \
    --thermompnn_dir "${THERMOMPNN_DIR}"

echo ""
echo "Task ${TASK_ID} finished at $(date)"
