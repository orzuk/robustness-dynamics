#!/bin/bash
#SBATCH --job-name=robustness_pmpnn
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:l4:1
#
# Compute ProteinMPNN log-likelihood robustness for proteins.
# Same pattern as 2_compute_robustness_thermompnn.sh but with --scorer proteinmpnn.
#
# Uses the same ThermoMPNN repo and env (vanilla_model_weights/v_48_020.pt).
#
# Usage:
#   sbatch --array=0-27 scripts/slurm/11_compute_robustness_proteinmpnn.sh
#   sbatch --array=0 scripts/slurm/11_compute_robustness_proteinmpnn.sh   # test
#
# Override dataset with environment variables:
#   PMPNN_ATLAS_DIR=... PMPNN_OUTPUT_DIR=... sbatch ...
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

# Allow overriding dataset via env vars (used by master_pipeline for BBFlow/PDB)
INPUT_DIR="${PMPNN_ATLAS_DIR:-${ATLAS_DIR}}"
OUTPUT_DIR="${PMPNN_OUTPUT_DIR:-${ROBUSTNESS_DIR}}"

echo "============================================"
echo "Robustness: ProteinMPNN"
echo "Date:       $(date)"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Array task: ${TASK_ID}"
echo "Proteins:   ${BATCH_START} to ${BATCH_END}"
echo "Input:      ${INPUT_DIR}"
echo "Output:     ${OUTPUT_DIR}"
echo "============================================"

python "${REPO_DIR}/scripts/compute_robustness.py" \
    --scorer proteinmpnn \
    --atlas_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch \
    --batch_start "${BATCH_START}" \
    --batch_end "${BATCH_END}" \
    --device cuda \
    --skip_existing \
    --thermompnn_dir "${THERMOMPNN_DIR}"

echo ""
echo "Task ${TASK_ID} finished at $(date)"
