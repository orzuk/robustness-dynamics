#!/bin/bash
#SBATCH --job-name=frust_mpnn
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:l4:1
#
# Compute per-residue FrustraMPNN frustration for a dataset.
# Same array/batch pattern as 11_compute_robustness_proteinmpnn.sh, but uses
# the dedicated FrustraMPNN env + checkpoint and the compute_frustration.py driver.
# Produces two scorer views per protein: frustrampnn/ and frustrampnn_native/.
#
# Prereq: run 0b_setup_frustrampnn_env.sh once, and place the checkpoint at
#         $FRUSTRAMPNN_CHECKPOINT.
#
# Usage (adjust array to ceil(n_proteins / CHUNK_SIZE) - 1):
#   sbatch --array=0 scripts/slurm/14_compute_robustness_frustrampnn.sh          # smoke: first 50
#   sbatch --array=0-38 scripts/slurm/14_compute_robustness_frustrampnn.sh        # ATLAS (1938/50)
#
# Override dataset via env vars (same convention as the ProteinMPNN script):
#   FRUST_INPUT_DIR=$BBFLOW_PROCESSED FRUST_OUTPUT_DIR=$BBFLOW_ROBUSTNESS sbatch ...
# ============================================================================
set -euo pipefail
if [[ -z "${REPO_DIR:-}" ]]; then
    _cfg="$(dirname "${BASH_SOURCE[0]}")/config.sh"
    [[ ! -f "$_cfg" ]] && _cfg="${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/config.sh"
    source "$_cfg"
fi

# FrustraMPNN uses its OWN env, not VENV_DIR.
source "${FRUSTRAMPNN_ENV}/bin/activate"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
BATCH_START=$((TASK_ID * CHUNK_SIZE))
BATCH_END=$((BATCH_START + CHUNK_SIZE))

# Default to ATLAS; override with FRUST_INPUT_DIR / FRUST_OUTPUT_DIR for other datasets.
INPUT_DIR="${FRUST_INPUT_DIR:-${ATLAS_DIR}}"
OUTPUT_DIR="${FRUST_OUTPUT_DIR:-${ROBUSTNESS_DIR}}"

echo "============================================"
echo "FrustraMPNN frustration"
echo "Date:       $(date)"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Array task: ${TASK_ID}   Proteins: ${BATCH_START}..${BATCH_END}"
echo "Input:      ${INPUT_DIR}"
echo "Output:     ${OUTPUT_DIR}  (scorer dirs frustrampnn/ + frustrampnn_native/)"
echo "Checkpoint: ${FRUSTRAMPNN_CHECKPOINT}"
echo "============================================"

python "${REPO_DIR}/scripts/compute_frustration.py" \
    --atlas_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch \
    --batch_start "${BATCH_START}" \
    --batch_end "${BATCH_END}" \
    --device cuda \
    --skip_existing \
    --frustrampnn_checkpoint "${FRUSTRAMPNN_CHECKPOINT}"

echo ""
echo "Task ${TASK_ID} finished at $(date)"
