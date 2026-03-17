#!/bin/bash
#SBATCH --job-name=pdb_des_rob
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:l4:1
#
# Compute ThermoMPNN robustness for PDB de novo designed proteins.
# Designed as a SLURM array job: each task processes a chunk of proteins.
# GPU job. Run after 8_pdb_designs_download.sh completes.
#
# Usage:
#   sbatch --array=0-6 scripts/slurm/9_pdb_designs_robustness.sh   # all ~318 proteins
#   sbatch --array=0   scripts/slurm/9_pdb_designs_robustness.sh   # test (first 50)
#
# CHUNK_SIZE is set in config.sh (default 50).
# ~318 proteins / 50 = 7 chunks -> array=0-6
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
echo "PDB De Novo Designs — Robustness (ThermoMPNN)"
echo "Date:       $(date)"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Array task: ${TASK_ID}"
echo "Proteins:   ${BATCH_START} to ${BATCH_END}"
echo "Input:      ${PDB_DESIGNS_DIR}"
echo "Output:     ${PDB_DESIGNS_ROBUSTNESS}"
echo "============================================"

python "${REPO_DIR}/scripts/compute_robustness.py" \
    --scorer thermompnn \
    --atlas_dir "${PDB_DESIGNS_DIR}" \
    --output_dir "${PDB_DESIGNS_ROBUSTNESS}" \
    --batch \
    --batch_start "${BATCH_START}" \
    --batch_end "${BATCH_END}" \
    --device cuda \
    --skip_existing \
    --thermompnn_dir "${THERMOMPNN_DIR}"

echo ""
echo "Task ${TASK_ID} finished at $(date)"
