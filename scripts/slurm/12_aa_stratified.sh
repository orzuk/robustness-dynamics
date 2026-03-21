#!/bin/bash
#SBATCH --job-name=aa_stratified
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#
# Run amino-acid-stratified analysis on all datasets.
# CPU-only, no GPU needed.
#
# Usage:
#   sbatch scripts/slurm/12_aa_stratified.sh
# ============================================================================

set -euo pipefail
if [[ -z "${REPO_DIR:-}" ]]; then
    _cfg="$(dirname "${BASH_SOURCE[0]}")/config.sh"
    [[ ! -f "$_cfg" ]] && _cfg="${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/config.sh"
    source "$_cfg"
fi

source "${VENV_DIR}/bin/activate"

SCRIPT="${REPO_DIR}/scripts/analyze_aa_stratified.py"

echo "============================================"
echo "AA-stratified analysis"
echo "Date: $(date)"
echo "============================================"

for SCORER in thermompnn proteinmpnn; do

# --- ATLAS + RMSF ---
echo ""
echo "=== ATLAS RMSF (${SCORER}) ==="
python "$SCRIPT" \
    --data-dir "${ATLAS_DIR}" \
    --robustness-dir "${ROBUSTNESS_DIR}" \
    --scorer ${SCORER} \
    --target rmsf \
    --output-dir "${ANALYSIS_DIR}/aa_stratified_rmsf_${SCORER}" \
    --dataset-name "ATLAS_RMSF_${SCORER}"

# --- ATLAS + B-factor ---
echo ""
echo "=== ATLAS B-factor (${SCORER}) ==="
python "$SCRIPT" \
    --data-dir "${ATLAS_DIR}" \
    --robustness-dir "${ROBUSTNESS_DIR}" \
    --scorer ${SCORER} \
    --target bfactor \
    --output-dir "${ANALYSIS_DIR}/aa_stratified_bfactor_${SCORER}" \
    --dataset-name "ATLAS_Bfactor_${SCORER}"

# --- BBFlow + RMSF ---
echo ""
echo "=== BBFlow RMSF (${SCORER}) ==="
python "$SCRIPT" \
    --data-dir "${BBFLOW_PROCESSED}" \
    --robustness-dir "${BBFLOW_ROBUSTNESS}" \
    --scorer ${SCORER} \
    --target rmsf \
    --output-dir "${BBFLOW_ANALYSIS}/aa_stratified_rmsf_${SCORER}" \
    --dataset-name "BBFlow_RMSF_${SCORER}"

# --- PDB designs + B-factor ---
echo ""
echo "=== PDB designs B-factor (${SCORER}) ==="
python "$SCRIPT" \
    --data-dir "${PDB_DESIGNS_DIR}" \
    --robustness-dir "${PDB_DESIGNS_ROBUSTNESS}" \
    --scorer ${SCORER} \
    --target bfactor \
    --output-dir "${PDB_DESIGNS_ANALYSIS}/aa_stratified_bfactor_${SCORER}" \
    --dataset-name "PDB_designs_Bfactor_${SCORER}"

done

echo ""
echo "All AA-stratified analyses done at $(date)"
