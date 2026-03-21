#!/bin/bash
#SBATCH --job-name=aa_stratified
#SBATCH --time=01:00:00
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

# --- ATLAS + RMSF ---
echo ""
echo "=== ATLAS RMSF ==="
python "$SCRIPT" \
    --data-dir "${ATLAS_DIR}" \
    --robustness-dir "${ROBUSTNESS_DIR}" \
    --scorer thermompnn \
    --target rmsf \
    --output-dir "${ANALYSIS_DIR}/aa_stratified_rmsf" \
    --dataset-name "ATLAS_RMSF"

# --- ATLAS + B-factor ---
echo ""
echo "=== ATLAS B-factor ==="
python "$SCRIPT" \
    --data-dir "${ATLAS_DIR}" \
    --robustness-dir "${ROBUSTNESS_DIR}" \
    --scorer thermompnn \
    --target bfactor \
    --output-dir "${ANALYSIS_DIR}/aa_stratified_bfactor" \
    --dataset-name "ATLAS_Bfactor"

# --- BBFlow + RMSF ---
echo ""
echo "=== BBFlow RMSF ==="
python "$SCRIPT" \
    --data-dir "${BBFLOW_PROCESSED}" \
    --robustness-dir "${BBFLOW_ROBUSTNESS}" \
    --scorer thermompnn \
    --target rmsf \
    --output-dir "${BBFLOW_ANALYSIS}/aa_stratified_rmsf" \
    --dataset-name "BBFlow_RMSF"

# --- PDB designs + B-factor ---
echo ""
echo "=== PDB designs B-factor ==="
python "$SCRIPT" \
    --data-dir "${PDB_DESIGNS_DIR}" \
    --robustness-dir "${PDB_DESIGNS_ROBUSTNESS}" \
    --scorer thermompnn \
    --target bfactor \
    --output-dir "${PDB_DESIGNS_ANALYSIS}/aa_stratified_bfactor" \
    --dataset-name "PDB_designs_Bfactor"

echo ""
echo "All AA-stratified analyses done at $(date)"
