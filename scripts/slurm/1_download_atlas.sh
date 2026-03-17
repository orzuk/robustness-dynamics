#!/bin/bash
#SBATCH --job-name=atlas_download
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#
# Download ATLAS database (~1,400 proteins).
# CPU-only job (no GPU needed), but takes many hours due to rate limiting.
#
# Usage:
#   sbatch scripts/slurm/1_download_atlas.sh
#   sbatch scripts/slurm/1_download_atlas.sh --test   # 50 proteins
#
# Resumable: if interrupted, just re-submit — skips already-downloaded proteins.
# ============================================================================

set -euo pipefail
if [[ -z "${REPO_DIR:-}" ]]; then
    _cfg="$(dirname "${BASH_SOURCE[0]}")/config.sh"
    [[ ! -f "$_cfg" ]] && _cfg="${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/config.sh"
    source "$_cfg"
fi

# Parse args
TEST_MODE=false
if [[ "${1:-}" == "--test" ]]; then
    TEST_MODE=true
fi

source "${VENV_DIR}/bin/activate"

echo "============================================"
echo "ATLAS Download"
echo "Date:       $(date)"
echo "Node:       $(hostname)"
echo "Output dir: ${ATLAS_DIR}"
echo "Test mode:  ${TEST_MODE}"
echo "============================================"

if [[ "${TEST_MODE}" == true ]]; then
    python "${REPO_DIR}/scripts/download_atlas.py" \
        --output_dir "${ATLAS_DIR}" \
        --max_proteins 50 \
        --delay 0.5
else
    python "${REPO_DIR}/scripts/download_atlas.py" \
        --output_dir "${ATLAS_DIR}" \
        --delay 0.5
fi

echo ""
echo "Download finished at $(date)"
echo "Data in: ${ATLAS_DIR}"
