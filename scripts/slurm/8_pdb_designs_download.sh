#!/bin/bash
#SBATCH --job-name=pdb_designs_dl
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#
# Download de novo designed protein structures from PDB.
# CPU-only job (no GPU needed). Downloads ~430 PDB files + extracts B-factors.
# Resumable: re-submit to continue from where it left off.
#
# Usage:
#   sbatch scripts/slurm/8_pdb_designs_download.sh
#   sbatch scripts/slurm/8_pdb_designs_download.sh --test   # 20 proteins
#
# ============================================================================

set -euo pipefail
if [[ -z "${REPO_DIR:-}" ]]; then
    _cfg="$(dirname "${BASH_SOURCE[0]}")/config.sh"
    [[ ! -f "$_cfg" ]] && _cfg="${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/config.sh"
    source "$_cfg"
fi

# Parse args
EXTRA_FLAGS=""
if [[ "${1:-}" == "--test" ]]; then
    EXTRA_FLAGS="--max_proteins 20"
fi

source "${VENV_DIR}/bin/activate"

echo "============================================"
echo "PDB De Novo Designs Download"
echo "Date:       $(date)"
echo "Node:       $(hostname)"
echo "Output dir: ${PDB_DESIGNS_DIR}"
echo "============================================"

python "${REPO_DIR}/scripts/download_pdb_designs.py" \
    --output_dir "${PDB_DESIGNS_DIR}" \
    ${EXTRA_FLAGS}

echo ""
echo "Download finished at $(date)"
echo "Data in: ${PDB_DESIGNS_DIR}"
