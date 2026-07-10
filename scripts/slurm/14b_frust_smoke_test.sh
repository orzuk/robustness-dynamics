#!/bin/bash
#SBATCH --job-name=frust_smoke
#SBATCH --time=0:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#
# Smoke test for FrustraMPNN. Runs compute_frustration.py on ONE ATLAS protein
# on a GPU and prints the head of both output views, to validate the
# frustrampnn predict() schema + env BEFORE launching the full arrays
# (14_compute_robustness_frustrampnn.sh).
#
#   sbatch scripts/slurm/14b_frust_smoke_test.sh
#   # then read the log:  tail -40 $PROJECT_DIR/logs/frust_smoke_*.out
#
# PASS criteria (see the log):
#   - NO "WARN: ... wildtype mismatches" line (position indexing aligns).
#   - frustrampnn_native std_ddg column = real F_native values (~ -4..+2), not all NaN.
#   - frustrampnn std_ddg column = small positive spreads.
# ============================================================================
#SBATCH --output=/sci/labs/orzuk/orzuk/projects/ProteinStability/logs/frust_smoke_%j.out
#SBATCH --error=/sci/labs/orzuk/orzuk/projects/ProteinStability/logs/frust_smoke_%j.err
set -euo pipefail

if [[ -z "${REPO_DIR:-}" ]]; then
    _cfg="$(dirname "${BASH_SOURCE[0]}")/config.sh"
    [[ ! -f "$_cfg" ]] && _cfg="${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/config.sh"
    source "$_cfg"
fi
source "${FRUSTRAMPNN_ENV}/bin/activate"
cd "${REPO_DIR}"

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Checkpoint: ${FRUSTRAMPNN_CHECKPOINT}"

PDB=$(ls ${ATLAS_DIR}/proteins/*/*.pdb | head -1)
echo "Test PDB: ${PDB}"
OUT=/tmp/frust_smoke_${SLURM_JOB_ID:-manual}

python scripts/compute_frustration.py \
    --pdb_file "${PDB}" --protein_id smoke_test \
    --output_dir "${OUT}" \
    --frustrampnn_checkpoint "${FRUSTRAMPNN_CHECKPOINT}" \
    --device cuda --no_skip_existing

echo ""
echo "=== frustrampnn_native (std_ddg column = F_native) ==="
head "${OUT}/frustrampnn_native/smoke_test_robustness.tsv"
echo ""
echo "=== frustrampnn (std_ddg column = std of frustration profile) ==="
head "${OUT}/frustrampnn/smoke_test_robustness.tsv"
echo ""
echo "Smoke test done. Output in ${OUT}"
