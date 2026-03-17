#!/bin/bash
# ============================================================================
# Master submission script for the ATLAS robustness-dynamics pipeline.
#
# Submits jobs with dependencies so they run in the correct order:
#   1. Download ATLAS data (CPU, ~24h)
#   2. Compute robustness with ESM-1v (GPU array, ~8h per chunk)
#   3. Compute robustness with ThermoMPNN (GPU array, ~8h per chunk)  [optional]
#   4. Run correlation analysis (CPU, ~1h)
#
# All paths come from config.sh — edit that file, not this one.
#
# Usage:
#   bash scripts/slurm/submit_pipeline.sh                  # ESM-1v only
#   bash scripts/slurm/submit_pipeline.sh --thermompnn     # + ThermoMPNN
#   bash scripts/slurm/submit_pipeline.sh --skip-download  # already downloaded
#   bash scripts/slurm/submit_pipeline.sh --test           # 50 proteins
#
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# Defaults
USE_THERMOMPNN=false
SKIP_DOWNLOAD=false
TEST_MODE=false
NUM_ARRAY_TASKS=$(( (1400 + CHUNK_SIZE - 1) / CHUNK_SIZE ))

# Parse args
for arg in "$@"; do
    case "$arg" in
        --thermompnn)     USE_THERMOMPNN=true ;;
        --skip-download)  SKIP_DOWNLOAD=true ;;
        --test)           TEST_MODE=true ;;
    esac
done

if [[ "${TEST_MODE}" == true ]]; then
    NUM_ARRAY_TASKS=1
fi

echo "============================================"
echo "ATLAS Pipeline Submission"
echo "Date:           $(date)"
echo "Repo:           ${REPO_DIR}"
echo "Project:        ${PROJECT_DIR}"
echo "ThermoMPNN:     ${USE_THERMOMPNN}"
echo "Skip download:  ${SKIP_DOWNLOAD}"
echo "Test mode:      ${TEST_MODE}"
echo "Array tasks:    0-$((NUM_ARRAY_TASKS - 1))"
echo "Logs:           ${LOG_DIR}"
echo "============================================"
echo ""

# Note: #SBATCH directives can't use shell variables, so we pass
# --output, --error, and --partition via sbatch command-line flags here.
# These override the (absent) #SBATCH lines in the individual scripts.

# --- Step 1: Download ---
if [[ "${SKIP_DOWNLOAD}" == false ]]; then
    DOWNLOAD_ARGS=(
        --parsable
        --partition="${CPU_PARTITION}"
        --output="${LOG_DIR}/atlas_download_%j.out"
        --error="${LOG_DIR}/atlas_download_%j.err"
    )
    if [[ "${TEST_MODE}" == true ]]; then
        DOWNLOAD_JOB=$(sbatch "${DOWNLOAD_ARGS[@]}" "${SCRIPT_DIR}/1_download_atlas.sh" --test)
    else
        DOWNLOAD_JOB=$(sbatch "${DOWNLOAD_ARGS[@]}" "${SCRIPT_DIR}/1_download_atlas.sh")
    fi
    echo "1. Download ATLAS:        job ${DOWNLOAD_JOB}"
    DEPEND_FLAG="--dependency=afterok:${DOWNLOAD_JOB}"
else
    echo "1. Download ATLAS:        SKIPPED"
    DEPEND_FLAG=""
fi

# --- Step 2a: ESM-1v robustness ---
ESM_ARGS=(
    --parsable
    --partition="${GPU_PARTITION}"
    --output="${LOG_DIR}/robustness_esm_%A_%a.out"
    --error="${LOG_DIR}/robustness_esm_%A_%a.err"
    --array="0-$((NUM_ARRAY_TASKS - 1))"
)
[[ -n "${DEPEND_FLAG}" ]] && ESM_ARGS+=("${DEPEND_FLAG}")

ESM_JOB=$(sbatch "${ESM_ARGS[@]}" "${SCRIPT_DIR}/2_compute_robustness_esm.sh")
echo "2. ESM-1v robustness:     array job ${ESM_JOB} (${NUM_ARRAY_TASKS} tasks)"
CORRELATE_DEPEND="--dependency=afterok:${ESM_JOB}"

# --- Step 2b: ThermoMPNN robustness (optional) ---
if [[ "${USE_THERMOMPNN}" == true ]]; then
    TMPNN_ARGS=(
        --parsable
        --partition="${GPU_PARTITION}"
        --output="${LOG_DIR}/robustness_tmpnn_%A_%a.out"
        --error="${LOG_DIR}/robustness_tmpnn_%A_%a.err"
        --array="0-$((NUM_ARRAY_TASKS - 1))"
    )
    [[ -n "${DEPEND_FLAG}" ]] && TMPNN_ARGS+=("${DEPEND_FLAG}")

    TMPNN_JOB=$(sbatch "${TMPNN_ARGS[@]}" "${SCRIPT_DIR}/2_compute_robustness_thermompnn.sh")
    echo "3. ThermoMPNN robustness: array job ${TMPNN_JOB} (${NUM_ARRAY_TASKS} tasks)"
    CORRELATE_DEPEND="--dependency=afterok:${ESM_JOB}:${TMPNN_JOB}"
    CORRELATE_FLAG="--both"
else
    echo "3. ThermoMPNN:            SKIPPED"
    CORRELATE_FLAG=""
fi

# --- Step 3: Correlation analysis ---
CORR_ARGS=(
    --parsable
    --partition="${CPU_PARTITION}"
    --output="${LOG_DIR}/correlate_%j.out"
    --error="${LOG_DIR}/correlate_%j.err"
    "${CORRELATE_DEPEND}"
)

CORR_JOB=$(sbatch "${CORR_ARGS[@]}" "${SCRIPT_DIR}/3_correlate.sh" ${CORRELATE_FLAG})
echo "4. Correlation:           job ${CORR_JOB}"

echo ""
echo "============================================"
echo "Pipeline submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/atlas_download_*.out"
echo "  tail -f ${LOG_DIR}/robustness_esm_*.out"
echo "============================================"
