#!/bin/bash
# ============================================================================
# Master pipeline: runs ALL stages from data download to final paper outputs.
#
# ONE command to rule them all. Each stage automatically waits for the
# previous stage via SLURM --dependency=afterok chaining.
#
# Stages:
#   1: Data download (ATLAS, PDB designs)
#   2: Data preprocessing (BBFlow, RCI-S2)
#   3: Robustness computation (ESM-1v + ThermoMPNN, all datasets)  [GPU]
#   4: pLDDT computation (ESMFold for BBFlow)                      [GPU]
#   5: Correlation analysis (all dataset x scorer x target)        [CPU]
#   6: Multi-DDG regression (ThermoMPNN, all datasets)             [CPU]
#   7: Collect results + generate tables/figures                   [CPU]
#
# Usage:
#   bash scripts/slurm/master_pipeline.sh --all          # full pipeline
#   bash scripts/slurm/master_pipeline.sh --from 5       # start from stage 5
#   bash scripts/slurm/master_pipeline.sh --only 7       # run only stage 7
#   bash scripts/slurm/master_pipeline.sh --all --test   # test mode (small subset)
#
# Notes:
#   - All scripts are resumable/idempotent (--skip_existing).
#   - If any job in a stage fails, subsequent stages won't run
#     (afterok dependency). Check logs in $PROJECT_DIR/logs/.
#   - Stage 7 is submitted as a SLURM job too (so it chains properly).
#   - Requires -A orzuk on this cluster.
#
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

ACCOUNT="-A orzuk"

# --- PDB designs: exclude natural proteins that passed keyword filter ---
PDB_EXCLUDE="2GAR_A 2IP6_A 2QSB_A 3F4M_A 4GXT_A 5ZEO_A 7AM3_A 7AM4_A 3NED_A 3NF0_A 3U8V_A 8A3K_UNK"

# --- RCI-S2 paths ---
RCI_CSV="${PROJECT_DIR}/data/gradation_nmr/zenodo_submission_v2/rci/rci_final.csv"
RCI_PDB_DIR="${PROJECT_DIR}/data/gradation_nmr/zenodo_submission_v2/rci/pdb_files"
RCI_OUTPUT_DIR="${PROJECT_DIR}/data/rci_s2_processed"
RCI_ROBUSTNESS_DIR="${PROJECT_DIR}/data/rci_s2_robustness"
RCI_ANALYSIS_DIR="${PROJECT_DIR}/data/rci_s2_analysis"

# --- Parse arguments ---
FROM_STAGE=99
ONLY_STAGE=""
TEST_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)   FROM_STAGE=1; shift ;;
        --from)  FROM_STAGE="$2"; shift 2 ;;
        --only)  ONLY_STAGE="$2"; FROM_STAGE="$2"; shift 2 ;;
        --test)  TEST_MODE=true; shift ;;
        -h|--help)
            head -35 "$0" | tail -n +2 | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ "$FROM_STAGE" -eq 99 ]]; then
    echo "Usage: bash $0 --all | --from N | --only N [--test]"
    echo "Run with -h for details."
    exit 0
fi

should_run() {
    local stage=$1
    if [[ -n "$ONLY_STAGE" ]]; then
        [[ "$stage" == "$ONLY_STAGE" ]]
    else
        [[ "$stage" -ge "$FROM_STAGE" ]]
    fi
}

# Collect job IDs per stage for dependency chaining
STAGE_JOBS=()           # all jobs from current stage
PREV_STAGE_DEP=""       # --dependency string from previous stage

# Helper: submit a job, capture its ID, add to STAGE_JOBS
# Usage: submit_job [sbatch args...]
submit_job() {
    local job_id
    job_id=$(sbatch --parsable "$@" 2>&1)
    if [[ "$job_id" =~ ^[0-9]+$ ]]; then
        STAGE_JOBS+=("$job_id")
        echo "    job $job_id"
    else
        echo "    ERROR: $job_id"
    fi
}

# Build dependency flag from STAGE_JOBS, then reset for next stage
advance_stage() {
    if [[ ${#STAGE_JOBS[@]} -gt 0 ]]; then
        local dep_ids
        dep_ids=$(IFS=:; echo "${STAGE_JOBS[*]}")
        PREV_STAGE_DEP="--dependency=afterok:${dep_ids}"
    fi
    STAGE_JOBS=()
}

# Common sbatch flags
dep_flag() {
    # Return dependency flag if we have one from a previous stage
    if [[ -n "$PREV_STAGE_DEP" ]]; then
        echo "$PREV_STAGE_DEP"
    fi
}

echo "============================================================"
echo "  MASTER PIPELINE -- Robustness vs. Dynamics Paper"
echo "============================================================"
echo "Date:        $(date)"
echo "Repo:        ${REPO_DIR}"
echo "Project:     ${PROJECT_DIR}"
echo "From stage:  ${FROM_STAGE}"
echo "Only stage:  ${ONLY_STAGE:-all}"
echo "Test mode:   ${TEST_MODE}"
echo "============================================================"
echo ""

# Check venv
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "ERROR: Venv not found at ${VENV_DIR}"
    echo "Run: sbatch ${ACCOUNT} ${SCRIPT_DIR}/0_setup_env.sh"
    exit 1
fi

mkdir -p "${LOG_DIR}"

# ============================================================================
# STAGE 1: Data download (ATLAS + PDB designs)
# ============================================================================
if should_run 1; then
    echo "=== STAGE 1: Data download ==="

    # ATLAS
    N_ATLAS=$(ls "${ATLAS_DIR}/proteins/" 2>/dev/null | wc -l)
    if [[ "$N_ATLAS" -gt 1900 ]]; then
        echo "  ATLAS: already have ${N_ATLAS} proteins, skipping"
    else
        TEST_FLAG=""
        [[ "$TEST_MODE" == true ]] && TEST_FLAG="--test"
        echo -n "  ATLAS download: "
        submit_job ${ACCOUNT} --partition=${CPU_PARTITION} --time=24:00:00 --mem=4G \
            --job-name=dl_atlas --output="${LOG_DIR}/dl_atlas_%j.out" \
            $(dep_flag) \
            "${SCRIPT_DIR}/1_download_atlas.sh" ${TEST_FLAG}
    fi

    # PDB designs
    N_PDB=$(ls "${PDB_DESIGNS_DIR}/proteins/" 2>/dev/null | wc -l)
    if [[ "$N_PDB" -gt 300 ]]; then
        echo "  PDB designs: already have ${N_PDB} proteins, skipping"
    else
        TEST_FLAG=""
        [[ "$TEST_MODE" == true ]] && TEST_FLAG="--test"
        echo -n "  PDB designs download: "
        submit_job ${ACCOUNT} --partition=${CPU_PARTITION} --time=04:00:00 --mem=4G \
            --job-name=dl_pdb --output="${LOG_DIR}/dl_pdb_%j.out" \
            $(dep_flag) \
            "${SCRIPT_DIR}/8_pdb_designs_download.sh" ${TEST_FLAG}
    fi

    echo "  (BBFlow + RCI-S2: external datasets, must be provided manually)"
    advance_stage
    echo ""
fi

# ============================================================================
# STAGE 2: Data preprocessing (BBFlow RMSF extraction, RCI-S2 formatting)
# ============================================================================
if should_run 2; then
    echo "=== STAGE 2: Data preprocessing ==="

    # BBFlow
    N_BB=$(ls "${BBFLOW_PROCESSED}/proteins/" 2>/dev/null | wc -l)
    if [[ "$N_BB" -gt 90 ]]; then
        echo "  BBFlow: already have ${N_BB} proteins, skipping"
    else
        echo -n "  BBFlow preprocessing: "
        submit_job ${ACCOUNT} --partition=${CPU_PARTITION} --time=01:00:00 --mem=8G \
            --job-name=prep_bb --output="${LOG_DIR}/prep_bb_%j.out" \
            $(dep_flag) \
            --wrap="bash -c 'source ${VENV_DIR}/bin/activate && cd ${REPO_DIR} && python scripts/prepare_bbflow.py --bbflow_dir ${BBFLOW_RAW} --output_dir ${BBFLOW_PROCESSED}'"
    fi

    # RCI-S2
    N_RCI=$(ls "${RCI_OUTPUT_DIR}/proteins/" 2>/dev/null | wc -l)
    if [[ "$N_RCI" -gt 700 ]]; then
        echo "  RCI-S2: already have ${N_RCI} proteins, skipping"
    else
        echo -n "  RCI-S2 preprocessing: "
        submit_job ${ACCOUNT} --partition=${CPU_PARTITION} --time=00:15:00 --mem=8G \
            --job-name=prep_rci --output="${LOG_DIR}/prep_rci_%j.out" \
            $(dep_flag) \
            --wrap="bash -c 'source ${VENV_DIR}/bin/activate && cd ${REPO_DIR} && python scripts/preprocess_rci_dataset.py --rci_csv ${RCI_CSV} --pdb_dir ${RCI_PDB_DIR} --output_dir ${RCI_OUTPUT_DIR}'"
    fi

    advance_stage
    echo ""
fi

# ============================================================================
# STAGE 3: Robustness computation (GPU array jobs)
# ============================================================================
if should_run 3; then
    echo "=== STAGE 3: Robustness computation (GPU) ==="

    DEP=$(dep_flag)

    # --- ATLAS ---
    N_ATLAS_PROTS=$(ls "${ATLAS_DIR}/proteins/" 2>/dev/null | wc -l)
    ATLAS_CHUNKS=$(( (N_ATLAS_PROTS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
    ATLAS_MAX=$(( ATLAS_CHUNKS - 1 ))
    [[ "$TEST_MODE" == true ]] && ATLAS_MAX=0

    echo -n "  ATLAS ThermoMPNN (array 0-${ATLAS_MAX}): "
    submit_job ${ACCOUNT} --array=0-${ATLAS_MAX} --partition=${GPU_PARTITION} \
        --job-name=rob_a_t --output="${LOG_DIR}/rob_atlas_tmpnn_%A_%a.out" \
        $DEP \
        "${SCRIPT_DIR}/2_compute_robustness_thermompnn.sh"

    echo -n "  ATLAS ESM-1v (array 0-${ATLAS_MAX}): "
    submit_job ${ACCOUNT} --array=0-${ATLAS_MAX} --partition=${GPU_PARTITION} \
        --job-name=rob_a_e --output="${LOG_DIR}/rob_atlas_esm_%A_%a.out" \
        $DEP \
        "${SCRIPT_DIR}/2_compute_robustness_esm.sh"

    # --- BBFlow ---
    N_BB_PROTS=$(ls "${BBFLOW_PROCESSED}/proteins/" 2>/dev/null | wc -l)
    BB_CHUNKS=$(( (N_BB_PROTS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
    BB_MAX=$(( BB_CHUNKS - 1 ))
    [[ "$TEST_MODE" == true ]] && BB_MAX=0

    echo -n "  BBFlow ThermoMPNN (array 0-${BB_MAX}): "
    submit_job ${ACCOUNT} --array=0-${BB_MAX} --partition=${GPU_PARTITION} \
        --time=04:00:00 --mem=16G --cpus-per-task=2 --gres=${GPU_GRES} \
        --job-name=rob_b_t --output="${LOG_DIR}/rob_bb_tmpnn_%A_%a.out" \
        $DEP \
        --wrap="bash -c 'source ${VENV_DIR}/bin/activate && cd ${REPO_DIR} && python scripts/compute_robustness.py --scorer thermompnn --atlas_dir ${BBFLOW_PROCESSED} --output_dir ${BBFLOW_ROBUSTNESS} --batch --batch_start \$((\${SLURM_ARRAY_TASK_ID}*${CHUNK_SIZE})) --batch_end \$((\${SLURM_ARRAY_TASK_ID}*${CHUNK_SIZE}+${CHUNK_SIZE})) --device cuda --skip_existing --thermompnn_dir ${THERMOMPNN_DIR}'"

    echo -n "  BBFlow ESM-1v (array 0-${BB_MAX}): "
    submit_job ${ACCOUNT} --array=0-${BB_MAX} --partition=${GPU_PARTITION} \
        --time=04:00:00 --mem=16G --cpus-per-task=2 --gres=${GPU_GRES} \
        --job-name=rob_b_e --output="${LOG_DIR}/rob_bb_esm_%A_%a.out" \
        $DEP \
        --wrap="bash -c 'source ${VENV_DIR}/bin/activate && cd ${REPO_DIR} && python scripts/compute_robustness.py --scorer esm1v --atlas_dir ${BBFLOW_PROCESSED} --output_dir ${BBFLOW_ROBUSTNESS} --batch --batch_start \$((\${SLURM_ARRAY_TASK_ID}*${CHUNK_SIZE})) --batch_end \$((\${SLURM_ARRAY_TASK_ID}*${CHUNK_SIZE}+${CHUNK_SIZE})) --device cuda --skip_existing'"

    # --- PDB designs ---
    N_PDB_PROTS=$(ls "${PDB_DESIGNS_DIR}/proteins/" 2>/dev/null | wc -l)
    PDB_CHUNKS=$(( (N_PDB_PROTS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
    PDB_MAX=$(( PDB_CHUNKS - 1 ))
    [[ "$TEST_MODE" == true ]] && PDB_MAX=0

    echo -n "  PDB designs ThermoMPNN (array 0-${PDB_MAX}): "
    submit_job ${ACCOUNT} --array=0-${PDB_MAX} --partition=${GPU_PARTITION} \
        --job-name=rob_p_t --output="${LOG_DIR}/rob_pdb_tmpnn_%A_%a.out" \
        $DEP \
        "${SCRIPT_DIR}/9_pdb_designs_robustness.sh"

    echo -n "  PDB designs ESM-1v (array 0-${PDB_MAX}): "
    submit_job ${ACCOUNT} --array=0-${PDB_MAX} --partition=${GPU_PARTITION} \
        --time=08:00:00 --mem=16G --cpus-per-task=2 --gres=${GPU_GRES} \
        --job-name=rob_p_e --output="${LOG_DIR}/rob_pdb_esm_%A_%a.out" \
        $DEP \
        --wrap="bash -c 'source ${VENV_DIR}/bin/activate && cd ${REPO_DIR} && python scripts/compute_robustness.py --scorer esm1v --atlas_dir ${PDB_DESIGNS_DIR} --output_dir ${PDB_DESIGNS_ROBUSTNESS} --batch --batch_start \$((\${SLURM_ARRAY_TASK_ID}*${CHUNK_SIZE})) --batch_end \$((\${SLURM_ARRAY_TASK_ID}*${CHUNK_SIZE}+${CHUNK_SIZE})) --device cuda --skip_existing'"

    # --- RCI-S2 ---
    N_RCI_PROTS=$(ls "${RCI_OUTPUT_DIR}/proteins/" 2>/dev/null | wc -l)
    RCI_CHUNKS=$(( (N_RCI_PROTS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
    RCI_MAX=$(( RCI_CHUNKS - 1 ))
    [[ "$TEST_MODE" == true ]] && RCI_MAX=0

    # Build PDB list for pdb_list mode
    PDB_LIST="${RCI_OUTPUT_DIR}/pdb_list.txt"
    find "${RCI_OUTPUT_DIR}/proteins" -name "*.pdb" -type l 2>/dev/null | sort > "${PDB_LIST}" || true

    echo -n "  RCI-S2 ThermoMPNN (array 0-${RCI_MAX}): "
    submit_job ${ACCOUNT} --array=0-${RCI_MAX} --partition=${GPU_PARTITION} \
        --time=08:00:00 --mem=16G --cpus-per-task=2 --gres=${GPU_GRES} \
        --job-name=rob_r_t --output="${LOG_DIR}/rob_rci_tmpnn_%A_%a.out" \
        $DEP \
        --wrap="bash -c 'source ${VENV_DIR}/bin/activate && cd ${REPO_DIR} && START=\$((\${SLURM_ARRAY_TASK_ID}*${CHUNK_SIZE}+1)) && END=\$((START+${CHUNK_SIZE}-1)) && BATCH=/tmp/rci_batch_\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}.txt && sed -n \"\${START},\${END}p\" ${PDB_LIST} > \$BATCH && python scripts/compute_robustness.py --scorer thermompnn --pdb_list \$BATCH --output_dir ${RCI_ROBUSTNESS_DIR} --skip_existing'"

    echo -n "  RCI-S2 ESM-1v (array 0-${RCI_MAX}): "
    submit_job ${ACCOUNT} --array=0-${RCI_MAX} --partition=${GPU_PARTITION} \
        --time=08:00:00 --mem=16G --cpus-per-task=2 --gres=${GPU_GRES} \
        --job-name=rob_r_e --output="${LOG_DIR}/rob_rci_esm_%A_%a.out" \
        $DEP \
        --wrap="bash -c 'source ${VENV_DIR}/bin/activate && cd ${REPO_DIR} && START=\$((\${SLURM_ARRAY_TASK_ID}*${CHUNK_SIZE}+1)) && END=\$((START+${CHUNK_SIZE}-1)) && BATCH=/tmp/rci_esm_\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}.txt && sed -n \"\${START},\${END}p\" ${PDB_LIST} > \$BATCH && python scripts/compute_robustness.py --scorer esm1v --pdb_list \$BATCH --output_dir ${RCI_ROBUSTNESS_DIR} --device cuda --skip_existing'"

    advance_stage
    echo ""
fi

# ============================================================================
# STAGE 4: pLDDT computation (ESMFold for BBFlow)
# ============================================================================
if should_run 4; then
    echo "=== STAGE 4: pLDDT computation (ESMFold for BBFlow) ==="

    PLDDT_COUNT=$(find "${BBFLOW_PROCESSED}/proteins" -name "plddt.tsv" 2>/dev/null | wc -l)
    if [[ "$PLDDT_COUNT" -gt 90 ]]; then
        echo "  BBFlow pLDDT: ${PLDDT_COUNT} files exist, skipping"
    else
        echo -n "  BBFlow ESMFold pLDDT: "
        submit_job ${ACCOUNT} --partition=${GPU_PARTITION} \
            --job-name=plddt_bb --output="${LOG_DIR}/plddt_bb_%j.out" \
            $(dep_flag) \
            "${SCRIPT_DIR}/5_compute_plddt_esmfold.sh"
    fi

    advance_stage
    echo ""
fi

# ============================================================================
# STAGE 5: Correlation analysis (CPU, all dataset x scorer x target)
# ============================================================================
if should_run 5; then
    echo "=== STAGE 5: Correlation analysis ==="

    DEP=$(dep_flag)

    CONSURF_FLAG=""
    if [[ -d "${CONSURF_DIR}/files" ]] && find "${CONSURF_DIR}/files" -maxdepth 1 -name "*.json" -print -quit | grep -q .; then
        CONSURF_FLAG="--consurf_dir ${CONSURF_DIR}"
        N_CONSURF=$(find "${CONSURF_DIR}/files" -maxdepth 1 -name "*.json" | wc -l)
        echo "  ConSurf data found (${N_CONSURF} JSONs in files/)"
    elif [[ -d "${CONSURF_DIR}" ]] && find "${CONSURF_DIR}" -maxdepth 1 -name "*.json" -print -quit | grep -q .; then
        CONSURF_FLAG="--consurf_dir ${CONSURF_DIR}"
        echo "  ConSurf data found"
    else
        echo "  WARNING: No ConSurf data at ${CONSURF_DIR}"
    fi

    # ATLAS (both scorers; script handles both RMSF + B-factor targets)
    for SCORER in thermompnn esm1v proteinmpnn; do
        echo -n "  ATLAS ${SCORER}: "
        submit_job ${ACCOUNT} --partition=${CPU_PARTITION} --time=02:00:00 --mem=16G --cpus-per-task=4 \
            --job-name=corr_a_${SCORER:0:1} --output="${LOG_DIR}/corr_atlas_${SCORER}_%j.out" \
            $DEP \
            --wrap="bash -c 'source ${VENV_DIR}/bin/activate && cd ${REPO_DIR} && python scripts/correlate_robustness_dynamics.py --atlas_dir ${ATLAS_DIR} --robustness_dir ${ROBUSTNESS_DIR} --scorer ${SCORER} --output_dir ${ANALYSIS_DIR} ${CONSURF_FLAG}'"
    done

    # BBFlow
    for SCORER in thermompnn esm1v proteinmpnn; do
        echo -n "  BBFlow ${SCORER}: "
        submit_job ${ACCOUNT} --partition=${CPU_PARTITION} --time=02:00:00 --mem=16G --cpus-per-task=4 \
            --job-name=corr_b_${SCORER:0:1} --output="${LOG_DIR}/corr_bb_${SCORER}_%j.out" \
            $DEP \
            --wrap="bash -c 'source ${VENV_DIR}/bin/activate && cd ${REPO_DIR} && python scripts/correlate_robustness_dynamics.py --atlas_dir ${BBFLOW_PROCESSED} --robustness_dir ${BBFLOW_ROBUSTNESS} --scorer ${SCORER} --output_dir ${BBFLOW_ANALYSIS} --no_dssp ${CONSURF_FLAG}'"
    done

    # PDB designs (exclude natural proteins that passed keyword filter)
    for SCORER in thermompnn esm1v proteinmpnn; do
        echo -n "  PDB designs ${SCORER}: "
        submit_job ${ACCOUNT} --partition=${CPU_PARTITION} --time=02:00:00 --mem=16G --cpus-per-task=4 \
            --job-name=corr_p_${SCORER:0:1} --output="${LOG_DIR}/corr_pdb_${SCORER}_%j.out" \
            $DEP \
            --wrap="bash -c 'source ${VENV_DIR}/bin/activate && cd ${REPO_DIR} && python scripts/correlate_robustness_dynamics.py --atlas_dir ${PDB_DESIGNS_DIR} --robustness_dir ${PDB_DESIGNS_ROBUSTNESS} --scorer ${SCORER} --output_dir ${PDB_DESIGNS_ANALYSIS} --target bfactor --exclude ${PDB_EXCLUDE} ${CONSURF_FLAG}'"
    done

    # RCI-S2
    for SCORER in thermompnn esm1v proteinmpnn; do
        echo -n "  RCI-S2 ${SCORER}: "
        submit_job ${ACCOUNT} --partition=${CPU_PARTITION} --time=02:00:00 --mem=16G --cpus-per-task=4 \
            --job-name=corr_r_${SCORER:0:1} --output="${LOG_DIR}/corr_rci_${SCORER}_%j.out" \
            $DEP \
            --wrap="bash -c 'source ${VENV_DIR}/bin/activate && cd ${REPO_DIR} && python scripts/correlate_robustness_dynamics.py --atlas_dir ${RCI_OUTPUT_DIR} --robustness_dir ${RCI_ROBUSTNESS_DIR} --scorer ${SCORER} --output_dir ${RCI_ANALYSIS_DIR} --target bfactor ${CONSURF_FLAG}'"
    done

    # RelaxDB (hetNOE + R2 + R2/R1 targets)
    RELAXDB_DATA="${PROJECT_DIR}/data/relaxdb_processed"
    RELAXDB_ROB="${PROJECT_DIR}/data/relaxdb_robustness"
    RELAXDB_ANALYSIS="${PROJECT_DIR}/data/relaxdb_analysis"
    for SCORER in thermompnn esm1v proteinmpnn; do
        echo -n "  RelaxDB ${SCORER}: "
        submit_job ${ACCOUNT} --partition=${CPU_PARTITION} --time=02:00:00 --mem=16G --cpus-per-task=4 \
            --job-name=corr_x_${SCORER:0:1} --output="${LOG_DIR}/corr_relaxdb_${SCORER}_%j.out" \
            $DEP \
            --wrap="bash -c 'source ${VENV_DIR}/bin/activate && cd ${REPO_DIR} && python scripts/correlate_robustness_dynamics.py --atlas_dir ${RELAXDB_DATA} --robustness_dir ${RELAXDB_ROB} --scorer ${SCORER} --output_dir ${RELAXDB_ANALYSIS} --target bfactor'"
    done
    # RelaxDB virtual targets (R2, R2/R1)
    for SUFFIX_TAG in "_R2.tsv:R2" "_R2R1.tsv:R2R1"; do
        SUFFIX="${SUFFIX_TAG%%:*}"
        TAG="${SUFFIX_TAG##*:}"
        for SCORER in thermompnn proteinmpnn; do
            echo -n "  RelaxDB ${TAG} ${SCORER}: "
            submit_job ${ACCOUNT} --partition=${CPU_PARTITION} --time=02:00:00 --mem=16G --cpus-per-task=4 \
                --job-name=corr_x${TAG:0:1}_${SCORER:0:1} --output="${LOG_DIR}/corr_relaxdb_${TAG}_${SCORER}_%j.out" \
                $DEP \
                --wrap="bash -c 'source ${VENV_DIR}/bin/activate && cd ${REPO_DIR} && python scripts/correlate_robustness_dynamics.py --atlas_dir ${RELAXDB_DATA} --robustness_dir ${RELAXDB_ROB} --scorer ${SCORER} --output_dir ${PROJECT_DIR}/data/relaxdb_analysis_${TAG} --target bfactor --bfactor_suffix ${SUFFIX}'"
        done
    done

    # S2 experimental
    S2EXP_DATA="${PROJECT_DIR}/data/s2_exp_processed"
    S2EXP_ROB="${PROJECT_DIR}/data/s2_exp_robustness"
    S2EXP_ANALYSIS="${PROJECT_DIR}/data/s2_exp_analysis"
    for SCORER in thermompnn esm1v proteinmpnn; do
        echo -n "  S2-exp ${SCORER}: "
        submit_job ${ACCOUNT} --partition=${CPU_PARTITION} --time=02:00:00 --mem=16G --cpus-per-task=4 \
            --job-name=corr_s_${SCORER:0:1} --output="${LOG_DIR}/corr_s2exp_${SCORER}_%j.out" \
            $DEP \
            --wrap="bash -c 'source ${VENV_DIR}/bin/activate && cd ${REPO_DIR} && python scripts/correlate_robustness_dynamics.py --atlas_dir ${S2EXP_DATA} --robustness_dir ${S2EXP_ROB} --scorer ${SCORER} --output_dir ${S2EXP_ANALYSIS} --target bfactor'"
    done

    advance_stage
    echo ""
fi

# ============================================================================
# STAGE 6: Multi-DDG regression (CPU, ThermoMPNN + ProteinMPNN)
# ============================================================================
if should_run 6; then
    echo "=== STAGE 6: Multi-DDG regression ==="

    DEP=$(dep_flag)

    for SCORER in thermompnn proteinmpnn; do
        echo -n "  ATLAS RMSF (${SCORER}): "
        submit_job ${ACCOUNT} --partition=${CPU_PARTITION} --time=04:00:00 --mem=32G --cpus-per-task=4 \
            --job-name=mddg_a_r_${SCORER:0:1} --output="${LOG_DIR}/mddg_atlas_rmsf_${SCORER}_%j.out" \
            $DEP \
            --wrap="bash -c 'source ${VENV_DIR}/bin/activate && cd ${REPO_DIR} && python scripts/multi_ddg_regression.py --atlas_dir ${ATLAS_DIR} --robustness_dir ${ROBUSTNESS_DIR} --scorer ${SCORER} --target rmsf --output_dir ${ANALYSIS_DIR}'"

        echo -n "  ATLAS B-factor (${SCORER}): "
        submit_job ${ACCOUNT} --partition=${CPU_PARTITION} --time=04:00:00 --mem=32G --cpus-per-task=4 \
            --job-name=mddg_a_b_${SCORER:0:1} --output="${LOG_DIR}/mddg_atlas_bfac_${SCORER}_%j.out" \
            $DEP \
            --wrap="bash -c 'source ${VENV_DIR}/bin/activate && cd ${REPO_DIR} && python scripts/multi_ddg_regression.py --atlas_dir ${ATLAS_DIR} --robustness_dir ${ROBUSTNESS_DIR} --scorer ${SCORER} --target bfactor --output_dir ${ANALYSIS_DIR}'"

        echo -n "  BBFlow RMSF (${SCORER}): "
        submit_job ${ACCOUNT} --partition=${CPU_PARTITION} --time=01:00:00 --mem=16G --cpus-per-task=4 \
            --job-name=mddg_b_${SCORER:0:1} --output="${LOG_DIR}/mddg_bbflow_${SCORER}_%j.out" \
            $DEP \
            --wrap="bash -c 'source ${VENV_DIR}/bin/activate && cd ${REPO_DIR} && python scripts/multi_ddg_regression.py --atlas_dir ${BBFLOW_PROCESSED} --robustness_dir ${BBFLOW_ROBUSTNESS} --scorer ${SCORER} --target rmsf --output_dir ${BBFLOW_ANALYSIS}'"

        echo -n "  PDB designs B-factor (${SCORER}): "
        submit_job ${ACCOUNT} --partition=${CPU_PARTITION} --time=01:00:00 --mem=16G --cpus-per-task=4 \
            --job-name=mddg_p_${SCORER:0:1} --output="${LOG_DIR}/mddg_pdb_${SCORER}_%j.out" \
            $DEP \
            --wrap="bash -c 'source ${VENV_DIR}/bin/activate && cd ${REPO_DIR} && python scripts/multi_ddg_regression.py --atlas_dir ${PDB_DESIGNS_DIR} --robustness_dir ${PDB_DESIGNS_ROBUSTNESS} --scorer ${SCORER} --target bfactor --output_dir ${PDB_DESIGNS_ANALYSIS} --exclude ${PDB_EXCLUDE}'"

        echo -n "  RCI-S2 B-factor (${SCORER}): "
        submit_job ${ACCOUNT} --partition=${CPU_PARTITION} --time=04:00:00 --mem=32G --cpus-per-task=4 \
            --job-name=mddg_r_${SCORER:0:1} --output="${LOG_DIR}/mddg_rci_${SCORER}_%j.out" \
            $DEP \
            --wrap="bash -c 'source ${VENV_DIR}/bin/activate && cd ${REPO_DIR} && python scripts/multi_ddg_regression.py --atlas_dir ${RCI_OUTPUT_DIR} --robustness_dir ${RCI_ROBUSTNESS_DIR} --scorer ${SCORER} --target bfactor --output_dir ${RCI_ANALYSIS_DIR}'"
    done

    advance_stage
    echo ""
fi

# ============================================================================
# STAGE 8: ProteinMPNN robustness (GPU array jobs, all datasets)
# ============================================================================
if should_run 8; then
    echo "=== STAGE 8: ProteinMPNN robustness (GPU) ==="

    DEP=$(dep_flag)

    # --- ATLAS ---
    N_ATLAS_PROTS=$(ls "${ATLAS_DIR}/proteins/" 2>/dev/null | wc -l)
    ATLAS_CHUNKS=$(( (N_ATLAS_PROTS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
    ATLAS_MAX=$(( ATLAS_CHUNKS - 1 ))
    [[ "$TEST_MODE" == true ]] && ATLAS_MAX=0

    echo -n "  ATLAS ProteinMPNN (array 0-${ATLAS_MAX}): "
    submit_job ${ACCOUNT} --array=0-${ATLAS_MAX} --partition=${GPU_PARTITION} \
        --job-name=rob_a_pm --output="${LOG_DIR}/rob_atlas_pmpnn_%A_%a.out" \
        $DEP \
        --export="ALL,PMPNN_ATLAS_DIR=${ATLAS_DIR},PMPNN_OUTPUT_DIR=${ROBUSTNESS_DIR}" \
        "${SCRIPT_DIR}/11_compute_robustness_proteinmpnn.sh"

    # --- BBFlow ---
    N_BB_PROTS=$(ls "${BBFLOW_PROCESSED}/proteins/" 2>/dev/null | wc -l)
    BB_CHUNKS=$(( (N_BB_PROTS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
    BB_MAX=$(( BB_CHUNKS - 1 ))
    [[ "$TEST_MODE" == true ]] && BB_MAX=0

    echo -n "  BBFlow ProteinMPNN (array 0-${BB_MAX}): "
    submit_job ${ACCOUNT} --array=0-${BB_MAX} --partition=${GPU_PARTITION} \
        --job-name=rob_b_pm --output="${LOG_DIR}/rob_bb_pmpnn_%A_%a.out" \
        $DEP \
        --export="ALL,PMPNN_ATLAS_DIR=${BBFLOW_PROCESSED},PMPNN_OUTPUT_DIR=${BBFLOW_ROBUSTNESS}" \
        "${SCRIPT_DIR}/11_compute_robustness_proteinmpnn.sh"

    # --- PDB designs ---
    N_PDB_PROTS=$(ls "${PDB_DESIGNS_DIR}/proteins/" 2>/dev/null | wc -l)
    PDB_CHUNKS=$(( (N_PDB_PROTS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
    PDB_MAX=$(( PDB_CHUNKS - 1 ))
    [[ "$TEST_MODE" == true ]] && PDB_MAX=0

    echo -n "  PDB designs ProteinMPNN (array 0-${PDB_MAX}): "
    submit_job ${ACCOUNT} --array=0-${PDB_MAX} --partition=${GPU_PARTITION} \
        --job-name=rob_p_pm --output="${LOG_DIR}/rob_pdb_pmpnn_%A_%a.out" \
        $DEP \
        --export="ALL,PMPNN_ATLAS_DIR=${PDB_DESIGNS_DIR},PMPNN_OUTPUT_DIR=${PDB_DESIGNS_ROBUSTNESS}" \
        "${SCRIPT_DIR}/11_compute_robustness_proteinmpnn.sh"

    # --- RCI-S2 ---
    N_RCI_PROTS=$(ls "${RCI_OUTPUT_DIR}/proteins/" 2>/dev/null | wc -l)
    RCI_CHUNKS=$(( (N_RCI_PROTS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
    RCI_MAX=$(( RCI_CHUNKS - 1 ))
    [[ "$TEST_MODE" == true ]] && RCI_MAX=0

    echo -n "  RCI-S2 ProteinMPNN (array 0-${RCI_MAX}): "
    submit_job ${ACCOUNT} --array=0-${RCI_MAX} --partition=${GPU_PARTITION} \
        --job-name=rob_r_pm --output="${LOG_DIR}/rob_rci_pmpnn_%A_%a.out" \
        $DEP \
        --export="ALL,PMPNN_ATLAS_DIR=${RCI_OUTPUT_DIR},PMPNN_OUTPUT_DIR=${RCI_ROBUSTNESS_DIR}" \
        "${SCRIPT_DIR}/11_compute_robustness_proteinmpnn.sh"

    # --- RelaxDB ---
    RELAXDB_DATA="${PROJECT_DIR}/data/relaxdb_processed"
    RELAXDB_ROB="${PROJECT_DIR}/data/relaxdb_robustness"
    N_RELAXDB_PROTS=$(ls "${RELAXDB_DATA}/proteins/" 2>/dev/null | wc -l)
    RELAXDB_CHUNKS=$(( (N_RELAXDB_PROTS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
    RELAXDB_MAX=$(( RELAXDB_CHUNKS - 1 ))
    [[ "$TEST_MODE" == true ]] && RELAXDB_MAX=0

    echo -n "  RelaxDB ProteinMPNN (array 0-${RELAXDB_MAX}): "
    submit_job ${ACCOUNT} --array=0-${RELAXDB_MAX} --partition=${GPU_PARTITION} \
        --job-name=rob_x_pm --output="${LOG_DIR}/rob_relaxdb_pmpnn_%A_%a.out" \
        $DEP \
        --export="ALL,PMPNN_ATLAS_DIR=${RELAXDB_DATA},PMPNN_OUTPUT_DIR=${RELAXDB_ROB}" \
        "${SCRIPT_DIR}/11_compute_robustness_proteinmpnn.sh"

    # --- S2 experimental ---
    S2EXP_DATA="${PROJECT_DIR}/data/s2_exp_processed"
    S2EXP_ROB="${PROJECT_DIR}/data/s2_exp_robustness"
    N_S2EXP_PROTS=$(ls "${S2EXP_DATA}/proteins/" 2>/dev/null | wc -l)
    S2EXP_CHUNKS=$(( (N_S2EXP_PROTS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
    S2EXP_MAX=$(( S2EXP_CHUNKS - 1 ))
    [[ "$TEST_MODE" == true ]] && S2EXP_MAX=0

    echo -n "  S2-exp ProteinMPNN (array 0-${S2EXP_MAX}): "
    submit_job ${ACCOUNT} --array=0-${S2EXP_MAX} --partition=${GPU_PARTITION} \
        --job-name=rob_s_pm --output="${LOG_DIR}/rob_s2exp_pmpnn_%A_%a.out" \
        $DEP \
        --export="ALL,PMPNN_ATLAS_DIR=${S2EXP_DATA},PMPNN_OUTPUT_DIR=${S2EXP_ROB}" \
        "${SCRIPT_DIR}/11_compute_robustness_proteinmpnn.sh"

    advance_stage
    echo ""
fi

# ============================================================================
# STAGE 9: AA-stratified analysis (CPU)
# ============================================================================
if should_run 9; then
    echo "=== STAGE 9: AA-stratified analysis ==="

    echo -n "  All datasets: "
    submit_job ${ACCOUNT} --partition=${CPU_PARTITION} --time=01:00:00 --mem=16G --cpus-per-task=4 \
        --job-name=aa_strat --output="${LOG_DIR}/aa_stratified_%j.out" \
        $(dep_flag) \
        "${SCRIPT_DIR}/12_aa_stratified.sh"

    advance_stage
    echo ""
fi

# ============================================================================
# STAGE 7: Collect results + generate tables and figures
# ============================================================================
if should_run 7; then
    echo "=== STAGE 7: Collect + generate paper outputs ==="

    RESULTS_JSON="${PROJECT_DIR}/data/paper_results/unified_results.json"
    OUTPUT_BASE="${PROJECT_DIR}/data/paper_results"
    TABLES_DIR="${OUTPUT_BASE}/Tables"
    FIGURES_DIR="${OUTPUT_BASE}/Figures"

    # Submit as SLURM job so it chains with previous stages
    echo -n "  Collect + tables + figures: "
    submit_job ${ACCOUNT} --partition=${CPU_PARTITION} --time=00:30:00 --mem=8G --cpus-per-task=2 \
        --job-name=paper_out --output="${LOG_DIR}/paper_output_%j.out" \
        $(dep_flag) \
        --wrap="bash -c 'source ${VENV_DIR}/bin/activate && cd ${REPO_DIR} && mkdir -p ${TABLES_DIR} ${FIGURES_DIR} && python scripts/collect_results.py --output ${RESULTS_JSON} --verbose && python scripts/generate_latex_tables.py --results ${RESULTS_JSON} --output-dir ${OUTPUT_BASE} && python scripts/generate_paper_figures.py --results ${RESULTS_JSON} --output-dir ${OUTPUT_BASE} && echo Done: tables in ${TABLES_DIR}, figures in ${FIGURES_DIR}'"

    advance_stage
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================
echo "============================================================"
echo "  All jobs submitted! Monitor with:"
echo "    squeue -u \$USER"
echo "    watch -n 30 squeue -u \$USER"
echo ""
echo "  Logs: ${LOG_DIR}/"
echo ""
echo "  When everything finishes, outputs will be in:"
echo "    ${PROJECT_DIR}/data/paper_results/unified_results.json"
echo "    ${PROJECT_DIR}/data/paper_results/Tables/"
echo "    ${PROJECT_DIR}/data/paper_results/Figures/"
echo "============================================================"
