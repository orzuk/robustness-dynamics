#!/bin/bash
#
# Submit SLURM jobs for missing analyses:
#   1. BBFlow stratified (re-run correlation with DSSP + ConSurf)
#   2. NMR-APP correlations with DSSP (4 targets)
#   3. RCI-S2 + RelaxDB hetNOE correlations with ConSurf
#   4. Multi-DDG regression for RelaxDB R2, R2/R1, NMR-APP (4 targets)
#
# Usage: bash scripts/slurm/run_missing_analyses.sh
#   (from repo root, on the cluster)

set -euo pipefail

PROJECT="/sci/labs/orzuk/orzuk/projects/ProteinStability"
REPO="/sci/labs/orzuk/orzuk/github/robustness-dynamics"
VENV="${PROJECT}/envs/robustness"
LOGDIR="${PROJECT}/logs"
CONSURF="${PROJECT}/data/ConSurf"
SCRIPTS="${REPO}/scripts"

mkdir -p "$LOGDIR"

submit() {
    local NAME="$1"
    local TIME="$2"
    local MEM="$3"
    local CMD="$4"

    sbatch --job-name="$NAME" \
           --output="${LOGDIR}/${NAME}_%j.out" \
           --error="${LOGDIR}/${NAME}_%j.err" \
           --time="$TIME" --mem="$MEM" --cpus-per-task=4 \
           --partition=glacier \
           --wrap=". ${VENV}/bin/activate && cd ${REPO} && ${CMD}"
    echo "  Submitted: $NAME"
}

echo "=== 1. BBFlow correlation (DSSP + ConSurf) ==="
submit "bbflow_corr" "02:00:00" "16G" \
    "python ${SCRIPTS}/correlate_robustness_dynamics.py \
        --atlas_dir ${PROJECT}/data/bbflow_processed \
        --robustness_dir ${PROJECT}/data/bbflow_robustness \
        --scorer thermompnn \
        --output_dir ${PROJECT}/data/bbflow_analysis \
        --target rmsf \
        --consurf_dir ${CONSURF}"

echo "=== 2. RCI-S2 correlation (with ConSurf) ==="
submit "rci_s2_corr" "02:00:00" "16G" \
    "python ${SCRIPTS}/correlate_robustness_dynamics.py \
        --atlas_dir ${PROJECT}/data/rci_s2_processed \
        --robustness_dir ${PROJECT}/data/rci_s2_robustness \
        --scorer thermompnn \
        --output_dir ${PROJECT}/data/rci_s2_analysis \
        --target bfactor \
        --consurf_dir ${CONSURF}"

echo "=== 3. RelaxDB hetNOE correlation (with ConSurf) ==="
submit "relaxdb_corr" "02:00:00" "16G" \
    "python ${SCRIPTS}/correlate_robustness_dynamics.py \
        --atlas_dir ${PROJECT}/data/relaxdb_processed \
        --robustness_dir ${PROJECT}/data/relaxdb_robustness \
        --scorer thermompnn \
        --output_dir ${PROJECT}/data/relaxdb_analysis \
        --target bfactor \
        --consurf_dir ${CONSURF}"

echo "=== 4. NMR-APP correlations (DSSP, 4 targets) ==="
for SUFFIX in _hetNOE.tsv _R1.tsv _R2.tsv _R2R1.tsv; do
    TAG=${SUFFIX#_}; TAG=${TAG%.tsv}
    submit "nmrapp_corr_${TAG}" "02:00:00" "16G" \
        "python ${SCRIPTS}/correlate_robustness_dynamics.py \
            --atlas_dir ${PROJECT}/data/nmr_app_processed \
            --robustness_dir ${PROJECT}/data/nmr_app_robustness \
            --scorer thermompnn \
            --output_dir ${PROJECT}/data/nmr_app_analysis/${TAG} \
            --target bfactor \
            --bfactor_suffix ${SUFFIX} \
            --consurf_dir ${CONSURF}"
done

echo "=== 5. Multi-DDG: RelaxDB R2 ==="
submit "relaxdb_R2_multiddg" "04:00:00" "32G" \
    "python ${SCRIPTS}/multi_ddg_regression.py \
        --atlas_dir ${PROJECT}/data/relaxdb_processed \
        --robustness_dir ${PROJECT}/data/relaxdb_robustness \
        --scorer thermompnn \
        --output_dir ${PROJECT}/data/relaxdb_analysis \
        --target bfactor \
        --bfactor_suffix _R2.tsv"

echo "=== 6. Multi-DDG: RelaxDB R2/R1 ==="
submit "relaxdb_R2R1_multiddg" "04:00:00" "32G" \
    "python ${SCRIPTS}/multi_ddg_regression.py \
        --atlas_dir ${PROJECT}/data/relaxdb_processed \
        --robustness_dir ${PROJECT}/data/relaxdb_robustness \
        --scorer thermompnn \
        --output_dir ${PROJECT}/data/relaxdb_analysis \
        --target bfactor \
        --bfactor_suffix _R2R1.tsv"

echo "=== 7. Multi-DDG: NMR-APP (4 targets) ==="
for SUFFIX in _hetNOE.tsv _R1.tsv _R2.tsv _R2R1.tsv; do
    TAG=${SUFFIX#_}; TAG=${TAG%.tsv}
    submit "nmrapp_multiddg_${TAG}" "04:00:00" "32G" \
        "python ${SCRIPTS}/multi_ddg_regression.py \
            --atlas_dir ${PROJECT}/data/nmr_app_processed \
            --robustness_dir ${PROJECT}/data/nmr_app_robustness \
            --scorer thermompnn \
            --output_dir ${PROJECT}/data/nmr_app_analysis/${TAG} \
            --target bfactor \
            --bfactor_suffix ${SUFFIX}"
done

echo ""
echo "=== All jobs submitted. After completion, run: ==="
echo "  source ${VENV}/bin/activate && cd ${REPO}"
echo "  python scripts/collect_results.py"
echo "  python scripts/generate_latex_tables.py --output-dir ${PROJECT}/data/paper_results"
