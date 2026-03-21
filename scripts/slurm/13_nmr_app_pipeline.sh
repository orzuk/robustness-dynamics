#!/bin/bash
# ============================================================================
# Full pipeline for NMR-APP designed protein relaxation data (Hiller lab).
#
# Three stages, chained via SLURM dependencies:
#   Stage 1: Preprocess xlsx + fold with ESMFold (GPU) — parse data, predict structures
#   Stage 2: Compute robustness (GPU) — ThermoMPNN + ProteinMPNN
#   Stage 3: Correlate + regression (CPU) — all NMR targets (hetNOE, R1, R2, R2/R1)
#
# Usage:
#   bash scripts/slurm/13_nmr_app_pipeline.sh          # run all 3 stages
#   bash scripts/slurm/13_nmr_app_pipeline.sh 2         # start from stage 2
#   bash scripts/slurm/13_nmr_app_pipeline.sh 3         # start from stage 3
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

START_STAGE=${1:-1}
P="$PROJECT_DIR"
T="$THERMOMPNN_DIR"
R="$REPO_DIR"
ACCOUNT="-A orzuk"

# --- NMR-APP paths ---
NMR_APP_XLSX="${P}/data/NMR_relaxation/Relaxation_Data_9_proteins_NMR_APP.xlsx"
NMR_APP_DIR="${P}/data/nmr_app_processed"
NMR_APP_ROB="${P}/data/nmr_app_robustness"
NMR_APP_ANALYSIS="${P}/data/nmr_app_analysis"

# Check xlsx exists
if [[ ! -f "$NMR_APP_XLSX" ]]; then
    # Try Google Drive path
    GDRIVE_XLSX="/mnt/g/My Drive/Students/MeiraBarron/Dynamics/data/NMR_relaxation/Relaxation_Data_9_proteins_NMR_APP.xlsx"
    if [[ -f "$GDRIVE_XLSX" ]]; then
        mkdir -p "$(dirname "$NMR_APP_XLSX")"
        cp "$GDRIVE_XLSX" "$NMR_APP_XLSX"
        echo "Copied xlsx from Google Drive to ${NMR_APP_XLSX}"
    else
        echo "ERROR: xlsx not found at ${NMR_APP_XLSX}"
        echo "Copy it to: ${P}/data/NMR_relaxation/"
        exit 1
    fi
fi

echo "=== NMR-APP Pipeline (stages ${START_STAGE}-3) ==="
echo "  xlsx:    ${NMR_APP_XLSX}"
echo "  output:  ${NMR_APP_DIR}"
echo ""

TMPDIR_SLURM=$(mktemp -d)
trap "rm -rf $TMPDIR_SLURM" EXIT

mkdir -p "${LOG_DIR}"

# ---- Stage 1: Preprocess + fold ----
JOB1=""
if [[ $START_STAGE -le 1 ]]; then
    cat > "$TMPDIR_SLURM/stage1.sh" << 'STAGE1_EOF'
#!/bin/bash
#SBATCH --job-name=nmrapp_fold
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
STAGE1_EOF

    cat >> "$TMPDIR_SLURM/stage1.sh" << EOF
#SBATCH --gres=${GPU_GRES}
#SBATCH --partition=${GPU_PARTITION}
#SBATCH --output=${LOG_DIR}/nmrapp_fold_%j.out
#SBATCH --error=${LOG_DIR}/nmrapp_fold_%j.err

source ${VENV_DIR}/bin/activate
cd ${R}

echo "Stage 1: Preprocess + ESMFold — \$(date)"

python scripts/preprocess_nmr_app.py \\
    --xlsx ${NMR_APP_XLSX} \\
    --output_dir ${NMR_APP_DIR} \\
    --fold

echo "Stage 1 done — \$(date)"
EOF
    JOB1=$(sbatch --parsable ${ACCOUNT} "$TMPDIR_SLURM/stage1.sh")
    echo "Stage 1 (preprocess + fold): job $JOB1"
else
    echo "Stage 1 (preprocess + fold): SKIPPED"
fi

# ---- Stage 2: Compute robustness (ThermoMPNN + ProteinMPNN) ----
JOB2=""
if [[ $START_STAGE -le 2 ]]; then
    DEP=""
    [[ -n "$JOB1" ]] && DEP="--dependency=afterok:$JOB1"

    cat > "$TMPDIR_SLURM/stage2.sh" << 'STAGE2_EOF'
#!/bin/bash
#SBATCH --job-name=nmrapp_rob
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
STAGE2_EOF

    cat >> "$TMPDIR_SLURM/stage2.sh" << EOF
#SBATCH --gres=${GPU_GRES}
#SBATCH --partition=${GPU_PARTITION}
#SBATCH --output=${LOG_DIR}/nmrapp_robustness_%j.out
#SBATCH --error=${LOG_DIR}/nmrapp_robustness_%j.err

source ${VENV_DIR}/bin/activate
cd ${R}

echo "Stage 2: Robustness computation — \$(date)"

# ThermoMPNN
python scripts/compute_robustness.py \\
    --scorer thermompnn \\
    --atlas_dir ${NMR_APP_DIR} \\
    --output_dir ${NMR_APP_ROB} \\
    --batch \\
    --device cuda --skip_existing --thermompnn_dir ${T}

# ProteinMPNN
python scripts/compute_robustness.py \\
    --scorer proteinmpnn \\
    --atlas_dir ${NMR_APP_DIR} \\
    --output_dir ${NMR_APP_ROB} \\
    --batch \\
    --device cuda --skip_existing --thermompnn_dir ${T}

echo "Stage 2 done — \$(date)"
EOF
    JOB2=$(sbatch --parsable ${ACCOUNT} $DEP "$TMPDIR_SLURM/stage2.sh")
    echo "Stage 2 (robustness): job $JOB2"
else
    echo "Stage 2 (robustness): SKIPPED"
fi

# ---- Stage 3: Correlate + multi-DDG regression ----
DEP3=""
[[ -n "$JOB2" ]] && DEP3="--dependency=afterok:$JOB2"

cat > "$TMPDIR_SLURM/stage3.sh" << 'STAGE3_EOF'
#!/bin/bash
#SBATCH --job-name=nmrapp_analyze
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
STAGE3_EOF

cat >> "$TMPDIR_SLURM/stage3.sh" << EOF
#SBATCH --partition=${CPU_PARTITION}
#SBATCH --output=${LOG_DIR}/nmrapp_analyze_%j.out
#SBATCH --error=${LOG_DIR}/nmrapp_analyze_%j.err

source ${VENV_DIR}/bin/activate
cd ${R}

echo "Stage 3: Correlation analysis — \$(date)"

# Primary target: hetNOE (stored as 1-hetNOE in _Bfactor.tsv)
for SCORER in thermompnn proteinmpnn; do
    echo "=== \${SCORER}: hetNOE (primary) ==="
    python scripts/correlate_robustness_dynamics.py \\
        --atlas_dir ${NMR_APP_DIR} \\
        --robustness_dir ${NMR_APP_ROB} \\
        --scorer \${SCORER} \\
        --output_dir ${NMR_APP_ANALYSIS}/hetNOE \\
        --target bfactor \\
        --no_dssp
done

# Additional NMR targets: R2 and R2/R1
for SUFFIX in _R2.tsv _R2R1.tsv _R1.tsv; do
    TAG=\$(echo \$SUFFIX | sed 's/^_//;s/\.tsv//')
    for SCORER in thermompnn proteinmpnn; do
        echo "=== \${SCORER}: \${TAG} ==="
        python scripts/correlate_robustness_dynamics.py \\
            --atlas_dir ${NMR_APP_DIR} \\
            --robustness_dir ${NMR_APP_ROB} \\
            --scorer \${SCORER} \\
            --output_dir ${NMR_APP_ANALYSIS}/\${TAG} \\
            --target bfactor \\
            --bfactor_suffix \${SUFFIX} \\
            --no_dssp
    done
done

# Multi-DDG regression on hetNOE
for SCORER in thermompnn proteinmpnn; do
    echo "=== Multi-DDG regression (hetNOE, \${SCORER}) ==="
    python scripts/multi_ddg_regression.py \\
        --atlas_dir ${NMR_APP_DIR} \\
        --robustness_dir ${NMR_APP_ROB} \\
        --scorer \${SCORER} \\
        --target bfactor \\
        --output_dir ${NMR_APP_ANALYSIS}/multi_ddg_hetNOE
done

echo "Stage 3 done — \$(date)"
EOF

if [[ $START_STAGE -le 3 ]]; then
    JOB3=$(sbatch --parsable ${ACCOUNT} $DEP3 "$TMPDIR_SLURM/stage3.sh")
    echo "Stage 3 (analyze): job $JOB3"
else
    echo "Stage 3 (analyze): SKIPPED"
fi

echo ""
echo "=== All jobs submitted. Monitor with: squeue -u \$USER ==="
echo "  Logs: ${LOG_DIR}/nmrapp_*.out"
echo ""
echo "  Results will be in: ${NMR_APP_ANALYSIS}/"
