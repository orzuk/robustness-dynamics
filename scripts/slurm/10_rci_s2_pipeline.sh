#!/bin/bash
#
# Pipeline for RCI-S2 dataset: preprocess, compute ThermoMPNN, run correlations.
# All steps submitted via sbatch -- nothing blocks your terminal.
#
# Usage:
#   bash scripts/slurm/10_rci_s2_pipeline.sh preprocess     # Step 1: ~1 min
#   bash scripts/slurm/10_rci_s2_pipeline.sh thermompnn     # Step 2: ~hours (GPU array job)
#   bash scripts/slurm/10_rci_s2_pipeline.sh all_analysis   # Step 3+4: after ThermoMPNN done
#   bash scripts/slurm/10_rci_s2_pipeline.sh collect        # Step 5: after analysis done
#

set -e

# Source shared config (partitions, paths, etc.)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# RCI data paths
RCI_CSV=$PROJECT_DIR/data/gradation_nmr/zenodo_submission_v2/rci/rci_final.csv
PDB_DIR=$PROJECT_DIR/data/gradation_nmr/zenodo_submission_v2/rci/pdb_files
OUTPUT_DIR=$PROJECT_DIR/data/rci_s2_processed
RCI_ROBUSTNESS_DIR=$PROJECT_DIR/data/rci_s2_robustness
RCI_ANALYSIS_DIR=$PROJECT_DIR/data/rci_s2_analysis
CONSURF_DIR=$PROJECT_DIR/data/ConSurf

STEP=${1:?Usage: bash scripts/slurm/10_rci_s2_pipeline.sh [preprocess|thermompnn|esm1v|all_analysis|collect]}

mkdir -p $LOG_DIR

# ============================================================
# Step 1: Preprocess RCI dataset (sbatch, ~1 min)
# ============================================================
if [[ "$STEP" == "preprocess" ]]; then
    cat > $LOG_DIR/rci_preprocess.slurm << EOF
#!/bin/bash
#SBATCH --job-name=rci_prep
#SBATCH --output=$LOG_DIR/rci_prep_%j.out
#SBATCH --error=$LOG_DIR/rci_prep_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=$CPU_PARTITION

source $VENV_DIR/bin/activate
cd $REPO_DIR

echo "=== Preprocess RCI dataset ==="
echo "Started: \$(date)"

python scripts/preprocess_rci_dataset.py \\
    --rci_csv $RCI_CSV \\
    --pdb_dir $PDB_DIR \\
    --output_dir $OUTPUT_DIR

echo "Finished: \$(date)"
echo "Protein dirs:"
ls $OUTPUT_DIR/proteins/ | wc -l
EOF

    sbatch $LOG_DIR/rci_preprocess.slurm
    echo "Submitted preprocessing job. Check: tail -n 20 $LOG_DIR/rci_prep_*.out"
fi

# ============================================================
# Step 2: Compute ThermoMPNN robustness (GPU array job)
# ============================================================
if [[ "$STEP" == "thermompnn" ]]; then
    echo "=== Step 2: Submit ThermoMPNN SLURM job ==="

    # Create PDB list file for --pdb_list mode
    PDB_LIST=$OUTPUT_DIR/pdb_list.txt
    find $OUTPUT_DIR/proteins -name "*.pdb" -type l | sort > $PDB_LIST
    N_PDBS=$(wc -l < $PDB_LIST)
    echo "  Found $N_PDBS PDB files"

    BATCH_SIZE=$CHUNK_SIZE
    N_BATCHES=$(( (N_PDBS + BATCH_SIZE - 1) / BATCH_SIZE ))
    MAX_IDX=$(( N_BATCHES - 1 ))

    cat > $LOG_DIR/rci_thermompnn.slurm << EOF
#!/bin/bash
#SBATCH --job-name=rci_thermo
#SBATCH --output=$LOG_DIR/rci_thermo_%a_%j.out
#SBATCH --error=$LOG_DIR/rci_thermo_%a_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=$GPU_PARTITION
#SBATCH --gres=$GPU_GRES
#SBATCH --array=0-${MAX_IDX}

source $VENV_DIR/bin/activate
cd $REPO_DIR

BATCH_SIZE=$BATCH_SIZE
START=\$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE + 1 ))
END=\$(( START + BATCH_SIZE - 1 ))

# Extract batch of PDB paths
BATCH_FILE=/tmp/rci_batch_\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}.txt
sed -n "\${START},\${END}p" $PDB_LIST > \$BATCH_FILE

N=\$(wc -l < \$BATCH_FILE)
if [ "\$N" -eq 0 ]; then
    echo "No PDBs in batch \$SLURM_ARRAY_TASK_ID, exiting."
    exit 0
fi

echo "=== Batch \$SLURM_ARRAY_TASK_ID: \$N PDBs (lines \$START to \$END) ==="
echo "Started: \$(date)"

python scripts/compute_robustness.py \\
    --scorer thermompnn \\
    --pdb_list \$BATCH_FILE \\
    --output_dir $RCI_ROBUSTNESS_DIR \\
    --skip_existing

echo "Finished: \$(date)"
EOF

    echo "  Submitting $N_BATCHES array jobs (partition=$GPU_PARTITION)..."
    sbatch $LOG_DIR/rci_thermompnn.slurm
    echo "  Monitor: squeue -u \$USER | grep rci_thermo"
    echo "  Progress: ls $RCI_ROBUSTNESS_DIR/thermompnn/ | wc -l"
fi

# ============================================================
# Step 2b: Compute ESM-1v robustness (GPU array job)
# ============================================================
if [[ "$STEP" == "esm1v" ]]; then
    echo "=== Step 2b: Submit ESM-1v SLURM job ==="

    PDB_LIST=$OUTPUT_DIR/pdb_list.txt
    if [[ ! -f "$PDB_LIST" ]]; then
        find $OUTPUT_DIR/proteins -name "*.pdb" -type l | sort > $PDB_LIST
    fi
    N_PDBS=$(wc -l < $PDB_LIST)
    echo "  Found $N_PDBS PDB files"

    BATCH_SIZE=$CHUNK_SIZE
    N_BATCHES=$(( (N_PDBS + BATCH_SIZE - 1) / BATCH_SIZE ))
    MAX_IDX=$(( N_BATCHES - 1 ))

    cat > $LOG_DIR/rci_esm1v.slurm << EOF
#!/bin/bash
#SBATCH --job-name=rci_esm1v
#SBATCH --output=$LOG_DIR/rci_esm1v_%a_%j.out
#SBATCH --error=$LOG_DIR/rci_esm1v_%a_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=$GPU_PARTITION
#SBATCH --gres=$GPU_GRES
#SBATCH --array=0-${MAX_IDX}

source $VENV_DIR/bin/activate
cd $REPO_DIR

BATCH_SIZE=$BATCH_SIZE
START=\$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE + 1 ))
END=\$(( START + BATCH_SIZE - 1 ))

BATCH_FILE=/tmp/rci_esm1v_batch_\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}.txt
sed -n "\${START},\${END}p" $PDB_LIST > \$BATCH_FILE

N=\$(wc -l < \$BATCH_FILE)
if [ "\$N" -eq 0 ]; then
    echo "No PDBs in batch \$SLURM_ARRAY_TASK_ID, exiting."
    exit 0
fi

echo "=== ESM-1v Batch \$SLURM_ARRAY_TASK_ID: \$N PDBs (lines \$START to \$END) ==="
echo "Started: \$(date)"

python scripts/compute_robustness.py \\
    --scorer esm1v \\
    --pdb_list \$BATCH_FILE \\
    --output_dir $RCI_ROBUSTNESS_DIR \\
    --device cuda \\
    --skip_existing

echo "Finished: \$(date)"
EOF

    echo "  Submitting $N_BATCHES array jobs (partition=$GPU_PARTITION)..."
    sbatch $LOG_DIR/rci_esm1v.slurm
    echo "  Monitor: squeue -u \$USER | grep rci_esm1v"
    echo "  Progress: ls $RCI_ROBUSTNESS_DIR/esm1v/ 2>/dev/null | wc -l"
fi

# ============================================================
# Step 3+4: Correlation + Multi-DDG (CPU sbatch)
# ============================================================
if [[ "$STEP" == "all_analysis" ]]; then
    # Correlation
    cat > $LOG_DIR/rci_correlate.slurm << EOF
#!/bin/bash
#SBATCH --job-name=rci_corr
#SBATCH --output=$LOG_DIR/rci_corr_%j.out
#SBATCH --error=$LOG_DIR/rci_corr_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=$CPU_PARTITION

source $VENV_DIR/bin/activate
cd $REPO_DIR

echo "=== RCI-S2 Correlation Analysis ==="
echo "Started: \$(date)"

python scripts/correlate_robustness_dynamics.py \\
    --atlas_dir $OUTPUT_DIR \\
    --robustness_dir $RCI_ROBUSTNESS_DIR \\
    --scorer thermompnn \\
    --output_dir $RCI_ANALYSIS_DIR \\
    --target bfactor \\
    --consurf_dir $CONSURF_DIR

echo "Finished: \$(date)"
EOF

    sbatch $LOG_DIR/rci_correlate.slurm
    echo "Submitted ThermoMPNN correlation job"

    # ESM-1v correlation (if robustness files exist)
    if [[ -d "$RCI_ROBUSTNESS_DIR/esm1v" ]]; then
        cat > $LOG_DIR/rci_correlate_esm.slurm << EOF
#!/bin/bash
#SBATCH --job-name=rci_corr_esm
#SBATCH --output=$LOG_DIR/rci_corr_esm_%j.out
#SBATCH --error=$LOG_DIR/rci_corr_esm_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=$CPU_PARTITION

source $VENV_DIR/bin/activate
cd $REPO_DIR

echo "=== RCI-S2 ESM-1v Correlation Analysis ==="
echo "Started: \$(date)"

python scripts/correlate_robustness_dynamics.py \\
    --atlas_dir $OUTPUT_DIR \\
    --robustness_dir $RCI_ROBUSTNESS_DIR \\
    --scorer esm1v \\
    --output_dir $RCI_ANALYSIS_DIR \\
    --target bfactor \\
    --consurf_dir $CONSURF_DIR

echo "Finished: \$(date)"
EOF

        sbatch $LOG_DIR/rci_correlate_esm.slurm
        echo "Submitted ESM-1v correlation job"
    else
        echo "WARNING: No ESM-1v robustness dir found. Run 'esm1v' step first."
    fi

    # Multi-DDG
    cat > $LOG_DIR/rci_multi_ddg.slurm << EOF
#!/bin/bash
#SBATCH --job-name=rci_mddg
#SBATCH --output=$LOG_DIR/rci_mddg_%j.out
#SBATCH --error=$LOG_DIR/rci_mddg_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=$CPU_PARTITION

source $VENV_DIR/bin/activate
cd $REPO_DIR

echo "=== RCI-S2 Multi-DDG Regression ==="
echo "Started: \$(date)"

python scripts/multi_ddg_regression.py \\
    --atlas_dir $OUTPUT_DIR \\
    --robustness_dir $RCI_ROBUSTNESS_DIR \\
    --scorer thermompnn \\
    --output_dir $RCI_ANALYSIS_DIR \\
    --target bfactor

echo "Finished: \$(date)"
EOF

    sbatch $LOG_DIR/rci_multi_ddg.slurm
    echo "Submitted multi-DDG regression job"
fi

# ============================================================
# Step 5: Collect results (sbatch)
# ============================================================
if [[ "$STEP" == "collect" ]]; then
    cat > $LOG_DIR/rci_collect.slurm << EOF
#!/bin/bash
#SBATCH --job-name=rci_coll
#SBATCH --output=$LOG_DIR/rci_collect_%j.out
#SBATCH --error=$LOG_DIR/rci_collect_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=$CPU_PARTITION

source $VENV_DIR/bin/activate
cd $REPO_DIR

echo "=== Collect all results ==="
echo "Started: \$(date)"

python scripts/collect_results.py \\
    --output $PROJECT_DIR/data/paper_results/unified_results.json \\
    --verbose

echo "Finished: \$(date)"
EOF

    sbatch $LOG_DIR/rci_collect.slurm
    echo "Submitted collect job"
fi
