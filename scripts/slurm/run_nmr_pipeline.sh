#!/bin/bash
# ============================================================================
# Run the full robustness-dynamics pipeline for the two new NMR datasets:
#   1. RelaxDB (143 proteins, per-residue R1/R2/hetNOE)
#   2. S2 experimental (42 proteins, Lipari-Szabo order parameters)
#
# Three stages, chained via SLURM dependencies:
#   Stage 1: Preprocess (CPU) — parse raw data, download AF2 PDBs
#   Stage 2: Compute robustness (GPU) — ThermoMPNN DDG predictions
#   Stage 3: Correlate + regression (CPU) — run analysis pipeline
#
# Usage:
#   bash scripts/slurm/run_nmr_pipeline.sh          # run all 3 stages
#   bash scripts/slurm/run_nmr_pipeline.sh 2         # start from stage 2
#   bash scripts/slurm/run_nmr_pipeline.sh 3         # start from stage 3
# ============================================================================

set -euo pipefail

# Load shared config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

START_STAGE=${1:-1}

echo "=== NMR Pipeline (stages ${START_STAGE}-3) ==="
echo ""

# ---- Stage 1: Preprocess ----
if [[ $START_STAGE -le 1 ]]; then
    JOB1=$(sbatch --parsable <<'SLURM'
#!/bin/bash
#SBATCH --job-name=nmr_preprocess
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=glacier
#SBATCH --output=/sci/labs/orzuk/orzuk/projects/ProteinStability/logs/nmr_preprocess_%j.out
#SBATCH --error=/sci/labs/orzuk/orzuk/projects/ProteinStability/logs/nmr_preprocess_%j.err

source /sci/labs/orzuk/orzuk/projects/ProteinStability/envs/robustness/bin/activate
cd /sci/labs/orzuk/orzuk/github/robustness-dynamics
P=/sci/labs/orzuk/orzuk/projects/ProteinStability

echo "Stage 1: Preprocess — $(date)"

python scripts/preprocess_relaxdb.py \
    --relaxdb_csv $P/data/NMR_relaxation/relaxdb_data.csv \
    --output_dir $P/data/relaxdb_processed \
    --download_pdbs

python scripts/preprocess_s2_experimental.py \
    --s2_file $P/data/NMR_relaxation/s2_values.txt \
    --output_dir $P/data/s2_exp_processed \
    --download_pdbs

echo "Stage 1 done — $(date)"
SLURM
    echo "Stage 1 (preprocess): job $JOB1"
else
    JOB1=""
    echo "Stage 1 (preprocess): SKIPPED"
fi

# ---- Stage 2: Compute robustness (ThermoMPNN, GPU) ----
if [[ $START_STAGE -le 2 ]]; then
    DEP=""
    [[ -n "${JOB1:-}" ]] && DEP="--dependency=afterok:$JOB1"

    JOB2=$(sbatch --parsable $DEP <<'SLURM'
#!/bin/bash
#SBATCH --job-name=nmr_robustness
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:l4:1
#SBATCH --partition=catfish
#SBATCH --output=/sci/labs/orzuk/orzuk/projects/ProteinStability/logs/nmr_robustness_%j.out
#SBATCH --error=/sci/labs/orzuk/orzuk/projects/ProteinStability/logs/nmr_robustness_%j.err

source /sci/labs/orzuk/orzuk/projects/ProteinStability/envs/robustness/bin/activate
cd /sci/labs/orzuk/orzuk/github/robustness-dynamics
P=/sci/labs/orzuk/orzuk/projects/ProteinStability
T=/sci/labs/orzuk/orzuk/github/ThermoMPNN

echo "Stage 2: ThermoMPNN robustness — $(date)"

python scripts/compute_robustness.py \
    --scorer thermompnn \
    --atlas_dir $P/data/relaxdb_processed \
    --output_dir $P/data/relaxdb_robustness \
    --device cuda --skip_existing --thermompnn_dir $T

python scripts/compute_robustness.py \
    --scorer thermompnn \
    --atlas_dir $P/data/s2_exp_processed \
    --output_dir $P/data/s2_exp_robustness \
    --device cuda --skip_existing --thermompnn_dir $T

echo "Stage 2 done — $(date)"
SLURM
    echo "Stage 2 (robustness): job $JOB2"
else
    JOB2=""
    echo "Stage 2 (robustness): SKIPPED"
fi

# ---- Stage 3: Correlate + multi-DDG regression ----
DEP3=""
[[ -n "${JOB2:-}" ]] && DEP3="--dependency=afterok:$JOB2"

JOB3=$(sbatch --parsable $DEP3 <<'SLURM'
#!/bin/bash
#SBATCH --job-name=nmr_analyze
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=glacier
#SBATCH --output=/sci/labs/orzuk/orzuk/projects/ProteinStability/logs/nmr_analyze_%j.out
#SBATCH --error=/sci/labs/orzuk/orzuk/projects/ProteinStability/logs/nmr_analyze_%j.err

source /sci/labs/orzuk/orzuk/projects/ProteinStability/envs/robustness/bin/activate
cd /sci/labs/orzuk/orzuk/github/robustness-dynamics

echo "Stage 3: Correlate + regression — $(date)"

python scripts/run_all_analyses.py --only-dataset relaxdb --force
python scripts/run_all_analyses.py --only-dataset s2_experimental --force

echo "Stage 3 done — $(date)"
SLURM
echo "Stage 3 (analyze): job $JOB3"

echo ""
echo "=== All jobs submitted. Monitor with: squeue -u \$USER ==="
