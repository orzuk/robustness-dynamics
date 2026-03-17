#!/bin/bash
#
# Master SLURM script: run all paper analyses + post-processing.
#
# Usage:
#   # Dry run (see what would be submitted):
#   bash scripts/slurm/run_full_pipeline.sh --dry-run
#
#   # Run all missing analyses:
#   bash scripts/slurm/run_full_pipeline.sh
#
#   # Force re-run everything:
#   bash scripts/slurm/run_full_pipeline.sh --force
#
#   # Just regenerate tables + figures from existing results:
#   bash scripts/slurm/run_full_pipeline.sh --postprocess-only

set -euo pipefail

# Source config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh" 2>/dev/null || true

# Defaults (override from config.sh if available)
VENV="${VENV_DIR:-/sci/labs/orzuk/orzuk/projects/ProteinStability/envs/robustness}"
REPO="${REPO_DIR:-/sci/labs/orzuk/orzuk/github/robustness-dynamics}"

# Activate environment
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
fi

# Pass all arguments through to the Python orchestrator
python "${REPO}/scripts/run_all_analyses.py" "$@"
