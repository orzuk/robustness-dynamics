#!/bin/bash
# Re-run all multi-DDG regressions in parallel (4 sbatch jobs).
# Usage: bash scripts/slurm/rerun_multi_ddg.sh
set -e

P=/sci/labs/orzuk/orzuk/projects/ProteinStability
REPO=/sci/labs/orzuk/orzuk/github/robustness-dynamics
VENV=$P/envs/robustness/bin/activate

sbatch -A orzuk --time=01:00:00 --mem=16G --partition=glacier -o $P/logs/mddg_atlas_rmsf_%j.out --wrap="bash -c 'source $VENV && cd $REPO && python scripts/multi_ddg_regression.py --atlas_dir $P/data/atlas --robustness_dir $P/data/atlas_robustness --scorer thermompnn --target rmsf --output_dir $P/data/atlas_analysis'"

sbatch -A orzuk --time=01:00:00 --mem=16G --partition=glacier -o $P/logs/mddg_atlas_bfac_%j.out --wrap="bash -c 'source $VENV && cd $REPO && python scripts/multi_ddg_regression.py --atlas_dir $P/data/atlas --robustness_dir $P/data/atlas_robustness --scorer thermompnn --target bfactor --output_dir $P/data/atlas_analysis'"

sbatch -A orzuk --time=01:00:00 --mem=16G --partition=glacier -o $P/logs/mddg_bbflow_%j.out --wrap="bash -c 'source $VENV && cd $REPO && python scripts/multi_ddg_regression.py --atlas_dir $P/data/bbflow_processed --robustness_dir $P/data/bbflow_robustness --scorer thermompnn --target rmsf --output_dir $P/data/bbflow_analysis'"

sbatch -A orzuk --time=01:00:00 --mem=16G --partition=glacier -o $P/logs/mddg_pdb_%j.out --wrap="bash -c 'source $VENV && cd $REPO && python scripts/multi_ddg_regression.py --atlas_dir $P/data/pdb_designs --robustness_dir $P/data/pdb_designs_robustness --scorer thermompnn --target bfactor --output_dir $P/data/pdb_designs_analysis'"

echo "Submitted 4 multi-DDG jobs. After all finish, run:"
echo "  python scripts/collect_results.py --output $P/data/paper_results/unified_results.json --verbose"
echo "  python scripts/generate_latex_tables.py"
echo "  python scripts/generate_paper_figures.py"
