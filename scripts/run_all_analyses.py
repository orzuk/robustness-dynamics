#!/usr/bin/env python3
"""
Master orchestrator: run all analyses needed for the paper.

Checks which results exist, submits SLURM jobs for missing analyses,
then collects results and generates tables + figures.

Usage:
  # Dry run — show what would be submitted
  python run_all_analyses.py --dry-run

  # Run all missing analyses
  python run_all_analyses.py

  # Force re-run everything
  python run_all_analyses.py --force

  # Only run correlations for atlas
  python run_all_analyses.py --only-type correlation --only-dataset atlas

  # Skip analysis, just collect + generate outputs
  python run_all_analyses.py --postprocess-only
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

from paper_config import (
    CLUSTER, DATASETS, CORRELATION_RUNS, MULTI_DDG_RUNS,
    SLURM_DEFAULTS, AnalysisRun,
)


def check_output_exists(run: AnalysisRun, analysis_type: str) -> bool:
    """Check if the expected output file already exists."""
    if analysis_type == "correlation":
        return Path(run.pooled_json_path).exists()
    elif analysis_type == "multi_ddg":
        return Path(run.multi_ddg_json_path).exists()
    return False


def build_correlation_command(run: AnalysisRun) -> str:
    """Build the CLI command for a correlation analysis run."""
    ds = run.ds
    cmd_parts = [
        f"python {CLUSTER.scripts_dir}/correlate_robustness_dynamics.py",
        f"--atlas_dir {ds.data_dir}",
        f"--robustness_dir {ds.robustness_dir}",
        f"--scorer {run.scorer}",
        f"--output_dir {ds.analysis_dir}",
    ]
    if ds.bfactor_only:
        cmd_parts.append("--target bfactor")
    return " \\\n    ".join(cmd_parts)


def build_multi_ddg_command(run: AnalysisRun) -> str:
    """Build the CLI command for a multi-DDG regression run."""
    ds = run.ds
    cmd_parts = [
        f"python {CLUSTER.scripts_dir}/multi_ddg_regression.py",
        f"--atlas_dir {ds.data_dir}",
        f"--robustness_dir {ds.robustness_dir}",
        f"--scorer {run.scorer}",
        f"--output_dir {ds.analysis_dir}",
        f"--target {run.target}",
    ]
    return " \\\n    ".join(cmd_parts)


def build_slurm_script(cmd: str, run_key: str, analysis_type: str) -> str:
    """Build a SLURM batch script."""
    slurm_cfg = SLURM_DEFAULTS.get(analysis_type, SLURM_DEFAULTS["correlation"])
    log_dir = CLUSTER.log_dir
    return f"""#!/bin/bash
#SBATCH --job-name={run_key}
#SBATCH --output={log_dir}/{run_key}_%j.out
#SBATCH --error={log_dir}/{run_key}_%j.err
#SBATCH --time={slurm_cfg['time']}
#SBATCH --mem={slurm_cfg['mem']}
#SBATCH --cpus-per-task={slurm_cfg['cpus']}
#SBATCH --partition={slurm_cfg['partition']}

source {CLUSTER.venv}/bin/activate
cd {CLUSTER.repo_dir}

echo "=== {run_key} ({analysis_type}) ==="
echo "Started: $(date)"

{cmd}

echo "Finished: $(date)"
"""


def submit_job(slurm_script: str, run_key: str, dry_run: bool) -> str:
    """Submit a SLURM job. Returns job ID or 'DRY_RUN'."""
    if dry_run:
        print(f"  [DRY RUN] Would submit: {run_key}")
        return "DRY_RUN"

    script_path = Path(CLUSTER.log_dir) / f"{run_key}.slurm"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w") as f:
        f.write(slurm_script)

    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR submitting {run_key}: {result.stderr}")
        return ""

    # Parse job ID from "Submitted batch job 12345"
    job_id = result.stdout.strip().split()[-1]
    print(f"  Submitted {run_key}: job {job_id}")
    return job_id


def run_postprocess(dry_run: bool):
    """Run collect + table/figure generation."""
    results_path = f"{CLUSTER.paper_results_dir}/unified_results.json"
    tables_dir = f"{CLUSTER.paper_results_dir}/tables"
    figures_dir = f"{CLUSTER.paper_results_dir}/figures"

    cmds = [
        f"python {CLUSTER.scripts_dir}/collect_results.py --output {results_path}",
        f"python {CLUSTER.scripts_dir}/generate_latex_tables.py --results {results_path} --output-dir {tables_dir}",
        f"python {CLUSTER.scripts_dir}/generate_paper_figures.py --results {results_path} --output-dir {figures_dir}",
    ]

    for cmd in cmds:
        print(f"\n$ {cmd}")
        if not dry_run:
            os.system(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run all paper analyses")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without submitting")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results exist")
    parser.add_argument("--only-type", choices=["correlation", "multi_ddg"],
                        help="Run only this analysis type")
    parser.add_argument("--only-dataset", type=str,
                        help="Run only this dataset (atlas, bbflow, pdb_designs)")
    parser.add_argument("--postprocess-only", action="store_true",
                        help="Skip analysis, just collect + generate outputs")
    args = parser.parse_args()

    if args.postprocess_only:
        run_postprocess(args.dry_run)
        return

    job_ids = []

    # --- Correlation runs ---
    if args.only_type is None or args.only_type == "correlation":
        print("=== CORRELATION ANALYSES ===")
        for run in CORRELATION_RUNS:
            if args.only_dataset and run.dataset != args.only_dataset:
                continue
            exists = check_output_exists(run, "correlation")
            if exists and not args.force:
                print(f"  SKIP {run.key}: output exists")
                continue

            cmd = build_correlation_command(run)
            script = build_slurm_script(cmd, run.key, "correlation")

            if args.dry_run:
                print(f"\n--- {run.key} ---")
                print(cmd)
            job_id = submit_job(script, run.key, args.dry_run)
            if job_id:
                job_ids.append(job_id)

    # --- Multi-DDG runs ---
    if args.only_type is None or args.only_type == "multi_ddg":
        print("\n=== MULTI-DDG REGRESSION ===")
        for run in MULTI_DDG_RUNS:
            if args.only_dataset and run.dataset != args.only_dataset:
                continue
            exists = check_output_exists(run, "multi_ddg")
            if exists and not args.force:
                print(f"  SKIP {run.key}: output exists")
                continue

            cmd = build_multi_ddg_command(run)
            script = build_slurm_script(cmd, run.key, "multi_ddg")

            if args.dry_run:
                print(f"\n--- {run.key} ---")
                print(cmd)
            job_id = submit_job(script, run.key, args.dry_run)
            if job_id:
                job_ids.append(job_id)

    # --- Post-processing ---
    if job_ids and not args.dry_run:
        # Submit collection job dependent on all analysis jobs
        dep_str = ":".join(j for j in job_ids if j != "DRY_RUN")
        if dep_str:
            print(f"\nAll analysis jobs submitted. Run post-processing after they complete:")
            print(f"  python run_all_analyses.py --postprocess-only")
    elif not job_ids:
        print("\nNo new analyses needed. Running post-processing...")
        run_postprocess(args.dry_run)

    # Summary
    n_submitted = len([j for j in job_ids if j and j != "DRY_RUN"])
    n_skipped = (len(CORRELATION_RUNS) + len(MULTI_DDG_RUNS)) - len(job_ids)
    print(f"\nSummary: {n_submitted} jobs submitted, {n_skipped} skipped (already exist)")


if __name__ == "__main__":
    main()
