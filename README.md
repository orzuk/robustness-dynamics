# Mutational Robustness Predicts Protein Dynamics

Code accompanying the paper:

> **Mutational Robustness Predicts Protein Dynamics Across Natural and Designed Proteins**
> Meira Barron & Or Zuk, 2026

## Overview

This repository contains the analysis pipeline for testing whether per-residue mutational robustness (the standard deviation of predicted DDG values across all 19 single-amino-acid substitutions) predicts protein dynamics (RMSF, B-factors, NMR order parameters).

The analysis covers ~3,100 proteins across four datasets:
- **ATLAS** (~1,928 natural proteins with MD simulations)
- **BBFlow** (100 de novo designed proteins with MD)
- **PDB de novo designs** (306 crystallized designed proteins with B-factors)
- **NMR RCI-S2** (759 proteins with NMR-derived order parameters)

DDG values are computed using two predictors:
- **ThermoMPNN** (structure-conditioned, primary scorer)
- **ESM-1v** (sequence-only, secondary scorer)

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: numpy, scipy, pandas, matplotlib, seaborn, scikit-learn, biopython, torch

For DDG computation, you also need:
- [ThermoMPNN](https://github.com/Kuhlman-Lab/ThermoMPNN)
- [ESM](https://github.com/facebookresearch/esm) (for ESM-1v and ESMFold)

## Pipeline

The full pipeline is designed to run on a SLURM cluster. See `scripts/slurm/master_pipeline.sh` for the 7-stage pipeline:

1. **Data download**: ATLAS database, PDB designs (`download_atlas.py`, `download_pdb_designs.py`)
2. **Preprocessing**: BBFlow MD trajectories, NMR RCI-S2 dataset (`prepare_bbflow.py`, `preprocess_rci_dataset.py`)
3. **Robustness computation**: DDG matrices for all proteins (`compute_robustness.py`)
4. **pLDDT computation**: ESMFold predictions (`compute_plddt_esmfold.py`)
5. **Correlation analysis**: Robustness vs dynamics (`correlate_robustness_dynamics.py`)
6. **Multi-DDG regression**: 20-dimensional DDG feature models (`multi_ddg_regression.py`)
7. **Paper outputs**: Tables and figures (`generate_latex_tables.py`, `generate_paper_figures.py`)

### Quick start (cluster)

```bash
# Run the full pipeline
bash scripts/slurm/master_pipeline.sh

# Or run individual stages
python scripts/run_all_analyses.py --dry-run     # check what's missing
python scripts/run_all_analyses.py               # submit missing jobs
python scripts/run_all_analyses.py --postprocess-only  # generate tables/figures
```

### Configuration

All dataset paths and analysis parameters are defined in `scripts/paper_config.py`. Update the cluster paths to match your environment, or set environment variables:

```bash
export ROBUSTNESS_PROJECT_DIR=/path/to/your/project
export ROBUSTNESS_REPO_DIR=/path/to/this/repo
```

## Repository Structure

```
scripts/
  paper_config.py                  -- Central configuration
  compute_robustness.py            -- DDG matrix computation
  correlate_robustness_dynamics.py -- Core correlation analysis
  multi_ddg_regression.py          -- 20-dim DDG regression
  collect_results.py               -- Gather results into unified JSON
  generate_latex_tables.py         -- Auto-generate LaTeX tables
  generate_paper_figures.py        -- Auto-generate figures
  run_all_analyses.py              -- Master orchestrator
  download_atlas.py                -- ATLAS dataset download
  download_pdb_designs.py          -- PDB designs download + filtering
  prepare_bbflow.py                -- BBFlow preprocessing
  preprocess_rci_dataset.py        -- NMR RCI-S2 preprocessing
  compute_plddt_esmfold.py         -- pLDDT via ESMFold
  check_designs_consurf.py         -- ConSurf conservation validation
  diagnostic_nmr_r2.py             -- NMR diagnostic analysis
  plot_multi_ddg_results.py        -- Multi-DDG visualization
  megascale_atlas_overlap.py       -- MegaScale/ATLAS overlap check
  slurm/                           -- SLURM submission scripts
```

## Citation

If you use this code, please cite:

```
Barron, M. & Zuk, O. (2026). Mutational Robustness Predicts Protein Dynamics
Across Natural and Designed Proteins. [Journal TBD].
```

## License

MIT
