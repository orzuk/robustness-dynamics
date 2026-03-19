#!/usr/bin/env python3
"""
Paper configuration: defines all analyses, datasets, and output specifications
for the robustness-dynamics paper.

This is the single source of truth for what analyses are needed and where
data lives. Used by:
  - run_all_analyses.py   (run missing analyses on cluster)
  - collect_results.py    (gather results into unified JSON)
  - generate_paper_outputs.py (generate LaTeX tables + figures)

Usage:
  from paper_config import DATASETS, ANALYSIS_RUNS, CLUSTER_PATHS
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


# ============================================================================
# CLUSTER PATHS
# ============================================================================
# Override via environment variables:
#   ROBUSTNESS_PROJECT_DIR  (default: /sci/labs/orzuk/orzuk/projects/ProteinStability)
#   ROBUSTNESS_REPO_DIR     (default: /sci/labs/orzuk/orzuk/github/robustness-dynamics)

_DEFAULT_PROJECT = "/sci/labs/orzuk/orzuk/projects/ProteinStability"
_DEFAULT_REPO = "/sci/labs/orzuk/orzuk/github/robustness-dynamics"

@dataclass
class ClusterPaths:
    project_dir: str = os.environ.get("ROBUSTNESS_PROJECT_DIR", _DEFAULT_PROJECT)
    repo_dir: str = os.environ.get("ROBUSTNESS_REPO_DIR", _DEFAULT_REPO)

    @property
    def venv(self):
        return f"{self.project_dir}/envs/robustness"

    @property
    def log_dir(self):
        return f"{self.project_dir}/logs"

    @property
    def paper_results_dir(self):
        return f"{self.project_dir}/data/paper_results"

    @property
    def consurf_dir(self):
        return f"{self.project_dir}/data/ConSurf"

    @property
    def scripts_dir(self):
        return f"{self.repo_dir}/scripts"

CLUSTER = ClusterPaths()


# ============================================================================
# DATASET DEFINITIONS
# ============================================================================

@dataclass
class Dataset:
    name: str               # e.g., "atlas", "bbflow", "pdb_designs"
    display_name: str       # e.g., "ATLAS", "BBFlow", "PDB designs"
    data_dir: str           # where proteins/ directory lives
    robustness_dir: str     # where {scorer}/ robustness outputs live
    analysis_dir: str       # where correlation/regression outputs go
    dataset_type: str       # "natural" or "designed"
    available_targets: List[str]   # ["rmsf"], ["bfactor"], or ["rmsf", "bfactor"]
    available_scorers: List[str] = field(default_factory=lambda: ["esm1v", "thermompnn"])
    n_proteins_approx: int = 0
    bfactor_only: bool = False     # pass --target bfactor to correlate script
    has_plddt: bool = True
    exclude_proteins: List[str] = field(default_factory=list)
    bfactor_suffix: str = "_Bfactor.tsv"  # alternative TSV suffix for NMR targets
    virtual: bool = False  # if True, skip in generate_all_runs() (used for alt NMR targets)


# Natural proteins that passed the keyword-based PDB design filter
# but are actually natural or natural-scaffold variants (identified via
# ConSurf-DB coverage and PDB TITLE inspection).
PDB_DESIGNS_EXCLUDE = [
    "2GAR_A",   # natural enzyme (pH-dependent active site loop)
    "2IP6_A",   # natural protein (PEDB)
    "2QSB_A",   # natural protein (uncharacterized family UPF0147)
    "3F4M_A",   # natural protein (TIPE2)
    "4GXT_A",   # natural protein (conserved functionally unknown)
    "5ZEO_A",   # natural protein (sperm whale myoglobin mutant)
    "7AM3_A",   # natural enzyme variant (peptiligase mutant)
    "7AM4_A",   # natural enzyme variant (peptiligase mutant)
    "3NED_A",   # ambiguous (mRouge fluorescent protein, no design keywords)
    "3NF0_A",   # ambiguous (mPlum fluorescent protein variant)
    "3U8V_A",   # ambiguous (small metal binding protein)
    "8A3K_UNK", # ambiguous (no title, unknown chain)
]

DATASETS = {
    "atlas": Dataset(
        name="atlas",
        display_name="ATLAS",
        data_dir=f"{CLUSTER.project_dir}/data/atlas",
        robustness_dir=f"{CLUSTER.project_dir}/data/atlas_robustness",
        analysis_dir=f"{CLUSTER.project_dir}/data/atlas_analysis",
        dataset_type="natural",
        available_targets=["rmsf", "bfactor"],
        n_proteins_approx=1928,
    ),
    "bbflow": Dataset(
        name="bbflow",
        display_name="BBFlow",
        data_dir=f"{CLUSTER.project_dir}/data/bbflow_processed",
        robustness_dir=f"{CLUSTER.project_dir}/data/bbflow_robustness",
        analysis_dir=f"{CLUSTER.project_dir}/data/bbflow_analysis",
        dataset_type="designed",
        available_targets=["rmsf"],
        n_proteins_approx=100,
    ),
    "pdb_designs": Dataset(
        name="pdb_designs",
        display_name="PDB designs",
        data_dir=f"{CLUSTER.project_dir}/data/pdb_designs",
        robustness_dir=f"{CLUSTER.project_dir}/data/pdb_designs_robustness",
        analysis_dir=f"{CLUSTER.project_dir}/data/pdb_designs_analysis",
        dataset_type="designed",
        available_targets=["bfactor"],
        available_scorers=["esm1v", "thermompnn"],
        n_proteins_approx=306,
        bfactor_only=True,
        has_plddt=True,  # pLDDT from ESMFold predictions
        exclude_proteins=PDB_DESIGNS_EXCLUDE,
    ),
    "rci_s2": Dataset(
        name="rci_s2",
        display_name="NMR (RCI-S$^2$)",
        data_dir=f"{CLUSTER.project_dir}/data/rci_s2_processed",
        robustness_dir=f"{CLUSTER.project_dir}/data/rci_s2_robustness",
        analysis_dir=f"{CLUSTER.project_dir}/data/rci_s2_analysis",
        dataset_type="natural",
        available_targets=["bfactor"],  # stores 1 - rciS2 as "bfactor"
        available_scorers=["esm1v", "thermompnn"],
        n_proteins_approx=762,
        bfactor_only=True,
        has_plddt=True,  # AF2 pLDDT from Gavalda-Garcia dataset
    ),
    "relaxdb": Dataset(
        name="relaxdb",
        display_name="NMR (RelaxDB hetNOE)",
        data_dir=f"{CLUSTER.project_dir}/data/relaxdb_processed",
        robustness_dir=f"{CLUSTER.project_dir}/data/relaxdb_robustness",
        analysis_dir=f"{CLUSTER.project_dir}/data/relaxdb_analysis",
        dataset_type="natural",
        available_targets=["bfactor"],  # stores 1 - hetNOE as "bfactor"
        available_scorers=["esm1v", "thermompnn"],
        n_proteins_approx=143,
        bfactor_only=True,
        has_plddt=True,
    ),
    "s2_experimental": Dataset(
        name="s2_experimental",
        display_name="NMR (exp. S$^2$)",
        data_dir=f"{CLUSTER.project_dir}/data/s2_exp_processed",
        robustness_dir=f"{CLUSTER.project_dir}/data/s2_exp_robustness",
        analysis_dir=f"{CLUSTER.project_dir}/data/s2_exp_analysis",
        dataset_type="natural",
        available_targets=["bfactor"],  # stores 1 - S² as "bfactor"
        available_scorers=["esm1v", "thermompnn"],
        n_proteins_approx=42,
        bfactor_only=True,
        has_plddt=True,
    ),
    # Virtual datasets: same proteins as relaxdb but alternative NMR targets
    "relaxdb_R2": Dataset(
        name="relaxdb_R2",
        display_name="NMR (RelaxDB R$_2$)",
        data_dir=f"{CLUSTER.project_dir}/data/relaxdb_processed",
        robustness_dir=f"{CLUSTER.project_dir}/data/relaxdb_robustness",
        analysis_dir=f"{CLUSTER.project_dir}/data/relaxdb_analysis_R2",
        dataset_type="natural",
        available_targets=["bfactor"],
        available_scorers=["thermompnn"],
        n_proteins_approx=143,
        bfactor_only=True,
        has_plddt=True,
        bfactor_suffix="_R2.tsv",
        virtual=True,
    ),
    "relaxdb_R2R1": Dataset(
        name="relaxdb_R2R1",
        display_name="NMR (RelaxDB R$_2$/R$_1$)",
        data_dir=f"{CLUSTER.project_dir}/data/relaxdb_processed",
        robustness_dir=f"{CLUSTER.project_dir}/data/relaxdb_robustness",
        analysis_dir=f"{CLUSTER.project_dir}/data/relaxdb_analysis_R2R1",
        dataset_type="natural",
        available_targets=["bfactor"],
        available_scorers=["thermompnn"],
        n_proteins_approx=143,
        bfactor_only=True,
        has_plddt=True,
        bfactor_suffix="_R2R1.tsv",
        virtual=True,
    ),
}


# ============================================================================
# ANALYSIS RUN DEFINITIONS
# ============================================================================

@dataclass
class AnalysisRun:
    """One specific analysis run = (dataset, scorer, target)."""
    dataset: str
    scorer: str
    target: str  # "rmsf" or "bfactor"

    @property
    def key(self) -> str:
        return f"{self.dataset}_{self.scorer}_{self.target}"

    @property
    def ds(self) -> Dataset:
        return DATASETS[self.dataset]

    @property
    def pooled_json_path(self) -> str:
        return f"{self.ds.analysis_dir}/{self.scorer}/pooled_results_std_ddg.json"

    @property
    def stratified_json_path(self) -> str:
        return f"{self.ds.analysis_dir}/{self.scorer}/stratified_results_std_ddg.json"

    @property
    def per_protein_tsv_path(self) -> str:
        return f"{self.ds.analysis_dir}/{self.scorer}/per_protein_correlations_std_ddg.tsv"

    @property
    def multi_ddg_json_path(self) -> str:
        return f"{self.ds.analysis_dir}/{self.scorer}/multi_ddg_{self.target}_results.json"


def generate_all_runs() -> List[AnalysisRun]:
    """Generate all valid (dataset, scorer, target) combinations.

    Skips virtual datasets (e.g. relaxdb_R2) which share data with their
    parent but use alternative bfactor suffixes for figure generation only.
    """
    runs = []
    for ds in DATASETS.values():
        if ds.virtual:
            continue
        for scorer in ds.available_scorers:
            for target in ds.available_targets:
                runs.append(AnalysisRun(dataset=ds.name, scorer=scorer, target=target))
    return runs


# All correlation analysis runs (excludes virtual datasets like relaxdb_R2)
CORRELATION_RUNS = generate_all_runs()

# Multi-DDG regression runs (ThermoMPNN only, as ESM-1v was skipped)
MULTI_DDG_RUNS = [r for r in CORRELATION_RUNS if r.scorer == "thermompnn"]

# All runs INCLUDING virtual datasets (for collect_results and figure generation)
ALL_RUNS_WITH_VIRTUAL = CORRELATION_RUNS + [
    AnalysisRun(dataset=ds.name, scorer=scorer, target=target)
    for ds in DATASETS.values() if ds.virtual
    for scorer in ds.available_scorers
    for target in ds.available_targets
]
ALL_MULTI_DDG_WITH_VIRTUAL = [r for r in ALL_RUNS_WITH_VIRTUAL if r.scorer == "thermompnn"]


# ============================================================================
# PAPER TABLE SPECIFICATIONS
# ============================================================================

# Column order for Table 1 (main results) — NMR moved to supplementary
TABLE1_COLUMNS = [
    ("atlas", "rmsf"),
    ("bbflow", "rmsf"),
    ("atlas", "bfactor"),
    ("pdb_designs", "bfactor"),
]

# NMR panels — supplementary material (5 targets across 3 datasets)
NMR_COLUMNS = [
    ("rci_s2", "bfactor"),
    ("relaxdb", "bfactor"),
    ("relaxdb_R2", "bfactor"),
    ("relaxdb_R2R1", "bfactor"),
    ("s2_experimental", "bfactor"),
]

# All columns including NMR (for tables that still need it)
TABLE1_COLUMNS_ALL = TABLE1_COLUMNS + NMR_COLUMNS

# Table 1 predictor rows
TABLE1_PREDICTORS = ["esm1v", "thermompnn", "plddt", "sasa"]

# Table 2 strata
TABLE2_SS_STRATA = ["H", "E", "C"]
TABLE2_BURIAL_STRATA = ["core", "boundary", "surface"]

# Table 3 models for multi-DDG comparison (in display order)
TABLE3_MODEL_ORDER = [
    # Scalar baselines
    "ols_std_ddg",       # OLS on std(DDG) - same as our primary index
    "ols_mean_abs_ddg",
    "ols_sasa",
    "ols_plddt",
    "ols_std_plddt",     # OLS on std(DDG) + pLDDT
    # Ridge models
    "ridge_20ddg",
    "ridge_nonlinear_only",
    "ridge_20ddg_nonlinear",
    "ridge_20ddg_plddt",
    "ridge_20ddg_nonlinear_plddt",
]

# Alternative robustness scalar measures (for Table 3 top half)
ALT_ROBUSTNESS_MEASURES = [
    ("std_ddg", r"$\operatorname{std}(\Delta\Delta G)$"),
    ("mean_ddg", r"mean $\Delta\Delta G$"),
    ("max_ddg", r"$\max|\Delta\Delta G|$"),
    ("frac_destab", r"frac.\ destabilizing"),
    ("mean_abs_ddg", r"mean $|\Delta\Delta G|$"),
    ("frac_neutral", r"frac.\ neutral"),
]


# ============================================================================
# FIGURE SPECIFICATIONS
# ============================================================================

# Figure 1: 4-panel per-protein correlation distributions (2x2, no NMR)
FIG1_PANELS = TABLE1_COLUMNS

# Figure 2: 4-panel 2D density scatter with marginals (2x2, no NMR)
FIG2_PANELS = TABLE1_COLUMNS

# Figure 3: Multi-DDG model comparison (2 rows: RMSF + B-factor, no NMR)
FIG3_PANELS = TABLE1_COLUMNS

# Supplementary NMR panels (same figure types, separate files)
FIG_NMR_PANELS = NMR_COLUMNS

# Display labels for NMR target axes
NMR_TARGET_LABELS = {
    "rci_s2": r"$1 - S^2_\mathrm{RCI}$",
    "relaxdb": r"$1 - \mathrm{hetNOE}$",
    "relaxdb_R2": r"R$_2$ (s$^{-1}$)",
    "relaxdb_R2R1": r"R$_2$/R$_1$",
    "s2_experimental": r"$1 - S^2$",
}

# Figure 4: DDG coefficients (3 panels: ATLAS RMSF vs B-fac, ATLAS vs BBFlow, PDB designs)
FIG4_PANELS = [
    ("atlas", "rmsf"),
    ("atlas", "bfactor"),
    ("bbflow", "rmsf"),
    ("pdb_designs", "bfactor"),
]

# Figure 5: Case study proteins (line plot + structure panels)
CASE_STUDY_PROTEINS_MAIN = ["5ygh_A"]          # main paper Fig 5
CASE_STUDY_PROTEINS_SUPP = ["5ezu_A", "4rp5_A"]  # supplementary
CASE_STUDY_PROTEINS = CASE_STUDY_PROTEINS_MAIN + CASE_STUDY_PROTEINS_SUPP
CASE_STUDY_SCORER = "thermompnn"


# ============================================================================
# SLURM SETTINGS
# ============================================================================

SLURM_DEFAULTS = {
    "correlation": {
        "time": "02:00:00",
        "mem": "16G",
        "cpus": 4,
        "partition": "glacier",
    },
    "multi_ddg": {
        "time": "04:00:00",
        "mem": "32G",
        "cpus": 4,
        "partition": "glacier",
    },
}
