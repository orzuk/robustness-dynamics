#!/usr/bin/env python3
"""
Generate all paper figures from analysis outputs.

Reads unified_results.json for summary statistics and per-protein TSV
files for distribution/scatter figures.

Usage:
  python generate_paper_figures.py \
      --results unified_results.json \
      --output-dir /path/to/paper_results
  python generate_paper_figures.py --results ... --figure fig1
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats as scipy_stats

from paper_config import (
    DATASETS,
    FIG1_PANELS, FIG2_PANELS, FIG_NMR_PANELS,
    TABLE1_COLUMNS_ALL,
    CASE_STUDY_PROTEINS, CASE_STUDY_SCORER,
)

# Global font settings for publication readability
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})


# ============================================================================
# HELPERS
# ============================================================================

# Panel labels for each dataset-target combo (main figures: a-d, NMR in supp)
PANEL_LABELS = {
    ("atlas", "rmsf"): "a",
    ("bbflow", "rmsf"): "b",
    ("atlas", "bfactor"): "c",
    ("pdb_designs", "bfactor"): "d",
    ("rci_s2", "bfactor"): "a",  # labeled (a) within its own supp figure
}

PANEL_SUFFIXES = {
    ("atlas", "rmsf"): "atlas_rmsf",
    ("bbflow", "rmsf"): "bbflow_rmsf",
    ("atlas", "bfactor"): "atlas_bfac",
    ("pdb_designs", "bfactor"): "pdb_designs_bfac",
    ("rci_s2", "bfactor"): "nmr_rci_s2",
}


def _add_panel_label(ax, label: str):
    """Add a bold panel label (a), (b), etc. to top-left of axes."""
    ax.text(0.02, 0.98, f"({label})", transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top", ha="left")


def _load_per_protein_tsv(dataset, scorer, target) -> pd.DataFrame:
    """Load per-protein correlations TSV for a run."""
    from paper_config import AnalysisRun
    run = AnalysisRun(dataset=dataset, scorer=scorer, target=target)
    path = Path(run.per_protein_tsv_path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _load_pooled_data(dataset, scorer) -> pd.DataFrame:
    """Load merged per-residue data for pooled scatter/density plots.

    This loads the per-protein data files and concatenates them.
    Each protein's data is z-scored within-protein.
    Returns DataFrame with columns: robustness_z, target_z, protein_id.
    """
    ds = DATASETS[dataset]
    rob_dir = Path(ds.robustness_dir) / scorer
    data_dir = Path(ds.data_dir) / "proteins"

    if not data_dir.exists():
        return pd.DataFrame()

    rows = []
    for protein_dir in sorted(data_dir.iterdir()):
        if not protein_dir.is_dir():
            continue
        pid = protein_dir.name

        # Load robustness
        rob_path = rob_dir / f"{pid}_robustness.tsv"
        if not rob_path.exists():
            continue
        rob_df = pd.read_csv(rob_path, sep="\t")
        if "std_ddg" not in rob_df.columns:
            continue

        # Load target
        rmsf_path = list(protein_dir.glob("*_RMSF.tsv"))
        bfac_path = list(protein_dir.glob("*_Bfactor.tsv"))

        targets = {}
        if rmsf_path:
            rmsf_df = pd.read_csv(rmsf_path[0], sep="\t")
            rmsf_cols = [c for c in rmsf_df.columns
                         if c.lower().startswith("rmsf") or "r1" in c.lower()]
            if rmsf_cols:
                targets["rmsf"] = rmsf_df[rmsf_cols].mean(axis=1).values
        if bfac_path:
            bfac_df = pd.read_csv(bfac_path[0], sep="\t")
            bfac_cols = [c for c in bfac_df.columns
                         if "bfactor" in c.lower() or "b_factor" in c.lower()]
            if bfac_cols:
                targets["bfactor"] = bfac_df[bfac_cols[0]].values

        rob = rob_df["std_ddg"].values
        for tname, tvals in targets.items():
            n = min(len(rob), len(tvals))
            if n < 10:
                continue
            r, t = rob[:n], tvals[:n]
            # Z-score within protein
            r_z = (r - np.nanmean(r)) / (np.nanstd(r) + 1e-10)
            t_z = (t - np.nanmean(t)) / (np.nanstd(t) + 1e-10)
            for i in range(n):
                if np.isfinite(r_z[i]) and np.isfinite(t_z[i]):
                    rows.append({
                        "robustness_z": r_z[i],
                        "target_z": t_z[i],
                        "robustness_raw": float(r[i]),
                        "target_raw": float(t[i]),
                        "target_type": tname,
                        "protein_id": pid,
                    })

    return pd.DataFrame(rows)


# ============================================================================
# FIGURE 1: Per-protein correlation distributions (4 panels)
# ============================================================================

def generate_fig1(results: dict, output_dir: Path):
    """Per-protein rho histograms + scatter plots, one row per dataset-target."""
    n_panels = len(FIG1_PANELS)
    fig, axes = plt.subplots(n_panels, 2, figsize=(12, 4 * n_panels))
    if n_panels == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (ds_name, target) in enumerate(FIG1_PANELS):
        ds = DATASETS[ds_name]
        if target == "rmsf":
            target_label = "RMSF"
        elif ds_name == "rci_s2":
            target_label = r"$1{-}S^2_\mathrm{RCI}$"
        else:
            target_label = "B-factor"
        panel_label = f"{ds.display_name} {target_label}"

        # Histogram panel
        ax_hist = axes[row_idx, 0]
        ax_scat = axes[row_idx, 1]

        # Load per-protein data for each scorer
        for scorer, color, label in [
            ("thermompnn", "tab:blue", "ThMPNN"),
            ("esm1v", "tab:green", "ESM-1v"),
        ]:
            if scorer not in ds.available_scorers:
                continue
            pp = _load_per_protein_tsv(ds_name, scorer, target)
            if pp.empty:
                continue

            # Determine the rho column name
            rho_candidates = [
                f"rho_std_ddg_{target}",
                "rho_robustness_bfactor_target" if target == "bfactor" else "rho_std_ddg_rmsf",
            ]
            rho_col = next((c for c in rho_candidates if c in pp.columns), None)
            if rho_col is None:
                for c in pp.columns:
                    if "rho" in c and ("std_ddg" in c or "robustness_bfactor" in c):
                        rho_col = c
                        break
            if rho_col is None or rho_col not in pp.columns:
                continue

            vals = pp[rho_col].dropna()
            ax_hist.hist(vals, bins=30, alpha=0.5, color=color, label=label)

        # pLDDT (from ThermoMPNN run)
        if ds.has_plddt:
            pp_th = _load_per_protein_tsv(ds_name, "thermompnn", target)
            if not pp_th.empty:
                plddt_col = f"rho_plddt_{target}"
                if plddt_col not in pp_th.columns:
                    plddt_col = "rho_plddt_rmsf" if target == "rmsf" else "rho_plddt_bfactor"
                if plddt_col in pp_th.columns:
                    vals = pp_th[plddt_col].dropna()
                    ax_hist.hist(vals, bins=30, alpha=0.5, color="tab:orange", label="pLDDT")

                    # Scatter: robustness rho vs pLDDT rho
                    rob_candidates = [
                        f"rho_std_ddg_{target}",
                        "rho_robustness_bfactor_target" if target == "bfactor" else "rho_std_ddg_rmsf",
                    ]
                    rob_col = next((c for c in rob_candidates if c in pp_th.columns), None)
                    if rob_col is None:
                        for c in pp_th.columns:
                            if "rho" in c and ("std_ddg" in c or "robustness_bfactor" in c):
                                rob_col = c
                                break
                    if rob_col:
                        both = pp_th[[rob_col, plddt_col]].dropna()
                        ax_scat.scatter(both[rob_col], both[plddt_col],
                                        alpha=0.3, s=10, c="tab:blue")
                        lim = [-1, 1]
                        ax_scat.plot(lim, lim, "k--", alpha=0.5)
                        ax_scat.set_xlim(lim)
                        ax_scat.set_ylim(lim)
                        ax_scat.set_xlabel(r"$\rho$(rob, target)")
                        ax_scat.set_ylabel(r"$\rho$(pLDDT, target)")
        else:
            ax_scat.text(0.5, 0.5, "No pLDDT\navailable",
                         transform=ax_scat.transAxes,
                         ha="center", va="center", fontsize=13, color="gray")
            ax_scat.set_xlim([-1, 1])
            ax_scat.set_ylim([-1, 1])
            ax_scat.set_xlabel(r"$\rho$(rob, target)")
            ax_scat.set_ylabel(r"$\rho$(pLDDT, target)")

        ax_hist.set_title(panel_label, fontweight="bold")
        ax_hist.set_xlabel(r"Per-protein Spearman $\rho$")
        ax_hist.set_ylabel("Count")
        ax_hist.set_xlim([-1, 1])

        # Legend only on first panel, no frame
        if row_idx == 0:
            ax_hist.legend(frameon=False)

    plt.tight_layout()
    # Save combined figure (backward compat)
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"fig1_per_protein_correlations.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save individual panels as standalone figures
    for row_idx, (ds_name, target) in enumerate(FIG1_PANELS):
        ds = DATASETS[ds_name]
        letter = PANEL_LABELS.get((ds_name, target), chr(ord('a') + row_idx))
        suffix = PANEL_SUFFIXES.get((ds_name, target), f"panel_{letter}")
        if target == "rmsf":
            target_label = "RMSF"
        elif ds_name == "rci_s2":
            target_label = r"$1{-}S^2_\mathrm{RCI}$"
        else:
            target_label = "B-factor"
        panel_label = f"{ds.display_name} {target_label}"

        fig_single, (ax_h, ax_s) = plt.subplots(1, 2, figsize=(10, 4))

        for scorer, color, label in [
            ("thermompnn", "tab:blue", "ThMPNN"),
            ("esm1v", "tab:green", "ESM-1v"),
        ]:
            if scorer not in ds.available_scorers:
                continue
            pp = _load_per_protein_tsv(ds_name, scorer, target)
            if pp.empty:
                continue
            rho_candidates = [
                f"rho_std_ddg_{target}",
                "rho_robustness_bfactor_target" if target == "bfactor" else "rho_std_ddg_rmsf",
            ]
            rho_col = next((c for c in rho_candidates if c in pp.columns), None)
            if rho_col is None:
                for c in pp.columns:
                    if "rho" in c and ("std_ddg" in c or "robustness_bfactor" in c):
                        rho_col = c
                        break
            if rho_col is None or rho_col not in pp.columns:
                continue
            vals = pp[rho_col].dropna()
            ax_h.hist(vals, bins=30, alpha=0.5, color=color, label=label)

        if ds.has_plddt:
            pp_th = _load_per_protein_tsv(ds_name, "thermompnn", target)
            if not pp_th.empty:
                plddt_col = f"rho_plddt_{target}"
                if plddt_col not in pp_th.columns:
                    plddt_col = "rho_plddt_rmsf" if target == "rmsf" else "rho_plddt_bfactor"
                if plddt_col in pp_th.columns:
                    vals = pp_th[plddt_col].dropna()
                    ax_h.hist(vals, bins=30, alpha=0.5, color="tab:orange", label="pLDDT")

                    rob_candidates = [
                        f"rho_std_ddg_{target}",
                        "rho_robustness_bfactor_target" if target == "bfactor" else "rho_std_ddg_rmsf",
                    ]
                    rob_col = next((c for c in rob_candidates if c in pp_th.columns), None)
                    if rob_col is None:
                        for c in pp_th.columns:
                            if "rho" in c and ("std_ddg" in c or "robustness_bfactor" in c):
                                rob_col = c
                                break
                    if rob_col:
                        both = pp_th[[rob_col, plddt_col]].dropna()
                        ax_s.scatter(both[rob_col], both[plddt_col],
                                     alpha=0.3, s=10, c="tab:blue")
                        lim = [-1, 1]
                        ax_s.plot(lim, lim, "k--", alpha=0.5)
                        ax_s.set_xlim(lim); ax_s.set_ylim(lim)
                        ax_s.set_xlabel(r"$\rho$(rob, target)")
                        ax_s.set_ylabel(r"$\rho$(pLDDT, target)")

        ax_h.set_title(panel_label, fontweight="bold")
        ax_h.set_xlabel(r"Per-protein Spearman $\rho$")
        ax_h.set_ylabel("Count")
        ax_h.set_xlim([-1, 1])
        ax_h.legend(frameon=False)
        _add_panel_label(ax_h, letter)

        plt.tight_layout()
        for ext in ["pdf", "png"]:
            fig_single.savefig(output_dir / f"fig1{letter}_{suffix}.{ext}",
                               dpi=200, bbox_inches="tight")
        plt.close(fig_single)
    print("  Generated fig1_per_protein_correlations (combined + individual panels)")


# ============================================================================
# FIGURE 2: 2D density scatter with marginals (4 panels)
# ============================================================================

TARGET_UNITS = {
    "rmsf": r"RMSF ($\AA$)",
    "bfactor": r"B-factor ($\AA^2$)",
}


def _density_scatter_panels(panels, fig, use_raw: bool):
    """Shared logic for Fig 2 and Supp Fig 1: hexbin + marginals.

    Parameters
    ----------
    panels : list of (ds_name, target) tuples
    fig : matplotlib Figure
    use_raw : if True, plot raw units; if False, plot z-scored
    """
    n_panels = len(panels)
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols

    for panel_idx, (ds_name, target) in enumerate(panels):
        ds = DATASETS[ds_name]
        if target == "rmsf":
            target_label = "RMSF"
        elif ds_name == "rci_s2":
            target_label = r"$1{-}S^2_\mathrm{RCI}$"
        else:
            target_label = "B-factor"

        pooled = _load_pooled_data(ds_name, "thermompnn")
        if pooled.empty:
            continue

        target_data = pooled[pooled["target_type"] == target]
        if target_data.empty:
            continue

        n_max = 50000
        if len(target_data) > n_max:
            target_data = target_data.sample(n_max, random_state=42)

        if use_raw:
            x = target_data["robustness_raw"].values
            y = target_data["target_raw"].values
            x_label = r"$\operatorname{std}(\Delta\Delta G)$ (kcal/mol)"
            if ds_name == "rci_s2":
                y_label = r"$1 - S^2_\mathrm{RCI}$"
            else:
                y_label = TARGET_UNITS.get(target, target_label)
        else:
            x = target_data["robustness_z"].values
            y = target_data["target_z"].values
            x_label = r"$\operatorname{std}(\Delta\Delta G)$ (z-scored)"
            y_label = f"{target_label} (z-scored)"

        y_clip = np.percentile(y, 99)
        y_floor = np.percentile(y, 1)

        row = panel_idx // n_cols
        col = panel_idx % n_cols

        gs_inner = GridSpec(
            4, 4, figure=fig,
            left=0.07 + 0.48 * col, right=0.07 + 0.48 * col + 0.40,
            bottom=0.07 + (1.0 / n_rows) * (n_rows - 1 - row),
            top=0.07 + (1.0 / n_rows) * (n_rows - 1 - row) + (0.85 / n_rows),
            hspace=0.05, wspace=0.05,
        )

        ax_main = fig.add_subplot(gs_inner[1:, :-1])
        ax_top = fig.add_subplot(gs_inner[0, :-1], sharex=ax_main)
        ax_right = fig.add_subplot(gs_inner[1:, -1], sharey=ax_main)

        mask = (y >= y_floor) & (y <= y_clip)
        hb = ax_main.hexbin(x[mask], y[mask], gridsize=40, cmap="Blues",
                             mincnt=1, linewidths=0.2)
        cb = fig.colorbar(hb, ax=ax_right, pad=0.1, shrink=0.8)
        cb.set_label("Count", fontsize=11)
        cb.ax.tick_params(labelsize=10)
        ax_main.set_xlabel(x_label)
        ax_main.set_ylabel(y_label)
        ax_main.set_ylim(y_floor, y_clip)

        ax_top.hist(x, bins=50, color="tab:blue", alpha=0.7, density=True)
        ax_top.set_ylabel("Density")
        plt.setp(ax_top.get_xticklabels(), visible=False)

        ax_right.hist(y[mask], bins=50, orientation="horizontal",
                       color="tab:orange", alpha=0.7, density=True)
        ax_right.set_xlabel("Density")
        plt.setp(ax_right.get_yticklabels(), visible=False)

        rho = scipy_stats.spearmanr(x, y)[0]
        ax_top.set_title(f"{ds.display_name} {target_label} "
                         f"($\\rho = {rho:.3f}$, $n = {len(x):,}$)",
                         fontsize=14, fontweight="bold")


def _single_density_scatter(ds_name, target, use_raw, output_dir, fig_num, letter, suffix):
    """Generate a single density scatter panel as its own figure."""
    ds = DATASETS[ds_name]
    if target == "rmsf":
        target_label = "RMSF"
    elif ds_name == "rci_s2":
        target_label = r"$1{-}S^2_\mathrm{RCI}$"
    else:
        target_label = "B-factor"

    pooled = _load_pooled_data(ds_name, "thermompnn")
    if pooled.empty:
        return
    target_data = pooled[pooled["target_type"] == target]
    if target_data.empty:
        return
    n_max = 50000
    if len(target_data) > n_max:
        target_data = target_data.sample(n_max, random_state=42)

    if use_raw:
        x = target_data["robustness_raw"].values
        y = target_data["target_raw"].values
        x_label = r"$\operatorname{std}(\Delta\Delta G)$ (kcal/mol)"
        y_label = TARGET_UNITS.get(target, target_label) if ds_name != "rci_s2" else r"$1 - S^2_\mathrm{RCI}$"
    else:
        x = target_data["robustness_z"].values
        y = target_data["target_z"].values
        x_label = r"$\operatorname{std}(\Delta\Delta G)$ (z-scored)"
        y_label = f"{target_label} (z-scored)"

    y_clip = np.percentile(y, 99)
    y_floor = np.percentile(y, 1)
    mask = (y >= y_floor) & (y <= y_clip)

    fig = plt.figure(figsize=(7, 6))
    gs = GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.05)
    ax_main = fig.add_subplot(gs[1:, :-1])
    ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)

    hb = ax_main.hexbin(x[mask], y[mask], gridsize=40, cmap="Blues",
                         mincnt=1, linewidths=0.2)
    cb = fig.colorbar(hb, ax=ax_right, pad=0.1, shrink=0.8)
    cb.set_label("Count", fontsize=11)
    cb.ax.tick_params(labelsize=10)
    ax_main.set_xlabel(x_label)
    ax_main.set_ylabel(y_label)
    ax_main.set_ylim(y_floor, y_clip)

    ax_top.hist(x, bins=50, color="tab:blue", alpha=0.7, density=True)
    ax_top.set_ylabel("Density")
    plt.setp(ax_top.get_xticklabels(), visible=False)

    ax_right.hist(y[mask], bins=50, orientation="horizontal",
                   color="tab:orange", alpha=0.7, density=True)
    ax_right.set_xlabel("Density")
    plt.setp(ax_right.get_yticklabels(), visible=False)

    rho = scipy_stats.spearmanr(x, y)[0]
    ax_top.set_title(f"{ds.display_name} {target_label} "
                     f"($\\rho = {rho:.3f}$, $n = {len(x):,}$)",
                     fontsize=14, fontweight="bold")
    _add_panel_label(ax_top, letter)

    for ext in ["pdf", "png"]:
        fname = f"fig{fig_num}{letter}_{suffix}.{ext}"
        fig.savefig(output_dir / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_fig2(results: dict, output_dir: Path):
    """2D density scatter plots with marginal distributions, one per dataset-target."""
    n_panels = len(FIG2_PANELS)
    n_rows = (n_panels + 1) // 2
    # Combined figure (backward compat)
    fig = plt.figure(figsize=(20, 9 * n_rows))
    _density_scatter_panels(FIG2_PANELS, fig, use_raw=False)
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"fig2_density_scatter.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Individual panels
    for ds_name, target in FIG2_PANELS:
        letter = PANEL_LABELS.get((ds_name, target), "x")
        suffix = PANEL_SUFFIXES.get((ds_name, target), "panel")
        _single_density_scatter(ds_name, target, use_raw=False,
                                output_dir=output_dir, fig_num=2,
                                letter=letter, suffix=suffix)
    print("  Generated fig2_density_scatter (combined + individual panels)")


# ============================================================================
# FIGURE 3 (merged): Model comparison (left) + Ridge coefficients (right)
#   3 rows: RMSF, B-factor, NMR.  2 columns: CV R², coefficients.
# ============================================================================

def generate_fig3(results: dict, output_dir: Path):
    """3x2 merged figure: model CV R² (left) and 24-feature Ridge
    coefficients with error bars (right), one row per target type."""

    model_display = {
        "ols_std_ddg": r"std($\Delta\Delta G$)",
        "ols_mean_abs_ddg": r"mean|$\Delta\Delta G$|",
        "ols_plddt": "pLDDT",
        "ols_sasa": "SASA",
        "ols_std_plddt": "std+pLDDT",
        "ridge_20ddg": r"20 $\Delta\Delta G$",
        "ridge_nonlinear_only": "4 NL",
        "ridge_20ddg_nonlinear": "20+NL",
        "ridge_20ddg_plddt": "20+pLDDT",
        "ridge_20ddg_nonlinear_plddt": "20+NL+pLDDT",
    }

    AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
    NONLINEAR_NAMES = ["std_ddg", "mean|DDG|", "max|DDG|", "min_ddg"]
    NONLINEAR_LABELS = [
        r"std($\Delta\Delta G$)",
        r"mean|$\Delta\Delta G$|",
        r"max|$\Delta\Delta G$|",
        r"min($\Delta\Delta G$)",
    ]
    ALL_FEATURES = AA_ORDER + NONLINEAR_NAMES
    ALL_LABELS = AA_ORDER + NONLINEAR_LABELS
    COEF_MODEL = "ridge_20ddg_nonlinear"

    # 2 rows: RMSF, B-factor (NMR moved to supplementary)
    row_configs = [
        ("RMSF", "rmsf", [
            ("atlas", "tab:blue", "ATLAS"),
            ("bbflow", "tab:orange", "BBFlow"),
        ]),
        ("B-factor", "bfactor", [
            ("atlas", "tab:blue", "ATLAS"),
            ("pdb_designs", "tab:green", "PDB designs"),
        ]),
    ]

    fig, axes = plt.subplots(len(row_configs), 2, figsize=(22, 12),
                             gridspec_kw={"width_ratios": [1, 1.8]})

    for row_idx, (title, target, dataset_list) in enumerate(row_configs):
        # ---- Left: CV R² model comparison ----
        ax_r2 = axes[row_idx, 0]

        all_model_names = []
        for mname in model_display:
            for ds_name, _, _ in dataset_list:
                run_key = f"{ds_name}_thermompnn_{target}"
                run = results.get("runs", {}).get(run_key, {})
                models = run.get("multi_ddg", {}).get("models", {})
                if mname in models:
                    if mname not in all_model_names:
                        all_model_names.append(mname)
                    break

        if all_model_names:
            n_models = len(all_model_names)
            n_datasets = len(dataset_list)
            bar_width = 0.8 / n_datasets
            x = np.arange(n_models)

            for ds_idx, (ds_name, color, label) in enumerate(dataset_list):
                run_key = f"{ds_name}_thermompnn_{target}"
                run = results.get("runs", {}).get(run_key, {})
                models = run.get("multi_ddg", {}).get("models", {})

                r2_vals = []
                r2_stds = []
                for mname in all_model_names:
                    m = models.get(mname, {})
                    r2_vals.append(m.get("cv_r2_mean", 0) or 0)
                    r2_stds.append(m.get("cv_r2_std", 0) or 0)

                offset = (ds_idx - (n_datasets - 1) / 2) * bar_width
                ax_r2.bar(x + offset, r2_vals, bar_width, yerr=r2_stds,
                          color=color, alpha=0.8, capsize=2, label=label)

            display_names = [model_display[m] for m in all_model_names]
            ax_r2.set_xticks(x)
            ax_r2.set_xticklabels(display_names, rotation=45, ha="right")

        ax_r2.set_ylabel("CV $R^2$")
        ax_r2.set_title(f"{title}: model comparison", fontweight="bold")
        # Legend on every left panel (datasets differ by row)
        ax_r2.legend(frameon=False)

        # ---- Right: 24-feature Ridge coefficients with error bars ----
        ax_coef = axes[row_idx, 1]
        n_series = len(dataset_list)
        width = 0.8 / n_series
        # Add gap between AA and nonlinear features
        x_coef = np.arange(len(ALL_FEATURES), dtype=float)
        x_coef[len(AA_ORDER):] += 1.0  # shift NL features right by 1 unit

        for series_idx, (ds_name, color, label) in enumerate(dataset_list):
            run_key = f"{ds_name}_thermompnn_{target}"
            run = results.get("runs", {}).get(run_key, {})
            models = run.get("multi_ddg", {}).get("models", {})
            ridge = models.get(COEF_MODEL, {})
            coefs = ridge.get("feature_coefs_mean")
            if not coefs:
                continue

            feat_names = ridge.get("feature_names", [])
            coef_dict = dict(zip(feat_names, coefs))
            vals = [coef_dict.get(f, 0) for f in ALL_FEATURES]

            # Error bars: prefer theoretical SE, fallback to CV std
            coefs_se = ridge.get("feature_coefs_se")
            coefs_std = ridge.get("feature_coefs_std")
            err_source = coefs_se or coefs_std
            if err_source:
                err_dict = dict(zip(feat_names, err_source))
                errs = [2 * err_dict.get(f, 0) for f in ALL_FEATURES]
            else:
                errs = None

            offset = (series_idx - (n_series - 1) / 2) * width
            ax_coef.bar(x_coef + offset, vals, width, yerr=errs,
                        color=color, alpha=0.8, capsize=2, label=label)

        ax_coef.set_xticks(x_coef)
        ax_coef.set_xticklabels(ALL_LABELS, rotation=45, ha="right")
        ax_coef.set_ylabel("Ridge coefficient")
        ax_coef.set_title(f"{title}: Ridge coefficients (20 AA + 4 NL)",
                          fontweight="bold")
        ax_coef.axhline(0, color="gray", linewidth=0.5)
        # Vertical separator: darker line in the gap between AA and NL
        sep_x = len(AA_ORDER) - 0.5 + 0.5  # midpoint of the gap
        ax_coef.axvline(sep_x, color="black", linewidth=1.2,
                        linestyle="-", alpha=0.7)
        # No legend on right panels (same colors as left)

    plt.tight_layout()
    # Combined figure (backward compat)
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"fig3_model_comparison.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save individual row panels
    row_labels = ["a", "c"]  # RMSF=a, Bfactor=c
    row_suffixes = ["rmsf", "bfactor"]
    for row_idx, (title, target, dataset_list) in enumerate(row_configs):
        rl = row_labels[row_idx]
        rs = row_suffixes[row_idx]

        # --- Left panel: model comparison ---
        fig_l, ax_l = plt.subplots(figsize=(7, 5))
        all_model_names_l = []
        for mname in model_display:
            for ds_name, _, _ in dataset_list:
                run_key = f"{ds_name}_thermompnn_{target}"
                run = results.get("runs", {}).get(run_key, {})
                models = run.get("multi_ddg", {}).get("models", {})
                if mname in models:
                    if mname not in all_model_names_l:
                        all_model_names_l.append(mname)
                    break
        if all_model_names_l:
            n_m = len(all_model_names_l)
            n_d = len(dataset_list)
            bw = 0.8 / n_d
            xp = np.arange(n_m)
            for di, (ds_name, color, label) in enumerate(dataset_list):
                run_key = f"{ds_name}_thermompnn_{target}"
                run = results.get("runs", {}).get(run_key, {})
                models = run.get("multi_ddg", {}).get("models", {})
                r2v = [models.get(mn, {}).get("cv_r2_mean", 0) or 0 for mn in all_model_names_l]
                r2s = [models.get(mn, {}).get("cv_r2_std", 0) or 0 for mn in all_model_names_l]
                off = (di - (n_d - 1) / 2) * bw
                ax_l.bar(xp + off, r2v, bw, yerr=r2s, color=color, alpha=0.8, capsize=2, label=label)
            ax_l.set_xticks(xp)
            ax_l.set_xticklabels([model_display[m] for m in all_model_names_l], rotation=45, ha="right")
        ax_l.set_ylabel("CV $R^2$")
        ax_l.set_title(f"{title}: model comparison", fontweight="bold")
        ax_l.legend(frameon=False)
        _add_panel_label(ax_l, rl)
        plt.tight_layout()
        for ext in ["pdf", "png"]:
            fig_l.savefig(output_dir / f"fig3{rl}_{rs}_models.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig_l)

        # --- Right panel: coefficients ---
        rl2 = chr(ord(rl) + 1)  # b, d, f
        fig_r, ax_r = plt.subplots(figsize=(11, 5))
        n_s = len(dataset_list)
        w = 0.8 / n_s
        x_c = np.arange(len(ALL_FEATURES), dtype=float)
        x_c[len(AA_ORDER):] += 1.0
        for si, (ds_name, color, label) in enumerate(dataset_list):
            run_key = f"{ds_name}_thermompnn_{target}"
            run = results.get("runs", {}).get(run_key, {})
            models = run.get("multi_ddg", {}).get("models", {})
            ridge = models.get(COEF_MODEL, {})
            coefs = ridge.get("feature_coefs_mean")
            if not coefs:
                continue
            feat_names = ridge.get("feature_names", [])
            coef_dict = dict(zip(feat_names, coefs))
            vals = [coef_dict.get(f, 0) for f in ALL_FEATURES]
            coefs_se = ridge.get("feature_coefs_se")
            coefs_std = ridge.get("feature_coefs_std")
            err_source = coefs_se or coefs_std
            errs = [2 * dict(zip(feat_names, err_source)).get(f, 0) for f in ALL_FEATURES] if err_source else None
            off = (si - (n_s - 1) / 2) * w
            ax_r.bar(x_c + off, vals, w, yerr=errs, color=color, alpha=0.8, capsize=2, label=label)
        ax_r.set_xticks(x_c)
        ax_r.set_xticklabels(ALL_LABELS, rotation=45, ha="right")
        ax_r.set_ylabel("Ridge coefficient")
        ax_r.set_title(f"{title}: Ridge coefficients (20 AA + 4 NL)", fontweight="bold")
        ax_r.axhline(0, color="gray", linewidth=0.5)
        sep_x = len(AA_ORDER) - 0.5 + 0.5
        ax_r.axvline(sep_x, color="black", linewidth=1.2, linestyle="-", alpha=0.7)
        _add_panel_label(ax_r, rl2)
        plt.tight_layout()
        for ext in ["pdf", "png"]:
            fig_r.savefig(output_dir / f"fig3{rl2}_{rs}_coefs.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig_r)

    print("  Generated fig3_model_comparison (combined + individual panels)")


def generate_fig4(results: dict, output_dir: Path):
    """Kept as no-op; merged into fig3."""
    print("  (fig4 merged into fig3, skipping)")


def generate_supp_fig1(results: dict, output_dir: Path):
    """Raw (un-normalized) density scatter -- same layout as Fig 2 but raw units."""
    n_panels = len(FIG2_PANELS)
    n_rows = (n_panels + 1) // 2
    # Combined figure (backward compat)
    fig = plt.figure(figsize=(20, 9 * n_rows))
    _density_scatter_panels(FIG2_PANELS, fig, use_raw=True)
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"supp_fig1_raw_scatter.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Individual panels
    for ds_name, target in FIG2_PANELS:
        letter = PANEL_LABELS.get((ds_name, target), "x")
        suffix = PANEL_SUFFIXES.get((ds_name, target), "panel")
        _single_density_scatter(ds_name, target, use_raw=True,
                                output_dir=output_dir, fig_num="s1",
                                letter=letter, suffix=suffix)
    print("  Generated supp_fig1_raw_scatter (combined + individual panels)")


def generate_supp_fig2(results: dict, output_dir: Path):
    """When does robustness beat pLDDT? + Stratify by protein flexibility."""

    # -- Panel A: characterize proteins where |rho_rob| > |rho_plddt| --
    # -- Panel B: rho vs mean target (protein flexibility) --

    panels = []
    for ds_name, target in FIG2_PANELS:
        ds = DATASETS[ds_name]
        pp = _load_per_protein_tsv(ds_name, "thermompnn", target)
        if pp.empty:
            continue

        # Determine rho column names
        if target == "rmsf":
            rho_rob_col = "rho_std_ddg_rmsf"
            rho_plddt_col = "rho_plddt_rmsf"
        else:
            rho_rob_col = "rho_robustness_bfactor_target"
            if rho_rob_col not in pp.columns:
                rho_rob_col = "rho_std_ddg_bfactor"
            rho_plddt_col = "rho_plddt_bfactor"

        if rho_rob_col not in pp.columns or rho_plddt_col not in pp.columns:
            continue

        if target == "rmsf":
            target_label = "RMSF"
        elif ds_name == "rci_s2":
            target_label = r"$1{-}S^2_\mathrm{RCI}$"
        else:
            target_label = "B-factor"

        # Compute mean target per protein from raw per-residue data
        pooled = _load_pooled_data(ds_name, "thermompnn")
        if pooled.empty:
            continue
        target_data = pooled[pooled["target_type"] == target]
        if target_data.empty:
            continue
        mean_target = target_data.groupby("protein_id")["target_raw"].mean()

        panels.append({
            "ds_name": ds_name, "target": target,
            "ds": ds, "target_label": target_label,
            "pp": pp, "rho_rob_col": rho_rob_col,
            "rho_plddt_col": rho_plddt_col,
            "mean_target": mean_target,
        })

    if not panels:
        print("  No data for supp_fig2")
        return

    n = len(panels)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4.5 * n), squeeze=False)

    for i, p in enumerate(panels):
        pp = p["pp"]
        rho_rob = pp[p["rho_rob_col"]].values
        rho_plddt = pp[p["rho_plddt_col"]].values
        valid = np.isfinite(rho_rob) & np.isfinite(rho_plddt)
        rho_rob = rho_rob[valid]
        rho_plddt = rho_plddt[valid]
        pids = pp["protein_id"].values[valid]
        n_res = pp["n_residues_used"].values[valid]

        rob_wins = np.abs(rho_rob) > np.abs(rho_plddt)

        # --- Panel A: What characterizes proteins where robustness wins? ---
        ax_a = axes[i, 0]

        ax_a.scatter(n_res[~rob_wins], np.abs(rho_rob[~rob_wins]),
                     alpha=0.3, s=15, c="tab:blue", label="pLDDT wins")
        ax_a.scatter(n_res[rob_wins], np.abs(rho_rob[rob_wins]),
                     alpha=0.5, s=25, c="tab:red", marker="^",
                     label="Robustness wins")

        frac = rob_wins.sum() / len(rob_wins) * 100
        # Median size comparison
        med_win = np.median(n_res[rob_wins]) if rob_wins.sum() > 0 else 0
        med_lose = np.median(n_res[~rob_wins]) if (~rob_wins).sum() > 0 else 0

        ax_a.set_xlabel("Protein length (residues)")
        ax_a.set_ylabel(r"Per-protein $|\rho|$ (robustness, target)")
        ax_a.set_title(
            f"{p['ds'].display_name} {p['target_label']}\n"
            f"Robustness wins: {frac:.0f}% "
            f"(med. length {med_win:.0f} vs {med_lose:.0f})",
            fontsize=13)
        ax_a.legend(fontsize=11)

        # --- Panel B: rho vs mean target (protein flexibility) ---
        ax_b = axes[i, 1]

        # Match per-protein rho to mean target
        mean_tgt_vals = []
        rho_vals = []
        plddt_vals = []
        for j, pid in enumerate(pids):
            if pid in p["mean_target"].index:
                mean_tgt_vals.append(p["mean_target"][pid])
                rho_vals.append(rho_rob[j])
                plddt_vals.append(rho_plddt[j])
        mean_tgt_vals = np.array(mean_tgt_vals)
        rho_vals = np.array(rho_vals)
        plddt_vals = np.array(plddt_vals)

        if len(mean_tgt_vals) > 5:
            # Bin by flexibility terciles
            t1, t2 = np.percentile(mean_tgt_vals, [33, 67])
            rigid = mean_tgt_vals <= t1
            medium = (mean_tgt_vals > t1) & (mean_tgt_vals <= t2)
            flexible = mean_tgt_vals > t2

            categories = [
                ("Rigid", rigid, "tab:blue"),
                ("Medium", medium, "tab:gray"),
                ("Flexible", flexible, "tab:red"),
            ]

            positions = np.arange(len(categories))
            bar_w = 0.35
            rob_medians = []
            plddt_medians = []
            for label, mask, color in categories:
                rob_medians.append(np.median(np.abs(rho_vals[mask])))
                plddt_medians.append(np.median(np.abs(plddt_vals[mask])))

            ax_b.bar(positions - bar_w / 2, rob_medians, bar_w,
                     color="tab:blue", alpha=0.8, label="Robustness")
            ax_b.bar(positions + bar_w / 2, plddt_medians, bar_w,
                     color="tab:orange", alpha=0.8, label="pLDDT")

            cat_labels = []
            for label, mask, _ in categories:
                cat_labels.append(f"{label}\n(n={mask.sum()})")
            ax_b.set_xticks(positions)
            ax_b.set_xticklabels(cat_labels)
            ax_b.set_ylabel(r"Median per-protein $|\rho|$")
            ax_b.set_title(
                f"{p['ds'].display_name} {p['target_label']}\n"
                f"by protein flexibility tercile",
                fontsize=13)
            ax_b.legend(fontsize=11)

            # Add text with gap
            for k, (rob_m, plddt_m) in enumerate(zip(rob_medians,
                                                      plddt_medians)):
                gap = plddt_m - rob_m
                y_pos = max(rob_m, plddt_m) + 0.02
                ax_b.text(k, y_pos, f"gap={gap:.2f}",
                          ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"supp_fig2_robustness_vs_plddt.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Generated supp_fig2_robustness_vs_plddt")


# ============================================================================
# SUPPLEMENTARY NMR FIGURES (same types as main, for rci_s2 dataset)
# ============================================================================

def generate_supp_nmr_fig1(results: dict, output_dir: Path):
    """NMR per-protein correlation histogram + scatter (same as fig1 panel)."""
    for ds_name, target in FIG_NMR_PANELS:
        letter = PANEL_LABELS.get((ds_name, target), "a")
        suffix = PANEL_SUFFIXES.get((ds_name, target), "panel")

        ds = DATASETS[ds_name]
        target_label = r"$1{-}S^2_\mathrm{RCI}$"
        panel_label = f"{ds.display_name} {target_label}"

        fig_single, (ax_h, ax_s) = plt.subplots(1, 2, figsize=(10, 4))

        for scorer, color, label in [
            ("thermompnn", "tab:blue", "ThMPNN"),
            ("esm1v", "tab:green", "ESM-1v"),
        ]:
            if scorer not in ds.available_scorers:
                continue
            pp = _load_per_protein_tsv(ds_name, scorer, target)
            if pp.empty:
                continue
            rho_candidates = [
                f"rho_std_ddg_{target}",
                "rho_robustness_bfactor_target" if target == "bfactor" else "rho_std_ddg_rmsf",
            ]
            rho_col = next((c for c in rho_candidates if c in pp.columns), None)
            if rho_col is None:
                for c in pp.columns:
                    if "rho" in c and ("std_ddg" in c or "robustness_bfactor" in c):
                        rho_col = c
                        break
            if rho_col is None or rho_col not in pp.columns:
                continue
            vals = pp[rho_col].dropna()
            ax_h.hist(vals, bins=30, alpha=0.5, color=color, label=label)

        if ds.has_plddt:
            pp_th = _load_per_protein_tsv(ds_name, "thermompnn", target)
            if not pp_th.empty:
                plddt_col = f"rho_plddt_{target}"
                if plddt_col not in pp_th.columns:
                    plddt_col = "rho_plddt_bfactor"
                if plddt_col in pp_th.columns:
                    vals = pp_th[plddt_col].dropna()
                    ax_h.hist(vals, bins=30, alpha=0.5, color="tab:orange", label="pLDDT")
                    rob_candidates = [
                        f"rho_std_ddg_{target}",
                        "rho_robustness_bfactor_target",
                    ]
                    rob_col = next((c for c in rob_candidates if c in pp_th.columns), None)
                    if rob_col is None:
                        for c in pp_th.columns:
                            if "rho" in c and ("std_ddg" in c or "robustness_bfactor" in c):
                                rob_col = c
                                break
                    if rob_col:
                        both = pp_th[[rob_col, plddt_col]].dropna()
                        ax_s.scatter(both[rob_col], both[plddt_col],
                                     alpha=0.3, s=10, c="tab:blue")
                        lim = [-1, 1]
                        ax_s.plot(lim, lim, "k--", alpha=0.5)
                        ax_s.set_xlim(lim); ax_s.set_ylim(lim)
                        ax_s.set_xlabel(r"$\rho$(rob, target)")
                        ax_s.set_ylabel(r"$\rho$(pLDDT, target)")

        ax_h.set_title(panel_label, fontweight="bold")
        ax_h.set_xlabel(r"Per-protein Spearman $\rho$")
        ax_h.set_ylabel("Count")
        ax_h.set_xlim([-1, 1])
        ax_h.legend(frameon=False)
        _add_panel_label(ax_h, letter)

        plt.tight_layout()
        for ext in ["pdf", "png"]:
            fig_single.savefig(output_dir / f"supp_nmr_fig1{letter}_{suffix}.{ext}",
                               dpi=200, bbox_inches="tight")
        plt.close(fig_single)
    print("  Generated supp_nmr_fig1 (NMR per-protein correlations)")


def generate_supp_nmr_fig2(results: dict, output_dir: Path):
    """NMR 2D density scatter (same as fig2 panel)."""
    for ds_name, target in FIG_NMR_PANELS:
        letter = PANEL_LABELS.get((ds_name, target), "a")
        suffix = PANEL_SUFFIXES.get((ds_name, target), "panel")
        # Z-scored
        _single_density_scatter(ds_name, target, use_raw=False,
                                output_dir=output_dir, fig_num="s_nmr2",
                                letter=letter, suffix=suffix)
        # Raw
        _single_density_scatter(ds_name, target, use_raw=True,
                                output_dir=output_dir, fig_num="s_nmr2_raw",
                                letter=letter, suffix=suffix)
    print("  Generated supp_nmr_fig2 (NMR density scatter)")


def generate_supp_nmr_fig3(results: dict, output_dir: Path):
    """NMR model comparison + Ridge coefficients (same as fig3 row)."""
    model_display = {
        "ols_std_ddg": r"std($\Delta\Delta G$)",
        "ols_mean_abs_ddg": r"mean|$\Delta\Delta G$|",
        "ols_plddt": "pLDDT",
        "ols_sasa": "SASA",
        "ols_std_plddt": "std+pLDDT",
        "ridge_20ddg": r"20 $\Delta\Delta G$",
        "ridge_nonlinear_only": "4 NL",
        "ridge_20ddg_nonlinear": "20+NL",
        "ridge_20ddg_plddt": "20+pLDDT",
        "ridge_20ddg_nonlinear_plddt": "20+NL+pLDDT",
    }
    AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
    NONLINEAR_NAMES = ["std_ddg", "mean|DDG|", "max|DDG|", "min_ddg"]
    NONLINEAR_LABELS = [
        r"std($\Delta\Delta G$)",
        r"mean|$\Delta\Delta G$|",
        r"max|$\Delta\Delta G$|",
        r"min($\Delta\Delta G$)",
    ]
    ALL_FEATURES = AA_ORDER + NONLINEAR_NAMES
    ALL_LABELS = AA_ORDER + NONLINEAR_LABELS
    COEF_MODEL = "ridge_20ddg_nonlinear"

    title = r"NMR ($1 - S^2_\mathrm{RCI}$)"
    target = "bfactor"
    dataset_list = [("rci_s2", "tab:purple", r"$S^2_\mathrm{RCI}$")]

    # --- Left panel: model comparison ---
    fig_l, ax_l = plt.subplots(figsize=(7, 5))
    all_model_names_l = []
    for mname in model_display:
        for ds_name, _, _ in dataset_list:
            run_key = f"{ds_name}_thermompnn_{target}"
            run = results.get("runs", {}).get(run_key, {})
            models = run.get("multi_ddg", {}).get("models", {})
            if mname in models:
                if mname not in all_model_names_l:
                    all_model_names_l.append(mname)
                break
    if all_model_names_l:
        n_m = len(all_model_names_l)
        n_d = len(dataset_list)
        bw = 0.8 / n_d
        xp = np.arange(n_m)
        for di, (ds_name, color, label) in enumerate(dataset_list):
            run_key = f"{ds_name}_thermompnn_{target}"
            run = results.get("runs", {}).get(run_key, {})
            models = run.get("multi_ddg", {}).get("models", {})
            r2v = [models.get(mn, {}).get("cv_r2_mean", 0) or 0 for mn in all_model_names_l]
            r2s = [models.get(mn, {}).get("cv_r2_std", 0) or 0 for mn in all_model_names_l]
            off = (di - (n_d - 1) / 2) * bw
            ax_l.bar(xp + off, r2v, bw, yerr=r2s, color=color, alpha=0.8, capsize=2, label=label)
        ax_l.set_xticks(xp)
        ax_l.set_xticklabels([model_display[m] for m in all_model_names_l], rotation=45, ha="right")
    ax_l.set_ylabel("CV $R^2$")
    ax_l.set_title(f"{title}: model comparison", fontweight="bold")
    ax_l.legend(frameon=False)
    _add_panel_label(ax_l, "a")
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig_l.savefig(output_dir / f"supp_nmr_fig3a_models.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig_l)

    # --- Right panel: coefficients ---
    fig_r, ax_r = plt.subplots(figsize=(11, 5))
    n_s = len(dataset_list)
    w = 0.8 / n_s
    x_c = np.arange(len(ALL_FEATURES), dtype=float)
    x_c[len(AA_ORDER):] += 1.0
    for si, (ds_name, color, label) in enumerate(dataset_list):
        run_key = f"{ds_name}_thermompnn_{target}"
        run = results.get("runs", {}).get(run_key, {})
        models = run.get("multi_ddg", {}).get("models", {})
        ridge = models.get(COEF_MODEL, {})
        coefs = ridge.get("feature_coefs_mean")
        if not coefs:
            continue
        feat_names = ridge.get("feature_names", [])
        coef_dict = dict(zip(feat_names, coefs))
        vals = [coef_dict.get(f, 0) for f in ALL_FEATURES]
        coefs_se = ridge.get("feature_coefs_se")
        coefs_std = ridge.get("feature_coefs_std")
        err_source = coefs_se or coefs_std
        errs = [2 * dict(zip(feat_names, err_source)).get(f, 0) for f in ALL_FEATURES] if err_source else None
        off = (si - (n_s - 1) / 2) * w
        ax_r.bar(x_c + off, vals, w, yerr=errs, color=color, alpha=0.8, capsize=2, label=label)
    ax_r.set_xticks(x_c)
    ax_r.set_xticklabels(ALL_LABELS, rotation=45, ha="right")
    ax_r.set_ylabel("Ridge coefficient")
    ax_r.set_title(f"{title}: Ridge coefficients (20 AA + 4 NL)", fontweight="bold")
    ax_r.axhline(0, color="gray", linewidth=0.5)
    sep_x = len(AA_ORDER) - 0.5 + 0.5
    ax_r.axvline(sep_x, color="black", linewidth=1.2, linestyle="-", alpha=0.7)
    _add_panel_label(ax_r, "b")
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig_r.savefig(output_dir / f"supp_nmr_fig3b_coefs.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig_r)

    print("  Generated supp_nmr_fig3 (NMR model comparison + coefficients)")


# ============================================================================
# FIGURE 5: Case study (line plots + structure panels)
# ============================================================================

def generate_fig5(results: dict, output_dir: Path):
    """Case study figures: per-residue line plots and structure panels.

    Generates separate panel files for each case study protein.
    Structure panels (PyMOL renders) are only generated if --run-pymol
    is used via the standalone script; here we generate line plots and
    wrap any existing PyMOL renders into standalone panels.
    """
    from types import SimpleNamespace
    from generate_case_study_figure import process_protein

    atlas_ds = DATASETS["atlas"]

    # Check if --run-pymol was passed to the main script
    run_pymol = getattr(generate_fig5, '_run_pymol', False)

    args = SimpleNamespace(
        atlas_dir=atlas_ds.data_dir,
        robustness_dir=atlas_ds.robustness_dir,
        scorer=CASE_STUDY_SCORER,
        output_dir=str(output_dir),
        domains=None,              # auto-discover from data/domains/
        smooth_window=0,
        line_plot_only=not run_pymol,
        run_pymol=run_pymol,
        separate_panels=True,      # generate standalone labeled panels
        no_composite=True,         # skip composite (using LaTeX instead)
        clip_pct=10.0,
        trim_termini=0,
        width=2400,
        height=1800,
    )

    for protein_id in CASE_STUDY_PROTEINS:
        try:
            process_protein(protein_id, args)
        except Exception as e:
            print(f"  ERROR processing {protein_id}: {e}")

    print(f"  Generated case study panels for {len(CASE_STUDY_PROTEINS)} proteins")


# ============================================================================
# MAIN
# ============================================================================

FIGURE_GENERATORS = {
    "fig1": ("Fig 1 (per-protein correlations, 4 panels)", generate_fig1),
    "fig2": ("Fig 2 (density scatter, 4 panels)", generate_fig2),
    "fig3": ("Fig 3 (model comparison, 2 rows)", generate_fig3),
    "fig4": ("Fig 4 (DDG coefficients)", generate_fig4),
    "fig5": ("Fig 5 (case study)", generate_fig5),
    "supp_fig1": ("Supp Fig 1 (raw scatter)", generate_supp_fig1),
    "supp_fig2": ("Supp Fig 2 (robustness vs pLDDT characterization)",
                  generate_supp_fig2),
    "supp_nmr_fig1": ("Supp NMR Fig 1 (per-protein correlations)",
                      generate_supp_nmr_fig1),
    "supp_nmr_fig2": ("Supp NMR Fig 2 (density scatter)",
                      generate_supp_nmr_fig2),
    "supp_nmr_fig3": ("Supp NMR Fig 3 (model comparison + coefficients)",
                      generate_supp_nmr_fig3),
}


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    from paper_config import CLUSTER
    default_base = CLUSTER.paper_results_dir
    parser.add_argument("--results", type=str,
                        default=f"{default_base}/unified_results.json",
                        help="Path to unified_results.json")
    parser.add_argument("--output-dir", type=str, default=default_base,
                        help="Base output directory (figures go into Figures/ subdirectory)")
    parser.add_argument("--figure", type=str, default=None,
                        help="Generate only this figure (e.g., 'fig1', 'fig5')")
    parser.add_argument("--no-case-study", action="store_true",
                        help="Skip case study figure (fig5) generation")
    parser.add_argument("--run-pymol", action="store_true",
                        help="Run PyMOL for case study structure panels "
                             "(requires pymol in PATH)")
    parser.add_argument("--sbatch", action="store_true",
                        help="Submit each figure as a separate SLURM job "
                             "instead of running interactively")
    parser.add_argument("--sbatch-time", type=str, default="01:00:00",
                        help="SLURM time limit per figure job (default: 01:00:00)")
    parser.add_argument("--sbatch-mem", type=str, default="16G",
                        help="SLURM memory per figure job (default: 16G)")
    parser.add_argument("--sbatch-partition", type=str, default="glacier",
                        help="SLURM partition (default: glacier)")
    parser.add_argument("--no-tar", action="store_true",
                        help="Skip creating tar.gz archive of all figures "
                             "(archive is created by default)")
    args = parser.parse_args()

    # Determine which figures to generate
    figs_to_gen = FIGURE_GENERATORS
    if args.figure:
        figs_to_gen = {args.figure: FIGURE_GENERATORS[args.figure]}
    elif args.no_case_study:
        figs_to_gen = {k: v for k, v in FIGURE_GENERATORS.items() if k != "fig5"}

    if args.sbatch:
        # Submit each figure as a separate SLURM job
        import subprocess
        from paper_config import CLUSTER

        script_path = Path(__file__).resolve()
        log_dir = Path(CLUSTER.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        for fig_id in figs_to_gen:
            job_name = f"fig_{fig_id}"
            log_file = log_dir / f"{job_name}_%j.log"

            # Build the command that this job will run
            cmd_parts = [
                f"python {script_path}",
                f"--results {args.results}",
                f"--output-dir {args.output_dir}",
                f"--figure {fig_id}",
            ]
            if args.run_pymol:
                cmd_parts.append("--run-pymol")

            inner_cmd = " ".join(cmd_parts)

            # Activate venv and run
            sbatch_script = (
                f"#!/bin/bash\n"
                f"#SBATCH --job-name={job_name}\n"
                f"#SBATCH --output={log_file}\n"
                f"#SBATCH --time={args.sbatch_time}\n"
                f"#SBATCH --mem={args.sbatch_mem}\n"
                f"#SBATCH --partition={args.sbatch_partition}\n"
                f"#SBATCH --cpus-per-task=2\n"
                f"\n"
                f"source {CLUSTER.venv}/bin/activate\n"
                f"cd {CLUSTER.repo_dir}\n"
                f"{inner_cmd}\n"
            )

            result = subprocess.run(
                ["sbatch"], input=sbatch_script, capture_output=True,
                text=True)
            if result.returncode == 0:
                print(f"  Submitted {fig_id}: {result.stdout.strip()}")
            else:
                print(f"  FAILED to submit {fig_id}: {result.stderr.strip()}")

        print(f"\nSubmitted {len(figs_to_gen)} SLURM jobs. "
              f"Logs in: {log_dir}/fig_*_*.log")
        return

    # Interactive mode: run directly
    with open(args.results) as f:
        results = json.load(f)

    out_dir = Path(args.output_dir) / "Figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass run_pymol flag to fig5 via function attribute
    generate_fig5._run_pymol = args.run_pymol

    for fig_id, (description, generator) in figs_to_gen.items():
        print(f"Generating {description}...")
        try:
            generator(results, out_dir)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Create tar.gz archive of all figures
    if not args.no_tar:
        import tarfile
        tar_path = Path(args.output_dir) / "figures.tar.gz"
        print(f"Creating archive: {tar_path}")
        with tarfile.open(tar_path, "w:gz") as tar:
            for f in sorted(out_dir.iterdir()):
                if f.is_file() and f.suffix in (".pdf", ".png"):
                    tar.add(f, arcname=f.name)
        n_files = sum(1 for f in out_dir.iterdir()
                      if f.is_file() and f.suffix in (".pdf", ".png"))
        print(f"  Archived {n_files} files -> {tar_path}")

    print("Done.")


if __name__ == "__main__":
    main()
