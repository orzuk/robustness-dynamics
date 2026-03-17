#!/usr/bin/env python3
"""
Diagnostic: why does NMR show CV R² >> pooled R² for OLS std_ddg?

Generates plots comparing per-protein correlation strength vs protein size
across all 5 dataset-target combinations. This helps explain the discrepancy
between residue-weighted pooled R² and fold-averaged CV R².

Usage:
  python diagnostic_nmr_r2.py \
      --results unified_results.json \
      --output-dir figures/
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from paper_config import DATASETS, TABLE1_COLUMNS


def load_per_protein(dataset, target):
    """Load per-protein correlations TSV."""
    ds = DATASETS[dataset]
    analysis_dir = Path(ds.analysis_dir) / "thermompnn"
    tsv = analysis_dir / "per_protein_correlations_std_ddg.tsv"
    if not tsv.exists():
        return pd.DataFrame()
    df = pd.read_csv(tsv, sep="\t")
    # Pick the right rho column
    rho_col = f"rho_std_ddg_{target}"
    if rho_col not in df.columns:
        rho_col = "rho_robustness_bfactor_target" if target == "bfactor" else "rho_std_ddg_rmsf"
    if rho_col not in df.columns:
        for c in df.columns:
            if "rho" in c and "std_ddg" in c:
                rho_col = c
                break
    if rho_col not in df.columns:
        return pd.DataFrame()
    df["rho"] = df[rho_col]
    return df


def target_label(ds_name, target):
    if target == "rmsf":
        return "RMSF"
    if ds_name == "rci_s2":
        return r"$1{-}S^2_\mathrm{RCI}$"
    return "B-factor"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_panels = len(TABLE1_COLUMNS)

    # ---- Plot 1: |rho| vs protein length ----
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), sharey=True)

    for i, (ds_name, target) in enumerate(TABLE1_COLUMNS):
        ax = axes[i]
        ds = DATASETS[ds_name]
        df = load_per_protein(ds_name, target)
        if df.empty:
            ax.set_title(f"{ds.display_name}\n(no data)")
            continue

        length_col = "n_residues_used" if "n_residues_used" in df.columns else "seq_length"
        if length_col not in df.columns:
            ax.set_title(f"{ds.display_name}\n(no length)")
            continue

        valid = df[[length_col, "rho"]].dropna()
        x = valid[length_col].values
        y = np.abs(valid["rho"].values)

        ax.scatter(x, y, alpha=0.3, s=8, c="tab:blue")
        # Binned means
        bins = np.percentile(x, np.linspace(0, 100, 11))
        bins = np.unique(bins)
        for b0, b1 in zip(bins[:-1], bins[1:]):
            mask = (x >= b0) & (x < b1)
            if mask.sum() > 2:
                ax.plot((b0 + b1) / 2, np.median(y[mask]), "rs", markersize=8)

        r_sp, p_sp = scipy_stats.spearmanr(x, y)
        tl = target_label(ds_name, target)
        ax.set_title(f"{ds.display_name} {tl}\n"
                     f"$\\rho(n, |\\rho|) = {r_sp:.2f}$, n={len(x)}")
        ax.set_xlabel("Protein length (residues)")
        if i == 0:
            ax.set_ylabel(r"Per-protein $|\rho|$ (std $\Delta\Delta G$, target)")

    plt.tight_layout()
    fig.savefig(out_dir / "diagnostic_rho_vs_length.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "diagnostic_rho_vs_length.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Generated diagnostic_rho_vs_length")

    # ---- Plot 2: Distribution of per-protein rho (all 5 panels) ----
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), sharey=True)

    for i, (ds_name, target) in enumerate(TABLE1_COLUMNS):
        ax = axes[i]
        ds = DATASETS[ds_name]
        df = load_per_protein(ds_name, target)
        if df.empty:
            ax.set_title(f"{ds.display_name}\n(no data)")
            continue

        rho_vals = df["rho"].dropna().values
        ax.hist(rho_vals, bins=40, alpha=0.7, color="tab:blue", density=True)
        ax.axvline(np.median(rho_vals), color="red", linewidth=2,
                   label=f"median={np.median(rho_vals):.3f}")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

        # Fraction positive (wrong sign)
        frac_pos = np.mean(rho_vals > 0)
        tl = target_label(ds_name, target)
        ax.set_title(f"{ds.display_name} {tl}\n"
                     f"med={np.median(rho_vals):.3f}, "
                     f"{frac_pos:.0%} positive")
        ax.set_xlabel(r"Per-protein $\rho$")
        ax.legend(fontsize=9)
        if i == 0:
            ax.set_ylabel("Density")

    plt.tight_layout()
    fig.savefig(out_dir / "diagnostic_rho_distributions.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "diagnostic_rho_distributions.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Generated diagnostic_rho_distributions")

    # ---- Plot 3: Weighted vs unweighted R² comparison ----
    # Simulate residue-weighted vs equal-weight averaging
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for i, (ds_name, target) in enumerate(TABLE1_COLUMNS):
        ds = DATASETS[ds_name]
        df = load_per_protein(ds_name, target)
        if df.empty:
            continue

        length_col = "n_residues_used" if "n_residues_used" in df.columns else "seq_length"
        if length_col not in df.columns:
            continue

        valid = df[[length_col, "rho"]].dropna()
        lengths = valid[length_col].values
        rhos = valid["rho"].values

        # Unweighted median |rho|
        med_rho = np.median(np.abs(rhos))
        # Length-weighted mean |rho|
        weights = lengths / lengths.sum()
        weighted_mean_rho = np.sum(weights * np.abs(rhos))
        # Length-weighted mean rho (signed)
        weighted_signed = np.sum(weights * rhos)

        tl = target_label(ds_name, target)
        label = f"{ds.display_name} {tl}"
        ax.scatter(med_rho, weighted_mean_rho, s=100, zorder=5, label=label)
        ax.annotate(label, (med_rho, weighted_mean_rho),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)

    lim = [0, 0.8]
    ax.plot(lim, lim, "k--", alpha=0.5)
    ax.set_xlabel(r"Unweighted median $|\rho|$")
    ax.set_ylabel(r"Length-weighted mean $|\rho|$")
    ax.set_title("Effect of protein-size weighting on correlation strength")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    plt.tight_layout()
    fig.savefig(out_dir / "diagnostic_weighting_effect.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "diagnostic_weighting_effect.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Generated diagnostic_weighting_effect")

    # ---- Print summary stats ----
    print("\n=== Summary statistics ===")
    for ds_name, target in TABLE1_COLUMNS:
        ds = DATASETS[ds_name]
        df = load_per_protein(ds_name, target)
        if df.empty:
            continue
        tl = target_label(ds_name, target)
        rho_vals = df["rho"].dropna().values
        length_col = "n_residues_used" if "n_residues_used" in df.columns else "seq_length"
        lengths = df[length_col].dropna().values if length_col in df.columns else np.array([])

        # Get CV R² and pooled R² from results
        run_key = f"{ds_name}_thermompnn_{target}"
        run = results.get("runs", {}).get(run_key, {})
        pooled_r2 = run.get("pooled", {}).get("r2_robustness_rmsf", None)
        if pooled_r2 is None:
            pooled_r2 = run.get("pooled", {}).get("r2_robustness_bfactor", None)
        cv_r2 = None
        models = run.get("multi_ddg", {}).get("models", {})
        if "ols_std_ddg" in models:
            cv_r2 = models["ols_std_ddg"].get("cv_r2_mean")

        print(f"\n{ds.display_name} {tl}:")
        print(f"  n_proteins = {len(rho_vals)}")
        print(f"  median rho = {np.median(rho_vals):.3f}")
        print(f"  mean |rho| = {np.mean(np.abs(rho_vals)):.3f}")
        print(f"  frac positive = {np.mean(rho_vals > 0):.1%}")
        if len(lengths) > 0:
            print(f"  median length = {np.median(lengths):.0f}")
            print(f"  length range = [{np.min(lengths):.0f}, {np.max(lengths):.0f}]")
            r_len, _ = scipy_stats.spearmanr(lengths[:len(rho_vals)], np.abs(rho_vals))
            print(f"  corr(length, |rho|) = {r_len:.3f}")
        if pooled_r2 is not None:
            print(f"  pooled R² = {pooled_r2:.3f}")
        if cv_r2 is not None:
            print(f"  CV R² (OLS std_ddg) = {cv_r2:.3f}")
        if pooled_r2 is not None and cv_r2 is not None:
            print(f"  ratio CV/pooled = {cv_r2/pooled_r2:.1f}x")


if __name__ == "__main__":
    main()
