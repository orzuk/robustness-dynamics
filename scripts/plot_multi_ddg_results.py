#!/usr/bin/env python3
"""
Generate figures for multi-DDG regression results.

Reads the JSON result files and produces:
  1. Model comparison: R² bar chart across all models
     - ATLAS mode: RMSF vs B-factor side by side
     - BBFlow mode: BBFlow RMSF vs ATLAS RMSF side by side
  2. Coefficient bar plot: 20 AA coefficients
     - ATLAS mode: RMSF vs B-factor
     - BBFlow mode: BBFlow RMSF vs ATLAS RMSF

Usage:
  # ATLAS only (original behavior):
  python plot_multi_ddg_results.py \
      --results_dir /path/to/atlas_analysis/thermompnn \
      --output_dir /path/to/output

  # BBFlow (RMSF only, compared against ATLAS):
  python plot_multi_ddg_results.py \
      --results_dir /path/to/bbflow_analysis/thermompnn \
      --output_dir /path/to/output \
      --atlas_results_dir /path/to/atlas_analysis/thermompnn \
      --dataset bbflow
"""

import json
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

MODEL_ORDER = [
    "ols_mean_abs_ddg", "ols_sasa", "ols_plddt", "ols_mean_plddt",
    "ridge_nonlinear_only", "ridge_20ddg",
    "ridge_20ddg_nonlinear", "ridge_20ddg_plddt",
    "ridge_20ddg_nonlinear_plddt",
]
MODEL_LABELS = [
    "mean|DDG|", "SASA", "pLDDT", "mean|DDG|\n+pLDDT",
    "4 NL", "20 DDG",
    "20 DDG\n+4 NL", "20 DDG\n+pLDDT",
    "20 DDG\n+4 NL+pLDDT",
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_coefficients_dual(data_a, data_b, label_a, label_b,
                           output_dir, prefix, title_suffix=""):
    """Bar plot of 20-AA Ridge coefficients for two conditions."""
    coefs_a = data_a["ridge_20ddg"]["feature_coefs_mean"]
    coefs_b = data_b["ridge_20ddg"]["feature_coefs_mean"]

    x = np.arange(len(AA_LIST))
    width = 0.38

    # Sort by first condition's coefficient magnitude
    order = np.argsort(coefs_a)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, [coefs_a[i] for i in order], width,
           label=label_a, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.bar(x + width/2, [coefs_b[i] for i in order], width,
           label=label_b, color="#DD8452", edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([AA_LIST[i] for i in order], fontsize=11,
                       fontweight="bold")
    ax.set_ylabel("Ridge coefficient (z-scored)", fontsize=12)
    ax.set_xlabel("Target amino acid", fontsize=12)
    ax.set_title("Per-amino-acid DDG coefficients for predicting dynamics\n"
                 f"(Ridge regression, 20 DDG features, ThermoMPNN{title_suffix})",
                 fontsize=13)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = Path(output_dir) / f"{prefix}_multi_ddg_coefficients.{ext}"
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def plot_model_comparison_dual(data_a, data_b, label_a, label_b,
                               output_dir, prefix, title_suffix="",
                               n_proteins_a="", n_proteins_b=""):
    """Grouped bar chart comparing R² across models for two conditions."""
    r2_a = [data_a[m]["cv_r2_mean"] for m in MODEL_ORDER]
    r2_b = [data_b[m]["cv_r2_mean"] for m in MODEL_ORDER]
    std_a = [data_a[m]["cv_r2_std"] for m in MODEL_ORDER]
    std_b = [data_b[m]["cv_r2_std"] for m in MODEL_ORDER]

    x = np.arange(len(MODEL_ORDER))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, r2_a, width, yerr=std_a,
           label=label_a, color="#4C72B0", edgecolor="white",
           linewidth=0.5, capsize=3)
    ax.bar(x + width/2, r2_b, width, yerr=std_b,
           label=label_b, color="#DD8452", edgecolor="white",
           linewidth=0.5, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=9, ha="center")
    ax.set_ylabel("Cross-validated $R^2$", fontsize=12)

    n_info = ""
    if n_proteins_a and n_proteins_b:
        n_info = f"\n({label_a}: {n_proteins_a} proteins, {label_b}: {n_proteins_b} proteins)"
    elif n_proteins_a:
        n_info = f"\n({n_proteins_a} proteins)"
    ax.set_title(f"Multi-DDG regression: model comparison"
                 f"\n(ThermoMPNN, 5-fold protein-level CV{title_suffix}){n_info}",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    y_max = max(max(r2_a), max(r2_b))
    ax.set_ylim(0, y_max * 1.3)

    # Delta-R² annotations for both conditions
    best_a = r2_a[-1]
    plddt_a = r2_a[2]
    best_b = r2_b[-1]
    plddt_b = r2_b[2]
    ax.annotate(f"$\\Delta R^2 = +{best_a - plddt_a:.3f}$",
                xy=(len(MODEL_ORDER)-1 - width/2, best_a),
                xytext=(len(MODEL_ORDER)-2.5, best_a + y_max * 0.08),
                fontsize=9, color="#4C72B0",
                arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=1.2))
    ax.annotate(f"$\\Delta R^2 = +{best_b - plddt_b:.3f}$",
                xy=(len(MODEL_ORDER)-1 + width/2, best_b),
                xytext=(len(MODEL_ORDER)-1.5, best_b + y_max * 0.12),
                fontsize=9, color="#DD8452",
                arrowprops=dict(arrowstyle="->", color="#DD8452", lw=1.2))

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = Path(output_dir) / f"{prefix}_multi_ddg_model_comparison.{ext}"
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory with multi_ddg_*_results.json files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for output figures")
    parser.add_argument("--scorer", type=str, default="thermompnn",
                        help="Scorer name for figure filename prefix")
    parser.add_argument("--atlas_results_dir", type=str, default="",
                        help="ATLAS results dir (for BBFlow comparison)")
    parser.add_argument("--dataset", type=str, default="atlas",
                        choices=["atlas", "bbflow"],
                        help="Dataset mode: atlas (RMSF vs Bfactor) or "
                             "bbflow (BBFlow RMSF vs ATLAS RMSF)")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    rmsf_path = Path(args.results_dir) / "multi_ddg_rmsf_results.json"
    rmsf = load_json(rmsf_path)

    if args.dataset == "atlas":
        # Original mode: RMSF vs B-factor on ATLAS
        bf_path = Path(args.results_dir) / "multi_ddg_bfactor_results.json"
        bf = load_json(bf_path)

        plot_coefficients_dual(
            rmsf, bf, "RMSF", "B-factor",
            args.output_dir, args.scorer)
        plot_model_comparison_dual(
            rmsf, bf, "RMSF", "B-factor",
            args.output_dir, args.scorer,
            n_proteins_a="1,938")

    elif args.dataset == "bbflow":
        # BBFlow mode: BBFlow RMSF vs ATLAS RMSF
        if not args.atlas_results_dir:
            raise ValueError("--atlas_results_dir required for bbflow mode")

        atlas_rmsf = load_json(
            Path(args.atlas_results_dir) / "multi_ddg_rmsf_results.json")

        prefix = f"bbflow_{args.scorer}"

        plot_coefficients_dual(
            atlas_rmsf, rmsf, "ATLAS", "BBFlow",
            args.output_dir, prefix,
            title_suffix=", ATLAS vs BBFlow")
        plot_model_comparison_dual(
            atlas_rmsf, rmsf, "ATLAS RMSF", "BBFlow RMSF",
            args.output_dir, prefix,
            title_suffix="",
            n_proteins_a="1,938", n_proteins_b="100")

    print("\nDone.")


if __name__ == "__main__":
    main()
