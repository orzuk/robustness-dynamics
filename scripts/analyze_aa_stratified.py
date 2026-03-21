#!/usr/bin/env python3
"""
Amino-acid-stratified analysis of robustness-dynamics correlations.

Tests whether the robustness-dynamics correlation is driven (partly or fully)
by wild-type amino acid identity.  Produces:
  1. Per-AA summary: mean std(DDG), mean RMSF, mean pLDDT, n_residues
  2. Per-AA Spearman rho(robustness, dynamics) — the key control
  3. Pooled correlation with and without AA as covariate (partial corr)
  4. Figures: per-AA bar charts + within-AA scatter facets

Standalone script — does not modify the main analysis pipeline.

Usage (on cluster):
  python analyze_aa_stratified.py \
      --data-dir /path/to/atlas/proteins \
      --robustness-dir /path/to/atlas_robustness \
      --scorer thermompnn \
      --output-dir /path/to/aa_analysis
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Standard amino acids in conventional order
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)

# Amino acid groupings for summary
AA_GROUPS = {
    "large_hydrophobic": set("FILMVW"),
    "small": set("AGS"),
    "polar": set("NQTY"),
    "charged": set("DEKRH"),
    "special": set("CP"),
}
AA_TO_GROUP = {}
for grp, aas in AA_GROUPS.items():
    for a in aas:
        AA_TO_GROUP[a] = grp


# ============================================================================
# Data loading (mirrors correlate_robustness_dynamics.py loaders)
# ============================================================================

def load_robustness_tsv(robustness_dir: Path, scorer: str, pid: str):
    path = robustness_dir / scorer / f"{pid}_robustness.tsv"
    if not path.exists():
        return None
    return pd.read_csv(path, sep="\t")


def load_target_tsv(protein_dir: Path, suffix: str, col_hint: str):
    """Load a per-residue TSV (RMSF, Bfactor, pLDDT) from protein dir."""
    matches = list(protein_dir.glob(f"*{suffix}"))
    if not matches:
        return None
    df = pd.read_csv(matches[0], sep="\t")
    # Find the target column
    candidates = [c for c in df.columns if col_hint.lower() in c.lower()]
    if not candidates:
        # Fallback: last numeric column
        candidates = [c for c in df.columns
                      if df[c].dtype in (np.float64, np.float32, float)]
        candidates = candidates[-1:] if candidates else []
    if not candidates:
        return None
    result = pd.DataFrame({"value": df[candidates[0]].values})
    if "position" in df.columns:
        result["position"] = df["position"].values
    else:
        result["position"] = np.arange(1, len(df) + 1)
    return result


def load_sasa_tsv(protein_dir: Path):
    """Load SASA if available."""
    matches = list(protein_dir.glob("*_SASA.tsv")) + list(protein_dir.glob("*_sasa.tsv"))
    if not matches:
        return None
    df = pd.read_csv(matches[0], sep="\t")
    sasa_cols = [c for c in df.columns if "sasa" in c.lower() or "asa" in c.lower()]
    if not sasa_cols:
        return None
    result = pd.DataFrame({"sasa": df[sasa_cols[0]].values})
    if "position" in df.columns:
        result["position"] = df["position"].values
    else:
        result["position"] = np.arange(1, len(df) + 1)
    return result


# ============================================================================
# Main data collection
# ============================================================================

def collect_residue_data(data_dir: Path, robustness_dir: Path, scorer: str,
                         target_suffix: str, target_col_hint: str,
                         max_proteins: int = 0) -> pd.DataFrame:
    """Collect per-residue data across all proteins.

    Returns DataFrame with columns:
        protein_id, position, wt_aa, std_ddg, mean_ddg, mean_abs_ddg,
        target, plddt, sasa (where available)
    """
    proteins_dir = data_dir / "proteins"
    if not proteins_dir.exists():
        print(f"ERROR: {proteins_dir} not found")
        return pd.DataFrame()

    protein_ids = sorted([
        d.name for d in proteins_dir.iterdir()
        if d.is_dir()
    ])
    if max_proteins > 0:
        protein_ids = protein_ids[:max_proteins]

    all_rows = []
    n_loaded = 0

    for pid in protein_ids:
        protein_dir = proteins_dir / pid

        # Load robustness
        rob_df = load_robustness_tsv(robustness_dir, scorer, pid)
        if rob_df is None or "wt_aa" not in rob_df.columns:
            continue

        # Load target (RMSF or Bfactor)
        tgt_df = load_target_tsv(protein_dir, target_suffix, target_col_hint)
        if tgt_df is None:
            continue

        # Load pLDDT (optional)
        plddt_df = load_target_tsv(protein_dir, "_pLDDT.tsv", "plddt")

        # Position-based merge
        rob_positions = rob_df["position"].values if "position" in rob_df.columns \
            else np.arange(1, len(rob_df) + 1)
        tgt_positions = tgt_df["position"].values

        common = sorted(set(rob_positions) & set(tgt_positions))
        if len(common) < 20:
            continue

        rob_idx = {p: i for i, p in enumerate(rob_positions)}
        tgt_idx = {p: i for i, p in enumerate(tgt_positions)}
        plddt_idx = None
        if plddt_df is not None:
            plddt_idx = {p: i for i, p in enumerate(plddt_df["position"].values)}

        for pos in common:
            ri = rob_idx[pos]
            ti = tgt_idx[pos]

            aa = rob_df["wt_aa"].iloc[ri]
            if aa not in AA_SET:
                continue

            std_ddg = rob_df["std_ddg"].iloc[ri]
            target_val = tgt_df["value"].iloc[ti]

            if not (np.isfinite(std_ddg) and np.isfinite(target_val)):
                continue

            row = {
                "protein_id": pid,
                "position": pos,
                "wt_aa": aa,
                "std_ddg": std_ddg,
                "mean_ddg": rob_df["mean_ddg"].iloc[ri] if "mean_ddg" in rob_df.columns else np.nan,
                "mean_abs_ddg": rob_df["mean_abs_ddg"].iloc[ri] if "mean_abs_ddg" in rob_df.columns else np.nan,
                "target": target_val,
            }

            if plddt_idx is not None and pos in plddt_idx:
                pi = plddt_idx[pos]
                row["plddt"] = plddt_df["value"].iloc[pi]
            else:
                row["plddt"] = np.nan

            row["aa_group"] = AA_TO_GROUP.get(aa, "other")
            all_rows.append(row)

        n_loaded += 1
        if n_loaded % 100 == 0:
            print(f"  Loaded {n_loaded} proteins ({len(all_rows)} residues)...")

    print(f"Collected {len(all_rows)} residues from {n_loaded} proteins")
    return pd.DataFrame(all_rows)


# ============================================================================
# Analysis
# ============================================================================

def per_aa_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean std(DDG), target, pLDDT per amino acid."""
    rows = []
    for aa in AA_ORDER:
        sub = df[df["wt_aa"] == aa]
        if len(sub) < 10:
            continue
        rows.append({
            "aa": aa,
            "group": AA_TO_GROUP.get(aa, "other"),
            "n": len(sub),
            "frac": len(sub) / len(df),
            "mean_std_ddg": sub["std_ddg"].mean(),
            "median_std_ddg": sub["std_ddg"].median(),
            "mean_target": sub["target"].mean(),
            "median_target": sub["target"].median(),
            "mean_plddt": sub["plddt"].mean() if "plddt" in sub.columns else np.nan,
        })
    return pd.DataFrame(rows)


def per_aa_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman rho(robustness, target) stratified by wild-type AA.

    Two modes:
      - pooled: all residues of that AA across all proteins
      - within-protein: compute per-protein rho, then take median
    """
    rows = []
    for aa in AA_ORDER:
        sub = df[df["wt_aa"] == aa]
        if len(sub) < 30:
            rows.append({"aa": aa, "n": len(sub),
                         "rho_pooled": np.nan, "p_pooled": np.nan,
                         "rho_plddt_pooled": np.nan,
                         "median_rho_per_protein": np.nan,
                         "n_proteins": 0})
            continue

        # Pooled correlation
        rho, p = sp_stats.spearmanr(sub["std_ddg"], sub["target"])

        # pLDDT pooled correlation
        plddt_mask = sub["plddt"].notna()
        if plddt_mask.sum() > 30:
            rho_plddt, _ = sp_stats.spearmanr(
                sub.loc[plddt_mask, "plddt"], sub.loc[plddt_mask, "target"])
        else:
            rho_plddt = np.nan

        # Per-protein within-AA correlations
        per_prot_rhos = []
        for pid, grp in sub.groupby("protein_id"):
            if len(grp) >= 5:  # need enough residues of this AA in one protein
                r, _ = sp_stats.spearmanr(grp["std_ddg"], grp["target"])
                if np.isfinite(r):
                    per_prot_rhos.append(r)

        rows.append({
            "aa": aa,
            "group": AA_TO_GROUP.get(aa, "other"),
            "n": len(sub),
            "rho_pooled": rho,
            "p_pooled": p,
            "rho_plddt_pooled": rho_plddt,
            "median_rho_per_protein": np.nanmedian(per_prot_rhos) if per_prot_rhos else np.nan,
            "n_proteins": len(per_prot_rhos),
        })

    return pd.DataFrame(rows)


def overall_with_aa_control(df: pd.DataFrame) -> dict:
    """Compare robustness-target correlation with and without AA covariate."""
    from sklearn.linear_model import LinearRegression

    results = {}

    # Overall pooled Spearman
    rho_all, _ = sp_stats.spearmanr(df["std_ddg"], df["target"])
    results["rho_overall"] = rho_all

    # OLS: target ~ std_ddg
    X_rob = df[["std_ddg"]].values
    y = df["target"].values
    mask = np.isfinite(X_rob.ravel()) & np.isfinite(y)
    lr1 = LinearRegression().fit(X_rob[mask], y[mask])
    results["R2_robustness_only"] = lr1.score(X_rob[mask], y[mask])

    # OLS: target ~ AA dummies
    aa_dummies = pd.get_dummies(df["wt_aa"], prefix="aa", drop_first=True)
    X_aa = aa_dummies.values[mask]
    lr_aa = LinearRegression().fit(X_aa, y[mask])
    results["R2_aa_only"] = lr_aa.score(X_aa, y[mask])

    # OLS: target ~ std_ddg + AA dummies
    X_both = np.column_stack([X_rob[mask], X_aa])
    lr_both = LinearRegression().fit(X_both, y[mask])
    results["R2_robustness_plus_aa"] = lr_both.score(X_both, y[mask])
    results["delta_R2_rob_beyond_aa"] = results["R2_robustness_plus_aa"] - results["R2_aa_only"]
    results["delta_R2_aa_beyond_rob"] = results["R2_robustness_plus_aa"] - results["R2_robustness_only"]

    # Same with pLDDT
    plddt_mask = mask & df["plddt"].notna().values
    if plddt_mask.sum() > 100:
        X_plddt = df[["plddt"]].values[plddt_mask]
        lr_plddt = LinearRegression().fit(X_plddt, y[plddt_mask])
        results["R2_plddt_only"] = lr_plddt.score(X_plddt, y[plddt_mask])

        X_plddt_aa = np.column_stack([X_plddt, aa_dummies.values[plddt_mask]])
        lr_plddt_aa = LinearRegression().fit(X_plddt_aa, y[plddt_mask])
        results["R2_plddt_plus_aa"] = lr_plddt_aa.score(X_plddt_aa, y[plddt_mask])
        results["delta_R2_plddt_beyond_aa"] = results["R2_plddt_plus_aa"] - results["R2_aa_only"]

    # Partial Spearman: residualize target and robustness on AA, then correlate
    resid_rob = X_rob[mask].ravel() - lr_aa.predict(X_aa).ravel()  # wrong, need to residualize rob on AA too
    # Proper partial: regress both rob and target on AA, correlate residuals
    lr_rob_on_aa = LinearRegression().fit(X_aa, X_rob[mask].ravel())
    rob_resid = X_rob[mask].ravel() - lr_rob_on_aa.predict(X_aa)
    lr_tgt_on_aa = LinearRegression().fit(X_aa, y[mask])
    tgt_resid = y[mask] - lr_tgt_on_aa.predict(X_aa)
    rho_partial, _ = sp_stats.spearmanr(rob_resid, tgt_resid)
    results["rho_partial_controlling_aa"] = rho_partial

    results["n_residues"] = int(mask.sum())

    return results


# ============================================================================
# Figures
# ============================================================================

def plot_per_aa_bars(summary_df: pd.DataFrame, corr_df: pd.DataFrame,
                     output_dir: Path, target_label: str):
    """Three-panel bar chart: mean std(DDG), mean target, within-AA rho."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Color by AA group
    group_colors = {
        "large_hydrophobic": "#2166AC",
        "small": "#67A9CF",
        "polar": "#D1E5F0",
        "charged": "#F4A582",
        "special": "#B2182B",
    }

    # Panel (a): mean std(DDG) per AA
    ax = axes[0]
    colors = [group_colors.get(summary_df.iloc[i]["group"], "gray")
              for i in range(len(summary_df))]
    ax.bar(range(len(summary_df)), summary_df["mean_std_ddg"], color=colors)
    ax.set_xticks(range(len(summary_df)))
    ax.set_xticklabels(summary_df["aa"], fontsize=11)
    ax.set_ylabel(r"Mean $\operatorname{std}(\Delta\Delta G)$")
    ax.set_title("(a) Robustness by AA", fontweight="bold")

    # Panel (b): mean target per AA
    ax = axes[1]
    ax.bar(range(len(summary_df)), summary_df["mean_target"], color=colors)
    ax.set_xticks(range(len(summary_df)))
    ax.set_xticklabels(summary_df["aa"], fontsize=11)
    ax.set_ylabel(f"Mean {target_label}")
    ax.set_title(f"(b) {target_label} by AA", fontweight="bold")

    # Panel (c): within-AA pooled rho
    ax = axes[2]
    merged = corr_df[corr_df["aa"].isin(summary_df["aa"].values)].copy()
    merged = merged.set_index("aa").loc[summary_df["aa"].values].reset_index()
    c2 = [group_colors.get(merged.iloc[i]["group"], "gray")
          for i in range(len(merged))]
    ax.bar(range(len(merged)), merged["rho_pooled"], color=c2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(merged)))
    ax.set_xticklabels(merged["aa"], fontsize=11)
    ax.set_ylabel(r"Spearman $\rho$ (within-AA)")
    ax.set_title(r"(c) $\rho$(robustness, target) by AA", fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=group_colors[g], label=g.replace("_", " "))
                       for g in group_colors]
    axes[2].legend(handles=legend_elements, fontsize=9, loc="lower right")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"aa_stratified_bars.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)


def plot_robustness_vs_aa_heatmap(summary_df: pd.DataFrame,
                                   corr_df: pd.DataFrame,
                                   output_dir: Path, target_label: str):
    """Heatmap: AA (rows) x metric (cols) showing that AA identity
    doesn't fully explain the correlation."""

    merged = summary_df.merge(corr_df[["aa", "rho_pooled", "rho_plddt_pooled"]],
                               on="aa")
    merged = merged.sort_values("mean_std_ddg", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 7))
    cols_to_show = ["mean_std_ddg", "mean_target", "mean_plddt",
                    "rho_pooled", "rho_plddt_pooled"]
    col_labels = [r"mean std($\Delta\Delta G$)", f"mean {target_label}",
                  "mean pLDDT", r"$\rho$(rob, target)", r"$\rho$(pLDDT, target)"]

    data = merged[cols_to_show].values.astype(float)
    # Normalize each column to [0,1] for coloring
    data_norm = (data - np.nanmin(data, axis=0)) / (np.nanmax(data, axis=0) - np.nanmin(data, axis=0) + 1e-10)

    im = ax.imshow(data_norm, aspect="auto", cmap="RdBu_r")
    ax.set_yticks(range(len(merged)))
    ax.set_yticklabels(merged["aa"], fontsize=12)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10, rotation=30, ha="right")

    # Annotate with actual values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    ax.set_title("Per-AA summary (sorted by robustness)", fontweight="bold")
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"aa_stratified_heatmap.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Amino-acid-stratified robustness-dynamics analysis")
    parser.add_argument("--data-dir", required=True,
                        help="Dataset directory (contains proteins/ subdir)")
    parser.add_argument("--robustness-dir", required=True,
                        help="Robustness output directory")
    parser.add_argument("--scorer", default="thermompnn",
                        choices=["thermompnn", "esm1v"])
    parser.add_argument("--target", default="rmsf",
                        choices=["rmsf", "bfactor"],
                        help="Dynamics target to analyze")
    parser.add_argument("--bfactor-suffix", default="_Bfactor.tsv",
                        help="Suffix for bfactor/target TSV files")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for results")
    parser.add_argument("--max-proteins", type=int, default=0,
                        help="Limit number of proteins (0 = all)")
    parser.add_argument("--dataset-name", default="",
                        help="Dataset label for output files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine target suffix and column hint
    if args.target == "rmsf":
        target_suffix = "_RMSF.tsv"
        target_col_hint = "rmsf"
        target_label = "RMSF"
    else:
        target_suffix = args.bfactor_suffix
        target_col_hint = "bfactor"
        target_label = "B-factor"

    print(f"=== AA-stratified analysis: {args.dataset_name or args.data_dir} ===")
    print(f"Scorer: {args.scorer}, Target: {args.target}")

    # 1. Collect data
    df = collect_residue_data(
        data_dir=Path(args.data_dir),
        robustness_dir=Path(args.robustness_dir),
        scorer=args.scorer,
        target_suffix=target_suffix,
        target_col_hint=target_col_hint,
        max_proteins=args.max_proteins,
    )
    if df.empty:
        print("No data collected. Exiting.")
        sys.exit(1)

    # Save raw data
    df.to_csv(output_dir / "residue_data.tsv", sep="\t", index=False)

    # 2. Per-AA summary
    summary = per_aa_summary(df)
    summary.to_csv(output_dir / "per_aa_summary.tsv", sep="\t", index=False)
    print("\n=== Per-AA Summary ===")
    print(summary.to_string(index=False))

    # 3. Per-AA correlations
    corr = per_aa_correlations(df)
    corr.to_csv(output_dir / "per_aa_correlations.tsv", sep="\t", index=False)
    print("\n=== Per-AA Correlations ===")
    print(corr.to_string(index=False))

    # 4. Overall with AA control
    control = overall_with_aa_control(df)
    print("\n=== AA Covariate Control ===")
    for k, v in sorted(control.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    with open(output_dir / "aa_control_results.json", "w") as f:
        json.dump(control, f, indent=2)

    # 5. Figures
    plot_per_aa_bars(summary, corr, output_dir, target_label)
    plot_robustness_vs_aa_heatmap(summary, corr, output_dir, target_label)

    print(f"\nResults written to {output_dir}")


if __name__ == "__main__":
    main()
