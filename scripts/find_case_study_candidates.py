#!/usr/bin/env python3
"""Find case study proteins where robustness outperforms pLDDT.

Criteria:
  1. |rho_rob| > |rho_plddt| for RMSF (robustness wins)
  2. Strong overall correlation (|rho_rob| > 0.5)
  3. Relatively short (< 300 residues, ideally < 200)
  4. Single chain (no multimers — protein_id has single chain letter)
  5. Biologically interesting (manual check after filtering)

Usage:
  python scripts/find_case_study_candidates.py \
      --atlas_analysis /sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas_analysis \
      --scorer thermompnn
"""

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas_analysis", required=True)
    parser.add_argument("--scorer", default="thermompnn")
    parser.add_argument("--max_length", type=int, default=300)
    parser.add_argument("--min_rho", type=float, default=0.5,
                        help="Minimum |rho_rob| to consider")
    parser.add_argument("--top_n", type=int, default=30)
    args = parser.parse_args()

    # Load per-protein correlations for RMSF
    rmsf_path = Path(args.atlas_analysis) / args.scorer / "per_protein_correlations_std_ddg.tsv"
    if not rmsf_path.exists():
        print(f"ERROR: {rmsf_path} not found")
        return

    df = pd.read_csv(rmsf_path, sep="\t")
    print(f"Loaded {len(df)} proteins from {rmsf_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())

    # Use exact column names based on known schema
    id_col = ["protein_id"] if "protein_id" in df.columns else []
    n_col = ["seq_length"] if "seq_length" in df.columns else []
    rho_rob_col = ["rho_robustness_rmsf"] if "rho_robustness_rmsf" in df.columns else []
    rho_plddt_col = ["rho_plddt_rmsf"] if "rho_plddt_rmsf" in df.columns else []
    # Also grab B-factor correlations
    rho_rob_bfac_col = ["rho_robustness_bfactor_target"] if "rho_robustness_bfactor_target" in df.columns else []
    rho_plddt_bfac_col = ["rho_plddt_bfactor"] if "rho_plddt_bfactor" in df.columns else []

    if not (id_col and rho_rob_col):
        print("Could not find required columns. Available:")
        for c in df.columns:
            print(f"  {c}")
        return

    # Filter
    filt = df.copy()
    filt["abs_rho_rob"] = filt["rho_robustness_rmsf"].abs()
    filt = filt[filt["abs_rho_rob"] >= args.min_rho]
    print(f"\nAfter |rho_rob_rmsf| >= {args.min_rho}: {len(filt)} proteins")

    filt = filt[filt["seq_length"] <= args.max_length]
    print(f"After length <= {args.max_length}: {len(filt)} proteins")

    filt["abs_rho_plddt"] = filt["rho_plddt_rmsf"].abs()
    filt["rob_wins_rmsf"] = filt["abs_rho_rob"] > filt["abs_rho_plddt"]
    filt["adv_rmsf"] = filt["abs_rho_rob"] - filt["abs_rho_plddt"]

    # Also check B-factor
    if rho_rob_bfac_col and rho_plddt_bfac_col:
        filt["abs_rho_rob_bfac"] = filt["rho_robustness_bfactor_target"].abs()
        filt["abs_rho_plddt_bfac"] = filt["rho_plddt_bfactor"].abs()
        filt["rob_wins_bfac"] = filt["abs_rho_rob_bfac"] > filt["abs_rho_plddt_bfac"]
        filt["adv_bfac"] = filt["abs_rho_rob_bfac"] - filt["abs_rho_plddt_bfac"]
        filt["rob_wins_both"] = filt["rob_wins_rmsf"] & filt["rob_wins_bfac"]

    winners = filt[filt["rob_wins_rmsf"]].copy()
    print(f"Robustness beats pLDDT (RMSF): {len(winners)} proteins")

    if "rob_wins_both" in filt.columns:
        both = filt[filt["rob_wins_both"]]
        print(f"Robustness beats pLDDT (BOTH RMSF + B-factor): {len(both)} proteins")

    # Sort by RMSF advantage
    winners = winners.sort_values("adv_rmsf", ascending=False)

    cols = ["protein_id", "seq_length", "rho_robustness_rmsf", "rho_plddt_rmsf", "adv_rmsf"]
    if "adv_bfac" in winners.columns:
        cols += ["rho_robustness_bfactor_target", "rho_plddt_bfactor", "adv_bfac", "rob_wins_bfac"]

    print(f"\n=== Top {args.top_n} candidates (rob wins RMSF, sorted by advantage) ===")
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    print(winners[cols].head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
