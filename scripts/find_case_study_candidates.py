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

    # Try to identify relevant columns
    # Expected: protein_id, n_residues, rho_std_ddg (or similar), rho_plddt
    id_col = [c for c in df.columns if "protein" in c.lower() or "id" in c.lower()]
    n_col = [c for c in df.columns if "n_res" in c.lower() or "length" in c.lower() or "n_" in c.lower()]
    rho_rob_col = [c for c in df.columns if "rho" in c.lower() and ("std" in c.lower() or "rob" in c.lower())]
    rho_plddt_col = [c for c in df.columns if "rho" in c.lower() and "plddt" in c.lower()]

    print(f"\nDetected columns:")
    print(f"  ID: {id_col}")
    print(f"  N residues: {n_col}")
    print(f"  rho_rob: {rho_rob_col}")
    print(f"  rho_plddt: {rho_plddt_col}")

    if not (id_col and rho_rob_col):
        print("\nCould not auto-detect columns. Printing all column names:")
        for c in df.columns:
            print(f"  {c}: {df[c].dtype}  sample={df[c].iloc[0]}")
        return

    id_c = id_col[0]
    rho_rob_c = rho_rob_col[0]

    # Filter
    filt = df.copy()
    filt["abs_rho_rob"] = filt[rho_rob_c].abs()
    filt = filt[filt["abs_rho_rob"] >= args.min_rho]
    print(f"\nAfter |rho_rob| >= {args.min_rho}: {len(filt)} proteins")

    if n_col:
        n_c = n_col[0]
        filt = filt[filt[n_c] <= args.max_length]
        print(f"After length <= {args.max_length}: {len(filt)} proteins")

    if rho_plddt_col:
        rho_plddt_c = rho_plddt_col[0]
        filt["abs_rho_plddt"] = filt[rho_plddt_c].abs()
        filt["rob_wins"] = filt["abs_rho_rob"] > filt["abs_rho_plddt"]
        filt["rho_advantage"] = filt["abs_rho_rob"] - filt["abs_rho_plddt"]

        winners = filt[filt["rob_wins"]].copy()
        print(f"Robustness beats pLDDT: {len(winners)} proteins")

        # Sort by advantage (how much robustness wins by)
        winners = winners.sort_values("rho_advantage", ascending=False)

        cols_to_show = [id_c, rho_rob_c, rho_plddt_c, "rho_advantage"]
        if n_col:
            cols_to_show.insert(1, n_col[0])

        print(f"\n=== Top {args.top_n} candidates (robustness wins, sorted by advantage) ===")
        print(winners[cols_to_show].head(args.top_n).to_string(index=False))
    else:
        filt = filt.sort_values("abs_rho_rob", ascending=False)
        print(f"\n=== Top {args.top_n} by |rho_rob| (no pLDDT column found) ===")
        print(filt.head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
