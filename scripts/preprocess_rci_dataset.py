#!/usr/bin/env python3
"""
Preprocess the Gavalda-Garcia RCI/S2 dataset for the robustness-dynamics pipeline.

Reads the rci_final.csv from the Zenodo submission and creates per-protein
directory structures compatible with correlate_robustness_dynamics.py.

For each protein (UniProt ID):
  - Creates proteins/{uniprot_id}/ directory
  - Writes {uniprot_id}_Bfactor.tsv with (1 - rciS2) as "bfactor"
    (inverted so high = flexible, matching RMSF/B-factor sign convention)
  - Writes {uniprot_id}_pLDDT.tsv with AF2 pLDDT values
  - Symlinks the AF2 PDB file into the protein directory
  - Creates .done sentinel file

Input:
  rci_final.csv from:
    /sci/labs/orzuk/orzuk/projects/ProteinStability/data/gradation_nmr/
      zenodo_submission_v2/rci/rci_final.csv

  AF2 PDB files from:
    .../rci/pdb_files/{UniprotID}.pdb  (non-truncated)

Output:
  {output_dir}/proteins/{uniprot_id}/
    {uniprot_id}_Bfactor.tsv     # columns: position, bfactor  (= 1 - rciS2)
    {uniprot_id}_pLDDT.tsv       # columns: position, plddt
    {uniprot_id}.pdb             # symlink to AF2 model
    .done

Usage:
  python preprocess_rci_dataset.py \
      --rci_csv /path/to/rci_final.csv \
      --pdb_dir /path/to/rci/pdb_files \
      --output_dir /path/to/rci_s2_processed
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess RCI/S2 dataset for robustness-dynamics pipeline")
    parser.add_argument("--rci_csv", type=str, required=True,
                        help="Path to rci_final.csv")
    parser.add_argument("--pdb_dir", type=str, required=True,
                        help="Path to rci/pdb_files/ directory with AF2 PDBs")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory (will create proteins/ subdirectory)")
    parser.add_argument("--min_residues", type=int, default=30,
                        help="Minimum residues with valid rciS2 per protein (default: 30)")
    args = parser.parse_args()

    print(f"Reading {args.rci_csv} ...")
    df = pd.read_csv(args.rci_csv)
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Check required columns
    required = ["UniprotID", "seqIndex", "rciS2", "plddt"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Drop rows with missing rciS2
    df_valid = df.dropna(subset=["rciS2"])
    print(f"  Rows with valid rciS2: {len(df_valid)}")

    # Get unique proteins
    proteins = df_valid["UniprotID"].unique()
    print(f"  Unique proteins: {len(proteins)}")

    pdb_dir = Path(args.pdb_dir)
    proteins_dir = Path(args.output_dir) / "proteins"
    proteins_dir.mkdir(parents=True, exist_ok=True)

    n_processed = 0
    n_skipped_short = 0
    n_skipped_no_pdb = 0

    for uniprot_id in sorted(proteins):
        protein_df = df_valid[df_valid["UniprotID"] == uniprot_id].copy()

        # Check minimum length
        if len(protein_df) < args.min_residues:
            n_skipped_short += 1
            continue

        # Check PDB file exists
        pdb_path = pdb_dir / f"{uniprot_id}.pdb"
        if not pdb_path.exists():
            n_skipped_no_pdb += 1
            continue

        # Create protein directory
        prot_dir = proteins_dir / uniprot_id
        prot_dir.mkdir(exist_ok=True)

        # Sort by sequence index
        protein_df = protein_df.sort_values("seqIndex").reset_index(drop=True)

        # Write Bfactor TSV: use (1 - rciS2) so high = flexible
        # This matches the sign convention of RMSF and B-factor
        # Use seqIndex as position to match AF2 PDB residue numbering
        bfactor_df = pd.DataFrame({
            "position": protein_df["seqIndex"].values,
            "bfactor": 1.0 - protein_df["rciS2"].values,
        })
        bfactor_df.to_csv(prot_dir / f"{uniprot_id}_Bfactor.tsv",
                          sep="\t", index=False)

        # Write pLDDT TSV (same residues as Bfactor since both come from same filtered set)
        plddt_df = pd.DataFrame({
            "position": protein_df["seqIndex"].values,
            "plddt": protein_df["plddt"].values,
        })
        plddt_df.to_csv(prot_dir / f"{uniprot_id}_pLDDT.tsv",
                        sep="\t", index=False)

        # Symlink PDB file
        pdb_link = prot_dir / f"{uniprot_id}.pdb"
        if not pdb_link.exists():
            os.symlink(str(pdb_path.resolve()), str(pdb_link))

        # Create .done sentinel
        (prot_dir / ".done").touch()

        n_processed += 1

    print(f"\nDone:")
    print(f"  Processed: {n_processed} proteins")
    print(f"  Skipped (< {args.min_residues} residues): {n_skipped_short}")
    print(f"  Skipped (no PDB): {n_skipped_no_pdb}")
    print(f"  Output: {proteins_dir}")


if __name__ == "__main__":
    main()
