#!/usr/bin/env python3
"""
Download AF2 PDB files from the RelaxDB HuggingFace dataset.

RelaxDB includes pre-computed AF2 PDB files named by BMRB entry ID
(e.g., 4267.pdb). This script downloads them and creates symlinks
using UniProt IDs to match the preprocessing pipeline.

Usage:
  python scripts/download_relaxdb_pdbs.py \
      --relaxdb_csv $PROJECT_DIR/data/NMR_relaxation/relaxdb_data.csv \
      --output_dir $PROJECT_DIR/data/NMR_relaxation/af2_pdbs

  Then re-run preprocessing with --pdb_dir:
  python scripts/preprocess_relaxdb.py \
      --relaxdb_csv ... --output_dir ... \
      --pdb_dir $PROJECT_DIR/data/NMR_relaxation/af2_pdbs
"""

import os
import argparse
import pandas as pd
import urllib.request
from pathlib import Path


HF_BASE = "https://huggingface.co/datasets/gelnesr/RelaxDB/resolve/main"


def main():
    parser = argparse.ArgumentParser(
        description="Download RelaxDB AF2 PDB files from HuggingFace")
    parser.add_argument("--relaxdb_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save PDB files")
    args = parser.parse_args()

    df = pd.read_csv(args.relaxdb_csv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_fail = 0
    n_skip = 0

    for _, row in df.iterrows():
        bmrb_id = str(row["id"])
        uniprot_id = str(row.get("uniprot_id", ""))
        pdb_col = str(row.get("pdb", ""))

        # Output file named by UniProt ID (what preprocess_relaxdb expects)
        if not uniprot_id or uniprot_id == "nan":
            continue

        dest = out / f"{uniprot_id}.pdb"
        if dest.exists():
            n_skip += 1
            continue

        # Download from HuggingFace: relaxdb/af2_pdbs/{bmrb_id}.pdb
        url = f"{HF_BASE}/relaxdb/af2_pdbs/{bmrb_id}.pdb"
        try:
            urllib.request.urlretrieve(url, str(dest))
            n_ok += 1
            print(f"  OK: {uniprot_id} (BMRB {bmrb_id})")
        except Exception as e:
            n_fail += 1
            print(f"  FAIL: {uniprot_id} (BMRB {bmrb_id}): {e}")

    print(f"\nDone: {n_ok} downloaded, {n_skip} skipped, {n_fail} failed")
    print(f"PDBs in: {out}")


if __name__ == "__main__":
    main()
