#!/usr/bin/env python3
"""
Preprocess the RelaxDB dataset for the robustness-dynamics pipeline.

RelaxDB (gelnesr/RelaxDB on HuggingFace) contains per-residue NMR relaxation
data (R1, R2, hetNOE) for 143 proteins curated from BMRB.

This script:
  1. Parses the RelaxDB CSV (one row per protein, dict-in-cell for per-residue data)
  2. Extracts per-residue hetNOE values (primary dynamics target)
  3. Downloads AF2 PDB files from AlphaFold DB (by UniProt ID)
  4. Creates per-protein directories compatible with correlate_robustness_dynamics.py

For each protein:
  - Creates proteins/{protein_id}/ directory
  - Writes {protein_id}_Bfactor.tsv with (1 - hetNOE) as "bfactor"
    (inverted so high = flexible, matching RMSF/B-factor convention)
  - Writes {protein_id}_pLDDT.tsv with AF2 pLDDT values (from AF2 PDB B-factors)
  - Copies/symlinks the PDB file into the protein directory
  - Creates .done sentinel file

Input:
  relaxdb_data.csv from HuggingFace gelnesr/RelaxDB

Output:
  {output_dir}/proteins/{protein_id}/
    {protein_id}_Bfactor.tsv     # columns: position, bfactor  (= 1 - hetNOE)
    {protein_id}_pLDDT.tsv       # columns: position, plddt
    {protein_id}.pdb             # AF2 model (downloaded or symlinked)
    .done

Usage:
  python preprocess_relaxdb.py \
      --relaxdb_csv /path/to/relaxdb_data.csv \
      --output_dir /path/to/relaxdb_processed \
      [--pdb_dir /path/to/existing/pdbs]  \
      [--download_pdbs]  \
      [--min_residues 30] \
      [--min_noe_coverage 0.5]
"""

import os
import ast
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def parse_relaxation_dict(dict_str):
    """Parse the dict-in-cell string from RelaxDB CSV.

    Each cell contains a Python dict string like:
      "{'NOE': [None, None, 0.5, ...], 'R2': [...], 'R1': [...]}"

    Returns a dict of {key: list_of_values}.
    """
    if pd.isna(dict_str) or not dict_str:
        return {}
    try:
        return ast.literal_eval(dict_str)
    except (ValueError, SyntaxError):
        return {}


def extract_plddt_from_pdb(pdb_path):
    """Extract per-residue pLDDT from AF2 PDB B-factor column.

    AF2 stores pLDDT in the B-factor field of CA atoms.
    Returns list of (resnum, plddt) tuples.
    """
    plddt_vals = []
    seen_resnums = set()
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resnum = int(line[22:26].strip())
                if resnum not in seen_resnums:
                    bfac = float(line[60:66].strip())
                    plddt_vals.append((resnum, bfac))
                    seen_resnums.add(resnum)
    return plddt_vals


def download_af2_pdb(uniprot_id, output_path):
    """Download AF2 PDB from AlphaFold Protein Structure Database.

    Returns True if successful, False otherwise.
    """
    import urllib.request
    url = (f"https://alphafold.ebi.ac.uk/files/"
           f"AF-{uniprot_id}-F1-model_v4.pdb")
    try:
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"  WARNING: Failed to download AF2 PDB for {uniprot_id}: {e}")
        return False


def process_protein(row, pdb_dir, output_proteins_dir, download_pdbs,
                    min_residues, min_noe_coverage):
    """Process a single protein from RelaxDB.

    Returns (protein_id, status_string).
    """
    protein_id = row.get("uniprot_id") or row.get("pdb", "").replace("/", "_")
    if not protein_id or protein_id == "nan":
        return None, "no_id"

    # Parse relaxation data
    relax_data = parse_relaxation_dict(row.get("R2/R1/NOE", ""))
    if not relax_data:
        return protein_id, "no_relax_data"

    noe_values = relax_data.get("NOE", [])
    r1_values = relax_data.get("R1", [])
    r2_values = relax_data.get("R2", [])

    if not noe_values:
        return protein_id, "no_noe"

    seq_len = len(noe_values)

    # Count valid (non-None) NOE values
    valid_noe = [(i, v) for i, v in enumerate(noe_values) if v is not None]
    noe_coverage = len(valid_noe) / seq_len if seq_len > 0 else 0

    if len(valid_noe) < min_residues:
        return protein_id, "too_few_residues"

    if noe_coverage < min_noe_coverage:
        return protein_id, "low_coverage"

    # Resolve PDB file
    prot_dir = output_proteins_dir / protein_id
    prot_dir.mkdir(exist_ok=True)
    pdb_dest = prot_dir / f"{protein_id}.pdb"

    if not pdb_dest.exists():
        if pdb_dir:
            # Try to find PDB in provided directory
            candidates = [
                pdb_dir / f"{protein_id}.pdb",
                pdb_dir / f"AF-{protein_id}-F1-model_v4.pdb",
            ]
            found = False
            for c in candidates:
                if c.exists():
                    os.symlink(str(c.resolve()), str(pdb_dest))
                    found = True
                    break
            if not found and download_pdbs:
                if not download_af2_pdb(protein_id, str(pdb_dest)):
                    return protein_id, "no_pdb"
            elif not found:
                return protein_id, "no_pdb"
        elif download_pdbs:
            if not download_af2_pdb(protein_id, str(pdb_dest)):
                return protein_id, "no_pdb"
        else:
            return protein_id, "no_pdb"

    # Extract pLDDT from AF2 PDB
    plddt_data = extract_plddt_from_pdb(str(pdb_dest))

    # Build per-residue arrays (1-indexed positions)
    # hetNOE: use (1 - NOE) so high = flexible (matching RMSF convention)
    positions = list(range(1, seq_len + 1))
    bfactor_values = []
    for v in noe_values:
        if v is not None:
            bfactor_values.append(1.0 - v)
        else:
            bfactor_values.append(np.nan)

    # Write Bfactor TSV (1 - hetNOE)
    bfactor_df = pd.DataFrame({
        "position": positions,
        "bfactor": bfactor_values,
    })
    # Drop NaN rows — correlate script needs clean data
    bfactor_df = bfactor_df.dropna(subset=["bfactor"])
    bfactor_df.to_csv(prot_dir / f"{protein_id}_Bfactor.tsv",
                      sep="\t", index=False)

    # Write pLDDT TSV
    if plddt_data:
        plddt_df = pd.DataFrame(plddt_data, columns=["position", "plddt"])
        plddt_df.to_csv(prot_dir / f"{protein_id}_pLDDT.tsv",
                        sep="\t", index=False)

    # Also write raw R1, R2 TSVs for optional alternative targets
    for metric_name, values in [("R1", r1_values), ("R2", r2_values),
                                 ("hetNOE", noe_values)]:
        if values:
            metric_df = pd.DataFrame({
                "position": list(range(1, len(values) + 1)),
                metric_name.lower(): [v if v is not None else np.nan
                                      for v in values],
            })
            metric_df = metric_df.dropna(subset=[metric_name.lower()])
            metric_df.to_csv(prot_dir / f"{protein_id}_{metric_name}.tsv",
                             sep="\t", index=False)

    # Write R2/R1 ratio TSV (reports on μs-ms conformational exchange)
    if r1_values and r2_values and len(r1_values) == len(r2_values):
        r2r1_positions = []
        r2r1_values = []
        for i, (r1, r2) in enumerate(zip(r1_values, r2_values)):
            if r1 is not None and r2 is not None and r1 > 0:
                r2r1_positions.append(i + 1)
                r2r1_values.append(r2 / r1)
        if r2r1_values:
            r2r1_df = pd.DataFrame({
                "position": r2r1_positions,
                "bfactor": r2r1_values,  # use "bfactor" column so pipeline reads it
            })
            r2r1_df.to_csv(prot_dir / f"{protein_id}_R2R1.tsv",
                           sep="\t", index=False)

    # Create .done sentinel
    (prot_dir / ".done").touch()

    return protein_id, "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess RelaxDB for robustness-dynamics pipeline")
    parser.add_argument("--relaxdb_csv", type=str, required=True,
                        help="Path to relaxdb_data.csv")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory (will create proteins/ subdirectory)")
    parser.add_argument("--pdb_dir", type=str, default=None,
                        help="Directory with existing PDB files (optional)")
    parser.add_argument("--download_pdbs", action="store_true",
                        help="Download AF2 PDBs from AlphaFold DB if not found locally")
    parser.add_argument("--min_residues", type=int, default=30,
                        help="Minimum residues with valid hetNOE (default: 30)")
    parser.add_argument("--min_noe_coverage", type=float, default=0.3,
                        help="Minimum fraction of residues with NOE data (default: 0.3)")
    args = parser.parse_args()

    print(f"Reading {args.relaxdb_csv} ...")
    df = pd.read_csv(args.relaxdb_csv)
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Handle duplicate UniProt IDs (same protein at different field strengths).
    # Keep the entry with the most valid NOE values.
    def count_valid_noe(row):
        d = parse_relaxation_dict(row.get("R2/R1/NOE", ""))
        noe = d.get("NOE", [])
        return sum(1 for v in noe if v is not None)

    df["_noe_count"] = df.apply(count_valid_noe, axis=1)
    df = df.sort_values("_noe_count", ascending=False).reset_index(drop=True)
    n_before = len(df)
    df = df.drop_duplicates(subset=["uniprot_id"], keep="first").reset_index(drop=True)
    n_dedup = n_before - len(df)
    if n_dedup > 0:
        print(f"  Deduplicated: {n_dedup} entries removed (kept best NOE coverage)")
    print(f"  Unique proteins: {len(df)}")

    pdb_dir = Path(args.pdb_dir) if args.pdb_dir else None
    proteins_dir = Path(args.output_dir) / "proteins"
    proteins_dir.mkdir(parents=True, exist_ok=True)

    stats = {"ok": 0, "no_id": 0, "no_relax_data": 0, "no_noe": 0,
             "too_few_residues": 0, "low_coverage": 0, "no_pdb": 0}

    for _, row in df.iterrows():
        protein_id, status = process_protein(
            row, pdb_dir, proteins_dir, args.download_pdbs,
            args.min_residues, args.min_noe_coverage)
        stats[status] = stats.get(status, 0) + 1
        if status == "ok":
            print(f"  OK: {protein_id}")

    print(f"\nDone:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  Output: {proteins_dir}")


if __name__ == "__main__":
    main()
