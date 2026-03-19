#!/usr/bin/env python3
"""
Preprocess the experimental S² order parameter dataset for the robustness-dynamics pipeline.

This dataset (from Gavalda-Garcia et al., Bitbucket: bio2byte/af2_analysis_datagen_and_plots)
contains Lipari-Szabo S² order parameters for 42 proteins, derived from NMR relaxation
model-free analysis (NOT RCI proxy).

Input: s2_values.txt — whitespace-delimited file with blocks per protein:
  bmr4245 . SP_ID P0C273 4.55e-46 PDB 1AAR MQIFV...
       1 M      .
       2 Q     0.745
       3 I     0.686
       ...

Output:
  {output_dir}/proteins/{pdb_id}/
    {pdb_id}_Bfactor.tsv     # columns: position, bfactor  (= 1 - S²)
    {pdb_id}_pLDDT.tsv       # columns: position, plddt (from AF2 PDB)
    {pdb_id}.pdb             # AF2 model (downloaded or symlinked)
    .done

Usage:
  python preprocess_s2_experimental.py \
      --s2_file /path/to/s2_values.txt \
      --output_dir /path/to/s2_exp_processed \
      [--pdb_dir /path/to/existing/pdbs] \
      [--download_pdbs] \
      [--min_residues 20]
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def parse_s2_file(s2_path):
    """Parse the s2_values.txt file.

    Returns list of dicts with keys:
      bmrb_id, uniprot_id, pdb_id, sequence, residues [(resnum, aa, s2_or_nan)]
    """
    proteins = []
    current = None

    with open(s2_path, "r") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue

            # Header line: starts with "bmr" prefix
            if line.startswith("bmr"):
                if current is not None:
                    proteins.append(current)
                parts = line.split()
                # Format: bmrID . SP_ID UniprotID evalue PDB pdbID SEQUENCE
                current = {
                    "bmrb_id": parts[0] if len(parts) > 0 else "",
                    "uniprot_id": parts[3] if len(parts) > 3 else "",
                    "pdb_id": parts[6] if len(parts) > 6 else "",
                    "sequence": parts[7] if len(parts) > 7 else "",
                    "residues": [],
                }
            elif current is not None:
                # Residue line: "     1 M      .   " or "     2 Q     0.745"
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        resnum = int(parts[0])
                        aa = parts[1]
                        s2_str = parts[2]
                        s2_val = np.nan if s2_str == "." else float(s2_str)
                        current["residues"].append((resnum, aa, s2_val))
                    except (ValueError, IndexError):
                        pass

    if current is not None:
        proteins.append(current)

    return proteins


def extract_plddt_from_pdb(pdb_path):
    """Extract per-residue pLDDT from AF2 PDB B-factor column."""
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
    """Download AF2 PDB from AlphaFold Protein Structure Database."""
    import urllib.request
    url = (f"https://alphafold.ebi.ac.uk/files/"
           f"AF-{uniprot_id}-F1-model_v4.pdb")
    try:
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"  WARNING: Failed to download AF2 PDB for {uniprot_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess experimental S² dataset for robustness-dynamics pipeline")
    parser.add_argument("--s2_file", type=str, required=True,
                        help="Path to s2_values.txt")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory (will create proteins/ subdirectory)")
    parser.add_argument("--pdb_dir", type=str, default=None,
                        help="Directory with existing PDB files (optional)")
    parser.add_argument("--download_pdbs", action="store_true",
                        help="Download AF2 PDBs from AlphaFold DB if not found locally")
    parser.add_argument("--min_residues", type=int, default=20,
                        help="Minimum residues with valid S² (default: 20)")
    args = parser.parse_args()

    print(f"Parsing {args.s2_file} ...")
    proteins = parse_s2_file(args.s2_file)
    print(f"  Found {len(proteins)} proteins")

    pdb_dir = Path(args.pdb_dir) if args.pdb_dir else None
    proteins_dir = Path(args.output_dir) / "proteins"
    proteins_dir.mkdir(parents=True, exist_ok=True)

    n_processed = 0
    n_skipped_short = 0
    n_skipped_no_pdb = 0

    for prot in proteins:
        uniprot_id = prot["uniprot_id"]
        pdb_code = prot["pdb_id"]
        # Use UniProt ID as primary identifier (for AF2 PDB download)
        protein_id = uniprot_id if uniprot_id else pdb_code
        if not protein_id:
            continue

        residues = prot["residues"]
        valid_residues = [(r, a, s) for r, a, s in residues if not np.isnan(s)]

        if len(valid_residues) < args.min_residues:
            n_skipped_short += 1
            continue

        # Resolve PDB
        prot_dir = proteins_dir / protein_id
        prot_dir.mkdir(exist_ok=True)
        pdb_dest = prot_dir / f"{protein_id}.pdb"

        if not pdb_dest.exists():
            found = False
            if pdb_dir:
                for name in [f"{protein_id}.pdb", f"{pdb_code}.pdb",
                             f"AF-{uniprot_id}-F1-model_v4.pdb"]:
                    candidate = pdb_dir / name
                    if candidate.exists():
                        os.symlink(str(candidate.resolve()), str(pdb_dest))
                        found = True
                        break
            if not found and args.download_pdbs and uniprot_id:
                found = download_af2_pdb(uniprot_id, str(pdb_dest))
            if not found:
                n_skipped_no_pdb += 1
                continue

        # Write Bfactor TSV: (1 - S²) so high = flexible
        bfactor_df = pd.DataFrame({
            "position": [r for r, _, s in residues if not np.isnan(s)],
            "bfactor": [1.0 - s for _, _, s in residues if not np.isnan(s)],
        })
        bfactor_df.to_csv(prot_dir / f"{protein_id}_Bfactor.tsv",
                          sep="\t", index=False)

        # Extract and write pLDDT
        plddt_data = extract_plddt_from_pdb(str(pdb_dest))
        if plddt_data:
            plddt_df = pd.DataFrame(plddt_data, columns=["position", "plddt"])
            plddt_df.to_csv(prot_dir / f"{protein_id}_pLDDT.tsv",
                            sep="\t", index=False)

        # Also write raw S² for reference
        s2_df = pd.DataFrame({
            "position": [r for r, _, s in residues if not np.isnan(s)],
            "s2": [s for _, _, s in residues if not np.isnan(s)],
        })
        s2_df.to_csv(prot_dir / f"{protein_id}_S2.tsv",
                     sep="\t", index=False)

        # Sentinel
        (prot_dir / ".done").touch()

        n_processed += 1
        print(f"  OK: {protein_id} ({pdb_code}, {len(valid_residues)} residues)")

    print(f"\nDone:")
    print(f"  Processed: {n_processed} proteins")
    print(f"  Skipped (< {args.min_residues} residues): {n_skipped_short}")
    print(f"  Skipped (no PDB): {n_skipped_no_pdb}")
    print(f"  Output: {proteins_dir}")


if __name__ == "__main__":
    main()
