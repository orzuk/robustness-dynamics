#!/usr/bin/env python3
"""
Prepare the BBFlow de novo MD dataset for the robustness-dynamics pipeline.

Reads BBFlow MD trajectories (PDB + XTC), computes per-residue Cα RMSF,
and outputs ATLAS-compatible directory structure so the existing
correlate_robustness_dynamics.py works unchanged.

BBFlow data layout (input):
    bbflow-de-novo-dataset/
        metadata.csv              # pdb_name, pdb_path, design_method
        MD/
            sample_60_1/
                sample_60_1.pdb
                sample_60_1_R1.xtc
                sample_60_1_R2.xtc
                sample_60_1_R3.xtc
            sample_70_1/
                ...

ATLAS-compatible output:
    output_dir/
        proteins/
            sample_60_1/
                sample_60_1.pdb
                sample_60_1_RMSF.tsv
                .done
            sample_70_1/
                ...

Usage:
    python prepare_bbflow.py \
        --bbflow_dir /path/to/bbflow-de-novo-dataset \
        --output_dir /path/to/bbflow_processed

Dependencies:
    pip install mdtraj numpy pandas
"""

import os
import sys
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
import time

try:
    import mdtraj
except ImportError:
    sys.exit("ERROR: mdtraj is required. Install with: pip install mdtraj")


THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def compute_rmsf_for_protein(protein_dir: Path, protein_name: str,
                              output_dir: Path) -> bool:
    """Compute per-residue Cα RMSF from replicate XTC trajectories.

    Returns True on success.
    """
    pdb_path = protein_dir / f"{protein_name}.pdb"
    if not pdb_path.exists():
        print(f"  SKIP {protein_name}: no PDB at {pdb_path}")
        return False

    # Find replicate trajectories (naming: name_R1.xtc, name_R2.xtc, ...)
    xtc_files = sorted(protein_dir.glob(f"{protein_name}_R*.xtc"))
    if not xtc_files:
        print(f"  SKIP {protein_name}: no XTC trajectories found")
        return False

    # Load reference structure and select Cα atoms
    try:
        ref = mdtraj.load(str(pdb_path))
    except Exception as e:
        print(f"  SKIP {protein_name}: cannot load PDB: {e}")
        return False

    ca_indices = ref.topology.select("name CA")
    if len(ca_indices) == 0:
        print(f"  SKIP {protein_name}: no Cα atoms")
        return False

    # Compute RMSF per replicate
    rmsf_per_rep = []
    for xtc_path in xtc_files:
        try:
            traj = mdtraj.load(str(xtc_path), top=str(pdb_path))
            # Superpose on Cα atoms to remove translational/rotational motion
            traj.superpose(ref, atom_indices=ca_indices)
            # mdtraj.rmsf returns nm; multiply by 10 to get Angstroms
            rmsf_vals = mdtraj.rmsf(traj, ref, atom_indices=ca_indices) * 10.0
            rmsf_per_rep.append(rmsf_vals)
        except Exception as e:
            print(f"  WARNING {protein_name}: failed {xtc_path.name}: {e}")

    if not rmsf_per_rep:
        print(f"  SKIP {protein_name}: all trajectories failed")
        return False

    # Get residue names for each Cα
    residue_names = []
    for idx in ca_indices:
        atom = ref.topology.atom(idx)
        residue_names.append(atom.residue.name)

    # Build ATLAS-compatible RMSF TSV
    df = pd.DataFrame()
    df["resid"] = range(1, len(ca_indices) + 1)
    df["resname"] = residue_names
    for i, rmsf_vals in enumerate(rmsf_per_rep):
        df[f"RMSF_R{i+1}"] = rmsf_vals

    # Write output
    out_protein_dir = output_dir / "proteins" / protein_name
    out_protein_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = out_protein_dir / f"{protein_name}_RMSF.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)

    # Copy the PDB file
    out_pdb = out_protein_dir / f"{protein_name}.pdb"
    if not out_pdb.exists():
        shutil.copy2(str(pdb_path), str(out_pdb))

    # Write .done marker (pipeline compatibility)
    (out_protein_dir / ".done").touch()

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare BBFlow de novo MD dataset for robustness-dynamics pipeline")
    parser.add_argument("--bbflow_dir", type=str, required=True,
                        help="Path to bbflow-de-novo-dataset/ (contains MD/ and metadata.csv)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory (ATLAS-compatible structure)")
    parser.add_argument("--max_proteins", type=int, default=0,
                        help="Limit number of proteins (0 = all)")
    args = parser.parse_args()

    bbflow_dir = Path(args.bbflow_dir)
    output_dir = Path(args.output_dir)
    md_dir = bbflow_dir / "MD"

    if not md_dir.exists():
        sys.exit(f"ERROR: MD directory not found at {md_dir}")

    # Copy metadata.csv to output dir
    metadata_src = bbflow_dir / "metadata.csv"
    if metadata_src.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(metadata_src), str(output_dir / "metadata.csv"))

    # Find protein directories
    protein_dirs = sorted([
        d for d in md_dir.iterdir()
        if d.is_dir() and list(d.glob("*.pdb"))
    ])

    if args.max_proteins > 0:
        protein_dirs = protein_dirs[:args.max_proteins]

    print(f"BBFlow de novo MD dataset")
    print(f"Found {len(protein_dirs)} proteins in {md_dir}")
    print(f"Output: {output_dir}")
    print()

    n_ok, n_fail = 0, 0
    t_start = time.time()

    for i, pdir in enumerate(protein_dirs):
        protein_name = pdir.name
        t0 = time.time()
        print(f"[{i+1}/{len(protein_dirs)}] {protein_name} ...", end=" ",
              flush=True)

        success = compute_rmsf_for_protein(pdir, protein_name, output_dir)
        elapsed = time.time() - t0

        if success:
            n_ok += 1
            n_res = len(list((output_dir / "proteins" / protein_name).glob("*_RMSF.tsv")))
            print(f"OK ({elapsed:.1f}s)", flush=True)
        else:
            n_fail += 1

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.0f}s: {n_ok} succeeded, {n_fail} failed")
    print(f"Output: {output_dir}/proteins/")


if __name__ == "__main__":
    main()
