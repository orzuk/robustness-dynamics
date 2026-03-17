#!/usr/bin/env python3
"""
Compute per-residue pLDDT using ESMFold for proteins that lack it (e.g., BBFlow).

Produces ATLAS-compatible *_pLDDT.tsv files so the existing
correlate_robustness_dynamics.py can use them without changes.

Usage:
    python compute_plddt_esmfold.py \
        --proteins_dir /path/to/bbflow_processed/proteins \
        --device cuda \
        --skip_existing

Output per protein:
    {protein_name}_pLDDT.tsv   (columns: position, plddt)

Dependencies:
    pip install esm torch
    Requires ~4 GB GPU VRAM for proteins up to ~500 residues.
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def extract_sequence_from_pdb(pdb_path: str) -> str:
    """Extract single-chain protein sequence from a PDB file."""
    three_to_one = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }
    residues = []
    seen = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
                resname = line[17:20].strip()
                chain = line[21]
                resnum = line[22:27].strip()
                key = (chain, resnum)
                if key not in seen and resname in three_to_one:
                    seen.add(key)
                    residues.append(three_to_one[resname])
    return "".join(residues)


def compute_plddt_esmfold(sequence: str, model, tokenizer, backend: str,
                          device: str) -> np.ndarray:
    """Run ESMFold on a sequence and return per-residue pLDDT (0-100 scale).

    Uses model.infer_pdb() which works for both Meta and HuggingFace backends,
    then extracts pLDDT from the B-factor column of the output PDB string.
    """
    with torch.no_grad():
        pdb_str = model.infer_pdb(sequence)

    # Extract per-residue pLDDT from B-factor column of CA atoms
    plddt_values = []
    seen = set()
    for line in pdb_str.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            resnum = line[22:27].strip()
            chain = line[21]
            key = (chain, resnum)
            if key not in seen:
                seen.add(key)
                bfactor = float(line[60:66].strip())
                plddt_values.append(bfactor)

    plddt = np.array(plddt_values)
    # Convert to 0-100 scale if needed (ESMFold typically outputs 0-100)
    if len(plddt) > 0 and plddt.max() <= 1.0:
        plddt = plddt * 100.0
    return plddt


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-residue pLDDT using ESMFold for BBFlow/designed proteins")
    parser.add_argument("--proteins_dir", type=str, required=True,
                        help="Path to proteins/ directory (ATLAS-compatible structure)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device for ESMFold inference")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip proteins that already have pLDDT files")
    parser.add_argument("--max_proteins", type=int, default=0,
                        help="Limit number of proteins (0 = all)")
    args = parser.parse_args()

    proteins_dir = Path(args.proteins_dir)
    if not proteins_dir.exists():
        sys.exit(f"ERROR: proteins directory not found: {proteins_dir}")

    # Find protein directories with .done marker
    protein_dirs = sorted([
        d for d in proteins_dir.iterdir()
        if d.is_dir() and (d / ".done").exists()
    ])
    if args.max_proteins > 0:
        protein_dirs = protein_dirs[:args.max_proteins]

    print(f"ESMFold pLDDT computation")
    print(f"Found {len(protein_dirs)} proteins in {proteins_dir}")
    print(f"Device: {args.device}")
    print(f"Skip existing: {args.skip_existing}")
    print()

    # Load ESMFold model (try Meta fair-esm first, fallback to HuggingFace)
    print("Loading ESMFold model...", flush=True)
    t0 = time.time()
    device = args.device if torch.cuda.is_available() else "cpu"
    tokenizer = None
    try:
        import esm
        model = esm.pretrained.esmfold_v1().eval()
        backend = "meta-esmfold"
        print("  Using Meta fair-esm backend", flush=True)
    except Exception:
        from transformers import EsmForProteinFolding, AutoTokenizer
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").eval()
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        backend = "hf-esmfold"
        print("  Using HuggingFace transformers backend (no openfold needed)",
              flush=True)
    if device == "cuda":
        model = model.cuda()
        # Only half-precision the language model; the structure module breaks in fp16
        if hasattr(model, "esm"):
            model.esm = model.esm.half()
    print(f"Model loaded in {time.time() - t0:.1f}s on {device} ({backend})",
          flush=True)
    print()

    n_ok, n_skip, n_fail = 0, 0, 0
    t_start = time.time()

    for i, pdir in enumerate(protein_dirs):
        protein_name = pdir.name
        plddt_tsv = pdir / f"{protein_name}_pLDDT.tsv"

        if args.skip_existing and plddt_tsv.exists():
            n_skip += 1
            continue

        print(f"[{i+1}/{len(protein_dirs)}] {protein_name} ...", end=" ", flush=True)

        # Find PDB file
        pdb_files = list(pdir.glob("*.pdb"))
        if not pdb_files:
            print("SKIP (no PDB)", flush=True)
            n_fail += 1
            continue

        # Extract sequence
        sequence = extract_sequence_from_pdb(str(pdb_files[0]))
        if len(sequence) < 10:
            print(f"SKIP (seq too short: {len(sequence)})", flush=True)
            n_fail += 1
            continue

        # Run ESMFold
        t0 = time.time()
        try:
            plddt = compute_plddt_esmfold(sequence, model, tokenizer, backend,
                                          device)
        except Exception as e:
            print(f"FAIL ({e})", flush=True)
            n_fail += 1
            continue

        # Sanity check: length match
        if len(plddt) != len(sequence):
            print(f"FAIL (pLDDT length {len(plddt)} != seq length {len(sequence)})",
                  flush=True)
            n_fail += 1
            continue

        # Write ATLAS-compatible pLDDT TSV
        df = pd.DataFrame({
            "position": range(1, len(plddt) + 1),
            "plddt": plddt,
        })
        df.to_csv(plddt_tsv, sep="\t", index=False)

        elapsed = time.time() - t0
        print(f"OK (L={len(sequence)}, mean_pLDDT={plddt.mean():.1f}, {elapsed:.1f}s)",
              flush=True)
        n_ok += 1

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.0f}s: {n_ok} computed, {n_skip} skipped, {n_fail} failed")


if __name__ == "__main__":
    main()
