#!/usr/bin/env python3
"""
Find the overlap between MegaScale (Tsuboyama 2023) and ATLAS protein datasets.

MegaScale natural proteins are identified by 4-character PDB IDs (e.g., "3A4Z").
ATLAS proteins are identified as "{pdb_id}_{chain}" (e.g., "3a4z_A").

This script:
  1. Reads the ATLAS protein list
  2. Reads MegaScale Dataset1 CSV to extract unique WT_name values
  3. Separates natural (PDB-ID-based) from designed proteins
  4. Computes overlap at the PDB-ID level (ignoring chain)

Usage:
    python scripts/megascale_atlas_overlap.py \
        --atlas_list /path/to/ATLAS_pdb_list.txt \
        --megascale_csv /path/to/Tsuboyama2023_Dataset1_20230416.csv

    # Or if the CSV is too large / not available, use the processed K50 tables:
    python scripts/megascale_atlas_overlap.py \
        --atlas_list /path/to/ATLAS_pdb_list.txt \
        --megascale_dir /path/to/Processed_K50_dG_datasets
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path


def load_atlas_ids(atlas_list_path: str) -> dict:
    """Load ATLAS protein IDs and build a PDB->chains mapping.

    Returns dict: {uppercase_pdb_id: [chain1, chain2, ...]}
    """
    pdb_to_chains = {}
    with open(atlas_list_path) as f:
        for line in f:
            prot_id = line.strip()
            if not prot_id:
                continue
            # Format: "16pk_A" -> pdb="16pk", chain="A"
            parts = prot_id.rsplit("_", 1)
            if len(parts) == 2:
                pdb_id = parts[0].upper()
                chain = parts[1]
            else:
                pdb_id = prot_id.upper()
                chain = "?"
            pdb_to_chains.setdefault(pdb_id, []).append(chain)
    return pdb_to_chains


def load_megascale_from_csv(csv_path: str) -> set:
    """Load unique protein identifiers from MegaScale Dataset1 CSV.

    Tries columns in order: 'WT_name', 'name', 'pdb_name'.
    For 'name', extracts the base protein name (e.g., '1A32.pdb' from
    a row that might encode mutations in other columns).
    """
    wt_names = set()
    col_candidates = ["WT_name", "name", "pdb_name"]
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        # Find which column exists
        col_name = None
        first_row = next(reader, None)
        if first_row is None:
            return wt_names
        for c in col_candidates:
            if c in first_row:
                col_name = c
                break
        if col_name is None:
            print(f"  WARNING: None of {col_candidates} found in CSV columns: "
                  f"{list(first_row.keys())[:10]}")
            return wt_names
        print(f"  Using column '{col_name}' for protein identifiers")
        # Process first row
        val = first_row[col_name].strip()
        if val:
            # Extract base protein name: strip mutation suffixes if present
            # e.g., "1A32.pdb" -> "1A32.pdb", "1A32_A5G.pdb" keep as-is
            wt_names.add(val)
        # Process remaining rows
        for row in reader:
            val = row.get(col_name, "").strip()
            if val:
                wt_names.add(val)
    return wt_names


def load_megascale_from_dir(megascale_dir: str) -> set:
    """Infer WT_name values from filenames in Processed_K50_dG_datasets or
    K50_dG_tables directory."""
    wt_names = set()
    base = Path(megascale_dir)
    # Look for CSV files whose names might be WT_names
    for csv_file in base.rglob("*.csv"):
        name = csv_file.stem
        # Skip files that are clearly not protein identifiers
        if name.startswith("Tsuboyama") or name.startswith("NGS"):
            continue
        wt_names.add(name)
    # Also look for subdirectories
    for d in base.iterdir():
        if d.is_dir():
            wt_names.add(d.name)
    return wt_names


def classify_megascale_names(wt_names: set) -> tuple:
    """Separate MegaScale WT_names into natural (PDB-ID) and designed.

    Natural proteins: 4-character alphanumeric PDB IDs (e.g., "3A4Z", "1ABC")
    Designed proteins: longer names, often with .pdb extension
    """
    pdb_id_pattern = re.compile(r"^[0-9][A-Za-z0-9]{3}$")

    natural_pdb_ids = set()
    designed_names = set()

    for name in wt_names:
        # Strip .pdb extension if present
        clean = name.replace(".pdb", "").strip()
        if pdb_id_pattern.match(clean):
            natural_pdb_ids.add(clean.upper())
        else:
            designed_names.add(name)

    return natural_pdb_ids, designed_names


def main():
    parser = argparse.ArgumentParser(
        description="Find overlap between MegaScale and ATLAS datasets"
    )
    parser.add_argument(
        "--atlas_list",
        required=True,
        help="Path to ATLAS_pdb_list.txt",
    )
    parser.add_argument(
        "--megascale_csv",
        default="",
        help="Path to Tsuboyama2023_Dataset1_20230416.csv",
    )
    parser.add_argument(
        "--megascale_dir",
        default="",
        help="Path to MegaScale directory (e.g., Processed_K50_dG_datasets) "
        "to infer protein names from filenames",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Path to write overlap results (default: stdout)",
    )
    args = parser.parse_args()

    if not args.megascale_csv and not args.megascale_dir:
        parser.error("Provide --megascale_csv or --megascale_dir (or both)")

    # ---- Load ATLAS ----
    print(f"Loading ATLAS protein list from: {args.atlas_list}")
    atlas_pdb_to_chains = load_atlas_ids(args.atlas_list)
    atlas_pdb_ids = set(atlas_pdb_to_chains.keys())
    n_atlas_proteins = sum(len(v) for v in atlas_pdb_to_chains.values())
    n_atlas_pdbs = len(atlas_pdb_ids)
    print(f"  ATLAS: {n_atlas_proteins} protein chains from {n_atlas_pdbs} unique PDB IDs")

    # ---- Load MegaScale ----
    megascale_wt_names = set()
    if args.megascale_csv:
        print(f"Loading MegaScale from CSV: {args.megascale_csv}")
        megascale_wt_names |= load_megascale_from_csv(args.megascale_csv)
    if args.megascale_dir:
        print(f"Loading MegaScale from directory: {args.megascale_dir}")
        megascale_wt_names |= load_megascale_from_dir(args.megascale_dir)

    print(f"  MegaScale: {len(megascale_wt_names)} unique WT_name values")

    # ---- Classify ----
    mega_natural, mega_designed = classify_megascale_names(megascale_wt_names)
    print(f"  MegaScale natural (PDB IDs): {len(mega_natural)}")
    print(f"  MegaScale designed: {len(mega_designed)}")

    # ---- Overlap ----
    overlap_pdb_ids = atlas_pdb_ids & mega_natural
    n_overlap = len(overlap_pdb_ids)

    print(f"\n{'='*60}")
    print(f"OVERLAP: {n_overlap} PDB IDs found in both ATLAS and MegaScale")
    print(f"{'='*60}")

    if n_overlap == 0:
        print("\nNo overlap found.")
        print(f"  ATLAS PDB IDs (sample): {sorted(atlas_pdb_ids)[:10]}")
        print(f"  MegaScale natural PDB IDs (sample): {sorted(mega_natural)[:10]}")
        # Check if there's a case mismatch or format issue
        atlas_lower = {x.lower() for x in atlas_pdb_ids}
        mega_lower = {x.lower() for x in mega_natural}
        overlap_lower = atlas_lower & mega_lower
        if overlap_lower:
            print(f"\n  NOTE: {len(overlap_lower)} matches found when ignoring case!")
            print(f"  Sample: {sorted(overlap_lower)[:10]}")
    else:
        # List overlapping proteins with their ATLAS chain info
        out_lines = []
        header = f"{'PDB_ID':<8} {'ATLAS_chains':<20} {'ATLAS_IDs'}"
        out_lines.append(header)
        out_lines.append("-" * 60)

        for pdb_id in sorted(overlap_pdb_ids):
            chains = atlas_pdb_to_chains.get(pdb_id, [])
            chain_str = ",".join(sorted(chains))
            atlas_ids = ", ".join(
                f"{pdb_id.lower()}_{c}" for c in sorted(chains)
            )
            out_lines.append(f"{pdb_id:<8} {chain_str:<20} {atlas_ids}")

        for line in out_lines:
            print(line)

        # Count total ATLAS protein chains in the overlap
        total_chains = sum(
            len(atlas_pdb_to_chains[pdb_id]) for pdb_id in overlap_pdb_ids
        )
        print(f"\nTotal ATLAS protein chains with MegaScale data: {total_chains}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  ATLAS unique PDB IDs:            {n_atlas_pdbs}")
    print(f"  ATLAS total protein chains:      {n_atlas_proteins}")
    print(f"  MegaScale natural PDB IDs:       {len(mega_natural)}")
    print(f"  MegaScale designed proteins:     {len(mega_designed)}")
    print(f"  Overlapping PDB IDs:             {n_overlap}")
    if n_atlas_pdbs > 0:
        print(f"  Overlap / ATLAS PDBs:            {n_overlap/n_atlas_pdbs:.1%}")
    if mega_natural:
        print(f"  Overlap / MegaScale natural:     {n_overlap/len(mega_natural):.1%}")

    # ---- Write output ----
    if args.output:
        with open(args.output, "w") as f:
            for pdb_id in sorted(overlap_pdb_ids):
                chains = atlas_pdb_to_chains.get(pdb_id, [])
                for chain in sorted(chains):
                    f.write(f"{pdb_id.lower()}_{chain}\n")
        print(f"\nOverlap ATLAS IDs written to: {args.output}")


if __name__ == "__main__":
    main()
