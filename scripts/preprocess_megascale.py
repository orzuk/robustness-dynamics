#!/usr/bin/env python3
"""
Inventory MegaScale (Tsuboyama 2023, Dataset1) for the robustness-dynamics
revision. Diagnostic step — answers "how much data per protein, and which
proteins have crystal structures we can pair with B-factors?"

The Tsuboyama CSV (`Tsuboyama2023_Dataset1_20230416.csv`, ~1.2 GB) stores
one row per variant. The protein identity is in the `name` column (no
explicit `WT_name`). Format examples:
  1A32.pdb        — WT of natural protein 1A32
  1A32.pdb_M1A    — mutant M1A on 1A32
  HEEH_rd2_0726.pdb_full        — designed mini-protein (WT)
  HEEH_rd2_0726.pdb_full_K15I   — mutant on designed mini-protein
The ΔΔG estimate per variant is in the `deltaG` column (calibrated from
the trypsin + chymotrypsin proteolysis assay).

Output: a JSON inventory at $PROJECT_DIR/data/megascale/inventory.json with:
  - n_variants_total
  - n_proteins_total
  - n_natural (PDB-ID-style names) and n_designed
  - per-protein variant counts (TSV alongside the JSON)
  - histogram thresholds: how many proteins have >=50, >=100, >=500 variants
  - which natural-protein PDB IDs are in the dataset (sorted)

This is *not* the correlation analysis. It tells us whether the full
MegaScale × B-factor analysis (Tier-2 item #7 of the genetics-revision
plan) is worth scoping. The cutoff: if >=20 natural proteins have
>=100 variants each, the analysis is statistically viable.

Usage (with the symlink set up below):
    python scripts/preprocess_megascale.py
    # or with an explicit path:
    python scripts/preprocess_megascale.py --csv /path/to/Tsuboyama2023_Dataset1.csv

Setting up the symlink (one-time, on the cluster):
    mkdir -p $PROJECT_DIR/data/megascale
    ln -s /sci/labs/orzuk/meirab/protien_mega_exp_heatmaps/Processed_K50_dG_datasets/Processed_K50_dG_datasets/Tsuboyama2023_Dataset1_20230416.csv \
          $PROJECT_DIR/data/megascale/Tsuboyama2023_Dataset1.csv
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Lazy import: paper_config supplies the canonical project path
try:
    from paper_config import CLUSTER  # type: ignore
    _DEFAULT_CSV = f"{CLUSTER.project_dir}/data/megascale/Tsuboyama2023_Dataset1.csv"
except Exception:
    _DEFAULT_CSV = ""

# A natural-protein name starts with a 4-character PDB ID (digit + 3 alnum).
PDB_ID_RE = re.compile(r"^([0-9][A-Za-z0-9]{3})")


def base_protein_name(raw_name: str) -> str:
    """Strip mutation suffix and .pdb extension to get the WT protein name.

    Examples:
        '1A32.pdb'              -> '1A32'
        '1A32.pdb_M1A'          -> '1A32'
        'HEEH_rd2_0726.pdb_full' -> 'HEEH_rd2_0726'
        'HEEH_rd2_0726.pdb_full_K15I' -> 'HEEH_rd2_0726'
    """
    # Split at the first '.pdb' — everything after is mutation/scaffold metadata
    if ".pdb" in raw_name:
        return raw_name.split(".pdb", 1)[0]
    # Fallback: strip trailing _<mutation> if it matches a mutation pattern
    return re.sub(r"_[A-Z]\d+[A-Z]$", "", raw_name)


def classify(name: str) -> str:
    """Return 'natural' if the WT name is a 4-char PDB ID, else 'designed'."""
    return "natural" if PDB_ID_RE.match(name) else "designed"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=_DEFAULT_CSV,
                        help="Path to Tsuboyama2023_Dataset1_20230416.csv "
                             "(default: $PROJECT_DIR/data/megascale/Tsuboyama2023_Dataset1.csv)")
    parser.add_argument("--output_dir", default="",
                        help="Where to write inventory.json (default: alongside the CSV)")
    args = parser.parse_args()

    if not args.csv:
        sys.exit("ERROR: --csv not provided and paper_config default not set")
    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"ERROR: CSV not found at {csv_path}")

    out_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stream the CSV (1.2 GB) — only the `name` column is needed for the inventory.
    variant_counts: Counter = Counter()
    n_rows = 0
    print(f"Streaming {csv_path} ...", flush=True)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if "name" not in reader.fieldnames:
            sys.exit(f"ERROR: 'name' column missing. Headers: {reader.fieldnames[:6]}...")
        for row in reader:
            raw = (row.get("name") or "").strip()
            if not raw:
                continue
            wt = base_protein_name(raw)
            if wt:
                variant_counts[wt] += 1
            n_rows += 1
            if n_rows % 500_000 == 0:
                print(f"  {n_rows:,} rows, {len(variant_counts):,} proteins so far",
                      flush=True)

    # Classify and summarise.
    by_class = {"natural": [], "designed": []}
    for prot, k in variant_counts.most_common():
        by_class[classify(prot)].append((prot, k))

    def _coverage(records, thresholds=(50, 100, 200, 500, 1000)):
        return {f">= {t}": sum(1 for _, k in records if k >= t)
                for t in thresholds}

    inventory = {
        "csv_path": str(csv_path),
        "n_variants_total": n_rows,
        "n_proteins_total": len(variant_counts),
        "n_natural": len(by_class["natural"]),
        "n_designed": len(by_class["designed"]),
        "natural_coverage_by_variant_count": _coverage(by_class["natural"]),
        "designed_coverage_by_variant_count": _coverage(by_class["designed"]),
        "natural_pdb_ids_sample_top20": [p for p, _ in by_class["natural"][:20]],
        "designed_names_sample_top20":  [p for p, _ in by_class["designed"][:20]],
    }

    # Write JSON summary and a full per-protein TSV (variant counts).
    json_path = out_dir / "inventory.json"
    tsv_path  = out_dir / "variant_counts_per_protein.tsv"
    with open(json_path, "w") as f:
        json.dump(inventory, f, indent=2)
    with open(tsv_path, "w") as f:
        f.write("protein\tclass\tn_variants\n")
        for prot, k in variant_counts.most_common():
            f.write(f"{prot}\t{classify(prot)}\t{k}\n")

    print()
    print("="*60)
    print("MegaScale Dataset1 inventory")
    print("="*60)
    print(f"  Variants total:       {n_rows:,}")
    print(f"  Proteins total:       {len(variant_counts):,}")
    print(f"  Natural (4-char PDB): {len(by_class['natural']):,}")
    print(f"  Designed:             {len(by_class['designed']):,}")
    print()
    print("Natural proteins by variant-count threshold:")
    for thr, n in inventory["natural_coverage_by_variant_count"].items():
        print(f"  {thr:>6}: {n:,}")
    print()
    print(f"Wrote: {json_path}")
    print(f"Wrote: {tsv_path}")
    print()
    print("Decision rule (Tier-2 #7 of revision plan):")
    print(f"  >=20 natural proteins with >=100 variants -> analysis is viable.")
    n_viable = inventory["natural_coverage_by_variant_count"].get(">= 100", 0)
    if n_viable >= 20:
        print(f"  Currently: {n_viable} natural proteins -> VIABLE")
    else:
        print(f"  Currently: {n_viable} natural proteins -> below threshold")


if __name__ == "__main__":
    main()
