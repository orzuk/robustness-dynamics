#!/usr/bin/env python3
"""
Preprocess MegaScale natural-protein data into the ATLAS-compatible per-protein
directory layout, so the existing correlate_robustness_dynamics.py / multi-DDG
pipeline runs on experimental ΔΔG with no modification.

For each natural protein in MegaScale Dataset1 (Tsuboyama 2023):
  1. Aggregate per-variant ΔΔG → per-position experimental ΔΔG profile
     (sparse — not all 19 substitutions are measured at every position).
  2. Download the PDB from RCSB, keep only X-ray ≤ 2.0 Å (skip NMR; B-factors
     from NMR ensembles are not meaningful).
  3. Verify variant naming matches PDB residue numbering (>=80% of variants
     must map to a PDB residue whose WT identity matches the variant's
     reference AA; otherwise skip the protein).
  4. Write per-protein outputs in the ATLAS layout:
       {data_dir}/proteins/{pdb_chain}/
           {pdb_chain}.pdb
           {pdb_chain}_Bfactor.tsv  (position, bfactor)
       {robustness_dir}/experimental/
           {pdb_chain}_robustness.tsv  (position, std_ddg, mean_abs_ddg, ...)
       {robustness_dir}/experimental/
           {pdb_chain}_ddg_matrix.npy  (L x 20, NaN where not measured)

Once written, the existing pipeline runs unchanged with --scorer experimental
on a new "megascale" dataset (see paper_config.py).

Background: Genetics referee 2 argued the central ThermoMPNN→RMSF correlation
is structurally circular by construction (ThermoMPNN's inputs include
backbone coordinates; its outputs are learned packing features). The MegaScale
analysis answers that directly — both axes (ΔΔG and B-factor) become
experimental observables, with no ML in the loop. The MegaScale × B-factor
partial-ρ control over WCN then layers the structural control on top.

Usage:
    python scripts/preprocess_megascale_natural.py \\
        --min_variants 100   \\
        --min_mut_per_pos 8  \\
        --max_resolution 2.0 \\
        --max_proteins 0      # 0 = all qualifying

Outputs:
  $PROJECT_DIR/data/megascale_processed/proteins/{pdb}_{chain}/  (ATLAS layout)
  $PROJECT_DIR/data/megascale_robustness/experimental/           (per-protein)
  $PROJECT_DIR/data/megascale_processed/inventory_processed.json (summary)
"""

import argparse
import csv
import json
import re
import sys
import time
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from paper_config import CLUSTER
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from paper_config import CLUSTER

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")     # 20-AA index used by the 20-DDG Ridge
AA_TO_IDX = {a: i for i, a in enumerate(AA_ORDER)}

# Variant name patterns:
#   {ID}.pdb         -> WT (no underscore mutation suffix)
#   {ID}.pdb_<AA><pos><AA>   -> single-mutant
PDB_ID_RE = re.compile(r"^([0-9][A-Za-z0-9]{3})\.pdb(?:_([A-Z]\d+[A-Z]))?$")
MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


# ============================================================================
# CSV streaming + variant grouping
# ============================================================================

def stream_natural_variants(csv_path: Path,
                            keep_pdb_ids: Optional[set] = None
                            ) -> dict:
    """One pass over the Tsuboyama CSV. Return dict:
        pdb_id -> list of (position, aa_wt, aa_mut, ddg)
    For WT rows (mutation_str == None) we also need the row to extract the
    WT deltaG used as the per-protein reference. So the function returns
    a second dict: pdb_id -> deltaG_wt.
    """
    variants: dict = defaultdict(list)
    deltaG_wt: dict = {}
    n_rows, n_natural_rows, n_wt_rows, n_bad = 0, 0, 0, 0

    print(f"Streaming {csv_path} ...", flush=True)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_rows += 1
            if n_rows % 500_000 == 0:
                print(f"  {n_rows:,} rows, "
                      f"{n_natural_rows:,} natural-protein rows, "
                      f"{len(variants):,} proteins so far", flush=True)
            raw_name = (row.get("name") or "").strip()
            m = PDB_ID_RE.match(raw_name)
            if not m:
                continue
            pdb_id = m.group(1).upper()
            if keep_pdb_ids is not None and pdb_id not in keep_pdb_ids:
                continue
            n_natural_rows += 1
            try:
                dg = float(row.get("deltaG", "nan"))
            except (TypeError, ValueError):
                dg = float("nan")
            mut_str = m.group(2)
            if mut_str is None:
                # WT row — record reference ΔG. Some proteins have multiple
                # WT measurements; we average them.
                if not np.isnan(dg):
                    deltaG_wt.setdefault(pdb_id, []).append(dg)
                n_wt_rows += 1
                continue
            mm = MUT_RE.match(mut_str)
            if mm is None:
                n_bad += 1
                continue
            aa_wt, pos_str, aa_mut = mm.group(1), mm.group(2), mm.group(3)
            try:
                pos = int(pos_str)
            except ValueError:
                n_bad += 1
                continue
            variants[pdb_id].append((pos, aa_wt, aa_mut, dg))

    # Reduce WT measurements to a per-protein mean
    deltaG_wt_mean = {p: float(np.nanmean(v)) for p, v in deltaG_wt.items()}
    print(f"Done streaming: {n_rows:,} rows, "
          f"{n_natural_rows:,} natural-protein rows, "
          f"{n_wt_rows:,} WT rows, {n_bad:,} unparseable variants")
    return variants, deltaG_wt_mean


# ============================================================================
# Per-protein DDG profile
# ============================================================================

def build_ddg_profile(variants: list, deltaG_wt: float,
                      min_mut_per_pos: int = 8
                      ) -> Optional[dict]:
    """Turn a list of (pos, aa_wt, aa_mut, deltaG_mut) variants into:
        - L x 20 ddg_matrix with NaN where not measured
        - per-position summary stats (std_ddg, mean_abs_ddg, ...)
    requiring at least `min_mut_per_pos` measured mutations per position.

    Returns None if no position passes the coverage filter or the variants
    are inconsistent (multiple WT identities at the same position).
    """
    if not variants or np.isnan(deltaG_wt):
        return None

    # Group by position, check WT consistency.
    by_pos: dict = defaultdict(dict)         # pos -> {aa_mut: ddg}
    wt_at_pos: dict = {}                     # pos -> aa_wt (consensus)
    for pos, aa_wt, aa_mut, dg in variants:
        if np.isnan(dg):
            continue
        ddg = dg - deltaG_wt                 # convention: positive = destabilising
        if pos in wt_at_pos and wt_at_pos[pos] != aa_wt:
            # Inconsistent WT at this position — skip this position entirely.
            wt_at_pos[pos] = None
            continue
        wt_at_pos[pos] = aa_wt
        by_pos[pos][aa_mut] = ddg

    valid_positions = sorted(p for p, wt in wt_at_pos.items()
                             if wt is not None and len(by_pos[p]) >= min_mut_per_pos)
    if not valid_positions:
        return None

    L = max(valid_positions)                  # 1-based max position
    ddg_matrix = np.full((L, 20), np.nan, dtype=np.float32)
    wt_sequence = ["X"] * L                  # placeholder for unknown WT
    for pos in valid_positions:
        wt_aa = wt_at_pos[pos]
        wt_sequence[pos - 1] = wt_aa
        # Self-substitution is by convention 0 (no mutation)
        if wt_aa in AA_TO_IDX:
            ddg_matrix[pos - 1, AA_TO_IDX[wt_aa]] = 0.0
        for aa_mut, ddg in by_pos[pos].items():
            if aa_mut in AA_TO_IDX:
                ddg_matrix[pos - 1, AA_TO_IDX[aa_mut]] = ddg

    # Per-position summary stats (mirror compute_robustness.py's TSV schema).
    # std_ddg is the headline (per-residue robustness index).
    rows = []
    for pos in valid_positions:
        col = ddg_matrix[pos - 1]
        # Exclude the WT entry (==0) from the spread / mean calculations.
        wt_aa = wt_at_pos[pos]
        if wt_aa in AA_TO_IDX:
            mask = np.ones(20, dtype=bool)
            mask[AA_TO_IDX[wt_aa]] = False
            mut_vals = col[mask]
        else:
            mut_vals = col
        mut_vals = mut_vals[~np.isnan(mut_vals)]
        if len(mut_vals) < min_mut_per_pos:
            continue
        rows.append({
            "position":          pos,
            "wt_aa":             wt_aa,
            "n_measured":        int(len(mut_vals)),
            "std_ddg":           float(np.std(mut_vals)),
            "mean_ddg":          float(np.mean(mut_vals)),
            "mean_abs_ddg":      float(np.mean(np.abs(mut_vals))),
            "max_ddg":           float(np.max(np.abs(mut_vals))),
            "frac_destabilizing": float(np.mean(mut_vals > 1.0)),
            "frac_neutral":      float(np.mean(np.abs(mut_vals) < 0.5)),
        })
    if not rows:
        return None
    return {
        "robustness_df": pd.DataFrame(rows),
        "ddg_matrix":    ddg_matrix,
        "wt_sequence":   "".join(wt_sequence),
        "L":             L,
        "n_positions":   len(rows),
    }


# ============================================================================
# PDB download + parsing
# ============================================================================

RCSB_URL = "https://files.rcsb.org/download/{pdb}.pdb"


def download_pdb(pdb_id: str, dest_dir: Path,
                 retries: int = 2, sleep_s: float = 0.5) -> Optional[Path]:
    """Download a PDB from RCSB. Returns path on success, None on failure."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{pdb_id.upper()}.pdb"
    if out.exists() and out.stat().st_size > 1000:
        return out
    url = RCSB_URL.format(pdb=pdb_id.upper())
    for attempt in range(retries + 1):
        try:
            urllib.request.urlretrieve(url, str(out))
            time.sleep(sleep_s)
            if out.stat().st_size > 1000:
                return out
        except (urllib.error.URLError, OSError):
            time.sleep(sleep_s * (attempt + 1))
    if out.exists():
        out.unlink()
    return None


def parse_pdb_header(pdb_path: Path) -> dict:
    """Read EXPDTA + RESOLUTION (REMARK 2) without any heavy parser."""
    method, resolution = "", None
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("EXPDTA"):
                method = line[10:].strip()
            elif line.startswith("REMARK   2 RESOLUTION"):
                m = re.search(r"([0-9]+\.[0-9]+)", line)
                if m:
                    resolution = float(m.group(1))
            elif line.startswith("ATOM"):  # past header
                break
    return {"method": method, "resolution": resolution}


def extract_bfactor_and_seq(pdb_path: Path, chain: str = "A"
                            ) -> Optional[pd.DataFrame]:
    """Read CA atoms of the chosen chain. Return DataFrame with columns:
        position (PDB residue number, int)
        wt_aa    (single-letter)
        bfactor  (float)
    Skips altloc != A/blank, dedups by (chain, resSeq, iCode).
    """
    three_to_one = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }
    rows = []
    seen = set()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            ch = line[21]
            if ch != chain:
                continue
            resname = line[17:20].strip()
            if resname not in three_to_one:
                continue
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            key = (ch, resseq, line[26])
            if key in seen:
                continue
            seen.add(key)
            try:
                bf = float(line[60:66].strip())
            except ValueError:
                continue
            rows.append((resseq, three_to_one[resname], bf))
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["position", "wt_aa", "bfactor"])


# ============================================================================
# Verify MegaScale variant numbering matches PDB residue numbering
# ============================================================================

def verify_position_match(rob_df: pd.DataFrame, pdb_df: pd.DataFrame,
                          min_match_frac: float = 0.80
                          ) -> Optional[pd.DataFrame]:
    """Verify that the MegaScale variant WT residues match the PDB WT residues
    at the same positions. Try both:
      - direct positional alignment (no offset)
      - simple integer offset search (sometimes Tsuboyama uses 1-indexed
        from the start of the cloned fragment rather than PDB numbering)

    Returns the rob_df with positions remapped to PDB numbering on success,
    or None if even the best offset has < min_match_frac matches.
    """
    pdb_pos_to_aa = dict(zip(pdb_df["position"], pdb_df["wt_aa"]))
    if not pdb_pos_to_aa:
        return None
    best_offset, best_frac, best_matches = None, 0.0, 0
    pdb_pos_min = min(pdb_pos_to_aa)
    rob_pos_min = int(rob_df["position"].min())
    candidate_offsets = sorted(set([0, pdb_pos_min - rob_pos_min,
                                    pdb_pos_min - 1]))
    for offset in candidate_offsets:
        matches = 0
        n_total = 0
        for _, row in rob_df.iterrows():
            shifted = int(row["position"]) + offset
            if shifted in pdb_pos_to_aa:
                n_total += 1
                if pdb_pos_to_aa[shifted] == row["wt_aa"]:
                    matches += 1
        frac = matches / max(n_total, 1)
        if frac > best_frac:
            best_offset, best_frac, best_matches = offset, frac, matches
    if best_frac < min_match_frac:
        return None
    out = rob_df.copy()
    out["position"] = out["position"].astype(int) + best_offset
    # Drop rows that fall outside the PDB's resolved residues
    out = out[out["position"].isin(pdb_pos_to_aa)].reset_index(drop=True)
    if out.empty:
        return None
    return out


# ============================================================================
# Main driver
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", default=f"{CLUSTER.megascale_dir}/Tsuboyama2023_Dataset1.csv",
                        help="Path to MegaScale Dataset1 CSV (default: symlinked under paper_config)")
    parser.add_argument("--inventory", default=f"{CLUSTER.megascale_dir}/inventory.json",
                        help="Path to inventory.json produced by preprocess_megascale.py")
    parser.add_argument("--data_dir", default=f"{CLUSTER.project_dir}/data/megascale_processed",
                        help="Output dataset dir (ATLAS-compatible layout)")
    parser.add_argument("--robustness_dir", default=f"{CLUSTER.project_dir}/data/megascale_robustness",
                        help="Output robustness dir (per-protein TSV/npy)")
    parser.add_argument("--pdb_cache", default=f"{CLUSTER.megascale_dir}/pdbs",
                        help="Where to cache downloaded PDB files")
    parser.add_argument("--min_variants", type=int, default=100,
                        help="Skip natural proteins with fewer than this many measured variants total")
    parser.add_argument("--min_mut_per_pos", type=int, default=8,
                        help="Per-position minimum measured mutations to keep that position")
    parser.add_argument("--max_resolution", type=float, default=2.0,
                        help="Skip PDBs with resolution above this (or NMR)")
    parser.add_argument("--chain", default="A",
                        help="Chain to extract from each PDB (Tsuboyama typically clones chain A)")
    parser.add_argument("--max_proteins", type=int, default=0,
                        help="Limit number of qualifying proteins (0 = all)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip proteins whose output dir already exists with .done marker")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"ERROR: MegaScale CSV not found: {csv_path}")

    # Load the inventory so we know which proteins to pull from the CSV.
    inventory_path = Path(args.inventory)
    if not inventory_path.exists():
        sys.exit(f"ERROR: inventory.json not found: {inventory_path}\n"
                 f"       Run preprocess_megascale.py first.")
    counts_tsv = inventory_path.parent / "variant_counts_per_protein.tsv"
    if not counts_tsv.exists():
        sys.exit(f"ERROR: variant_counts_per_protein.tsv not found alongside inventory.json")
    counts = pd.read_csv(counts_tsv, sep="\t")
    qualifying = counts[(counts["class"] == "natural") &
                        (counts["n_variants"] >= args.min_variants)]
    qualifying_ids = set(qualifying["protein"].str.upper())
    print(f"Qualifying natural proteins (>={args.min_variants} variants): "
          f"{len(qualifying_ids)}")

    # Stream the CSV once, grouping variants by natural-protein PDB ID.
    variants_by_pdb, deltaG_wt_by_pdb = stream_natural_variants(
        csv_path, keep_pdb_ids=qualifying_ids)
    print(f"Collected variants for {len(variants_by_pdb)} natural proteins")
    print(f"Found WT ΔG for {len(deltaG_wt_by_pdb)} of them")

    data_dir = Path(args.data_dir)
    rob_dir  = Path(args.robustness_dir) / "experimental"
    pdb_cache = Path(args.pdb_cache)
    (data_dir / "proteins").mkdir(parents=True, exist_ok=True)
    rob_dir.mkdir(parents=True, exist_ok=True)
    pdb_cache.mkdir(parents=True, exist_ok=True)

    # Diagnostic counters
    stats = defaultdict(int)
    written_ids = []

    pdb_ids = sorted(set(variants_by_pdb.keys()) & set(deltaG_wt_by_pdb.keys()))
    if args.max_proteins > 0:
        pdb_ids = pdb_ids[: args.max_proteins]

    print(f"\nProcessing {len(pdb_ids)} proteins ...")
    for i, pdb_id in enumerate(pdb_ids):
        prot_id = f"{pdb_id.lower()}_{args.chain}"
        protein_outdir = data_dir / "proteins" / prot_id
        done_marker = protein_outdir / ".done"
        if args.skip_existing and done_marker.exists():
            stats["skipped_existing"] += 1
            continue

        if i % 25 == 0:
            print(f"  [{i+1}/{len(pdb_ids)}] {pdb_id} ...", flush=True)

        # 1. Build per-position ΔΔG profile.
        prof = build_ddg_profile(variants_by_pdb[pdb_id],
                                 deltaG_wt_by_pdb[pdb_id],
                                 min_mut_per_pos=args.min_mut_per_pos)
        if prof is None:
            stats["no_positions"] += 1
            continue

        # 2. Download + filter PDB.
        pdb_path = download_pdb(pdb_id, pdb_cache)
        if pdb_path is None:
            stats["pdb_download_fail"] += 1
            continue
        header = parse_pdb_header(pdb_path)
        if "X-RAY" not in header["method"].upper():
            stats["not_xray"] += 1
            continue
        if header["resolution"] is None or header["resolution"] > args.max_resolution:
            stats["resolution_fail"] += 1
            continue

        # 3. Extract per-residue B-factors for the chosen chain.
        pdb_df = extract_bfactor_and_seq(pdb_path, chain=args.chain)
        if pdb_df is None or len(pdb_df) < 10:
            stats["no_bfactor"] += 1
            continue

        # 4. Verify variant numbering matches PDB numbering (with offset search).
        rob_df = verify_position_match(prof["robustness_df"], pdb_df,
                                       min_match_frac=0.80)
        if rob_df is None:
            stats["position_mismatch"] += 1
            continue

        # 5. Subset B-factor table to matched positions.
        bf_df = pdb_df[pdb_df["position"].isin(rob_df["position"])].reset_index(drop=True)
        bf_out = bf_df[["position", "bfactor"]].copy()

        # 6. Write per-protein outputs.
        protein_outdir.mkdir(parents=True, exist_ok=True)
        # PDB file: filter to the analyzed chain (default A). Multi-chain
        # PDBs from RCSB would otherwise cause SASA/packing to aggregate
        # across chains, producing wrong sums at shared resSeqs.
        out_pdb = protein_outdir / f"{prot_id}.pdb"
        with open(pdb_path) as src, open(out_pdb, "w") as dst:
            for line in src:
                if line.startswith(("ATOM", "HETATM")):
                    if line[21] == args.chain:
                        dst.write(line)
                else:
                    dst.write(line)
        # B-factor TSV
        bf_out.to_csv(protein_outdir / f"{prot_id}_Bfactor.tsv",
                      sep="\t", index=False)
        # Robustness TSV (the scalar summaries that drive the bivariate analysis)
        rob_out_path = rob_dir / f"{prot_id}_robustness.tsv"
        rob_df.to_csv(rob_out_path, sep="\t", index=False)
        # NOTE: we deliberately do NOT write a *_ddg_matrix.npy here.
        # MegaScale per-position coverage is sparse (not all 19 substitutions
        # measured at every position), so the dense (L,20) matrix the
        # multi_ddg_regression.py Ridge expects would be NaN-heavy. The
        # Ridge drops any row with NaN, leaving almost nothing. The
        # bivariate sd(ΔΔG) × B-factor analysis (the actual reason
        # MegaScale is in the paper) only needs the scalar columns above.
        # Robustness JSON (global metrics) — used by load_robustness_global
        with open(rob_dir / f"{prot_id}_robustness.json", "w") as f:
            json.dump({
                "protein_id":    prot_id,
                "L":             int(prof["L"]),
                "n_positions":   int(len(rob_df)),
                "global_metrics": {
                    "mean_std_ddg":      float(rob_df["std_ddg"].mean()),
                    "mean_abs_ddg":      float(rob_df["mean_abs_ddg"].mean()),
                    "n_positions_used":  int(len(rob_df)),
                    "resolution_A":      header["resolution"],
                    "method":            header["method"],
                },
            }, f, indent=2)
        done_marker.touch()
        written_ids.append(prot_id)
        stats["written"] += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"MegaScale natural-protein preprocessing summary")
    print(f"{'='*60}")
    for k in ("written", "skipped_existing", "no_positions",
              "pdb_download_fail", "not_xray", "resolution_fail",
              "no_bfactor", "position_mismatch"):
        print(f"  {k:24s}: {stats[k]}")
    summary_path = data_dir / "inventory_processed.json"
    with open(summary_path, "w") as f:
        json.dump({
            "params": vars(args),
            "stats":  dict(stats),
            "written_proteins": written_ids,
        }, f, indent=2)
    print(f"\nWrote summary: {summary_path}")
    print(f"\nNext steps:")
    print(f"  1. Add 'megascale' as a Dataset in paper_config.py "
          f"with available_scorers=['experimental'].")
    print(f"  2. Run correlation analysis (--target bfactor --scorer experimental).")


if __name__ == "__main__":
    main()
