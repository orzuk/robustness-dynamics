#!/usr/bin/env python3
"""
Download ATLAS database analysis files (RMSF, pLDDT, B-factors, Neq, PDB)
for Direction 7: robustness vs. dynamics analysis.

Usage (on HUJI cluster):
    python download_atlas.py --output_dir /sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas
    python download_atlas.py --output_dir .../data/atlas --max_proteins 50  # test run
    python download_atlas.py --output_dir .../data/atlas --metadata_only     # just metadata

This downloads the "analysis" ZIP for each protein, extracts the TSV files
(RMSF, pLDDT, Bfactor, Neq) and the PDB structure, then cleans up the ZIP.

Data source: https://www.dsimb.inserm.fr/ATLAS
Reference: Vander Meersche et al. 2023, Nucleic Acids Research, gkad1084
License: CC-BY-NC 4.0
"""

import os
import sys
import json
import time
import zipfile
import argparse
import urllib.request
import urllib.error
from pathlib import Path


# ATLAS URLs
ATLAS_BASE = "https://www.dsimb.inserm.fr/ATLAS"
PDB_LIST_URL = f"{ATLAS_BASE}/data/download/distributions/2024_11_18_ATLAS_pdb.txt"
INFO_TSV_URL = f"{ATLAS_BASE}/data/download/distributions/2024_11_18_ATLAS_info.tsv"
ANALYSIS_API = f"{ATLAS_BASE}/api/ATLAS/analysis"        # /{pdb_chain}
METADATA_API = f"{ATLAS_BASE}/api/ATLAS/metadata"        # /{pdb_chain}

# Files to extract from each analysis ZIP
KEEP_SUFFIXES = ["_RMSF.tsv", "_pLDDT.tsv", "_Bfactor.tsv", "_Neq.tsv", ".pdb",
                 "_corresp.tsv"]


def download_file(url, dest_path, retries=3, delay=2, timeout=60):
    """Download a file with retries, exponential backoff, and socket timeout."""
    for attempt in range(retries):
        try:
            resp = urllib.request.urlopen(url, timeout=timeout)
            with open(dest_path, 'wb') as f:
                f.write(resp.read())
            return True
        except urllib.error.HTTPError as e:
            if e.code == 429:  # rate limited
                wait = delay * (2 ** attempt)
                print(f"  Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
            elif e.code == 404:
                print(f"  404 Not Found: {url}", flush=True)
                return False
            else:
                print(f"  HTTP {e.code} for {url}, attempt {attempt+1}/{retries}", flush=True)
                time.sleep(delay)
        except Exception as e:
            print(f"  Error: {e}, attempt {attempt+1}/{retries}", flush=True)
            time.sleep(delay)
    return False


def fetch_protein_list(output_dir):
    """Download the master list of ATLAS protein chains."""
    list_path = output_dir / "ATLAS_pdb_list.txt"
    if list_path.exists():
        print(f"Using cached protein list: {list_path}")
    else:
        print(f"Downloading protein list from {PDB_LIST_URL}")
        if not download_file(PDB_LIST_URL, str(list_path)):
            sys.exit("Failed to download protein list")

    with open(list_path) as f:
        proteins = [line.strip() for line in f if line.strip()]
    print(f"Found {len(proteins)} proteins in ATLAS")
    return proteins


def fetch_info_tsv(output_dir):
    """Download the master info TSV with per-protein metadata."""
    info_path = output_dir / "ATLAS_info.tsv"
    if info_path.exists():
        print(f"Using cached info TSV: {info_path}")
    else:
        print(f"Downloading info TSV from {INFO_TSV_URL}")
        if not download_file(INFO_TSV_URL, str(info_path)):
            print("Warning: failed to download info TSV")
    return info_path


def download_and_extract_analysis(pdb_chain, output_dir, keep_zip=False):
    """Download analysis ZIP for one protein, extract relevant TSVs + PDB."""
    protein_dir = output_dir / "proteins" / pdb_chain
    protein_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded (marker file)
    done_marker = protein_dir / ".done"
    if done_marker.exists():
        return "skipped"

    # Download ZIP
    zip_path = protein_dir / f"{pdb_chain}_analysis.zip"
    url = f"{ANALYSIS_API}/{pdb_chain}"

    if not download_file(url, str(zip_path)):
        return "failed"

    # Extract relevant files
    try:
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            for member in zf.namelist():
                basename = os.path.basename(member)
                if any(basename.endswith(suffix) for suffix in KEEP_SUFFIXES):
                    # Extract to flat directory (no subdirs)
                    target = protein_dir / basename
                    with zf.open(member) as src, open(target, 'wb') as dst:
                        dst.write(src.read())
    except zipfile.BadZipFile:
        print(f"  Bad ZIP file for {pdb_chain}")
        zip_path.unlink(missing_ok=True)
        return "failed"

    # Clean up ZIP (save disk space)
    if not keep_zip:
        zip_path.unlink(missing_ok=True)

    # Mark as done
    done_marker.touch()
    return "ok"


def download_metadata_json(pdb_chain, output_dir):
    """Download metadata JSON for one protein (lightweight, no ZIP)."""
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    meta_path = meta_dir / f"{pdb_chain}.json"
    if meta_path.exists():
        return "skipped"

    url = f"{METADATA_API}/{pdb_chain}"
    tmp_path = meta_dir / f"{pdb_chain}.tmp"

    if not download_file(url, str(tmp_path)):
        return "failed"

    # Validate JSON
    try:
        with open(tmp_path) as f:
            json.load(f)
        tmp_path.rename(meta_path)
        return "ok"
    except json.JSONDecodeError:
        print(f"  Invalid JSON for {pdb_chain}")
        tmp_path.unlink(missing_ok=True)
        return "failed"


def main():
    parser = argparse.ArgumentParser(
        description="Download ATLAS protein dynamics data (RMSF, pLDDT, etc.)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store downloaded data")
    parser.add_argument("--max_proteins", type=int, default=0,
                        help="Limit number of proteins (0 = all, use e.g. 50 for testing)")
    parser.add_argument("--metadata_only", action="store_true",
                        help="Only download metadata JSON (no analysis ZIPs)")
    parser.add_argument("--keep_zip", action="store_true",
                        help="Keep ZIP files after extraction (default: delete)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between downloads in seconds (be nice to server)")
    parser.add_argument("--start_from", type=int, default=0,
                        help="Start from protein index N (for resuming)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get protein list and info TSV
    proteins = fetch_protein_list(output_dir)
    fetch_info_tsv(output_dir)

    # Apply limits
    if args.max_proteins > 0:
        proteins = proteins[:args.max_proteins]
        print(f"Limiting to first {args.max_proteins} proteins")

    if args.start_from > 0:
        proteins = proteins[args.start_from:]
        print(f"Starting from index {args.start_from}")

    # Step 2: Download per-protein data
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    total = len(proteins)

    for i, pdb_chain in enumerate(proteins):
        progress = f"[{i+1}/{total}]"

        if args.metadata_only:
            status = download_metadata_json(pdb_chain, output_dir)
            print(f"{progress} {pdb_chain}: metadata {status}", flush=True)
        else:
            # Download both metadata and analysis
            meta_status = download_metadata_json(pdb_chain, output_dir)
            analysis_status = download_and_extract_analysis(
                pdb_chain, output_dir, keep_zip=args.keep_zip)
            status = analysis_status
            print(f"{progress} {pdb_chain}: analysis={analysis_status} "
                  f"metadata={meta_status}", flush=True)

        counts[status] = counts.get(status, 0) + 1

        # Rate limiting
        if status == "ok":
            time.sleep(args.delay)

    # Summary
    print(f"\nDone! {counts['ok']} downloaded, {counts['skipped']} skipped, "
          f"{counts['failed']} failed")
    print(f"Data in: {output_dir}")
    print(f"\nPer-protein files in: {output_dir}/proteins/{{pdb_chain}}/")
    print(f"  *_RMSF.tsv    - per-residue RMSF (Angstroms, 3 replicates)")
    print(f"  *_pLDDT.tsv   - per-residue AlphaFold2 pLDDT")
    print(f"  *_Bfactor.tsv - per-residue experimental B-factors")
    print(f"  *_Neq.tsv     - per-residue backbone deformability")
    print(f"  *.pdb         - structure after minimization")


if __name__ == "__main__":
    main()
