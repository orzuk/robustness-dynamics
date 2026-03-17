#!/usr/bin/env python3
"""Check which PDB design proteins have ConSurf-DB entries.

Reports MSA depth from ConSurf JSON to help distinguish:
- Natural proteins (deep MSA, hundreds of homologs)
- Designed proteins on natural scaffolds (moderate MSA from parent)
- Truly novel designs (shallow/no MSA)

Usage:
    python scripts/check_designs_consurf.py
"""

import json
import gzip
from pathlib import Path
from paper_config import CLUSTER

PROJECT = CLUSTER.project_dir
CONSURF_FILES = Path(CLUSTER.consurf_dir) / "files"
DESIGNS_DIR = Path(PROJECT) / "data" / "pdb_designs" / "proteins"
MAP_FILE = CONSURF_FILES.parent / "identical_to_unique_dict.txt"
METADATA = Path(PROJECT) / "data" / "pdb_designs" / "metadata.tsv"


def load_mapping():
    mapping = {}
    if MAP_FILE.exists():
        with open(MAP_FILE) as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    k, v = line.split(":", 1)
                    mapping[k.strip().lower()] = v.strip()
    return mapping


def get_consurf_info(json_path):
    """Extract MSA depth and score stats from ConSurf JSON."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        scores = data.get("SCORE", [])
        n_residues = len(scores)
        non_none = [s for s in scores if s is not None]
        n_scored = len(non_none)
        mean_score = sum(non_none) / len(non_none) if non_none else None

        # MSA depth: try msa_ratio (top-level), or dig into MSA DATA
        n_seqs = "?"
        msa_ratio = data.get("msa_ratio")
        if msa_ratio is not None:
            n_seqs = f"ratio={msa_ratio:.2f}" if isinstance(msa_ratio, float) else str(msa_ratio)

        msa_raw = data.get("MSA_DATA", data.get("MSA DATA"))
        msa_keys = []
        if isinstance(msa_raw, dict):
            msa_keys = list(msa_raw.keys())
            for key in ("n_sequences", "num_sequences", "NUMBER_OF_SEQS",
                         "n_seqs", "number_of_sequences", "NSEQS"):
                if key in msa_raw:
                    n_seqs = msa_raw[key]
                    break
        elif isinstance(msa_raw, list):
            n_seqs = f"list[{len(msa_raw)}]"
            msa_keys = [f"list_len={len(msa_raw)}"]

        return {
            "n_seqs": n_seqs,
            "n_residues": n_residues,
            "n_scored": n_scored,
            "mean_score": mean_score,
            "msa_keys": msa_keys,
            "top_keys": list(data.keys()),
        }
    except Exception as e:
        return {"error": str(e)}


def get_pdb_title(protein_dir, pdb_id):
    """Extract TITLE and KEYWDS from PDB file in protein directory."""
    pdb_code = pdb_id.split("_")[0]
    candidates = [
        protein_dir / f"{pdb_id}.pdb",
        protein_dir / f"{pdb_code}.pdb",
        protein_dir / f"{pdb_id}.pdb.gz",
        protein_dir / f"{pdb_code}.pdb.gz",
        protein_dir / f"{pdb_id.upper()}.pdb",
        protein_dir / f"{pdb_id.lower()}.pdb",
    ]
    # Also try any .pdb or .cif file in the directory
    for p in protein_dir.glob("*.pdb"):
        if p not in candidates:
            candidates.append(p)
    for p in protein_dir.glob("*.pdb.gz"):
        if p not in candidates:
            candidates.append(p)
    for p in protein_dir.glob("*.cif"):
        if p not in candidates:
            candidates.append(p)

    for pdb_path in candidates:
        if not pdb_path.exists():
            continue
        try:
            opener = gzip.open if str(pdb_path).endswith(".gz") else open
            title_lines = []
            keywds_lines = []
            header = ""
            with opener(pdb_path, "rt") as f:
                for line in f:
                    if line.startswith("TITLE"):
                        title_lines.append(line[10:].strip())
                    elif line.startswith("KEYWDS"):
                        keywds_lines.append(line[10:].strip())
                    elif line.startswith("HEADER"):
                        header = line[10:].strip()
                    elif line.startswith("ATOM"):
                        break  # stop after header section
            title = " ".join(title_lines).strip()
            keywds = " ".join(keywds_lines).strip()
            return title, keywds, header
        except Exception:
            continue
    return "", "", ""


def main():
    mapping = load_mapping()

    design_ids = sorted([d.name for d in DESIGNS_DIR.iterdir() if d.is_dir()])
    print(f"Total PDB design proteins: {len(design_ids)}")

    # Load metadata if available
    meta = {}
    if METADATA.exists():
        with open(METADATA) as f:
            header = f.readline().strip().split("\t")
            for line in f:
                fields = line.strip().split("\t")
                if len(fields) >= 2:
                    row = dict(zip(header, fields))
                    pid = row.get("protein_id", row.get("pdb_chain", fields[0]))
                    meta[pid] = row

    results = []
    for pid in design_ids:
        parts = pid.split("_")
        if len(parts) != 2:
            continue
        pdb, chain = parts

        json_path = None
        # Direct match
        cand = CONSURF_FILES / f"{pdb.upper()}_{chain.upper()}_consurf_info.json"
        if cand.exists():
            json_path = cand
        else:
            # Via mapping
            pid_lower = pid.lower()
            if pid_lower in mapping:
                mapped = mapping[pid_lower]
                mp, mc = mapped.split("_")
                mc_file = CONSURF_FILES / f"{mp.upper()}_{mc.upper()}_consurf_info.json"
                if mc_file.exists():
                    json_path = mc_file

        if json_path is not None:
            info = get_consurf_info(json_path)
            results.append((pid, info))

    print(f"Have ConSurf scores: {len(results)} / {len(design_ids)}")

    # Get PDB titles for flagged proteins
    print(f"\n{'PDB_ID':12s} {'MSA_seqs':>10s} {'scored':>8s} {'mean_score':>10s}  TITLE")
    print("=" * 120)
    for pid, info in sorted(results, key=lambda x: -(x[1].get("n_seqs", 0)
                            if isinstance(x[1].get("n_seqs"), (int, float)) else 0)):
        n_seqs = info.get("n_seqs", "?")
        n_scored = info.get("n_scored", "?")
        mean_s = info.get("mean_score")
        mean_str = f"{mean_s:.3f}" if mean_s is not None else "?"
        protein_dir = DESIGNS_DIR / pid
        title, keywds, header = get_pdb_title(protein_dir, pid)
        title_short = title[:70] if title else "(no title)"
        print(f"{pid:12s} {str(n_seqs):>10s} {str(n_scored):>8s} {mean_str:>10s}  {title_short}")

    # If first entry has n_seqs="?", show MSA keys for debugging
    if results and results[0][1].get("n_seqs") == "?":
        msa_keys = results[0][1].get("msa_keys", [])
        print(f"\nWARNING: MSA depth not found. MSA keys in first file: {msa_keys}")
        # Also dump the full MSA DATA dict for the first file
        pid0 = results[0][0]
        pdb0, chain0 = pid0.split("_")
        cand0 = CONSURF_FILES / f"{pdb0.upper()}_{chain0.upper()}_consurf_info.json"
        if cand0.exists():
            with open(cand0) as f:
                d = json.load(f)
            msa_data = d.get("MSA_DATA", d.get("MSA DATA", {}))
            print(f"Full MSA DATA for {pid0}:")
            for k, v in msa_data.items():
                vstr = str(v)[:80] if not isinstance(v, (int, float)) else str(v)
                print(f"  {k}: {vstr}")

    # Summary: check for keywords suggesting natural protein
    print(f"\n\n--- Keyword analysis ---")
    natural_keywords = ["wild type", "wild-type", "wildtype", "native",
                        "mutant", "mutation", "variant", "crystal structure of",
                        "enzyme", "kinase", "protease", "lyase", "synthase",
                        "transferase", "oxidase", "reductase", "dehydrogenase"]
    design_keywords = ["de novo", "denovo", "designed", "design", "computational",
                       "rosetta", "hallucin", "artificial", "synthetic"]
    flagged_natural = []
    flagged_design = []
    for pid, info in results:
        protein_dir = DESIGNS_DIR / pid
        title, keywds, header = get_pdb_title(protein_dir, pid)
        combined = (title + " " + keywds + " " + header).lower()
        nat_hits = [kw for kw in natural_keywords if kw in combined]
        des_hits = [kw for kw in design_keywords if kw in combined]
        if nat_hits and not des_hits:
            flagged_natural.append((pid, nat_hits, title[:60]))
        elif des_hits:
            flagged_design.append((pid, des_hits, title[:60]))

    print(f"\nLikely NATURAL (has natural keywords, no design keywords): {len(flagged_natural)}")
    for pid, kws, title in flagged_natural:
        print(f"  {pid:12s}  keywords={kws}  title={title}")

    print(f"\nLikely DESIGNED (has design keywords): {len(flagged_design)}")
    for pid, kws, title in flagged_design:
        print(f"  {pid:12s}  keywords={kws}  title={title}")

    print(f"\nAmbiguous (neither clear natural nor design keywords): "
          f"{len(results) - len(flagged_natural) - len(flagged_design)}")

    # Show JSON structure from first file
    if results:
        info0 = results[0][1]
        print(f"\nFirst file JSON structure:")
        print(f"  Top-level keys: {info0.get('top_keys', '?')}")
        print(f"  MSA DATA type/keys: {info0.get('msa_keys', '?')}")


if __name__ == "__main__":
    main()
