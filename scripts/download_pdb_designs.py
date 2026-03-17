#!/usr/bin/env python3
"""
Download de novo designed protein structures from the PDB for B-factor analysis.

This script:
  1. Queries the RCSB PDB Search API for de novo designed proteins with
     quality filters matching ATLAS (X-ray, resolution <= 2.0 A, >= 38 residues).
  2. Fetches metadata for each entry (title, resolution, sequence, chain info).
  3. Filters: single protein entity, monomeric assembly, non-membrane.
  4. Downloads PDB files and extracts per-residue B-factors.
  5. Saves a master metadata TSV and per-protein B-factor TSV files.

Usage (on HUJI cluster):
    python scripts/download_pdb_designs.py --output_dir /path/to/pdb_designs
    python scripts/download_pdb_designs.py --output_dir /path/to/pdb_designs --max_proteins 20  # test
    python scripts/download_pdb_designs.py --output_dir /path/to/pdb_designs --metadata_only

Output structure:
    output_dir/
        metadata.tsv              # master table: pdb_id, chain, resolution, length, title, ...
        proteins/
            {pdb_id}_{chain}/
                {pdb_id}.pdb      # full PDB file
                {pdb_id}_{chain}_Bfactor.tsv   # per-residue B-factors (Ca)
                .done             # marker file
"""

import os
import sys
import json
import time
import gzip
import argparse
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path

# ============================================================================
# RCSB API endpoints
# ============================================================================
SEARCH_API = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_API = "https://data.rcsb.org/graphql"
PDB_DOWNLOAD = "https://files.rcsb.org/download/{pdb_id}.pdb.gz"

# Search terms for de novo designed proteins
SEARCH_TERMS = [
    "de novo designed protein",
    "de novo protein design",
    "computationally designed protein",
]

# Membrane protein keywords to exclude (in title or description)
MEMBRANE_KEYWORDS = [
    "membrane", "transmembrane", "channel", "transporter", "receptor",
    "pore", "ion channel", "gpcr", "porin",
]

# Title-based false-positive filtering.
# The full-text search matches papers where *ligands* or *drugs* were "designed",
# not the protein itself.  We exclude entries whose title contains a
# ligand/complex indicator keyword UNLESS the title also contains a genuine
# protein-design keyword.

# Keywords that suggest the entry is a natural protein with a designed ligand
FALSE_POSITIVE_KEYWORDS = [
    "in complex with", "bound to", "bound with", "complexed with",
    "ligand", "inhibitor", "inhibition", "drug", "compound",
    "substrate", "agonist", "antagonist", "antibody", "nanobody",
    "peptide inhibitor", "small molecule", "fragment",
    "binding of", "recognition of",
]

# Keywords indicating structural-genomics targets, natural proteins, or
# fusion constructs (caught by full-text search incidentally)
NATURAL_PROTEIN_KEYWORDS = [
    "structural genomics", "northeast structural", "new york sgx",
    "midwest center for structural genomics", "joint center for structural",
    "seattle structural genomics",
    "colicin", "bacteriocin", "immunity protein",
    "fusion mbp", "fusion with mbp", "mbp fusion",
    "fusion gfp", "fusion with gfp", "gfp fusion",
]

# Positive keywords — if the title contains one of these, it is likely a
# genuine de novo design even if a false-positive keyword also appears
DESIGN_POSITIVE_KEYWORDS = [
    "de novo design", "de novo protein", "de novo fold",
    "de novo-designed", "designed protein", "designed fold",
    "designed repeat", "designed helix", "designed helical",
    "designed beta", "designed bundle", "designed barrel",
    "designed coiled", "designed mini", "designed scaffold",
    "computationally designed", "computational design",
    "computational protein design",
    "rosetta", "proteinmpnn", "hallucination",
    "rfdiffusion", "protein design",
    "top7", "topology design",
]

# ATLAS-comparable filters
MAX_RESOLUTION = 2.0    # Angstroms (ATLAS uses <= 2.0)
MIN_CHAIN_LENGTH = 38   # residues (ATLAS criterion)
MAX_CHAIN_LENGTH = 2000 # exclude very large complexes


def search_pdb(search_term: str) -> list:
    """Query RCSB Search API for PDB IDs matching a search term + quality filters."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "full_text",
                    "parameters": {"value": search_term},
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": MAX_RESOLUTION,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 10000}},
    }
    url = SEARCH_API + "?json=" + urllib.parse.quote(json.dumps(query))
    try:
        resp = urllib.request.urlopen(url, timeout=30)
        result = json.loads(resp.read())
        pdb_ids = [r["identifier"] for r in result.get("result_set", [])]
        print(f"  '{search_term}': {len(pdb_ids)} entries")
        return pdb_ids
    except Exception as e:
        print(f"  Search failed for '{search_term}': {e}")
        return []


def fetch_metadata_batch(pdb_ids: list) -> dict:
    """Fetch metadata for a batch of PDB entries via GraphQL.

    Returns dict: pdb_id -> metadata dict.
    Processes in batches of 50 to avoid query size limits.
    """
    all_metadata = {}
    batch_size = 50

    for i in range(0, len(pdb_ids), batch_size):
        batch = pdb_ids[i : i + batch_size]
        # Build a multi-entry GraphQL query
        queries = []
        for j, pdb_id in enumerate(batch):
            queries.append(f'''
            e{j}: entry(entry_id: "{pdb_id}") {{
                rcsb_entry_container_identifiers {{ entry_id }}
                struct {{ title }}
                rcsb_entry_info {{
                    resolution_combined
                    polymer_entity_count_protein
                    assembly_count
                }}
                assemblies {{
                    rcsb_assembly_info {{
                        polymer_entity_instance_count
                    }}
                }}
                polymer_entities {{
                    entity_poly {{
                        pdbx_seq_one_letter_code_can
                        rcsb_entity_polymer_type
                        pdbx_strand_id
                    }}
                    rcsb_polymer_entity {{
                        pdbx_description
                    }}
                }}
            }}
            ''')

        gql = {"query": "{ " + " ".join(queries) + " }"}
        data = json.dumps(gql).encode()
        req = urllib.request.Request(
            GRAPHQL_API, data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            result = json.loads(resp.read())
            if "errors" in result:
                print(f"  GraphQL errors in batch {i//batch_size}: "
                      f"{result['errors'][0]['message'][:100]}")

            for j, pdb_id in enumerate(batch):
                entry = result.get("data", {}).get(f"e{j}")
                if entry is None:
                    continue

                # Extract protein entities
                protein_entities = []
                for pe in (entry.get("polymer_entities") or []):
                    ep = pe.get("entity_poly") or {}
                    if ep.get("rcsb_entity_polymer_type") == "Protein":
                        seq = ep.get("pdbx_seq_one_letter_code_can", "")
                        chains = ep.get("pdbx_strand_id", "")
                        desc = (pe.get("rcsb_polymer_entity") or {}).get(
                            "pdbx_description", "")
                        protein_entities.append({
                            "sequence": seq,
                            "chains": chains,
                            "description": desc,
                            "length": len(seq),
                        })

                # Assembly info: prefer monomeric
                assemblies = entry.get("assemblies") or []
                assembly_sizes = []
                for asm in assemblies:
                    info = asm.get("rcsb_assembly_info") or {}
                    count = info.get("polymer_entity_instance_count", 0)
                    assembly_sizes.append(count)

                info = entry.get("rcsb_entry_info") or {}
                res_list = info.get("resolution_combined") or []
                resolution = res_list[0] if res_list else None

                title = (entry.get("struct") or {}).get("title", "")

                all_metadata[pdb_id] = {
                    "pdb_id": pdb_id,
                    "title": title,
                    "resolution": resolution,
                    "n_protein_entities": info.get(
                        "polymer_entity_count_protein", 0),
                    "assembly_sizes": assembly_sizes,
                    "protein_entities": protein_entities,
                }

        except Exception as e:
            print(f"  GraphQL batch {i//batch_size} failed: {e}")
            time.sleep(2)

        if i + batch_size < len(pdb_ids):
            time.sleep(0.5)  # rate limit

    return all_metadata


def filter_entry(meta: dict) -> tuple:
    """Apply quality filters to a PDB entry.

    Returns (chain_id, sequence, reason) where reason is None if accepted,
    or a string explaining why it was rejected.
    """
    title_lower = meta["title"].lower()

    # False-positive check: title suggests a natural protein + designed ligand
    has_fp_keyword = any(kw in title_lower for kw in FALSE_POSITIVE_KEYWORDS)
    has_design_keyword = any(kw in title_lower for kw in DESIGN_POSITIVE_KEYWORDS)
    if has_fp_keyword and not has_design_keyword:
        matched = [kw for kw in FALSE_POSITIVE_KEYWORDS if kw in title_lower]
        return None, None, f"likely false positive (title: '{matched[0]}')"

    # Structural genomics / natural protein check
    has_natural_keyword = any(kw in title_lower for kw in NATURAL_PROTEIN_KEYWORDS)
    if has_natural_keyword and not has_design_keyword:
        matched = [kw for kw in NATURAL_PROTEIN_KEYWORDS if kw in title_lower]
        return None, None, f"natural protein (title: '{matched[0]}')"

    # Membrane protein check
    for kw in MEMBRANE_KEYWORDS:
        if kw in title_lower:
            return None, None, f"membrane keyword: {kw}"

    # Also check entity descriptions
    for pe in meta["protein_entities"]:
        desc_lower = pe["description"].lower()
        for kw in MEMBRANE_KEYWORDS:
            if kw in desc_lower:
                return None, None, f"membrane keyword in entity: {kw}"

    # Single protein entity (exclude hetero-oligomers with different chains)
    if meta["n_protein_entities"] != 1:
        return None, None, f"multi-entity ({meta['n_protein_entities']} proteins)"

    # Get the protein entity
    if not meta["protein_entities"]:
        return None, None, "no protein entity"
    pe = meta["protein_entities"][0]

    # Chain length
    seq_len = pe["length"]
    if seq_len < MIN_CHAIN_LENGTH:
        return None, None, f"too short ({seq_len} < {MIN_CHAIN_LENGTH})"
    if seq_len > MAX_CHAIN_LENGTH:
        return None, None, f"too long ({seq_len} > {MAX_CHAIN_LENGTH})"

    # Check for monomeric assembly (at least one assembly with 1 chain)
    has_monomer = any(s == 1 for s in meta["assembly_sizes"])
    if not has_monomer and meta["assembly_sizes"]:
        return None, None, f"no monomeric assembly (sizes: {meta['assembly_sizes']})"

    # Pick first chain
    chains = pe["chains"].split(",")
    chain_id = chains[0].strip() if chains else "A"

    return chain_id, pe["sequence"], None


def download_pdb(pdb_id: str, dest_path: str, retries: int = 3) -> bool:
    """Download a PDB file (gzipped) and decompress it."""
    url = PDB_DOWNLOAD.format(pdb_id=pdb_id)
    gz_path = dest_path + ".gz"
    for attempt in range(retries):
        try:
            resp = urllib.request.urlopen(url, timeout=60)
            with open(gz_path, "wb") as f:
                f.write(resp.read())
            # Decompress
            with gzip.open(gz_path, "rb") as gz:
                with open(dest_path, "wb") as out:
                    out.write(gz.read())
            os.remove(gz_path)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"  404: {url}")
                return False
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            print(f"  Download error ({pdb_id}): {e}")
            time.sleep(2 * (attempt + 1))
    return False


def extract_ca_bfactors(pdb_path: str, chain_id: str) -> list:
    """Extract per-residue Ca B-factors from a PDB file.

    Returns list of dicts: [{"resid": int, "resname": str, "bfactor": float}, ...]
    """
    bfactors = []
    seen_resids = set()
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            chain = line[21].strip()
            if chain != chain_id:
                continue
            resname = line[17:20].strip()
            # Skip non-standard residues
            if len(resname) != 3:
                continue
            try:
                resid = int(line[22:26].strip())
                bfac = float(line[60:66].strip())
            except (ValueError, IndexError):
                continue
            icode = line[26].strip()
            key = (resid, icode)
            if key in seen_resids:
                continue
            seen_resids.add(key)
            bfactors.append({
                "resid": resid,
                "resname": resname,
                "bfactor": bfac,
            })
    return bfactors


def main():
    parser = argparse.ArgumentParser(
        description="Download de novo designed proteins from PDB for B-factor analysis")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--max_proteins", type=int, default=0,
                        help="Max proteins to download (0=all, for testing)")
    parser.add_argument("--metadata_only", action="store_true",
                        help="Only fetch metadata, skip PDB downloads")
    parser.add_argument("--max_resolution", type=float, default=MAX_RESOLUTION,
                        help=f"Max resolution in Angstroms (default: {MAX_RESOLUTION})")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proteins_dir = out_dir / "proteins"
    proteins_dir.mkdir(exist_ok=True)

    # ---- Step 1: Search PDB ----
    print("=" * 60)
    print("Step 1: Searching RCSB PDB for de novo designed proteins")
    print("=" * 60)

    all_pdb_ids = set()
    for term in SEARCH_TERMS:
        ids = search_pdb(term)
        all_pdb_ids.update(ids)
    all_pdb_ids = sorted(all_pdb_ids)
    print(f"\nTotal unique PDB entries: {len(all_pdb_ids)}")

    # ---- Step 2: Fetch metadata ----
    print("\n" + "=" * 60)
    print("Step 2: Fetching metadata via GraphQL")
    print("=" * 60)

    metadata = fetch_metadata_batch(all_pdb_ids)
    print(f"Metadata fetched for {len(metadata)} entries")

    # ---- Step 3: Filter ----
    print("\n" + "=" * 60)
    print("Step 3: Applying quality filters")
    print("=" * 60)

    accepted = []
    reject_reasons = {}

    for pdb_id in sorted(metadata.keys()):
        meta = metadata[pdb_id]
        chain_id, sequence, reason = filter_entry(meta)
        if reason:
            reject_reasons[reason.split(":")[0].split("(")[0].strip()] = \
                reject_reasons.get(
                    reason.split(":")[0].split("(")[0].strip(), 0) + 1
            continue
        accepted.append({
            "pdb_id": pdb_id,
            "chain_id": chain_id,
            "sequence": sequence,
            "length": len(sequence),
            "resolution": meta["resolution"],
            "title": meta["title"],
        })

    print(f"\nAccepted: {len(accepted)}")
    print("Rejection reasons:")
    for reason, count in sorted(reject_reasons.items(),
                                 key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    if args.max_proteins > 0:
        accepted = accepted[: args.max_proteins]
        print(f"\nLimited to {len(accepted)} proteins (--max_proteins)")

    # ---- Step 4: Save metadata ----
    meta_path = out_dir / "metadata.tsv"
    with open(meta_path, "w") as f:
        f.write("pdb_id\tchain\tlength\tresolution\ttitle\n")
        for entry in accepted:
            f.write(f"{entry['pdb_id']}\t{entry['chain_id']}\t"
                    f"{entry['length']}\t{entry['resolution']:.2f}\t"
                    f"{entry['title']}\n")
    print(f"\nMetadata saved to {meta_path}")

    if args.metadata_only:
        print("\n--metadata_only: skipping PDB downloads.")
        return

    # ---- Step 5: Download PDB files and extract B-factors ----
    print("\n" + "=" * 60)
    print(f"Step 4: Downloading {len(accepted)} PDB files + extracting B-factors")
    print("=" * 60)

    n_ok, n_fail = 0, 0
    for i, entry in enumerate(accepted):
        pdb_id = entry["pdb_id"]
        chain_id = entry["chain_id"]
        protein_dir = proteins_dir / f"{pdb_id}_{chain_id}"
        done_marker = protein_dir / ".done"

        if done_marker.exists():
            n_ok += 1
            continue

        protein_dir.mkdir(exist_ok=True)
        pdb_path = protein_dir / f"{pdb_id}.pdb"

        # Download PDB
        if not pdb_path.exists():
            ok = download_pdb(pdb_id, str(pdb_path))
            if not ok:
                print(f"  [{i+1}/{len(accepted)}] FAIL {pdb_id}")
                n_fail += 1
                continue

        # Extract B-factors
        bfactors = extract_ca_bfactors(str(pdb_path), chain_id)
        if not bfactors:
            print(f"  [{i+1}/{len(accepted)}] No CA atoms for {pdb_id}_{chain_id}")
            n_fail += 1
            continue

        bfac_path = protein_dir / f"{pdb_id}_{chain_id}_Bfactor.tsv"
        with open(bfac_path, "w") as f:
            f.write("resid\tresname\tbfactor\n")
            for bf in bfactors:
                f.write(f"{bf['resid']}\t{bf['resname']}\t{bf['bfactor']:.2f}\n")

        # Save sequence
        seq_path = protein_dir / f"{pdb_id}_{chain_id}_sequence.txt"
        with open(seq_path, "w") as f:
            f.write(entry["sequence"])

        done_marker.touch()
        n_ok += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(accepted)}] downloaded: {n_ok}, failed: {n_fail}")
        time.sleep(0.3)  # rate limit

    print(f"\nDone: {n_ok} downloaded, {n_fail} failed")
    print(f"Results in: {out_dir}")


if __name__ == "__main__":
    main()
