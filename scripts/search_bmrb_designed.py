#!/usr/bin/env python3
"""
Search BMRB for designed proteins with NMR relaxation data.

Strategy:
  1. Query BMRB instant search API with design-related keywords
  2. From the returned metadata, check whether relaxation data
     (T1, T2, hetNOE, S2) is deposited
  3. Filter by title heuristics to identify designed proteins
  4. Report the intersection

BMRB API: https://api.bmrb.io/v2/instant?term=...
Returns list of dicts with: value (BMRB ID), citations, authors,
data_types (list of {type, count}), sub_date, link

Usage:
  python scripts/search_bmrb_designed.py [--output results_bmrb.json]
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

BMRB_API = "https://api.bmrb.io/v2"

# Same keyword strategy as our PDB design search
DESIGN_KEYWORDS = [
    "de novo designed protein",
    "de novo protein design",
    "computationally designed protein",
    "computational protein design",
    "designed protein",
    "Rosetta design",
    "ProteinMPNN",
    "RFdiffusion",
    "designed helical bundle",
    "designed repeat protein",
    "designed miniprotein",
    "hallucinated protein",
    "protein design",
    "de novo helix bundle",
    "de novo four-helix",
    "de novo three-helix",
    "binary patterned protein",
    "Top7",
    "alpha3D",
]

# BMRB data_types values indicating relaxation experiments
RELAXATION_TYPES = {
    "heteronucl_T1_relaxation",
    "heteronucl_T2_relaxation",
    "heteronucl_NOEs",
    "order_parameters",
    "heteronucl_T1rho_relaxation",
    "auto_relaxation",
}

# Negative keywords — titles about natural proteins with "designed" ligands
NEGATIVE_KEYWORDS = [
    "inhibitor",
    "in complex with",
    "drug design",
    "ligand design",
    "structure-based drug",
    "rational design of inhibitor",
    "designed peptide inhibitor",
    "drug candidate",
    "small molecule",
]

# Positive keywords that override negative hits
POSITIVE_OVERRIDE = [
    "de novo",
    "rosetta",
    "proteinmpnn",
    "helical bundle",
    "repeat protein design",
    "miniprotein",
    "top7",
    "hallucinated",
    "rfdiffusion",
    "binary pattern",
]


def search_bmrb_instant(query: str) -> list:
    """Search BMRB using the instant search endpoint.

    Returns list of dicts, each with:
      value, citations, authors, data_types, sub_date, link
    """
    url = f"{BMRB_API}/instant"
    params = {"term": query}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        return []
    except Exception as e:
        print(f"  Warning: search failed for '{query}': {e}")
        return []


def has_relaxation_data(entry: dict) -> tuple:
    """Check if a BMRB entry has relaxation data from data_types field.

    Returns (bool, list_of_relaxation_types).
    """
    data_types = entry.get("data_types", [])
    found = []
    for dt in data_types:
        if isinstance(dt, dict):
            dtype = dt.get("type", "")
        else:
            dtype = str(dt)
        if dtype in RELAXATION_TYPES:
            found.append(dtype)
    return (len(found) > 0, found)


def is_likely_designed(title: str) -> bool:
    """Heuristic: does this title suggest a designed protein?"""
    title_lower = title.lower()

    # Check negative keywords
    has_negative = any(neg in title_lower for neg in NEGATIVE_KEYWORDS)
    has_positive_override = any(pos in title_lower for pos in POSITIVE_OVERRIDE)

    if has_negative and not has_positive_override:
        return False

    # Check for design keywords
    design_signals = [
        "de novo design", "de novo protein", "designed protein",
        "computationally designed", "computational design",
        "protein design", "rosetta", "proteinmpnn",
        "designed helix", "designed helical", "designed bundle",
        "designed repeat", "designed mini", "top7",
        "hallucinated", "rfdiffusion",
        "binary pattern", "designed fold", "artificial protein",
        "alpha3d", "de novo four-helix", "de novo three-helix",
        "de novo helix", "designed coiled-coil", "designed coiled coil",
        "consensus designed", "consensus design",
        "rationally designed", "designed superfamily",
        "designed peptide", "designed variant", "designed sequence",
        "de novo four_helix", "de novo three_helix",
        "engineered protein", "designed scaffold",
    ]
    return any(sig in title_lower for sig in design_signals)


def main():
    parser = argparse.ArgumentParser(
        description="Search BMRB for designed proteins with relaxation data")
    parser.add_argument("--output", default="bmrb_designed_relaxation.json",
                        help="Output JSON file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("BMRB Search: Designed Proteins with NMR Relaxation Data")
    print("=" * 60)

    # Step 1: Collect all candidate entries from keyword searches
    print("\n--- Step 1: Keyword search ---")
    all_entries = {}  # bmrb_id -> entry dict

    for kw in DESIGN_KEYWORDS:
        results = search_bmrb_instant(kw)
        new_count = 0
        for entry in results:
            bmrb_id = entry.get("value", "")
            if bmrb_id and bmrb_id not in all_entries:
                all_entries[bmrb_id] = entry
                new_count += 1
        if results:
            print(f"  '{kw}': {len(results)} hits ({new_count} new)")
        time.sleep(0.5)  # rate limiting

    print(f"\n  Total unique entries: {len(all_entries)}")

    # Step 2: Classify each entry
    print("\n--- Step 2: Classifying entries ---")
    designed_entries = []
    relaxation_entries = []
    both_entries = []

    for bmrb_id, entry in sorted(all_entries.items()):
        # Get title from citations field
        citations = entry.get("citations", [])
        title = citations[0] if citations else ""
        authors = entry.get("authors", [])

        has_relax, relax_types = has_relaxation_data(entry)
        is_designed = is_likely_designed(title)

        record = {
            "bmrb_id": bmrb_id,
            "title": title,
            "authors": authors[:3],  # first 3 authors
            "sub_date": entry.get("sub_date", ""),
            "is_designed": is_designed,
            "has_relaxation": has_relax,
            "relaxation_types": relax_types,
            "all_data_types": [dt.get("type", "") if isinstance(dt, dict)
                               else str(dt) for dt in entry.get("data_types", [])],
        }

        if is_designed:
            designed_entries.append(record)
        if has_relax:
            relaxation_entries.append(record)
        if is_designed and has_relax:
            both_entries.append(record)
            print(f"  *** MATCH: BMRB {bmrb_id} ***")
            print(f"      Title: {title[:100]}")
            print(f"      Authors: {', '.join(authors[:3])}")
            print(f"      Relaxation: {relax_types}")

    # Step 3: Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nCandidate entries searched:          {len(all_entries)}")
    print(f"Entries with designed protein title:  {len(designed_entries)}")
    print(f"Entries with ANY relaxation data:     {len(relaxation_entries)}")
    print(f"INTERSECTION (designed + relaxation): {len(both_entries)}")

    if designed_entries:
        print(f"\n--- All designed protein entries ({len(designed_entries)}) ---")
        for e in designed_entries:
            relax_flag = " [HAS RELAXATION]" if e["has_relaxation"] else ""
            print(f"  BMRB {e['bmrb_id']}: {e['title'][:90]}{relax_flag}")

    if both_entries:
        print(f"\n--- DESIGNED + RELAXATION ({len(both_entries)}) ---")
        print("These entries have per-residue NMR relaxation data for designed proteins:")
        for e in both_entries:
            print(f"\n  BMRB {e['bmrb_id']}")
            print(f"    Title: {e['title']}")
            print(f"    Authors: {', '.join(e['authors'])}")
            print(f"    Date: {e['sub_date']}")
            print(f"    Relaxation: {e['relaxation_types']}")
            print(f"    All data: {e['all_data_types']}")

    # Save results
    output = {
        "n_candidates": len(all_entries),
        "n_designed": len(designed_entries),
        "n_with_relaxation": len(relaxation_entries),
        "n_designed_with_relaxation": len(both_entries),
        "designed_entries": designed_entries,
        "designed_with_relaxation": both_entries,
    }

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
