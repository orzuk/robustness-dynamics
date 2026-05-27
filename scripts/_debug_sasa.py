#!/usr/bin/env python3
"""Temporary diagnostic: figure out why compute_sasa_from_pdb returns None
on the MegaScale 1ubq_A test case while inline reproduction works.

Run:
    python scripts/_debug_sasa.py [pdb_path]

If no path is given, defaults to $PROJECT_DIR/data/megascale_processed/proteins/1ubq_A/1ubq_A.pdb.
"""

import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    import correlate_robustness_dynamics as crd
    import mdtraj
    from collections import OrderedDict
    import pandas as pd

    if len(sys.argv) > 1:
        pdb = sys.argv[1]
    else:
        project = os.environ.get("PROJECT_DIR")
        if not project:
            sys.exit("PROJECT_DIR not set; pass an explicit PDB path as argv")
        pdb = f"{project}/data/megascale_processed/proteins/1ubq_A/1ubq_A.pdb"

    print(f"PDB: {pdb}")
    print(f"_PROTEIN_AA_3 size: {len(crd._PROTEIN_AA_3)}")
    print(f"_PROTEIN_AA_3 sample: {list(crd._PROTEIN_AA_3)[:5]}")

    # -------- INLINE: reproduce the function body, exception-visible --------
    print("\n=== INLINE reproduction ===")
    clean_path = crd._clean_pdb_for_mdtraj(pdb)
    print(f"  clean_path: {clean_path}")
    print(f"  clean exists: {os.path.exists(clean_path)}")
    try:
        traj = mdtraj.load(clean_path)
        print(f"  loaded: {traj.topology.n_atoms} atoms, "
              f"{traj.topology.n_residues} residues")
        sasa_per_atom = mdtraj.shrake_rupley(traj, mode='atom')
        print(f"  shrake_rupley shape: {sasa_per_atom.shape}")

        sasa_by_resseq = OrderedDict()
        for atom in traj.topology.atoms:
            if atom.residue.name not in crd._PROTEIN_AA_3:
                continue
            rs = atom.residue.resSeq
            sasa_by_resseq[rs] = sasa_by_resseq.get(rs, 0.0) + sasa_per_atom[0, atom.index]
        print(f"  sasa_by_resseq entries: {len(sasa_by_resseq)}")
        if sasa_by_resseq:
            df = pd.DataFrame({"position": list(sasa_by_resseq.keys()),
                               "sasa": list(sasa_by_resseq.values())})
            print(f"  built df: {len(df)} rows")
            print(df.head().to_string(index=False))
        else:
            print("  EMPTY — inline would return None")
    except Exception as e:
        print(f"  INLINE EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        try:
            os.unlink(clean_path)
        except FileNotFoundError:
            pass

    # -------- ACTUAL: invoke crd.compute_sasa_from_pdb directly --------
    # Monkey-patch the bare except to surface the real error.
    print("\n=== ACTUAL compute_sasa_from_pdb call (exception-visible) ===")
    import functools, types

    def patched_compute(pdb_path):
        clean_path = crd._clean_pdb_for_mdtraj(pdb_path)
        try:
            traj = mdtraj.load(clean_path)
            sasa_per_atom = mdtraj.shrake_rupley(traj, mode='atom')
            sasa_by_resseq = OrderedDict()
            for atom in traj.topology.atoms:
                if atom.residue.name not in crd._PROTEIN_AA_3:
                    continue
                rs = atom.residue.resSeq
                sasa_by_resseq[rs] = sasa_by_resseq.get(rs, 0.0) + sasa_per_atom[0, atom.index]
            if not sasa_by_resseq:
                print("  patched_compute: empty sasa_by_resseq -> returning None")
                return None
            return pd.DataFrame({"position": list(sasa_by_resseq.keys()),
                                 "sasa":     list(sasa_by_resseq.values())})
        except Exception as e:
            print(f"  patched_compute EXCEPTION: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None
        finally:
            try:
                os.unlink(clean_path)
            except FileNotFoundError:
                pass

    r1 = patched_compute(pdb)
    print(f"  patched_compute result is None? {r1 is None}")
    if r1 is not None:
        print(r1.head().to_string(index=False))

    print("\n=== ACTUAL crd.compute_sasa_from_pdb call ===")
    r2 = crd.compute_sasa_from_pdb(pdb)
    print(f"  crd.compute_sasa_from_pdb result is None? {r2 is None}")
    if r2 is not None:
        print(r2.head().to_string(index=False))


if __name__ == "__main__":
    main()
