#!/usr/bin/env python
"""
compute_frustration.py — per-residue local energetic frustration via FrustraMPNN.

FrustraMPNN (Beining et al., bioRxiv 2026.01.22.701012) is a ThermoMPNN-style
message-passing network that predicts the *single-residue local energetic
frustration* index for every amino-acid identity at every position of a protein,
orders of magnitude faster than the physics-based FrustratometeR.

Why we want it (robustness-dynamics paper):
  - Frustration is the energy-landscape-theory quantity that mechanistically
    predicts where proteins move (highly frustrated = competing interactions =
    dynamic; minimally frustrated = optimized = rigid). It is a more principled
    "why is this residue flexible" probe than sd(ΔΔG).
  - IMPORTANT: it does NOT break the packing-circularity critique (still
    structure-conditioned). Its value is (a) a mechanistic reframe and (b) a
    complementary predictor whose behaviour under the contact-number control is
    reported honestly. See ref_reports/FrustraMPNN_summary.md in the paper repo.

This script deliberately REUSES the existing robustness-output contract so the
whole downstream pipeline (correlate_robustness_dynamics.py, collect_results.py,
generate_latex_tables.py, multi_ddg_regression.py) consumes it unchanged. It
writes TWO scorer views under {output_dir}/{scorer}/ :

  frustrampnn         -> the per-residue robustness index (std_ddg column) holds
                         std(frustration profile over the 19 substitutions);
                         ddg_matrix.npy holds the L×19 mutant-frustration profile.
                         This is the direct analogue of sd(ΔΔG) for the head-to-head.
  frustrampnn_native  -> the per-residue index (std_ddg column) holds F_i(native),
                         the canonical single-residue frustration of the wild-type
                         residue. This is the theoretically-primary predictor.
                         (ddg_matrix.npy is copied from the profile so the 20-D
                          Ridge, if run for this view, has a matrix to read.)

Because both views expose their quantity in the `std_ddg` column that
correlate_robustness_dynamics.py reads as "the robustness index", you get
ρ(frustration, dynamics), the partials vs pLDDT / SASA / packing, and the
incremental ΔR² for FREE — no change to the 108 KB correlation workhorse.

Sign convention (FrustratometeR / FrustraMPNN):
  frustration > +0.55  -> minimally frustrated (optimized)  -> expect RIGID (low RMSF)
  frustration < -1.0   -> highly frustrated (conflict)      -> expect FLEXIBLE (high RMSF)
  => expect ρ(F_native, RMSF) < 0, same sign as ρ(sd(ΔΔG), RMSF).

Usage (mirrors compute_robustness.py):
  # single PDB (smoke test — DO THIS FIRST on the cluster, see run plan):
  python compute_frustration.py --pdb_file 1ubq.pdb --protein_id 1ubq \
      --output_dir out/ --frustrampnn_checkpoint /path/to/checkpoint.ckpt

  # batch over a dataset (proteins/{id}/*.pdb with .done markers, like ATLAS):
  python compute_frustration.py --atlas_dir $ATLAS_DIR --output_dir $FRUST_DIR \
      --batch --batch_start 0 --batch_end 50 --device cuda \
      --frustrampnn_checkpoint $FRUSTRAMPNN_CHECKPOINT

Install (NOT in the ThermoMPNN env — use a dedicated env, see run plan):
  pip install "frustrampnn[all]"        # pulls torch + torch-geometric
  # checkpoint from the FrustraMPNN Zenodo (10.5281/zenodo.17978321)
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- Reuse the existing robustness-output machinery (single source of truth) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from compute_robustness import (          # noqa: E402
    AA_LIST, AA_TO_IDX, N_AA,
    compute_robustness_metrics,
    save_results,
    extract_sequence_from_pdb,
    find_atlas_proteins,
    _json_default,
)

SCORER_MAIN = "frustrampnn"          # std(frustration profile) — sd(ΔΔG) analogue
SCORER_NATIVE = "frustrampnn_native"  # F_i(native) — canonical single-residue frustration


# ==========================================================================
# FrustraMPNN wrapper
# ==========================================================================

class FrustraMPNNScorer:
    """Thin wrapper around the `frustrampnn` package.

    Loads the model once, then for each PDB returns:
      - mut_profile: (L, 19) frustration index for the 19 non-native identities,
        in AA_LIST order with the wild-type column removed at each row
        (same layout as compute_robustness.py's ddg_matrix).
      - native:      (L,)   F_i(native), the frustration of the wild-type residue.
    """

    def __init__(self, checkpoint: Optional[str] = None, chain_id: str = "A"):
        self._model = None
        self._device = "cuda"
        self._chain_id = chain_id
        self._checkpoint = checkpoint or os.environ.get("FRUSTRAMPNN_CHECKPOINT")

    @property
    def name(self) -> str:
        return SCORER_MAIN

    def load_model(self, device: str = "cuda"):
        self._device = device
        if self._checkpoint is None:
            raise ValueError(
                "FrustraMPNN checkpoint not set. Either:\n"
                "  1. Set FRUSTRAMPNN_CHECKPOINT env var, or\n"
                "  2. Pass --frustrampnn_checkpoint /path/to/checkpoint.ckpt\n"
                "Download from the FrustraMPNN Zenodo (10.5281/zenodo.17978321)."
            )
        try:
            from frustrampnn import FrustraMPNN
        except ImportError as e:
            raise ImportError(
                f"Failed to import frustrampnn: {e}\n"
                "Install into a dedicated env (NOT the ThermoMPNN env):\n"
                '  pip install "frustrampnn[all]"'
            )
        # from_pretrained loads architecture + weights from the checkpoint.
        self._model = FrustraMPNN.from_pretrained(self._checkpoint)
        try:
            self._model = self._model.to(device).eval()
        except Exception:
            # Some builds handle device internally in .predict(); ignore if so.
            pass
        print(f"  FrustraMPNN loaded from {self._checkpoint}")

    def predict_profile(self, pdb_path: str, seq: str,
                        chain_id: Optional[str] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """Run FrustraMPNN on one PDB and return (mut_profile Lx19, native L)."""
        chain = chain_id or self._chain_id
        L = len(seq)

        # frustrampnn returns a long-format DataFrame with columns:
        #   frustration_pred, position (0-indexed), wildtype, mutation, pdb, chain
        df = self._model.predict(pdb_path, chains=[chain], show_progress=False)

        # Pivot long -> dense (L, 20) in AA_LIST order.
        full = np.full((L, N_AA), np.nan, dtype=np.float32)
        wt_from_df: Dict[int, str] = {}
        n_bad_pos = 0
        for pos, wt, mut, val in zip(
            df["position"].to_numpy(),
            df["wildtype"].to_numpy(),
            df["mutation"].to_numpy(),
            df["frustration_pred"].to_numpy(),
        ):
            p = int(pos)
            if not (0 <= p < L):
                n_bad_pos += 1
                continue
            wt_from_df[p] = str(wt)
            j = AA_TO_IDX.get(str(mut))
            if j is not None:
                full[p, j] = np.float32(val)

        # Sanity: does df's wildtype agree with our parsed sequence?
        # (Logged, not fatal — the smoke test uses this to catch index/frame drift.)
        if wt_from_df:
            mism = sum(1 for p, wt in wt_from_df.items()
                       if p < L and wt != seq[p])
            if mism > 0.05 * L:
                print(f"    WARN: {mism}/{L} wildtype mismatches vs parsed seq "
                      f"(possible position-indexing drift); n_bad_pos={n_bad_pos}")

        # native = the wild-type identity's column at each position.
        native = np.full(L, np.nan, dtype=np.float32)
        for p in range(L):
            wt_idx = AA_TO_IDX.get(seq[p])
            if wt_idx is not None:
                native[p] = full[p, wt_idx]

        # mut_profile (L, 19): drop the wild-type column at each row.
        mut_profile = np.full((L, N_AA - 1), np.nan, dtype=np.float32)
        for p in range(L):
            wt_idx = AA_TO_IDX.get(seq[p])
            if wt_idx is None:
                continue
            row = full[p]
            mut_profile[p] = np.concatenate([row[:wt_idx], row[wt_idx + 1:]])

        return mut_profile, native


# ==========================================================================
# native-view writer (F_native in the std_ddg slot so correlate picks it up)
# ==========================================================================

def save_native_view(protein_id: str, seq: str, native: np.ndarray,
                     mut_profile: np.ndarray, output_dir: str):
    """Write the frustrampnn_native scorer dir.

    The per-residue robustness index that correlate_robustness_dynamics.py
    reads is the `std_ddg` column, so we place F_i(native) there. All other
    numeric columns are filled with the native value too (harmless — only
    std_ddg is consumed as the index; the ALT-robustness table would show
    F_native under every alias, which we do not surface for this view).
    """
    out_dir = Path(output_dir) / SCORER_NATIVE
    out_dir.mkdir(parents=True, exist_ok=True)
    L = len(seq)

    # ddg_matrix: reuse the mutant-frustration profile so a 20-D Ridge for this
    # view (if ever run) has a matrix; the native index itself lives in the TSV.
    np.save(str(out_dir / f"{protein_id}_ddg_matrix.npy"), mut_profile)

    per_residue = []
    for i in range(L):
        v = float(native[i]) if np.isfinite(native[i]) else np.nan
        per_residue.append({
            "position": i + 1, "wt_aa": seq[i],
            "mean_abs_ddg": v, "mean_ddg": v, "std_ddg": v,
            "max_ddg": v, "min_ddg": v,
            "frac_destabilizing": np.nan, "frac_neutral": np.nan,
            "n_valid": 1 if np.isfinite(native[i]) else 0,
        })

    valid = native[np.isfinite(native)]
    result = {
        "protein_id": protein_id, "scorer": SCORER_NATIVE,
        "sequence": seq, "sequence_length": L,
        "global_metrics": {
            "global_native_frust_mean": float(np.mean(valid)) if valid.size else np.nan,
            "global_native_frust_std": float(np.std(valid)) if valid.size else np.nan,
            "frac_highly_frustrated": float(np.mean(valid < -1.0)) if valid.size else np.nan,
            "frac_minimally_frustrated": float(np.mean(valid > 0.55)) if valid.size else np.nan,
            "sequence_length": L,
        },
        "per_residue_metrics": per_residue,
    }
    with open(out_dir / f"{protein_id}_robustness.json", "w") as f:
        json.dump(result, f, indent=2, default=_json_default)

    tsv_path = out_dir / f"{protein_id}_robustness.tsv"
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "position", "wt_aa", "mean_abs_ddg", "mean_ddg", "std_ddg",
            "max_ddg", "min_ddg", "frac_destabilizing", "frac_neutral", "n_valid"
        ], delimiter="\t")
        writer.writeheader()
        for row in per_residue:
            writer.writerow(row)


# ==========================================================================
# main
# ==========================================================================

def _gather_proteins(args) -> List[Dict[str, str]]:
    if args.pdb_file:
        pid = args.protein_id or Path(args.pdb_file).stem
        return [{"protein_id": pid, "pdb_path": args.pdb_file}]
    if args.atlas_dir:
        proteins = find_atlas_proteins(args.atlas_dir)
        if args.batch:
            end = args.batch_end if args.batch_end >= 0 else len(proteins)
            proteins = proteins[args.batch_start:end]
        return proteins
    raise ValueError("Provide --pdb_file or --atlas_dir")


def main():
    ap = argparse.ArgumentParser(description="FrustraMPNN per-residue frustration")
    ap.add_argument("--pdb_file", type=str, default=None)
    ap.add_argument("--protein_id", type=str, default=None)
    ap.add_argument("--chain_id", type=str, default="A")
    ap.add_argument("--atlas_dir", type=str, default=None,
                    help="Dataset dir with proteins/{id}/*.pdb (+ .done markers)")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--batch_start", type=int, default=0)
    ap.add_argument("--batch_end", type=int, default=-1)
    ap.add_argument("--output_dir", type=str, required=True,
                    help="Robustness dir; scorer subdirs frustrampnn/ + "
                         "frustrampnn_native/ are created under it")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--frustrampnn_checkpoint", type=str, default=None,
                    help="Path to FrustraMPNN checkpoint.ckpt "
                         "(or set FRUSTRAMPNN_CHECKPOINT)")
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    proteins = _gather_proteins(args)
    print(f"FrustraMPNN: {len(proteins)} protein(s) to process")

    scorer = FrustraMPNNScorer(checkpoint=args.frustrampnn_checkpoint,
                               chain_id=args.chain_id)
    scorer.load_model(device=args.device)

    n_done, n_skip, n_fail = 0, 0, 0
    for rec in proteins:
        pid = rec["protein_id"]
        pdb_path = rec["pdb_path"]
        native_json = Path(args.output_dir) / SCORER_NATIVE / f"{pid}_robustness.json"
        main_json = Path(args.output_dir) / SCORER_MAIN / f"{pid}_robustness.json"
        if args.skip_existing and native_json.exists() and main_json.exists():
            n_skip += 1
            continue
        try:
            seq = extract_sequence_from_pdb(pdb_path, chain_id=args.chain_id)
            if not seq:
                print(f"  SKIP {pid}: empty sequence")
                n_fail += 1
                continue
            mut_profile, native = scorer.predict_profile(pdb_path, seq, args.chain_id)

            # View 1: std(frustration profile) via the reused machinery.
            metrics = compute_robustness_metrics(mut_profile, seq)
            save_results(pid, seq, mut_profile, metrics, SCORER_MAIN, args.output_dir)
            # View 2: native single-residue frustration.
            save_native_view(pid, seq, native, mut_profile, args.output_dir)

            n_done += 1
            if n_done % 25 == 0:
                print(f"  ...{n_done} done")
        except Exception as e:
            print(f"  FAIL {pid}: {e}")
            n_fail += 1

    print(f"Done. processed={n_done} skipped={n_skip} failed={n_fail}")
    print(f"Outputs: {args.output_dir}/{SCORER_MAIN}/  and  "
          f"{args.output_dir}/{SCORER_NATIVE}/")


if __name__ == "__main__":
    main()
