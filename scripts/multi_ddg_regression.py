#!/usr/bin/env python3
"""
Multi-dimensional DDG regression: predict dynamics (RMSF or B-factor)
using the full per-residue DDG profile (20 target-AA features) instead
of a single robustness summary statistic.

Each residue position i has a 20-dimensional feature vector:
  x_i = [DDG(S_i -> A), DDG(S_i -> C), ..., DDG(S_i -> Y)]
where the self-mutation entry DDG(S_i -> S_i) = 0.

This gives each column a consistent meaning across all positions
("cost of mutating TO alanine") regardless of the wild-type amino acid.

We compare:
  1. Ridge regression with full 20-DDG features
  2. Single-feature baselines (mean|DDG|, pLDDT, SASA)
  3. Combined: 20-DDG + pLDDT
All evaluated via protein-level k-fold cross-validation.

Usage:
  python multi_ddg_regression.py \
      --atlas_dir /path/to/atlas \
      --robustness_dir /path/to/robustness \
      --scorer thermompnn \
      --output_dir /path/to/output \
      --target rmsf          # or bfactor
      --n_folds 5
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Canonical amino acid ordering (matches compute_robustness.py)
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
N_AA = len(AA_LIST)


# ======================================================================
# DATA LOADING
# ======================================================================

def load_ddg_matrix_20col(robustness_dir: str, scorer: str,
                          protein_id: str) -> Optional[Tuple[np.ndarray, str]]:
    """Load L x 19 DDG matrix and expand to L x 20 (with 0 for self-mutation).

    Returns (ddg_20, sequence) where ddg_20 is shape (L, 20).
    """
    npy_path = Path(robustness_dir) / scorer / f"{protein_id}_ddg_matrix.npy"
    json_path = Path(robustness_dir) / scorer / f"{protein_id}_robustness.json"

    if not npy_path.exists() or not json_path.exists():
        return None

    ddg_19 = np.load(str(npy_path))  # (L, 19)
    with open(json_path) as f:
        meta = json.load(f)

    seq = meta.get("sequence", "")
    if not seq or len(seq) != ddg_19.shape[0]:
        return None

    L = len(seq)
    ddg_20 = np.zeros((L, N_AA), dtype=np.float32)

    for i in range(L):
        wt_aa = seq[i]
        wt_idx = AA_TO_IDX.get(wt_aa)
        if wt_idx is None:
            ddg_20[i, :] = np.nan
            continue
        # Reconstruct 20-column from 19-column by inserting 0 at wt position
        col = 0
        for j in range(N_AA):
            if j == wt_idx:
                ddg_20[i, j] = 0.0  # self-mutation
            else:
                ddg_20[i, j] = ddg_19[i, col]
                col += 1

    return ddg_20, seq


def load_atlas_column(protein_dir: str, suffix: str,
                      col_name: str,
                      return_positions: bool = False):
    """Load a single column from an ATLAS TSV file.

    If return_positions=True, returns (values, positions) tuple.
    positions is None if no position column exists in the TSV.
    """
    protein_dir = Path(protein_dir)
    matches = list(protein_dir.glob(f"*{suffix}"))
    if not matches:
        return (None, None) if return_positions else None
    df = pd.read_csv(matches[0], sep="\t")
    # Find the right column
    candidates = [c for c in df.columns if col_name.lower() in c.lower()]
    if not candidates:
        numeric = [c for c in df.columns
                   if df[c].dtype in (np.float64, np.float32, float)]
        candidates = numeric[-1:] if numeric else []
    if not candidates:
        return (None, None) if return_positions else None
    vals = df[candidates[0]].values
    if return_positions:
        positions = df["position"].values if "position" in df.columns else None
        return vals, positions
    return vals


def load_rmsf(protein_dir: str) -> Optional[np.ndarray]:
    """Load RMSF, averaged across replicates."""
    protein_dir = Path(protein_dir)
    matches = list(protein_dir.glob("*_RMSF.tsv"))
    if not matches:
        return None
    df = pd.read_csv(matches[0], sep="\t")
    rmsf_cols = [c for c in df.columns if "rmsf" in c.lower()
                 or "r1" in c.lower() or "r2" in c.lower()
                 or "r3" in c.lower()]
    if not rmsf_cols:
        rmsf_cols = [c for c in df.columns
                     if df[c].dtype in (np.float64, np.float32, float)]
    if not rmsf_cols:
        return None
    return df[rmsf_cols].mean(axis=1).values


def compute_sasa(pdb_path: str, chain_id: Optional[str] = None) -> Optional[np.ndarray]:
    """Compute per-residue SASA using mdtraj.

    If chain_id is given, only return SASA for residues in that chain.
    """
    try:
        import mdtraj
        traj = mdtraj.load(pdb_path)
        sasa_per_atom = mdtraj.shrake_rupley(traj, mode='atom')
        sasa_per_residue = np.zeros(traj.topology.n_residues)
        for atom in traj.topology.atoms:
            sasa_per_residue[atom.residue.index] += sasa_per_atom[0, atom.index]

        if chain_id is not None:
            # Filter to residues in the specified chain
            chain_residue_indices = [
                r.index for r in traj.topology.residues
                if r.chain.index == _get_chain_index(traj.topology, chain_id)
            ]
            if chain_residue_indices:
                return sasa_per_residue[chain_residue_indices]
            return None

        return sasa_per_residue
    except Exception:
        return None


def _get_chain_index(topology, chain_id: str) -> int:
    """Get mdtraj chain index from PDB chain letter."""
    for chain in topology.chains:
        # mdtraj stores chain IDs as integers but we can match by index
        # Chain letters map to indices: A=0, B=1, etc.
        if chr(ord('A') + chain.index) == chain_id.upper():
            return chain.index
    return 0  # fallback to first chain


# ======================================================================
# REGRESSION
# ======================================================================

@dataclass
class RegressionResult:
    """Results from cross-validated regression."""
    model_name: str
    n_features: int
    n_proteins_train: int
    n_proteins_test: int
    n_residues_train: int
    n_residues_test: int

    # Cross-validated metrics (mean +/- std across folds)
    cv_r2_mean: float = np.nan
    cv_r2_std: float = np.nan
    cv_rho_mean: float = np.nan
    cv_rho_std: float = np.nan

    # Per-protein CV metrics
    cv_per_protein_rho_median: float = np.nan
    cv_per_protein_rho_mean: float = np.nan

    # Feature importance (for multi-DDG model)
    feature_names: List[str] = None
    feature_coefs_mean: List[float] = None
    feature_coefs_std: List[float] = None       # empirical std across CV folds
    feature_coefs_se: List[float] = None        # theoretical SE from Ridge covariance
    feature_coefs_per_fold: List[List[float]] = None  # [fold_idx][feature_idx]


def build_dataset(
    atlas_dir: str,
    robustness_dir: str,
    scorer: str,
    target: str,
    max_seq_length: int = 0,
    max_proteins: int = 0,
    exclude_proteins: Optional[set] = None,
) -> Tuple[List[dict], List[str]]:
    """Build list of per-protein data dicts for regression.

    Each dict has: ddg_20 (L,20), target (L,), plddt (L,), sasa (L,),
                   mean_abs_ddg (L,), protein_id, seq_length.
    """
    atlas_proteins_dir = Path(atlas_dir) / "proteins"
    protein_ids = sorted([
        d.name for d in atlas_proteins_dir.iterdir()
        if d.is_dir() and (d / ".done").exists()
    ])

    if exclude_proteins:
        n_before = len(protein_ids)
        protein_ids = [p for p in protein_ids if p not in exclude_proteins]
        n_excluded = n_before - len(protein_ids)
        if n_excluded > 0:
            print(f"Excluded {n_excluded} proteins")

    if max_proteins > 0:
        protein_ids = protein_ids[:max_proteins]

    dataset = []
    skipped = {"no_ddg": 0, "no_target": 0, "too_long": 0, "too_short": 0,
               "length_mismatch": 0}

    for pid in protein_ids:
        protein_dir = str(atlas_proteins_dir / pid)

        # Load DDG matrix
        result = load_ddg_matrix_20col(robustness_dir, scorer, pid)
        if result is None:
            skipped["no_ddg"] += 1
            continue

        ddg_20, seq = result
        L = len(seq)

        if max_seq_length > 0 and L >= max_seq_length:
            skipped["too_long"] += 1
            continue

        # Load target (with positions for NMR datasets that have NaN-dropped rows)
        target_positions = None
        if target == "rmsf":
            y = load_rmsf(protein_dir)
        elif target == "bfactor":
            y, target_positions = load_atlas_column(
                protein_dir, "_Bfactor.tsv", "bfactor", return_positions=True)
        else:
            raise ValueError(f"Unknown target: {target}")

        if y is None:
            skipped["no_target"] += 1
            continue

        # Position-based alignment: if target has positions, use them to
        # select matching rows from the full-length DDG matrix (1..L)
        if target_positions is not None and len(y) != L:
            # Convert 1-based positions to 0-based indices into the DDG matrix
            idx = target_positions.astype(int) - 1
            # Keep only positions within range
            valid_idx = (idx >= 0) & (idx < L)
            if valid_idx.sum() < 10:
                skipped["length_mismatch"] += 1
                continue
            idx = idx[valid_idx]
            y = y[valid_idx]
            ddg_20 = ddg_20[idx]
            seq = "".join(seq[i] for i in idx) if isinstance(seq, str) else seq[idx]
            L = len(y)
        elif len(y) != L:
            skipped["length_mismatch"] += 1
            continue

        # Load baselines — use same position subset if we did alignment
        plddt_raw = load_atlas_column(protein_dir, "_pLDDT.tsv", "plddt",
                                      return_positions=True)
        if plddt_raw[0] is not None:
            plddt_vals, plddt_pos = plddt_raw
            if target_positions is not None and plddt_pos is not None:
                # Intersect with target positions
                common = np.isin(plddt_pos, target_positions)
                plddt = plddt_vals[common] if common.sum() == L else None
            elif len(plddt_vals) == L:
                plddt = plddt_vals
            else:
                plddt = None
        else:
            plddt = None

        pdb_files = list(Path(protein_dir).glob("*.pdb"))
        # Extract chain ID from protein ID (e.g., "1NA0_A" -> "A")
        chain_id = pid.split("_")[1] if "_" in pid else None
        sasa = compute_sasa(str(pdb_files[0]), chain_id=chain_id) if pdb_files else None
        if sasa is not None and len(sasa) != L:
            sasa = None

        # mean_abs_ddg, mean_ddg, and std_ddg from robustness TSV
        rob_tsv = Path(robustness_dir) / scorer / f"{pid}_robustness.tsv"
        mean_abs_ddg = None
        mean_ddg = None
        std_ddg = None
        if rob_tsv.exists():
            rob_df = pd.read_csv(rob_tsv, sep="\t")
            n_rob = len(rob_df)
            if target_positions is not None and n_rob != L:
                # Align robustness TSV to target positions
                rob_pos = rob_df["position"].values if "position" in rob_df.columns else np.arange(1, n_rob + 1)
                rob_idx = np.isin(rob_pos, target_positions)
                if rob_idx.sum() == L:
                    if "mean_abs_ddg" in rob_df.columns:
                        mean_abs_ddg = rob_df["mean_abs_ddg"].values[rob_idx]
                    if "mean_ddg" in rob_df.columns:
                        mean_ddg = rob_df["mean_ddg"].values[rob_idx]
                    if "std_ddg" in rob_df.columns:
                        std_ddg = rob_df["std_ddg"].values[rob_idx]
            else:
                if "mean_abs_ddg" in rob_df.columns and n_rob == L:
                    mean_abs_ddg = rob_df["mean_abs_ddg"].values
                if "mean_ddg" in rob_df.columns and n_rob == L:
                    mean_ddg = rob_df["mean_ddg"].values
                if "std_ddg" in rob_df.columns and n_rob == L:
                    std_ddg = rob_df["std_ddg"].values

        # Filter valid rows (no NaN in DDG or target)
        valid = ~(np.isnan(ddg_20).any(axis=1) | np.isnan(y))
        n_valid = valid.sum()
        if n_valid < 10:
            skipped["too_short"] += 1
            continue

        # Compute nonlinear summary features from RAW DDG values
        # (before z-scoring, so std/mean/max/min are meaningful).
        ddg_raw = ddg_20[valid]
        ddg_masked = ddg_raw.copy()
        ddg_masked[ddg_masked == 0] = np.nan
        nl_std = np.nanstd(ddg_masked, axis=1)
        nl_mean_abs = np.nanmean(np.abs(ddg_masked), axis=1)
        nl_max_abs = np.nanmax(np.abs(ddg_masked), axis=1)
        nl_min = np.nanmin(ddg_masked, axis=1)
        nonlinear_4_raw = np.column_stack([nl_std, nl_mean_abs, nl_max_abs, nl_min])

        # Within-protein z-scoring of ALL features and target.
        # This matches the correlation analysis and removes protein-level
        # offsets so the regression captures residue-level signal only.
        def _zscore(arr):
            if arr is None:
                return None
            mu, sd = np.nanmean(arr), np.nanstd(arr)
            if sd < 1e-10:
                return arr - mu  # constant feature, will be ~0
            return (arr - mu) / sd

        def _zscore_cols(mat):
            """Z-score each column of a 2D array within this protein."""
            out = np.empty_like(mat, dtype=float)
            for c in range(mat.shape[1]):
                out[:, c] = _zscore(mat[:, c])
            return out

        ddg_valid = _zscore_cols(ddg_raw)
        y_valid = _zscore(y[valid])
        nonlinear_4 = _zscore_cols(nonlinear_4_raw)

        entry = {
            "protein_id": pid,
            "seq_length": L,
            "ddg_20": ddg_valid,
            "nonlinear_4": nonlinear_4,
            "target": y_valid,
            "plddt": _zscore(plddt[valid]) if plddt is not None else None,
            "sasa": _zscore(sasa[valid]) if sasa is not None else None,
            "mean_abs_ddg": _zscore(mean_abs_ddg[valid]) if mean_abs_ddg is not None else None,
            "mean_ddg": _zscore(mean_ddg[valid]) if mean_ddg is not None else None,
            "std_ddg": _zscore(std_ddg[valid]) if std_ddg is not None else None,
            "n_residues": int(n_valid),
        }
        dataset.append(entry)

    print(f"Loaded {len(dataset)} proteins for {scorer}/{target}")
    print(f"Skipped: {skipped}")

    return dataset


def run_cv_regression(
    dataset: List[dict],
    n_folds: int = 5,
    alpha: float = 1.0,
    seed: int = 42,
) -> Dict[str, RegressionResult]:
    """Run protein-level k-fold CV for multiple models.

    Models compared:
      1. ridge_20ddg: Ridge on 20 DDG features
      2. ridge_20ddg_plddt: Ridge on 20 DDG + pLDDT
      3. ols_mean_abs_ddg: OLS on mean|DDG| (1 feature)
      4. ols_plddt: OLS on pLDDT (1 feature)
      5. ols_mean_abs_ddg_plddt: OLS on mean|DDG| + pLDDT (2 features)
      6. ols_sasa: OLS on SASA (1 feature)
    """
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.model_selection import KFold
    from scipy import stats as scipy_stats

    np.random.seed(seed)
    n = len(dataset)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Protein indices
    indices = np.arange(n)

    nonlinear_names = ["std_ddg", "mean|DDG|", "max|DDG|", "min_ddg"]

    # --- Model definitions: name -> (feature_extractor, regularized) ---
    def extract_20ddg(entry):
        return entry["ddg_20"]

    def extract_20ddg_nonlinear(entry):
        """20 per-AA DDG + 4 nonlinear summary features = 24 features.
        Uses pre-computed nonlinear features from the entry (already
        z-scored within-protein during loading).
        """
        nl = entry.get("nonlinear_4")
        if nl is None:
            return None
        return np.column_stack([entry["ddg_20"], nl])

    def extract_20ddg_nonlinear_plddt(entry):
        """24 DDG features + pLDDT = 25 features."""
        if entry["plddt"] is None:
            return None
        nl = entry.get("nonlinear_4")
        if nl is None:
            return None
        return np.column_stack([entry["ddg_20"], nl, entry["plddt"]])

    def extract_nonlinear_only(entry):
        """4 nonlinear DDG summary features only (no per-AA detail)."""
        nl = entry.get("nonlinear_4")
        if nl is None:
            return None
        return nl

    def extract_20ddg_plddt(entry):
        if entry["plddt"] is None:
            return None
        return np.column_stack([entry["ddg_20"], entry["plddt"]])

    def extract_std_ddg(entry):
        if entry["std_ddg"] is None:
            return None
        return entry["std_ddg"].reshape(-1, 1)

    def extract_mean_abs_ddg(entry):
        if entry["mean_abs_ddg"] is None:
            return None
        return entry["mean_abs_ddg"].reshape(-1, 1)

    def extract_mean_ddg(entry):
        if entry["mean_ddg"] is None:
            return None
        return entry["mean_ddg"].reshape(-1, 1)

    def extract_plddt(entry):
        if entry["plddt"] is None:
            return None
        return entry["plddt"].reshape(-1, 1)

    def extract_std_plddt(entry):
        if entry["std_ddg"] is None or entry["plddt"] is None:
            return None
        return np.column_stack([entry["std_ddg"], entry["plddt"]])

    def extract_mean_plddt(entry):
        if entry["mean_abs_ddg"] is None or entry["plddt"] is None:
            return None
        return np.column_stack([entry["mean_abs_ddg"], entry["plddt"]])

    def extract_sasa(entry):
        if entry["sasa"] is None:
            return None
        return entry["sasa"].reshape(-1, 1)

    models = {
        "ridge_20ddg": {
            "extractor": extract_20ddg,
            "use_ridge": True,
            "feature_names": list(AA_LIST),
            "n_features": 20,
        },
        "ridge_20ddg_nonlinear": {
            "extractor": extract_20ddg_nonlinear,
            "use_ridge": True,
            "feature_names": list(AA_LIST) + nonlinear_names,
            "n_features": 24,
        },
        "ridge_20ddg_nonlinear_plddt": {
            "extractor": extract_20ddg_nonlinear_plddt,
            "use_ridge": True,
            "feature_names": list(AA_LIST) + nonlinear_names + ["pLDDT"],
            "n_features": 25,
        },
        "ridge_nonlinear_only": {
            "extractor": extract_nonlinear_only,
            "use_ridge": True,
            "feature_names": nonlinear_names,
            "n_features": 4,
        },
        "ridge_20ddg_plddt": {
            "extractor": extract_20ddg_plddt,
            "use_ridge": True,
            "feature_names": list(AA_LIST) + ["pLDDT"],
            "n_features": 21,
        },
        "ols_std_ddg": {
            "extractor": extract_std_ddg,
            "use_ridge": False,
            "feature_names": ["std_ddg"],
            "n_features": 1,
        },
        "ols_mean_abs_ddg": {
            "extractor": extract_mean_abs_ddg,
            "use_ridge": False,
            "feature_names": ["mean|DDG|"],
            "n_features": 1,
        },
        "ols_mean_ddg": {
            "extractor": extract_mean_ddg,
            "use_ridge": False,
            "feature_names": ["mean_DDG"],
            "n_features": 1,
        },
        "ols_plddt": {
            "extractor": extract_plddt,
            "use_ridge": False,
            "feature_names": ["pLDDT"],
            "n_features": 1,
        },
        "ols_std_plddt": {
            "extractor": extract_std_plddt,
            "use_ridge": False,
            "feature_names": ["std_ddg", "pLDDT"],
            "n_features": 2,
        },
        "ols_mean_plddt": {
            "extractor": extract_mean_plddt,
            "use_ridge": False,
            "feature_names": ["mean|DDG|", "pLDDT"],
            "n_features": 2,
        },
        "ols_sasa": {
            "extractor": extract_sasa,
            "use_ridge": False,
            "feature_names": ["SASA"],
            "n_features": 1,
        },
    }

    results = {}

    for model_name, model_def in models.items():
        print(f"\n  Model: {model_name}")
        extractor = model_def["extractor"]
        use_ridge = model_def["use_ridge"]

        fold_r2s = []
        fold_rhos = []
        fold_per_protein_rhos = []
        fold_coefs = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(indices)):
            # Collect train/test residues
            X_train_parts, y_train_parts = [], []
            X_test_parts, y_test_parts = [], []
            test_protein_boundaries = []  # for per-protein rho

            for idx in train_idx:
                entry = dataset[idx]
                X = extractor(entry)
                if X is None:
                    continue
                X_train_parts.append(X)
                y_train_parts.append(entry["target"])

            offset = 0
            for idx in test_idx:
                entry = dataset[idx]
                X = extractor(entry)
                if X is None:
                    continue
                X_test_parts.append(X)
                y_test_parts.append(entry["target"])
                n_res = len(entry["target"])
                test_protein_boundaries.append((offset, offset + n_res,
                                                entry["protein_id"]))
                offset += n_res

            if not X_train_parts or not X_test_parts:
                continue

            X_train = np.vstack(X_train_parts)
            y_train = np.concatenate(y_train_parts)
            X_test = np.vstack(X_test_parts)
            y_test = np.concatenate(y_test_parts)

            # Data is already z-scored within each protein during loading.
            # Just replace any remaining NaN with 0.
            X_train_z = np.nan_to_num(X_train, nan=0.0)
            X_test_z = np.nan_to_num(X_test, nan=0.0)
            y_train_z = y_train
            y_test_z = y_test

            # Fit
            if use_ridge:
                reg = Ridge(alpha=alpha).fit(X_train_z, y_train_z)
            else:
                reg = LinearRegression().fit(X_train_z, y_train_z)

            # Evaluate
            y_pred = reg.predict(X_test_z)
            r2 = 1 - np.sum((y_test_z - y_pred)**2) / np.sum((y_test_z - np.mean(y_test_z))**2)
            rho, _ = scipy_stats.spearmanr(y_test_z, y_pred)

            fold_r2s.append(r2)
            fold_rhos.append(rho)
            fold_coefs.append(reg.coef_.copy())

            # Per-protein rho on test set
            pp_rhos = []
            for start, end, pid in test_protein_boundaries:
                if end - start < 10:
                    continue
                rho_pp, _ = scipy_stats.spearmanr(y_test_z[start:end],
                                                   y_pred[start:end])
                if not np.isnan(rho_pp):
                    pp_rhos.append(rho_pp)
            fold_per_protein_rhos.extend(pp_rhos)

        if not fold_r2s:
            continue

        # Aggregate
        n_train_total = sum(len(dataset[i]["target"])
                            for i in range(n) if extractor(dataset[i]) is not None)
        res = RegressionResult(
            model_name=model_name,
            n_features=model_def["n_features"],
            n_proteins_train=int(n * (n_folds - 1) / n_folds),
            n_proteins_test=int(n / n_folds),
            n_residues_train=0,  # approximate
            n_residues_test=0,
            cv_r2_mean=float(np.mean(fold_r2s)),
            cv_r2_std=float(np.std(fold_r2s)),
            cv_rho_mean=float(np.mean(fold_rhos)),
            cv_rho_std=float(np.std(fold_rhos)),
            feature_names=model_def["feature_names"],
        )

        if fold_per_protein_rhos:
            res.cv_per_protein_rho_median = float(np.median(fold_per_protein_rhos))
            res.cv_per_protein_rho_mean = float(np.mean(fold_per_protein_rhos))

        if fold_coefs:
            res.feature_coefs_mean = [float(x) for x in np.mean(fold_coefs, axis=0)]
            res.feature_coefs_std = [float(x) for x in np.std(fold_coefs, axis=0)]
            res.feature_coefs_per_fold = [[float(x) for x in fold] for fold in fold_coefs]

        # Theoretical SE: fit on all data, compute Ridge covariance matrix
        # Var(beta) = sigma^2 * (X'X + lambda*I)^{-1} X'X (X'X + lambda*I)^{-1}
        try:
            X_all_parts, y_all_parts = [], []
            for entry in dataset:
                X = extractor(entry)
                if X is not None:
                    X_all_parts.append(X)
                    y_all_parts.append(entry["target"])
            if X_all_parts:
                X_all = np.nan_to_num(np.vstack(X_all_parts), nan=0.0)
                y_all = np.concatenate(y_all_parts)
                if use_ridge:
                    reg_full = Ridge(alpha=alpha).fit(X_all, y_all)
                else:
                    reg_full = LinearRegression().fit(X_all, y_all)
                resid = y_all - reg_full.predict(X_all)
                sigma2 = float(np.sum(resid**2) / (len(y_all) - X_all.shape[1]))
                XtX = X_all.T @ X_all
                if use_ridge:
                    A_inv = np.linalg.inv(XtX + alpha * np.eye(X_all.shape[1]))
                    cov_beta = sigma2 * A_inv @ XtX @ A_inv
                else:
                    cov_beta = sigma2 * np.linalg.inv(XtX)
                se = np.sqrt(np.diag(cov_beta))
                res.feature_coefs_se = [float(x) for x in se]
        except Exception as e:
            print(f"    Warning: could not compute theoretical SE: {e}")

        print(f"    CV R²: {res.cv_r2_mean:.4f} ± {res.cv_r2_std:.4f}")
        print(f"    CV rho: {res.cv_rho_mean:.4f} ± {res.cv_rho_std:.4f}")
        print(f"    Per-protein median rho: {res.cv_per_protein_rho_median:.4f}")

        if res.feature_coefs_mean and len(res.feature_coefs_mean) <= 25:
            print(f"    Coefficients (mean ± CV-std | theoretical SE):")
            se_list = res.feature_coefs_se or [None] * len(res.feature_coefs_mean)
            std_list = res.feature_coefs_std or [None] * len(res.feature_coefs_mean)
            for name, coef, sd, se in zip(res.feature_names, res.feature_coefs_mean,
                                           std_list, se_list):
                se_str = f"SE={se:.4f}" if se is not None else "SE=N/A"
                sd_str = f"±{sd:.4f}" if sd is not None else ""
                print(f"      {name:12s}: {coef:+.4f} {sd_str}  ({se_str})")

        results[model_name] = res

    return results


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-DDG regression for dynamics prediction",
    )
    parser.add_argument("--atlas_dir", type=str, required=True)
    parser.add_argument("--robustness_dir", type=str, required=True)
    parser.add_argument("--scorer", type=str, default="thermompnn")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target", type=str, default="rmsf",
                        choices=["rmsf", "bfactor"],
                        help="Target variable to predict")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regularization strength")
    parser.add_argument("--max_seq_length", type=int, default=0)
    parser.add_argument("--max_proteins", type=int, default=0)
    parser.add_argument("--exclude", type=str, nargs="*", default=None,
                        help="Protein IDs to exclude (e.g. 2GAR_A 3F4M_A)")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"Multi-DDG Regression: {args.scorer} -> {args.target}")
    print(f"{'='*60}")

    exclude_set = set(args.exclude) if args.exclude else set()

    dataset = build_dataset(
        atlas_dir=args.atlas_dir,
        robustness_dir=args.robustness_dir,
        scorer=args.scorer,
        target=args.target,
        max_seq_length=args.max_seq_length,
        max_proteins=args.max_proteins,
        exclude_proteins=exclude_set,
    )

    if not dataset:
        print("No data loaded!")
        return

    print(f"\nRunning {args.n_folds}-fold protein-level CV "
          f"(alpha={args.alpha})...")
    results = run_cv_regression(
        dataset,
        n_folds=args.n_folds,
        alpha=args.alpha,
    )

    # Save results
    out_dir = Path(args.output_dir) / args.scorer
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"multi_ddg_{args.target}_results.json"
    serializable = {k: asdict(v) for k, v in results.items()}
    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2, default=lambda x: None
                  if isinstance(x, float) and np.isnan(x) else x)
    print(f"\nResults saved to {out_file}")

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"COMPARISON TABLE ({args.scorer} -> {args.target})")
    print(f"{'='*60}")
    print(f"{'Model':<25s} {'n_feat':>6s} {'CV R²':>12s} "
          f"{'CV rho':>12s} {'PP med rho':>10s}")
    print("-" * 70)
    for name in ["ols_plddt", "ols_mean_ddg", "ols_mean_abs_ddg", "ols_sasa",
                  "ols_mean_plddt", "ridge_nonlinear_only",
                  "ridge_20ddg", "ridge_20ddg_plddt",
                  "ridge_20ddg_nonlinear", "ridge_20ddg_nonlinear_plddt"]:
        if name not in results:
            continue
        r = results[name]
        print(f"{r.model_name:<25s} {r.n_features:>6d} "
              f"{r.cv_r2_mean:>6.4f}±{r.cv_r2_std:.4f} "
              f"{r.cv_rho_mean:>6.4f}±{r.cv_rho_std:.4f} "
              f"{r.cv_per_protein_rho_median:>10.4f}")


if __name__ == "__main__":
    main()
