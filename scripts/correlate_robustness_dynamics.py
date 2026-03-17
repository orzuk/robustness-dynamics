#!/usr/bin/env python3
"""
Correlate per-residue mutational robustness with protein dynamics (RMSF)
using ATLAS database data.

This is the core analysis script for Direction 7:
"Does mutational robustness predict protein dynamics?"

Inputs:
  - ATLAS download directory (from download_atlas.py):
      proteins/{pdb_chain}/*_RMSF.tsv, *_pLDDT.tsv, *_Bfactor.tsv
  - Robustness output directory (from compute_robustness.py):
      {scorer}/*_robustness.tsv

Outputs:
  - Per-protein correlation results (TSV + JSON)
  - Pooled correlation analysis
  - Comparison: robustness vs. pLDDT vs. B-factor as dynamics predictors
  - Stratified analysis by secondary structure and burial
  - Publication-ready figures

Usage:
  python correlate_robustness_dynamics.py \
      --atlas_dir /sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas \
      --robustness_dir /sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas_robustness \
      --scorer esm1v \
      --output_dir /sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas_analysis

  # Multiple scorers:
  python correlate_robustness_dynamics.py \
      --atlas_dir /sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas \
      --robustness_dir /sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas_robustness \
      --scorer esm1v thermompnn \
      --output_dir /sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas_analysis
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats as scipy_stats
from dataclasses import dataclass, field, asdict

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ==========================================================================
# DATA LOADING
# ==========================================================================

def load_atlas_tsv(protein_dir: str, suffix: str) -> Optional[pd.DataFrame]:
    """Load a per-residue TSV file from an ATLAS protein directory."""
    protein_dir = Path(protein_dir)
    matches = list(protein_dir.glob(f"*{suffix}"))
    if not matches:
        return None
    df = pd.read_csv(matches[0], sep="\t")
    return df


def load_atlas_rmsf(protein_dir: str) -> Optional[pd.DataFrame]:
    """Load RMSF data. Returns DataFrame with columns: position, rmsf_avg.

    ATLAS RMSF files have RMSF for 3 replicates. We average them.
    """
    df = load_atlas_tsv(protein_dir, "_RMSF.tsv")
    if df is None:
        return None
    # ATLAS RMSF TSV typically has columns like:
    # resid, resname, RMSF_R1, RMSF_R2, RMSF_R3 (or similar)
    # Handle different possible column formats
    rmsf_cols = [c for c in df.columns if "rmsf" in c.lower() or "r1" in c.lower()
                 or "r2" in c.lower() or "r3" in c.lower()]
    if not rmsf_cols:
        # Try numeric columns (skip residue id/name)
        rmsf_cols = [c for c in df.columns if df[c].dtype in (np.float64, np.float32, float)]

    if not rmsf_cols:
        return None

    result = pd.DataFrame()
    result["position"] = range(1, len(df) + 1)
    result["rmsf_avg"] = df[rmsf_cols].mean(axis=1).values
    # Also keep individual replicates if available
    for i, col in enumerate(rmsf_cols):
        result[f"rmsf_r{i+1}"] = df[col].values

    return result


def load_atlas_pldt(protein_dir: str) -> Optional[pd.DataFrame]:
    """Load pLDDT data. Returns DataFrame with columns: position, plddt."""
    df = load_atlas_tsv(protein_dir, "_pLDDT.tsv")
    if df is None:
        return None
    # Find the pLDDT column
    plddt_cols = [c for c in df.columns if "plddt" in c.lower() or "confidence" in c.lower()]
    if not plddt_cols:
        numeric = [c for c in df.columns if df[c].dtype in (np.float64, np.float32, float)]
        plddt_cols = numeric[-1:] if numeric else []
    if not plddt_cols:
        return None

    result = pd.DataFrame()
    result["position"] = range(1, len(df) + 1)
    result["plddt"] = df[plddt_cols[0]].values
    return result


def load_atlas_bfactor(protein_dir: str) -> Optional[pd.DataFrame]:
    """Load B-factor data."""
    df = load_atlas_tsv(protein_dir, "_Bfactor.tsv")
    if df is None:
        return None
    bfac_cols = [c for c in df.columns if "bfactor" in c.lower() or "b_factor" in c.lower()]
    if not bfac_cols:
        numeric = [c for c in df.columns if df[c].dtype in (np.float64, np.float32, float)]
        bfac_cols = numeric[-1:] if numeric else []
    if not bfac_cols:
        return None
    result = pd.DataFrame()
    result["position"] = range(1, len(df) + 1)
    result["bfactor"] = df[bfac_cols[0]].values
    return result


def load_robustness(robustness_dir: str, scorer: str, protein_id: str
                    ) -> Optional[pd.DataFrame]:
    """Load per-residue robustness TSV from compute_robustness.py output."""
    tsv_path = Path(robustness_dir) / scorer / f"{protein_id}_robustness.tsv"
    if not tsv_path.exists():
        return None
    return pd.read_csv(tsv_path, sep="\t")


def load_robustness_global(robustness_dir: str, scorer: str, protein_id: str
                           ) -> Optional[Dict]:
    """Load global robustness metrics from JSON."""
    json_path = Path(robustness_dir) / scorer / f"{protein_id}_robustness.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        data = json.load(f)
    return data.get("global_metrics")


def _load_consurf_mapping(consurf_dir: str) -> dict:
    """Load identical_to_unique_dict.txt mapping from ConSurf-DB.

    Returns dict mapping pdb_chain (lowercase) -> unique pdb_chain (lowercase).
    """
    dict_path = Path(consurf_dir) / "identical_to_unique_dict.txt"
    mapping = {}
    if not dict_path.exists():
        return mapping
    with open(dict_path) as f:
        for line in f:
            line = line.strip()
            if ":" not in line:
                continue
            ident, unique = line.split(":", 1)
            if len(ident) >= 5 and len(unique) >= 5:
                # Format: "104LB" -> pdb="104L", chain="B" -> "104l_b"
                pdb_i, chain_i = ident[:-1].lower(), ident[-1].lower()
                pdb_u, chain_u = unique[:-1].lower(), unique[-1].lower()
                mapping[f"{pdb_i}_{chain_i}"] = f"{pdb_u}_{chain_u}"
    return mapping


# Module-level cache for ConSurf mapping (loaded once per run)
_CONSURF_MAPPING: Optional[dict] = None


def load_conservation(protein_dir: str, consurf_dir: Optional[str] = None,
                      protein_id: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load per-residue evolutionary conservation scores from ConSurf-DB.

    Looks for a ConSurf JSON file in consurf_dir (central ConSurf-DB directory).
    Falls back to per-protein TSV files in protein_dir for backward compatibility.

    ConSurf JSON contains SCORE (continuous Rate4Site score, negative=conserved)
    and COLOR (discrete 1-9 grade, 9=most conserved).
    We use SCORE as the conservation measure.

    Returns DataFrame with columns: position, conservation.
    """
    global _CONSURF_MAPPING

    # --- Try ConSurf-DB JSON (preferred) ---
    if consurf_dir is not None and protein_id is not None:
        consurf_path = Path(consurf_dir)
        # Look in both consurf_dir and consurf_dir/files/ (ConSurf-DB layout)
        search_dirs = [consurf_path]
        if (consurf_path / "files").is_dir():
            search_dirs.append(consurf_path / "files")

        # Try direct match (case-insensitive: try original, upper, various combos)
        pid_parts = protein_id.split("_")
        if len(pid_parts) == 2:
            pdb, chain = pid_parts
            candidates = [
                f"{pdb.upper()}_{chain.upper()}_consurf_info.json",
                f"{pdb.upper()}_{chain}_consurf_info.json",
                f"{pdb.lower()}_{chain}_consurf_info.json",
                f"{pdb}_{chain}_consurf_info.json",
            ]
            json_path = None
            for cand in candidates:
                for sdir in search_dirs:
                    p = sdir / cand
                    if p.exists():
                        json_path = p
                        break
                if json_path is not None:
                    break

            # If not found directly, try the identical_to_unique mapping
            if json_path is None:
                if _CONSURF_MAPPING is None:
                    _CONSURF_MAPPING = _load_consurf_mapping(consurf_dir)
                pid_lower = protein_id.lower()
                if pid_lower in _CONSURF_MAPPING:
                    mapped = _CONSURF_MAPPING[pid_lower]
                    mpdb, mchain = mapped.split("_")
                    mapped_file = f"{mpdb.upper()}_{mchain.upper()}_consurf_info.json"
                    for sdir in search_dirs:
                        p = sdir / mapped_file
                        if p.exists():
                            json_path = p
                            break

            if json_path is not None:
                try:
                    with open(json_path) as f:
                        data = json.load(f)
                    scores = data.get("SCORE", [])
                    if scores:
                        result = pd.DataFrame()
                        result["position"] = range(1, len(scores) + 1)
                        result["conservation"] = [
                            float(s) if s is not None else np.nan for s in scores
                        ]
                        return result
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"  Warning: ConSurf JSON parse error for {protein_id}: {e}")

    # --- Fallback: per-protein TSV files (legacy) ---
    for suffix in ("_conservation.tsv", "_consurf.tsv", "_ConSurf.tsv"):
        df = load_atlas_tsv(protein_dir, suffix)
        if df is not None:
            break
    else:
        return None

    # Find the conservation score column
    score_cols = [c for c in df.columns
                  if any(kw in c.lower()
                         for kw in ("conservation", "consurf", "score"))]
    if not score_cols:
        numeric = [c for c in df.columns
                   if df[c].dtype in (np.float64, np.float32, float, int, np.int64)]
        score_cols = numeric[-1:] if numeric else []
    if not score_cols:
        return None

    result = pd.DataFrame()
    result["position"] = range(1, len(df) + 1)
    result["conservation"] = pd.to_numeric(df[score_cols[0]], errors="coerce").values
    return result


# ==========================================================================
# PDB PREPROCESSING
# ==========================================================================

def _clean_pdb_for_mdtraj(pdb_path: str) -> str:
    """Write a cleaned PDB to a temp file, suitable for mdtraj/DSSP.

    Raw PDB files can contain alternate conformations (altloc A/B) and
    modified residues recorded as HETATM.  mdtraj's C extensions crash
    (calling exit()) when two atoms occupy the same coordinates.

    This function:
      - Keeps only the first alternate conformation (altloc ' ' or 'A')
      - Drops duplicate (atom_name, chain, resSeq, iCode) entries
      - Preserves HETATM records for modified amino acids
      - Writes the result to a NamedTemporaryFile (caller must delete)

    Returns path to the cleaned temp PDB file.
    """
    import tempfile
    seen = set()
    cleaned_lines = []
    for line in open(pdb_path):
        if line.startswith(("ATOM", "HETATM")):
            altloc = line[16] if len(line) > 16 else " "
            if altloc not in (" ", "A"):
                continue  # skip alternate conformations B, C, ...
            # Deduplicate by (atom_name, chain, resSeq, iCode)
            atom_name = line[12:16]
            chain = line[21]
            resseq = line[22:27]  # includes iCode at position 26
            key = (atom_name, chain, resseq)
            if key in seen:
                continue
            seen.add(key)
            # Clear the altloc field so mdtraj doesn't see it
            line = line[:16] + " " + line[17:]
            cleaned_lines.append(line)
        elif line.startswith(("END", "TER", "MODEL", "ENDMDL",
                              "CRYST1", "REMARK", "HEADER")):
            cleaned_lines.append(line)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False)
    tmp.writelines(cleaned_lines)
    tmp.close()
    return tmp.name


# ==========================================================================
# SECONDARY STRUCTURE FROM PDB
# ==========================================================================

def assign_secondary_structure(pdb_path: str) -> Optional[List[str]]:
    """Assign secondary structure using DSSP (BioPython) or mdtraj fallback.

    Returns list of 'H' (helix), 'E' (sheet), 'C' (coil) per residue.
    Uses a cleaned PDB (no alt-confs / duplicate atoms) to avoid C-level
    crashes in mkdssp and mdtraj.
    """
    clean_path = _clean_pdb_for_mdtraj(pdb_path)
    try:
        return _assign_ss_impl(clean_path)
    finally:
        os.unlink(clean_path)


def _assign_ss_impl(pdb_path: str) -> Optional[List[str]]:
    """Inner implementation of SS assignment (operates on a clean PDB)."""
    # Try BioPython DSSP first (requires external mkdssp binary)
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.DSSP import DSSP
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        model = structure[0]
        dssp = DSSP(model, pdb_path, dssp="mkdssp")
        ss_list = []
        for key in dssp.keys():
            ss = dssp[key][2]  # secondary structure code
            if ss in ("H", "G", "I"):
                ss_list.append("H")  # helix
            elif ss in ("E", "B"):
                ss_list.append("E")  # sheet
            else:
                ss_list.append("C")  # coil/loop
        return ss_list
    except Exception:
        pass

    # Fallback: mdtraj DSSP (no external binary needed)
    try:
        import mdtraj
        traj = mdtraj.load(pdb_path)
        dssp = mdtraj.compute_dssp(traj)[0]  # shape (n_residues,)
        ss_list = []
        for ss in dssp:
            if ss == 'H':
                ss_list.append('H')
            elif ss == 'E':
                ss_list.append('E')
            else:
                ss_list.append('C')
        return ss_list
    except Exception:
        return None


def compute_burial(pdb_path: str) -> Optional[List[float]]:
    """Compute per-residue relative solvent accessibility (RSA).

    Returns list of RSA values (0=buried, 1=exposed).
    Uses a cleaned PDB to avoid C-level crashes from overlapping atoms.
    """
    clean_path = _clean_pdb_for_mdtraj(pdb_path)
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.DSSP import DSSP
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", clean_path)
        model = structure[0]
        dssp = DSSP(model, clean_path, dssp="mkdssp")
        rsa_list = [dssp[key][3] for key in dssp.keys()]
        return rsa_list
    except Exception:
        return None
    finally:
        os.unlink(clean_path)


def compute_sasa_from_pdb(pdb_path: str) -> Optional[pd.DataFrame]:
    """Compute per-residue SASA from a PDB file using mdtraj Shrake-Rupley.

    This is independent of DSSP (works even with --no_dssp) and provides
    a uniform burial baseline for all protein types (natural and designed).
    Uses a cleaned PDB to avoid C-level crashes from overlapping atoms.

    Returns DataFrame with columns: position, sasa (in nm^2).
    """
    clean_path = _clean_pdb_for_mdtraj(pdb_path)
    try:
        import mdtraj
        traj = mdtraj.load(clean_path)
        # Compute per-atom SASA, then sum by residue
        sasa_per_atom = mdtraj.shrake_rupley(traj, mode='atom')  # shape (1, n_atoms)
        sasa_per_residue = np.zeros(traj.topology.n_residues)
        for atom in traj.topology.atoms:
            sasa_per_residue[atom.residue.index] += sasa_per_atom[0, atom.index]
        result = pd.DataFrame({
            "position": range(1, len(sasa_per_residue) + 1),
            "sasa": sasa_per_residue,
        })
        return result
    except Exception:
        return None
    finally:
        os.unlink(clean_path)


# ==========================================================================
# HELPER: PARTIAL CORRELATION
# ==========================================================================

def partial_spearman(x, y, z):
    """Compute partial Spearman correlation rho(X,Y|Z).

    Returns (partial_rho, p_value) or (nan, nan) if degenerate.
    """
    rho_xy, _ = scipy_stats.spearmanr(x, y)
    rho_xz, _ = scipy_stats.spearmanr(x, z)
    rho_yz, _ = scipy_stats.spearmanr(y, z)
    denom = np.sqrt((1 - rho_xz**2) * (1 - rho_yz**2))
    if denom < 1e-10:
        return np.nan, np.nan
    partial_rho = (rho_xy - rho_xz * rho_yz) / denom
    n = len(x)
    if n <= 3:
        return partial_rho, np.nan
    t_stat = partial_rho * np.sqrt((n - 3) / (1 - partial_rho**2 + 1e-10))
    from scipy.stats import t as t_dist
    pval = 2 * t_dist.sf(abs(t_stat), n - 3)
    return partial_rho, pval


def regression_delta_r2(X_base, X_joint, y):
    """Compute R² for base model and joint model, return (r2_base, r2_joint, delta_r2).

    X_base: ndarray (n, k1) — baseline features
    X_joint: ndarray (n, k2) — baseline + new features
    y: ndarray (n,) — target
    """
    from sklearn.linear_model import LinearRegression
    r2_base = LinearRegression().fit(X_base, y).score(X_base, y)
    r2_joint = LinearRegression().fit(X_joint, y).score(X_joint, y)
    return r2_base, r2_joint, r2_joint - r2_base


# ==========================================================================
# CORRELATION ANALYSIS
# ==========================================================================

@dataclass
class PerProteinResult:
    """Correlation results for a single protein."""
    protein_id: str
    seq_length: int
    n_residues_used: int  # after filtering NaN

    # Robustness vs RMSF (primary measure: mean |DDG|)
    rho_robustness_rmsf: float = np.nan
    pval_robustness_rmsf: float = np.nan
    r2_robustness_rmsf: float = np.nan

    # Alternative robustness measures vs RMSF
    rho_frac_destab_rmsf: float = np.nan     # fraction of destabilizing mutations (ddG > 1)
    rho_frac_neutral_rmsf: float = np.nan    # fraction of neutral mutations (|ddG| < 0.5)
    rho_std_ddg_rmsf: float = np.nan         # std of ddG (landscape ruggedness)
    rho_max_ddg_rmsf: float = np.nan         # worst-case mutation effect
    rho_mean_abs_ddg_rmsf: float = np.nan    # mean |DDG| per residue
    rho_mean_ddg_rmsf: float = np.nan        # mean DDG (signed) per residue

    # pLDDT vs RMSF (baseline)
    rho_plddt_rmsf: float = np.nan
    pval_plddt_rmsf: float = np.nan
    r2_plddt_rmsf: float = np.nan

    # B-factor vs RMSF (sanity check)
    rho_bfactor_rmsf: float = np.nan
    pval_bfactor_rmsf: float = np.nan

    # SASA vs RMSF (burial baseline)
    rho_sasa_rmsf: float = np.nan
    pval_sasa_rmsf: float = np.nan
    r2_sasa_rmsf: float = np.nan

    # Robustness vs pLDDT (are they correlated?)
    rho_robustness_plddt: float = np.nan

    # Robustness vs B-factor
    rho_robustness_bfactor: float = np.nan

    # Robustness vs SASA (are they correlated?)
    rho_robustness_sasa: float = np.nan

    # Partial correlation: robustness vs RMSF, controlling for SASA
    rho_robustness_rmsf_partial_sasa: float = np.nan
    pval_robustness_rmsf_partial_sasa: float = np.nan

    # Partial correlation: robustness vs RMSF, controlling for pLDDT
    rho_robustness_rmsf_partial_plddt: float = np.nan

    # Multiple regression: RMSF ~ robustness + pLDDT
    r2_joint: float = np.nan
    delta_r2_over_plddt: float = np.nan  # r2_joint - r2_plddt
    beta_robustness: float = np.nan      # regression coefficient
    beta_plddt: float = np.nan

    # Multiple regression: RMSF ~ robustness + SASA
    r2_joint_sasa: float = np.nan
    delta_r2_over_sasa: float = np.nan  # r2_joint_sasa - r2_sasa

    # Multiple regression with best robustness combo: RMSF ~ mean_abs_ddg + frac_destab + pLDDT
    r2_joint_multi_rob: float = np.nan
    delta_r2_multi_rob_over_plddt: float = np.nan

    # Alternative robustness measures vs B-factor
    rho_frac_destab_bfactor: float = np.nan
    rho_frac_neutral_bfactor: float = np.nan
    rho_std_ddg_bfactor: float = np.nan
    rho_max_ddg_bfactor: float = np.nan
    rho_mean_abs_ddg_bfactor: float = np.nan
    rho_mean_ddg_bfactor: float = np.nan

    # === B-factor as TARGET (predicting experimental dynamics) ===
    # Robustness vs B-factor (primary)
    rho_robustness_bfactor_target: float = np.nan
    pval_robustness_bfactor_target: float = np.nan
    r2_robustness_bfactor_target: float = np.nan

    # pLDDT vs B-factor (baseline)
    rho_plddt_bfactor: float = np.nan
    r2_plddt_bfactor: float = np.nan

    # SASA vs B-factor
    rho_sasa_bfactor: float = np.nan
    r2_sasa_bfactor: float = np.nan

    # Partial correlation: robustness vs B-factor, controlling for SASA
    rho_robustness_bfactor_partial_sasa: float = np.nan

    # Partial correlation: robustness vs B-factor, controlling for pLDDT
    rho_robustness_bfactor_partial_plddt: float = np.nan

    # Multiple regression: B-factor ~ robustness + pLDDT
    r2_bfactor_joint_plddt: float = np.nan
    delta_r2_bfactor_over_plddt: float = np.nan

    # Multiple regression: B-factor ~ robustness + SASA
    r2_bfactor_joint_sasa: float = np.nan
    delta_r2_bfactor_over_sasa: float = np.nan

    # === CONSERVATION AS COVARIATE ===
    # Conservation vs dynamics (baseline)
    rho_conservation_rmsf: float = np.nan
    r2_conservation_rmsf: float = np.nan
    rho_conservation_bfactor: float = np.nan
    r2_conservation_bfactor: float = np.nan

    # Robustness vs Conservation (collinearity check)
    rho_robustness_conservation: float = np.nan

    # Partial: robustness vs RMSF | conservation
    rho_robustness_rmsf_partial_conservation: float = np.nan
    # Partial: robustness vs B-factor | conservation
    rho_robustness_bfactor_partial_conservation: float = np.nan

    # Multiple regression: RMSF ~ robustness + conservation
    r2_joint_conservation: float = np.nan
    delta_r2_over_conservation: float = np.nan
    # Multiple regression: B-factor ~ robustness + conservation
    r2_bfactor_joint_conservation: float = np.nan
    delta_r2_bfactor_over_conservation: float = np.nan

    # Global robustness metrics
    global_mean_abs_ddg: float = np.nan
    global_mean_ddg: float = np.nan

    scorer: str = ""


def correlate_single_protein(
    protein_id: str,
    robustness_df: pd.DataFrame,
    rmsf_df: pd.DataFrame,
    plddt_df: Optional[pd.DataFrame],
    bfactor_df: Optional[pd.DataFrame],
    global_metrics: Optional[Dict],
    scorer: str,
    sasa_df: Optional[pd.DataFrame] = None,
    conservation_df: Optional[pd.DataFrame] = None,
    rob_col: str = "mean_abs_ddg",
) -> Optional[PerProteinResult]:
    """Compute all correlations for a single protein."""

    # Merge on position — include all available robustness measures
    merge_cols = ["position"]
    for c in ["mean_abs_ddg", "mean_ddg", "frac_destabilizing", "frac_neutral",
              "std_ddg", "max_ddg"]:
        if c in robustness_df.columns:
            merge_cols.append(c)
    merged = robustness_df[merge_cols].copy()
    merged = merged.merge(rmsf_df[["position", "rmsf_avg"]], on="position", how="inner")

    if plddt_df is not None:
        merged = merged.merge(plddt_df[["position", "plddt"]], on="position", how="left")
    else:
        merged["plddt"] = np.nan

    if bfactor_df is not None:
        merged = merged.merge(bfactor_df[["position", "bfactor"]], on="position", how="left")
    else:
        merged["bfactor"] = np.nan

    if sasa_df is not None:
        merged = merged.merge(sasa_df[["position", "sasa"]], on="position", how="left")
    else:
        merged["sasa"] = np.nan

    if conservation_df is not None:
        merged = merged.merge(conservation_df[["position", "conservation"]],
                              on="position", how="left")
    else:
        merged["conservation"] = np.nan

    # Drop rows with NaN in core columns
    core = merged.dropna(subset=[rob_col, "rmsf_avg"])
    if len(core) < 10:  # too few residues
        return None

    result = PerProteinResult(
        protein_id=protein_id,
        seq_length=len(robustness_df),
        n_residues_used=len(core),
        scorer=scorer,
    )

    # --- Robustness vs RMSF (primary) ---
    rho, pval = scipy_stats.spearmanr(core[rob_col], core["rmsf_avg"])
    result.rho_robustness_rmsf = rho
    result.pval_robustness_rmsf = pval
    # Pearson R^2
    r, _ = scipy_stats.pearsonr(core[rob_col], core["rmsf_avg"])
    result.r2_robustness_rmsf = r ** 2

    # --- Alternative robustness measures vs RMSF ---
    for col, attr in [("frac_destabilizing", "rho_frac_destab_rmsf"),
                       ("frac_neutral", "rho_frac_neutral_rmsf"),
                       ("std_ddg", "rho_std_ddg_rmsf"),
                       ("max_ddg", "rho_max_ddg_rmsf"),
                       ("mean_abs_ddg", "rho_mean_abs_ddg_rmsf"),
                       ("mean_ddg", "rho_mean_ddg_rmsf")]:
        if col in core.columns:
            valid_alt = core.dropna(subset=[col])
            if len(valid_alt) >= 10:
                rho_alt, _ = scipy_stats.spearmanr(valid_alt[col], valid_alt["rmsf_avg"])
                setattr(result, attr, rho_alt)

    # --- pLDDT vs RMSF ---
    plddt_valid = core.dropna(subset=["plddt"])
    if len(plddt_valid) >= 10:
        rho, pval = scipy_stats.spearmanr(plddt_valid["plddt"], plddt_valid["rmsf_avg"])
        result.rho_plddt_rmsf = rho
        result.pval_plddt_rmsf = pval
        r, _ = scipy_stats.pearsonr(plddt_valid["plddt"], plddt_valid["rmsf_avg"])
        result.r2_plddt_rmsf = r ** 2

    # --- B-factor vs RMSF ---
    bfac_valid = core.dropna(subset=["bfactor"])
    if len(bfac_valid) >= 10:
        rho, pval = scipy_stats.spearmanr(bfac_valid["bfactor"], bfac_valid["rmsf_avg"])
        result.rho_bfactor_rmsf = rho
        result.pval_bfactor_rmsf = pval

    # --- Robustness vs pLDDT ---
    if len(plddt_valid) >= 10:
        rho, _ = scipy_stats.spearmanr(plddt_valid[rob_col], plddt_valid["plddt"])
        result.rho_robustness_plddt = rho

    # --- Robustness vs B-factor ---
    if len(bfac_valid) >= 10:
        rho, _ = scipy_stats.spearmanr(bfac_valid[rob_col], bfac_valid["bfactor"])
        result.rho_robustness_bfactor = rho

    # --- SASA vs RMSF (burial baseline) ---
    sasa_valid = core.dropna(subset=["sasa"])
    if len(sasa_valid) >= 10:
        rho, pval = scipy_stats.spearmanr(sasa_valid["sasa"], sasa_valid["rmsf_avg"])
        result.rho_sasa_rmsf = rho
        result.pval_sasa_rmsf = pval
        r, _ = scipy_stats.pearsonr(sasa_valid["sasa"], sasa_valid["rmsf_avg"])
        result.r2_sasa_rmsf = r ** 2

        # Robustness vs SASA
        rho, _ = scipy_stats.spearmanr(sasa_valid[rob_col], sasa_valid["sasa"])
        result.rho_robustness_sasa = rho

        # Partial correlation: robustness vs RMSF, controlling for SASA
        pr, pval_pr = partial_spearman(
            sasa_valid[rob_col].values,
            sasa_valid["rmsf_avg"].values,
            sasa_valid["sasa"].values,
        )
        result.rho_robustness_rmsf_partial_sasa = pr
        result.pval_robustness_rmsf_partial_sasa = pval_pr

    # --- Partial: robustness vs RMSF | pLDDT ---
    if len(plddt_valid) >= 10:
        pr, _ = partial_spearman(
            plddt_valid[rob_col].values,
            plddt_valid["rmsf_avg"].values,
            plddt_valid["plddt"].values,
        )
        result.rho_robustness_rmsf_partial_plddt = pr

    # --- Multiple regression: RMSF ~ robustness + SASA ---
    if len(sasa_valid) >= 10:
        from sklearn.linear_model import LinearRegression
        y_s = sasa_valid["rmsf_avg"].values
        X_sasa_only = sasa_valid[["sasa"]].values
        X_rob_sasa = sasa_valid[[rob_col, "sasa"]].values
        r2_sasa_only = LinearRegression().fit(X_sasa_only, y_s).score(X_sasa_only, y_s)
        r2_rob_sasa = LinearRegression().fit(X_rob_sasa, y_s).score(X_rob_sasa, y_s)
        result.r2_joint_sasa = r2_rob_sasa
        result.delta_r2_over_sasa = r2_rob_sasa - r2_sasa_only

    # --- Multiple regression: RMSF ~ robustness + pLDDT ---
    joint_valid = core.dropna(subset=["plddt"])
    if len(joint_valid) >= 10:
        from sklearn.linear_model import LinearRegression

        y = joint_valid["rmsf_avg"].values
        X_plddt = joint_valid[["plddt"]].values
        X_joint = joint_valid[[rob_col, "plddt"]].values

        # pLDDT alone
        reg_plddt = LinearRegression().fit(X_plddt, y)
        r2_plddt_only = reg_plddt.score(X_plddt, y)

        # Joint
        reg_joint = LinearRegression().fit(X_joint, y)
        r2_joint = reg_joint.score(X_joint, y)

        result.r2_joint = r2_joint
        result.delta_r2_over_plddt = r2_joint - r2_plddt_only
        result.beta_robustness = float(reg_joint.coef_[0])
        result.beta_plddt = float(reg_joint.coef_[1])

        # Multi-robustness regression: RMSF ~ mean_abs_ddg + frac_destabilizing + pLDDT
        if "frac_destabilizing" in joint_valid.columns:
            multi_valid = joint_valid.dropna(subset=["frac_destabilizing"])
            if len(multi_valid) >= 10:
                X_multi = multi_valid[[rob_col, "frac_destabilizing", "plddt"]].values
                y_multi = multi_valid["rmsf_avg"].values
                r2_multi = LinearRegression().fit(X_multi, y_multi).score(X_multi, y_multi)
                result.r2_joint_multi_rob = r2_multi
                result.delta_r2_multi_rob_over_plddt = r2_multi - r2_plddt_only

    # --- Conservation vs RMSF and robustness | conservation ---
    cons_valid = core.dropna(subset=["conservation"])
    if len(cons_valid) >= 10:
        from sklearn.linear_model import LinearRegression

        # Conservation vs RMSF baseline
        rho, _ = scipy_stats.spearmanr(cons_valid["conservation"],
                                       cons_valid["rmsf_avg"])
        result.rho_conservation_rmsf = rho
        r, _ = scipy_stats.pearsonr(cons_valid["conservation"],
                                    cons_valid["rmsf_avg"])
        result.r2_conservation_rmsf = r ** 2

        # Robustness vs Conservation (collinearity check)
        rho, _ = scipy_stats.spearmanr(cons_valid[rob_col],
                                       cons_valid["conservation"])
        result.rho_robustness_conservation = rho

        # Partial: robustness vs RMSF | conservation
        pr, _ = partial_spearman(
            cons_valid[rob_col].values,
            cons_valid["rmsf_avg"].values,
            cons_valid["conservation"].values,
        )
        result.rho_robustness_rmsf_partial_conservation = pr

        # Multiple regression: RMSF ~ robustness + conservation
        y_c = cons_valid["rmsf_avg"].values
        X_cons_only = cons_valid[["conservation"]].values
        X_rob_cons = cons_valid[[rob_col, "conservation"]].values
        r2_cons_only = LinearRegression().fit(X_cons_only, y_c).score(
            X_cons_only, y_c)
        r2_rob_cons = LinearRegression().fit(X_rob_cons, y_c).score(
            X_rob_cons, y_c)
        result.r2_joint_conservation = r2_rob_cons
        result.delta_r2_over_conservation = r2_rob_cons - r2_cons_only

    # =====================================================================
    # B-FACTOR AS TARGET (predicting experimental dynamics)
    # =====================================================================
    bfac_target = core.dropna(subset=["bfactor"])
    if len(bfac_target) >= 10:
        from sklearn.linear_model import LinearRegression

        # --- Robustness vs B-factor (primary) ---
        rho, pval = scipy_stats.spearmanr(bfac_target[rob_col], bfac_target["bfactor"])
        result.rho_robustness_bfactor_target = rho
        result.pval_robustness_bfactor_target = pval
        r, _ = scipy_stats.pearsonr(bfac_target[rob_col], bfac_target["bfactor"])
        result.r2_robustness_bfactor_target = r ** 2

        # --- Alternative robustness measures vs B-factor ---
        for col, attr in [("frac_destabilizing", "rho_frac_destab_bfactor"),
                           ("frac_neutral", "rho_frac_neutral_bfactor"),
                           ("std_ddg", "rho_std_ddg_bfactor"),
                           ("max_ddg", "rho_max_ddg_bfactor"),
                           ("mean_abs_ddg", "rho_mean_abs_ddg_bfactor"),
                           ("mean_ddg", "rho_mean_ddg_bfactor")]:
            if col in bfac_target.columns:
                valid_alt = bfac_target.dropna(subset=[col])
                if len(valid_alt) >= 10:
                    rho_alt, _ = scipy_stats.spearmanr(valid_alt[col], valid_alt["bfactor"])
                    setattr(result, attr, rho_alt)

        # --- pLDDT vs B-factor (baseline) ---
        bf_plddt = bfac_target.dropna(subset=["plddt"])
        if len(bf_plddt) >= 10:
            rho, _ = scipy_stats.spearmanr(bf_plddt["plddt"], bf_plddt["bfactor"])
            result.rho_plddt_bfactor = rho
            r, _ = scipy_stats.pearsonr(bf_plddt["plddt"], bf_plddt["bfactor"])
            result.r2_plddt_bfactor = r ** 2

        # --- SASA vs B-factor ---
        bf_sasa = bfac_target.dropna(subset=["sasa"])
        if len(bf_sasa) >= 10:
            rho, _ = scipy_stats.spearmanr(bf_sasa["sasa"], bf_sasa["bfactor"])
            result.rho_sasa_bfactor = rho
            r, _ = scipy_stats.pearsonr(bf_sasa["sasa"], bf_sasa["bfactor"])
            result.r2_sasa_bfactor = r ** 2

        # --- Partial: robustness vs B-factor | SASA ---
        if len(bf_sasa) >= 10:
            pr, _ = partial_spearman(
                bf_sasa[rob_col].values,
                bf_sasa["bfactor"].values,
                bf_sasa["sasa"].values,
            )
            result.rho_robustness_bfactor_partial_sasa = pr

        # --- Partial: robustness vs B-factor | pLDDT ---
        if len(bf_plddt) >= 10:
            pr, _ = partial_spearman(
                bf_plddt[rob_col].values,
                bf_plddt["bfactor"].values,
                bf_plddt["plddt"].values,
            )
            result.rho_robustness_bfactor_partial_plddt = pr

        # --- Regression: B-factor ~ robustness + pLDDT ---
        if len(bf_plddt) >= 10:
            y_bf = bf_plddt["bfactor"].values
            _, r2_j, dr2 = regression_delta_r2(
                bf_plddt[["plddt"]].values,
                bf_plddt[[rob_col, "plddt"]].values,
                y_bf,
            )
            result.r2_bfactor_joint_plddt = r2_j
            result.delta_r2_bfactor_over_plddt = dr2

        # --- Regression: B-factor ~ robustness + SASA ---
        if len(bf_sasa) >= 10:
            y_bf = bf_sasa["bfactor"].values
            _, r2_j, dr2 = regression_delta_r2(
                bf_sasa[["sasa"]].values,
                bf_sasa[[rob_col, "sasa"]].values,
                y_bf,
            )
            result.r2_bfactor_joint_sasa = r2_j
            result.delta_r2_bfactor_over_sasa = dr2

        # --- Conservation vs B-factor ---
        bf_cons = bfac_target.dropna(subset=["conservation"])
        if len(bf_cons) >= 10:
            rho, _ = scipy_stats.spearmanr(bf_cons["conservation"],
                                           bf_cons["bfactor"])
            result.rho_conservation_bfactor = rho
            r, _ = scipy_stats.pearsonr(bf_cons["conservation"],
                                        bf_cons["bfactor"])
            result.r2_conservation_bfactor = r ** 2

            # Partial: robustness vs B-factor | conservation
            pr, _ = partial_spearman(
                bf_cons[rob_col].values,
                bf_cons["bfactor"].values,
                bf_cons["conservation"].values,
            )
            result.rho_robustness_bfactor_partial_conservation = pr

            # Regression: B-factor ~ robustness + conservation
            y_bf = bf_cons["bfactor"].values
            _, r2_j, dr2 = regression_delta_r2(
                bf_cons[["conservation"]].values,
                bf_cons[[rob_col, "conservation"]].values,
                y_bf,
            )
            result.r2_bfactor_joint_conservation = r2_j
            result.delta_r2_bfactor_over_conservation = dr2

    # --- Global metrics ---
    if global_metrics:
        result.global_mean_abs_ddg = global_metrics.get("global_mean_abs_ddg", np.nan)
        result.global_mean_ddg = global_metrics.get("global_mean_ddg", np.nan)

    return result


# ==========================================================================
# POOLED ANALYSIS
# ==========================================================================

@dataclass
class PooledResult:
    """Pooled correlation results across all proteins."""
    n_proteins: int = 0
    n_residues: int = 0
    scorer: str = ""

    # Pooled Spearman (z-scored per protein)
    pooled_rho_robustness_rmsf: float = np.nan
    pooled_pval_robustness_rmsf: float = np.nan
    pooled_rho_plddt_rmsf: float = np.nan
    pooled_pval_plddt_rmsf: float = np.nan

    # Pooled Pearson R^2
    pooled_r2_robustness_rmsf: float = np.nan
    pooled_r2_plddt_rmsf: float = np.nan

    # Pooled joint regression
    pooled_r2_joint: float = np.nan
    pooled_delta_r2: float = np.nan

    # Pooled SASA (burial baseline)
    pooled_rho_sasa_rmsf: float = np.nan
    pooled_r2_sasa_rmsf: float = np.nan
    pooled_rho_robustness_rmsf_partial_sasa: float = np.nan
    pooled_rho_robustness_rmsf_partial_plddt: float = np.nan
    pooled_r2_joint_sasa: float = np.nan
    pooled_delta_r2_over_sasa: float = np.nan

    # Pooled B-factor (experimental dynamics baseline — predicting RMSF)
    pooled_rho_bfactor_rmsf: float = np.nan
    pooled_r2_bfactor_rmsf: float = np.nan
    pooled_rho_robustness_bfactor: float = np.nan
    pooled_r2_joint_bfactor: float = np.nan
    pooled_delta_r2_over_bfactor: float = np.nan

    # === B-FACTOR AS TARGET (predicting experimental dynamics) ===
    # Pooled: robustness → B-factor
    pooled_rho_robustness_bfactor_target: float = np.nan
    pooled_r2_robustness_bfactor_target: float = np.nan
    # Pooled: pLDDT → B-factor (baseline)
    pooled_rho_plddt_bfactor: float = np.nan
    pooled_r2_plddt_bfactor: float = np.nan
    # Pooled: SASA → B-factor
    pooled_rho_sasa_bfactor: float = np.nan
    pooled_r2_sasa_bfactor: float = np.nan
    # Pooled partial: robustness vs B-factor | SASA
    pooled_rho_robustness_bfactor_partial_sasa: float = np.nan
    # Pooled partial: robustness vs B-factor | pLDDT
    pooled_rho_robustness_bfactor_partial_plddt: float = np.nan
    # Pooled regression: B-factor ~ robustness + pLDDT
    pooled_r2_bfactor_joint_plddt: float = np.nan
    pooled_delta_r2_bfactor_over_plddt: float = np.nan
    # Pooled regression: B-factor ~ robustness + SASA
    pooled_r2_bfactor_joint_sasa: float = np.nan
    pooled_delta_r2_bfactor_over_sasa: float = np.nan

    # Distribution of per-protein correlations
    median_rho_robustness_rmsf: float = np.nan
    mean_rho_robustness_rmsf: float = np.nan
    std_rho_robustness_rmsf: float = np.nan
    median_rho_plddt_rmsf: float = np.nan
    mean_rho_plddt_rmsf: float = np.nan
    median_rho_sasa_rmsf: float = np.nan
    median_rho_partial_sasa: float = np.nan
    median_rho_partial_plddt: float = np.nan
    median_rho_bfactor_rmsf: float = np.nan
    median_rho_robustness_bfactor: float = np.nan

    # B-factor-as-target per-protein medians
    median_rho_robustness_bfactor_target: float = np.nan
    median_rho_plddt_bfactor: float = np.nan
    median_rho_sasa_bfactor: float = np.nan
    median_rho_robustness_bfactor_partial_sasa: float = np.nan
    median_rho_robustness_bfactor_partial_plddt: float = np.nan

    # === CONSERVATION AS COVARIATE (pooled) ===
    pooled_rho_conservation_rmsf: float = np.nan
    pooled_r2_conservation_rmsf: float = np.nan
    pooled_rho_conservation_bfactor: float = np.nan
    pooled_r2_conservation_bfactor: float = np.nan
    pooled_rho_robustness_conservation: float = np.nan
    pooled_rho_robustness_rmsf_partial_conservation: float = np.nan
    pooled_rho_robustness_bfactor_partial_conservation: float = np.nan
    pooled_r2_joint_conservation: float = np.nan
    pooled_delta_r2_over_conservation: float = np.nan
    pooled_r2_bfactor_joint_conservation: float = np.nan
    pooled_delta_r2_bfactor_over_conservation: float = np.nan
    median_rho_conservation_rmsf: float = np.nan
    median_rho_conservation_bfactor: float = np.nan
    median_rho_robustness_conservation: float = np.nan
    median_rho_robustness_rmsf_partial_conservation: float = np.nan
    median_rho_robustness_bfactor_partial_conservation: float = np.nan

    # Fraction of proteins where robustness beats pLDDT
    frac_robustness_beats_plddt: float = np.nan
    frac_robustness_beats_plddt_bfactor: float = np.nan  # for B-factor target


def run_pooled_analysis(
    per_protein_data: List[Tuple[pd.DataFrame, str]],
    per_protein_results: List[PerProteinResult],
    scorer: str,
    rob_col: str = "mean_abs_ddg",
    transform: str = "none",
) -> PooledResult:
    """Run pooled analysis across all proteins.

    per_protein_data: list of (merged_df, protein_id) for pooling residues.
    per_protein_results: list of PerProteinResult for summary stats.
    """
    result = PooledResult(scorer=scorer)

    # Summary of per-protein correlations
    rhos_rob = [r.rho_robustness_rmsf for r in per_protein_results
                if not np.isnan(r.rho_robustness_rmsf)]
    rhos_plddt = [r.rho_plddt_rmsf for r in per_protein_results
                  if not np.isnan(r.rho_plddt_rmsf)]
    rhos_sasa = [r.rho_sasa_rmsf for r in per_protein_results
                 if not np.isnan(r.rho_sasa_rmsf)]
    rhos_partial = [r.rho_robustness_rmsf_partial_sasa for r in per_protein_results
                    if not np.isnan(r.rho_robustness_rmsf_partial_sasa)]
    rhos_partial_plddt = [r.rho_robustness_rmsf_partial_plddt for r in per_protein_results
                          if not np.isnan(r.rho_robustness_rmsf_partial_plddt)]
    rhos_bfactor = [r.rho_bfactor_rmsf for r in per_protein_results
                    if not np.isnan(r.rho_bfactor_rmsf)]
    rhos_rob_bf = [r.rho_robustness_bfactor for r in per_protein_results
                   if not np.isnan(r.rho_robustness_bfactor)]

    result.n_proteins = len(per_protein_results)

    if rhos_rob:
        result.median_rho_robustness_rmsf = float(np.median(rhos_rob))
        result.mean_rho_robustness_rmsf = float(np.mean(rhos_rob))
        result.std_rho_robustness_rmsf = float(np.std(rhos_rob))
    if rhos_plddt:
        result.median_rho_plddt_rmsf = float(np.median(rhos_plddt))
        result.mean_rho_plddt_rmsf = float(np.mean(rhos_plddt))
    if rhos_sasa:
        result.median_rho_sasa_rmsf = float(np.median(rhos_sasa))
    if rhos_partial:
        result.median_rho_partial_sasa = float(np.median(rhos_partial))
    if rhos_partial_plddt:
        result.median_rho_partial_plddt = float(np.median(rhos_partial_plddt))
    if rhos_bfactor:
        result.median_rho_bfactor_rmsf = float(np.median(rhos_bfactor))
    if rhos_rob_bf:
        result.median_rho_robustness_bfactor = float(np.median(rhos_rob_bf))

    # B-factor-as-target per-protein medians
    def _median_attr(attr):
        vals = [getattr(r, attr) for r in per_protein_results
                if not np.isnan(getattr(r, attr))]
        return float(np.median(vals)) if vals else np.nan

    result.median_rho_robustness_bfactor_target = _median_attr("rho_robustness_bfactor_target")
    result.median_rho_plddt_bfactor = _median_attr("rho_plddt_bfactor")
    result.median_rho_sasa_bfactor = _median_attr("rho_sasa_bfactor")
    result.median_rho_robustness_bfactor_partial_sasa = _median_attr("rho_robustness_bfactor_partial_sasa")
    result.median_rho_robustness_bfactor_partial_plddt = _median_attr("rho_robustness_bfactor_partial_plddt")

    # Conservation per-protein medians
    result.median_rho_conservation_rmsf = _median_attr("rho_conservation_rmsf")
    result.median_rho_conservation_bfactor = _median_attr("rho_conservation_bfactor")
    result.median_rho_robustness_conservation = _median_attr("rho_robustness_conservation")
    result.median_rho_robustness_rmsf_partial_conservation = _median_attr("rho_robustness_rmsf_partial_conservation")
    result.median_rho_robustness_bfactor_partial_conservation = _median_attr("rho_robustness_bfactor_partial_conservation")

    # Fraction where |rho_robustness| > |rho_plddt| (RMSF target)
    both = [(r.rho_robustness_rmsf, r.rho_plddt_rmsf) for r in per_protein_results
            if not np.isnan(r.rho_robustness_rmsf) and not np.isnan(r.rho_plddt_rmsf)]
    if both:
        beats = sum(1 for rr, rp in both if abs(rr) > abs(rp))
        result.frac_robustness_beats_plddt = beats / len(both)

    # Fraction where |rho_robustness| > |rho_plddt| (B-factor target)
    both_bf = [(r.rho_robustness_bfactor_target, r.rho_plddt_bfactor)
               for r in per_protein_results
               if not np.isnan(r.rho_robustness_bfactor_target)
               and not np.isnan(r.rho_plddt_bfactor)]
    if both_bf:
        beats_bf = sum(1 for rr, rp in both_bf if abs(rr) > abs(rp))
        result.frac_robustness_beats_plddt_bfactor = beats_bf / len(both_bf)

    # Pool all residues (z-scored per protein)
    rob_col_z = f"{rob_col}_z"
    all_rows = []
    for merged_df, pid in per_protein_data:
        df = merged_df.dropna(subset=[rob_col, "rmsf_avg"]).copy()
        if len(df) < 10:
            continue
        # Optionally log-transform response variables before z-scoring
        # (reduces heavy-tail effects for OLS R²; does not affect Spearman)
        if transform == "log1p":
            for resp_col in ["rmsf_avg", "bfactor"]:
                if resp_col in df.columns:
                    df[resp_col] = np.log1p(np.clip(df[resp_col].values, 0, None))
        # Z-score within protein to remove protein-level differences
        for col in [rob_col, "rmsf_avg", "plddt", "sasa", "bfactor", "conservation"]:
            if col in df.columns:
                mu, sigma = df[col].mean(), df[col].std()
                if sigma > 0:
                    df[f"{col}_z"] = (df[col] - mu) / sigma
                else:
                    df[f"{col}_z"] = 0.0
        df["protein_id"] = pid
        all_rows.append(df)

    if not all_rows:
        return result

    pooled = pd.concat(all_rows, ignore_index=True)
    result.n_residues = len(pooled)

    # Pooled correlations on z-scored values
    valid = pooled.dropna(subset=[rob_col_z, "rmsf_avg_z"])
    if len(valid) >= 20:
        rho, pval = scipy_stats.spearmanr(valid[rob_col_z], valid["rmsf_avg_z"])
        result.pooled_rho_robustness_rmsf = rho
        result.pooled_pval_robustness_rmsf = pval
        r, _ = scipy_stats.pearsonr(valid[rob_col_z], valid["rmsf_avg_z"])
        result.pooled_r2_robustness_rmsf = r ** 2

    if "plddt_z" in pooled.columns:
        valid_plddt = pooled.dropna(subset=["plddt_z", "rmsf_avg_z"])
        if len(valid_plddt) >= 20:
            rho, pval = scipy_stats.spearmanr(valid_plddt["plddt_z"], valid_plddt["rmsf_avg_z"])
            result.pooled_rho_plddt_rmsf = rho
            result.pooled_pval_plddt_rmsf = pval
            r, _ = scipy_stats.pearsonr(valid_plddt["plddt_z"], valid_plddt["rmsf_avg_z"])
            result.pooled_r2_plddt_rmsf = r ** 2

        # Pooled joint regression
        joint_valid = pooled.dropna(subset=[rob_col_z, "plddt_z", "rmsf_avg_z"])
        if len(joint_valid) >= 20:
            from sklearn.linear_model import LinearRegression
            y = joint_valid["rmsf_avg_z"].values
            X_plddt = joint_valid[["plddt_z"]].values
            X_joint = joint_valid[[rob_col_z, "plddt_z"]].values

            r2_p = LinearRegression().fit(X_plddt, y).score(X_plddt, y)
            r2_j = LinearRegression().fit(X_joint, y).score(X_joint, y)
            result.pooled_r2_joint = r2_j
            result.pooled_delta_r2 = r2_j - r2_p

    # Pooled SASA analysis (burial baseline)
    if "sasa_z" in pooled.columns:
        valid_sasa = pooled.dropna(subset=["sasa_z", "rmsf_avg_z"])
        if len(valid_sasa) >= 20:
            rho, _ = scipy_stats.spearmanr(valid_sasa["sasa_z"], valid_sasa["rmsf_avg_z"])
            result.pooled_rho_sasa_rmsf = rho
            r, _ = scipy_stats.pearsonr(valid_sasa["sasa_z"], valid_sasa["rmsf_avg_z"])
            result.pooled_r2_sasa_rmsf = r ** 2

        # Pooled partial correlation: robustness vs RMSF | SASA
        valid_all = pooled.dropna(subset=[rob_col_z, "sasa_z", "rmsf_avg_z"])
        if len(valid_all) >= 20:
            pr, _ = partial_spearman(
                valid_all[rob_col_z].values,
                valid_all["rmsf_avg_z"].values,
                valid_all["sasa_z"].values,
            )
            result.pooled_rho_robustness_rmsf_partial_sasa = pr

        # Pooled partial correlation: robustness vs RMSF | pLDDT
        if "plddt_z" in pooled.columns:
            valid_rp = pooled.dropna(subset=[rob_col_z, "rmsf_avg_z", "plddt_z"])
            if len(valid_rp) >= 20:
                pr, _ = partial_spearman(
                    valid_rp[rob_col_z].values,
                    valid_rp["rmsf_avg_z"].values,
                    valid_rp["plddt_z"].values,
                )
                result.pooled_rho_robustness_rmsf_partial_plddt = pr

        # Pooled regression: RMSF ~ robustness + SASA
        if len(valid_all) >= 20:
            _, r2_j, dr2 = regression_delta_r2(
                valid_all[["sasa_z"]].values,
                valid_all[[rob_col_z, "sasa_z"]].values,
                valid_all["rmsf_avg_z"].values,
            )
            result.pooled_r2_joint_sasa = r2_j
            result.pooled_delta_r2_over_sasa = dr2

    # Pooled B-factor analysis (predicting RMSF from B-factor)
    if "bfactor_z" in pooled.columns:
        valid_bf = pooled.dropna(subset=["bfactor_z", "rmsf_avg_z"])
        if len(valid_bf) >= 20:
            rho, _ = scipy_stats.spearmanr(valid_bf["bfactor_z"], valid_bf["rmsf_avg_z"])
            result.pooled_rho_bfactor_rmsf = rho
            r, _ = scipy_stats.pearsonr(valid_bf["bfactor_z"], valid_bf["rmsf_avg_z"])
            result.pooled_r2_bfactor_rmsf = r ** 2

        # Robustness vs B-factor (cross-predictor correlation)
        valid_rob_bf = pooled.dropna(subset=[rob_col_z, "bfactor_z"])
        if len(valid_rob_bf) >= 20:
            rho, _ = scipy_stats.spearmanr(valid_rob_bf[rob_col_z], valid_rob_bf["bfactor_z"])
            result.pooled_rho_robustness_bfactor = rho

        # Joint regression: RMSF ~ robustness + B-factor
        valid_all_bf = pooled.dropna(subset=[rob_col_z, "bfactor_z", "rmsf_avg_z"])
        if len(valid_all_bf) >= 20:
            _, r2_j, dr2 = regression_delta_r2(
                valid_all_bf[["bfactor_z"]].values,
                valid_all_bf[[rob_col_z, "bfactor_z"]].values,
                valid_all_bf["rmsf_avg_z"].values,
            )
            result.pooled_r2_joint_bfactor = r2_j
            result.pooled_delta_r2_over_bfactor = dr2

    # =================================================================
    # POOLED B-FACTOR AS TARGET (predicting experimental dynamics)
    # =================================================================
    if "bfactor_z" in pooled.columns:
        valid_bf_t = pooled.dropna(subset=[rob_col_z, "bfactor_z"])
        if len(valid_bf_t) >= 20:
            # Robustness → B-factor
            rho, _ = scipy_stats.spearmanr(valid_bf_t[rob_col_z], valid_bf_t["bfactor_z"])
            result.pooled_rho_robustness_bfactor_target = rho
            r, _ = scipy_stats.pearsonr(valid_bf_t[rob_col_z], valid_bf_t["bfactor_z"])
            result.pooled_r2_robustness_bfactor_target = r ** 2

        # pLDDT → B-factor (baseline)
        if "plddt_z" in pooled.columns:
            valid_pb = pooled.dropna(subset=["plddt_z", "bfactor_z"])
            if len(valid_pb) >= 20:
                rho, _ = scipy_stats.spearmanr(valid_pb["plddt_z"], valid_pb["bfactor_z"])
                result.pooled_rho_plddt_bfactor = rho
                r, _ = scipy_stats.pearsonr(valid_pb["plddt_z"], valid_pb["bfactor_z"])
                result.pooled_r2_plddt_bfactor = r ** 2

        # SASA → B-factor
        if "sasa_z" in pooled.columns:
            valid_sb = pooled.dropna(subset=["sasa_z", "bfactor_z"])
            if len(valid_sb) >= 20:
                rho, _ = scipy_stats.spearmanr(valid_sb["sasa_z"], valid_sb["bfactor_z"])
                result.pooled_rho_sasa_bfactor = rho
                r, _ = scipy_stats.pearsonr(valid_sb["sasa_z"], valid_sb["bfactor_z"])
                result.pooled_r2_sasa_bfactor = r ** 2

        # Partial: robustness vs B-factor | SASA
        if "sasa_z" in pooled.columns:
            valid_rbs = pooled.dropna(subset=[rob_col_z, "bfactor_z", "sasa_z"])
            if len(valid_rbs) >= 20:
                pr, _ = partial_spearman(
                    valid_rbs[rob_col_z].values,
                    valid_rbs["bfactor_z"].values,
                    valid_rbs["sasa_z"].values,
                )
                result.pooled_rho_robustness_bfactor_partial_sasa = pr

        # Partial: robustness vs B-factor | pLDDT
        if "plddt_z" in pooled.columns:
            valid_rbp = pooled.dropna(subset=[rob_col_z, "bfactor_z", "plddt_z"])
            if len(valid_rbp) >= 20:
                pr, _ = partial_spearman(
                    valid_rbp[rob_col_z].values,
                    valid_rbp["bfactor_z"].values,
                    valid_rbp["plddt_z"].values,
                )
                result.pooled_rho_robustness_bfactor_partial_plddt = pr

        # Regression: B-factor ~ robustness + pLDDT
        if "plddt_z" in pooled.columns:
            valid_bfp = pooled.dropna(subset=[rob_col_z, "plddt_z", "bfactor_z"])
            if len(valid_bfp) >= 20:
                _, r2_j, dr2 = regression_delta_r2(
                    valid_bfp[["plddt_z"]].values,
                    valid_bfp[[rob_col_z, "plddt_z"]].values,
                    valid_bfp["bfactor_z"].values,
                )
                result.pooled_r2_bfactor_joint_plddt = r2_j
                result.pooled_delta_r2_bfactor_over_plddt = dr2

        # Regression: B-factor ~ robustness + SASA
        if "sasa_z" in pooled.columns:
            valid_bfs = pooled.dropna(subset=[rob_col_z, "sasa_z", "bfactor_z"])
            if len(valid_bfs) >= 20:
                _, r2_j, dr2 = regression_delta_r2(
                    valid_bfs[["sasa_z"]].values,
                    valid_bfs[[rob_col_z, "sasa_z"]].values,
                    valid_bfs["bfactor_z"].values,
                )
                result.pooled_r2_bfactor_joint_sasa = r2_j
                result.pooled_delta_r2_bfactor_over_sasa = dr2

        # Conservation → B-factor
        if "conservation_z" in pooled.columns:
            valid_cb = pooled.dropna(subset=["conservation_z", "bfactor_z"])
            if len(valid_cb) >= 20:
                rho, _ = scipy_stats.spearmanr(valid_cb["conservation_z"],
                                               valid_cb["bfactor_z"])
                result.pooled_rho_conservation_bfactor = rho
                r, _ = scipy_stats.pearsonr(valid_cb["conservation_z"],
                                            valid_cb["bfactor_z"])
                result.pooled_r2_conservation_bfactor = r ** 2

            # Partial: robustness vs B-factor | conservation
            valid_rbc = pooled.dropna(subset=[rob_col_z, "bfactor_z",
                                              "conservation_z"])
            if len(valid_rbc) >= 20:
                pr, _ = partial_spearman(
                    valid_rbc[rob_col_z].values,
                    valid_rbc["bfactor_z"].values,
                    valid_rbc["conservation_z"].values,
                )
                result.pooled_rho_robustness_bfactor_partial_conservation = pr

                # Regression: B-factor ~ robustness + conservation
                _, r2_j, dr2 = regression_delta_r2(
                    valid_rbc[["conservation_z"]].values,
                    valid_rbc[[rob_col_z, "conservation_z"]].values,
                    valid_rbc["bfactor_z"].values,
                )
                result.pooled_r2_bfactor_joint_conservation = r2_j
                result.pooled_delta_r2_bfactor_over_conservation = dr2

    # =================================================================
    # POOLED CONSERVATION ANALYSIS (RMSF target)
    # =================================================================
    if "conservation_z" in pooled.columns:
        valid_cons = pooled.dropna(subset=["conservation_z", "rmsf_avg_z"])
        if len(valid_cons) >= 20:
            rho, _ = scipy_stats.spearmanr(valid_cons["conservation_z"],
                                           valid_cons["rmsf_avg_z"])
            result.pooled_rho_conservation_rmsf = rho
            r, _ = scipy_stats.pearsonr(valid_cons["conservation_z"],
                                        valid_cons["rmsf_avg_z"])
            result.pooled_r2_conservation_rmsf = r ** 2

        # Robustness vs Conservation (collinearity)
        valid_rc = pooled.dropna(subset=[rob_col_z, "conservation_z"])
        if len(valid_rc) >= 20:
            rho, _ = scipy_stats.spearmanr(valid_rc[rob_col_z],
                                           valid_rc["conservation_z"])
            result.pooled_rho_robustness_conservation = rho

        # Partial: robustness vs RMSF | conservation
        valid_rrc = pooled.dropna(subset=[rob_col_z, "rmsf_avg_z",
                                          "conservation_z"])
        if len(valid_rrc) >= 20:
            pr, _ = partial_spearman(
                valid_rrc[rob_col_z].values,
                valid_rrc["rmsf_avg_z"].values,
                valid_rrc["conservation_z"].values,
            )
            result.pooled_rho_robustness_rmsf_partial_conservation = pr

            # Regression: RMSF ~ robustness + conservation
            _, r2_j, dr2 = regression_delta_r2(
                valid_rrc[["conservation_z"]].values,
                valid_rrc[[rob_col_z, "conservation_z"]].values,
                valid_rrc["rmsf_avg_z"].values,
            )
            result.pooled_r2_joint_conservation = r2_j
            result.pooled_delta_r2_over_conservation = dr2

    return result


# ==========================================================================
# STRATIFIED ANALYSIS
# ==========================================================================

def run_stratified_analysis(
    per_protein_data: List[Tuple[pd.DataFrame, str]],
    stratify_col: str,
    rob_col: str = "mean_abs_ddg",
) -> Dict[str, Dict[str, float]]:
    """Run correlation analysis stratified by a categorical column.

    Computes correlations for both RMSF-as-target and B-factor-as-target.

    Args:
        per_protein_data: list of (merged_df, protein_id)
        stratify_col: column name to stratify by (e.g., "ss", "burial_class")

    Returns:
        Dict mapping category -> {rho_robustness_rmsf, rho_plddt_rmsf,
                                   rho_robustness_bfactor, rho_plddt_bfactor, ...}
    """
    all_rows = []
    for merged_df, pid in per_protein_data:
        df = merged_df.dropna(subset=[rob_col, "rmsf_avg"]).copy()
        if stratify_col not in df.columns or len(df) < 5:
            continue
        for col in [rob_col, "rmsf_avg", "plddt", "bfactor", "sasa"]:
            if col in df.columns:
                mu, sigma = df[col].mean(), df[col].std()
                df[f"{col}_z"] = (df[col] - mu) / sigma if sigma > 0 else 0.0
        all_rows.append(df)

    if not all_rows:
        return {}

    pooled = pd.concat(all_rows, ignore_index=True)
    rob_col_z = f"{rob_col}_z"
    results = {}

    for cat, group in pooled.groupby(stratify_col):
        if len(group) < 20:
            continue
        entry = {"n_residues": len(group)}

        # --- RMSF as target ---
        valid = group.dropna(subset=[rob_col_z, "rmsf_avg_z"])
        if len(valid) >= 20:
            rho, pval = scipy_stats.spearmanr(valid[rob_col_z], valid["rmsf_avg_z"])
            entry["rho_robustness_rmsf"] = rho
            entry["pval_robustness_rmsf"] = pval

        if "plddt_z" in group.columns:
            valid_p = group.dropna(subset=["plddt_z", "rmsf_avg_z"])
            if len(valid_p) >= 20:
                rho, _ = scipy_stats.spearmanr(valid_p["plddt_z"], valid_p["rmsf_avg_z"])
                entry["rho_plddt_rmsf"] = rho

        # --- B-factor as target ---
        if "bfactor_z" in group.columns:
            valid_rb = group.dropna(subset=[rob_col_z, "bfactor_z"])
            if len(valid_rb) >= 20:
                rho, _ = scipy_stats.spearmanr(valid_rb[rob_col_z], valid_rb["bfactor_z"])
                entry["rho_robustness_bfactor"] = rho

            if "plddt_z" in group.columns:
                valid_pb = group.dropna(subset=["plddt_z", "bfactor_z"])
                if len(valid_pb) >= 20:
                    rho, _ = scipy_stats.spearmanr(valid_pb["plddt_z"], valid_pb["bfactor_z"])
                    entry["rho_plddt_bfactor"] = rho

            if "sasa_z" in group.columns:
                valid_sb = group.dropna(subset=["sasa_z", "bfactor_z"])
                if len(valid_sb) >= 20:
                    rho, _ = scipy_stats.spearmanr(valid_sb["sasa_z"], valid_sb["bfactor_z"])
                    entry["rho_sasa_bfactor"] = rho

        results[str(cat)] = entry

    return results


# ==========================================================================
# FIGURE GENERATION
# ==========================================================================

def generate_figures(
    per_protein_results: List[PerProteinResult],
    per_protein_data: List[Tuple[pd.DataFrame, str]],
    pooled_result: PooledResult,
    stratified_ss: Dict,
    stratified_burial: Dict,
    output_dir: str,
    scorer: str,
    rob_col: str = "mean_abs_ddg",
    dataset_name: str = "",
):
    """Generate publication figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available, skipping figures")
        return

    fig_dir = Path(output_dir) / scorer / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Build a filename prefix: {dataset}_{scorer}_{rob_col}_
    rob_suffix = f"_{rob_col}" if rob_col != "mean_abs_ddg" else ""
    prefix = f"{dataset_name}_{scorer}{rob_suffix}" if dataset_name else f"{scorer}{rob_suffix}"

    # --- Fig A: Distribution of per-protein rho (robustness vs target) ---
    # Use B-factor-as-target fields if RMSF fields are all NaN
    rhos_rob = [r.rho_robustness_rmsf for r in per_protein_results
                if not np.isnan(r.rho_robustness_rmsf)]
    rhos_plddt = [r.rho_plddt_rmsf for r in per_protein_results
                  if not np.isnan(r.rho_plddt_rmsf)]
    target_label = "RMSF"
    if not rhos_rob:
        # Fall back to B-factor target fields
        rhos_rob = [r.rho_robustness_bfactor_target for r in per_protein_results
                    if not np.isnan(r.rho_robustness_bfactor_target)]
        rhos_plddt = [r.rho_plddt_bfactor for r in per_protein_results
                      if not np.isnan(r.rho_plddt_bfactor)]
        target_label = "B-factor"

    if rhos_rob:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram of rho values
        ax = axes[0]
        ax.hist(rhos_rob, bins=40, alpha=0.7, label=f"Robustness ({scorer})",
                color="steelblue", edgecolor="black", linewidth=0.5)
        if rhos_plddt:
            ax.hist(rhos_plddt, bins=40, alpha=0.5, label="pLDDT",
                    color="coral", edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel(f"Spearman rho (predictor vs. {target_label})")
        ax.set_ylabel("Number of proteins")
        ax.set_title("Per-protein correlation with dynamics")
        ax.legend()

        # Scatter: rho_robustness vs rho_plddt
        ax = axes[1]
        if rhos_plddt:
            if target_label == "B-factor":
                both = [(r.rho_robustness_bfactor_target, r.rho_plddt_bfactor)
                        for r in per_protein_results
                        if not np.isnan(r.rho_robustness_bfactor_target)
                        and not np.isnan(r.rho_plddt_bfactor)]
            else:
                both = [(r.rho_robustness_rmsf, r.rho_plddt_rmsf)
                        for r in per_protein_results
                        if not np.isnan(r.rho_robustness_rmsf)
                        and not np.isnan(r.rho_plddt_rmsf)]
            if both:
                x, y = zip(*both)
                ax.scatter(x, y, alpha=0.3, s=15, color="steelblue")
                lim = max(abs(min(min(x), min(y))), abs(max(max(x), max(y)))) + 0.1
                ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, label="y=x")
                ax.set_xlabel(f"rho (robustness vs {target_label})")
                ax.set_ylabel(f"rho (pLDDT vs {target_label})")
                ax.set_title("Robustness vs pLDDT as dynamics predictors")
                ax.legend()

        plt.tight_layout()
        plt.savefig(fig_dir / f"{prefix}_per_protein_correlations.png", dpi=150)
        plt.close()

    # --- Fig B: Pooled scatter (z-scored) ---
    rob_col_z = f"{rob_col}_z"
    all_rows = []
    for merged_df, pid in per_protein_data:
        df = merged_df.dropna(subset=[rob_col, "rmsf_avg"]).copy()
        if len(df) < 10:
            continue
        for col in [rob_col, "rmsf_avg"]:
            mu, sigma = df[col].mean(), df[col].std()
            df[f"{col}_z"] = (df[col] - mu) / sigma if sigma > 0 else 0.0
        all_rows.append(df)

    if all_rows:
        pooled_df = pd.concat(all_rows, ignore_index=True)
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))

        # Subsample for plotting if too many points
        if len(pooled_df) > 10000:
            plot_df = pooled_df.sample(10000, random_state=42)
        else:
            plot_df = pooled_df

        ax.scatter(plot_df[rob_col_z], plot_df["rmsf_avg_z"],
                   alpha=0.05, s=3, color="steelblue")
        ax.set_xlabel(f"Per-residue robustness ({rob_col}, z-scored)")
        ax.set_ylabel("RMSF from MD (z-scored)")
        ax.set_title(f"Pooled: robustness vs dynamics (N={len(pooled_df):,} residues, "
                     f"{pooled_result.n_proteins} proteins)\n"
                     f"rho={pooled_result.pooled_rho_robustness_rmsf:.3f}")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)

        plt.tight_layout()
        plt.savefig(fig_dir / f"{prefix}_pooled_scatter.png", dpi=150)
        plt.close()

    # --- Fig C: Stratified bar chart ---
    if stratified_ss:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        categories = sorted(stratified_ss.keys())
        labels = {"H": "Helix", "E": "Sheet", "C": "Coil"}
        x_labels = [labels.get(c, c) for c in categories]

        rho_rob = [stratified_ss[c].get("rho_robustness_rmsf", 0) for c in categories]
        rho_pld = [stratified_ss[c].get("rho_plddt_rmsf", 0) for c in categories]
        x = np.arange(len(categories))
        w = 0.35

        ax.bar(x - w/2, rho_rob, w, label=f"Robustness ({scorer})",
               color="steelblue")
        ax.bar(x + w/2, rho_pld, w, label="pLDDT", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("Spearman rho with RMSF")
        ax.set_title("Correlation with dynamics by secondary structure")
        ax.legend()
        ax.axhline(0, color="black", linewidth=0.5)

        plt.tight_layout()
        plt.savefig(fig_dir / f"{prefix}_stratified_ss.png", dpi=150)
        plt.close()

    print(f"Figures saved to {fig_dir}")


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Correlate mutational robustness with protein dynamics (RMSF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--atlas_dir", type=str, required=True,
                        help="ATLAS download directory (from download_atlas.py)")
    parser.add_argument("--robustness_dir", type=str, required=True,
                        help="Robustness output directory (from compute_robustness.py)")
    parser.add_argument("--scorer", type=str, nargs="+", default=["esm1v"],
                        help="Scorer name(s) to analyze (default: esm1v)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for analysis results")
    parser.add_argument("--no_figures", action="store_true",
                        help="Skip figure generation")
    parser.add_argument("--no_dssp", action="store_true",
                        help="Skip DSSP-based secondary structure assignment")
    parser.add_argument("--no_sasa", action="store_true",
                        help="Skip SASA computation (avoids mdtraj crashes "
                             "on raw PDB files with overlapping atoms)")
    parser.add_argument("--max_proteins", type=int, default=0,
                        help="Limit number of proteins (0=all, for testing)")
    parser.add_argument("--max_seq_length", type=int, default=0,
                        help="Skip proteins with length >= this (0=no limit, "
                             "1024=exclude >=1024 to match ESM-1v limit)")
    parser.add_argument("--robustness_col", type=str, default="std_ddg",
                        help="Column name for the per-residue robustness index "
                             "(default: mean_abs_ddg). Other options: std_ddg, "
                             "max_ddg, frac_destabilizing, frac_neutral")
    parser.add_argument("--target", type=str, default="rmsf",
                        choices=["rmsf", "bfactor"],
                        help="Primary dynamics target column (default: rmsf). "
                             "Use 'bfactor' for datasets without MD trajectories "
                             "(e.g. PDB de novo designs with only crystal B-factors).")
    parser.add_argument("--transform", type=str, default="none",
                        choices=["none", "log1p"],
                        help="Optional transform applied to the response variable "
                             "(RMSF or B-factor) before z-scoring for pooled OLS R². "
                             "'log1p' applies log(1+x) to reduce heavy-tail effects. "
                             "Does not affect Spearman correlations (rank-invariant). "
                             "(default: none)")
    parser.add_argument("--consurf_dir", type=str, default=None,
                        help="Path to ConSurf-DB directory containing JSON files "
                             "and identical_to_unique_dict.txt")
    parser.add_argument("--exclude", type=str, nargs="*", default=None,
                        help="Protein IDs to exclude (e.g. 2GAR_A 3F4M_A)")
    args = parser.parse_args()

    exclude_set = set(args.exclude) if args.exclude else set()

    for scorer in args.scorer:
        print(f"\n{'='*60}")
        print(f"ANALYSIS: scorer = {scorer}, robustness_col = {args.robustness_col}, "
              f"target = {args.target}")
        print(f"{'='*60}")
        run_analysis_for_scorer(
            atlas_dir=args.atlas_dir,
            robustness_dir=args.robustness_dir,
            scorer=scorer,
            output_dir=args.output_dir,
            make_figures=not args.no_figures,
            use_dssp=not args.no_dssp,
            compute_sasa=not args.no_sasa,
            max_proteins=args.max_proteins,
            max_seq_length=args.max_seq_length,
            robustness_col=args.robustness_col,
            target=args.target,
            transform=args.transform,
            consurf_dir=args.consurf_dir,
            exclude_proteins=exclude_set,
        )


def run_analysis_for_scorer(
    atlas_dir: str,
    robustness_dir: str,
    scorer: str,
    output_dir: str,
    make_figures: bool = True,
    use_dssp: bool = True,
    compute_sasa: bool = True,
    max_proteins: int = 0,
    max_seq_length: int = 0,
    robustness_col: str = "std_ddg",
    target: str = "rmsf",
    transform: str = "none",
    consurf_dir: Optional[str] = None,
    exclude_proteins: Optional[set] = None,
):
    """Run the full correlation analysis for one scorer."""
    rob_col = robustness_col  # short alias used throughout
    bfactor_only = (target == "bfactor")
    dataset_name = Path(atlas_dir).name  # e.g. "atlas", "pdb_designs", "bbflow_processed"
    out_dir = Path(output_dir) / scorer
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find proteins that have both data and robustness data
    atlas_proteins_dir = Path(atlas_dir) / "proteins"
    dataset_label = dataset_name  # alias used in print statements
    if not atlas_proteins_dir.exists():
        print(f"ERROR: proteins dir not found: {atlas_proteins_dir}")
        return

    protein_ids = sorted([
        d.name for d in atlas_proteins_dir.iterdir()
        if d.is_dir() and (d / ".done").exists()
    ])

    # Apply exclusion filter
    if exclude_proteins:
        n_before = len(protein_ids)
        protein_ids = [p for p in protein_ids if p not in exclude_proteins]
        n_excluded = n_before - len(protein_ids)
        if n_excluded > 0:
            print(f"Excluded {n_excluded} proteins from {dataset_label}")

    if max_proteins > 0:
        protein_ids = protein_ids[:max_proteins]

    if max_seq_length > 0:
        print(f"Found {len(protein_ids)} proteins in {dataset_label} "
              f"(max_seq_length={max_seq_length})")
    else:
        print(f"Found {len(protein_ids)} proteins in {dataset_label}")

    # Process each protein
    per_protein_results = []
    per_protein_data = []  # for pooled analysis
    n_skip_no_robustness = 0
    n_skip_no_rmsf = 0
    n_skip_too_short = 0
    n_skip_too_long = 0

    for idx, pid in enumerate(protein_ids):
        protein_dir = str(atlas_proteins_dir / pid)

        # Load robustness
        rob_df = load_robustness(robustness_dir, scorer, pid)
        if rob_df is None:
            n_skip_no_robustness += 1
            continue

        # Skip proteins at or exceeding max sequence length (e.g. ESM-1v 1024 limit)
        if max_seq_length > 0 and len(rob_df) >= max_seq_length:
            n_skip_too_long += 1
            continue

        # Load RMSF (required unless --target bfactor)
        rmsf_df = load_atlas_rmsf(protein_dir)
        if rmsf_df is None and not bfactor_only:
            n_skip_no_rmsf += 1
            continue

        # Load pLDDT and B-factor
        plddt_df = load_atlas_pldt(protein_dir)
        bfactor_df = load_atlas_bfactor(protein_dir)

        # In bfactor_only mode, B-factor is required
        if bfactor_only and bfactor_df is None:
            n_skip_no_rmsf += 1  # reuse counter
            continue

        # In bfactor_only mode, synthesize a fake rmsf_df from bfactor
        # so the downstream code (which keys on "rmsf_avg") works unchanged.
        if bfactor_only and rmsf_df is None:
            rmsf_df = bfactor_df.rename(columns={"bfactor": "rmsf_avg"}).copy()
        global_metrics = load_robustness_global(robustness_dir, scorer, pid)

        # Compute SASA from PDB (uniform burial baseline for all proteins)
        sasa_df = None
        if compute_sasa:
            pdb_files = list(Path(protein_dir).glob("*.pdb"))
            if pdb_files:
                try:
                    sasa_df = compute_sasa_from_pdb(str(pdb_files[0]))
                except Exception as e:
                    print(f"  Warning: SASA computation failed for {pid}: {e}")

        # Load conservation scores (if available)
        conservation_df = load_conservation(protein_dir, consurf_dir=consurf_dir,
                                            protein_id=pid)

        # Correlate
        result = correlate_single_protein(
            pid, rob_df, rmsf_df, plddt_df, bfactor_df, global_metrics, scorer,
            sasa_df=sasa_df, conservation_df=conservation_df, rob_col=rob_col,
        )
        if result is None:
            n_skip_too_short += 1
            continue

        per_protein_results.append(result)

        # Build merged DataFrame for pooled analysis
        rob_merge_cols = ["position"]
        for extra in ["mean_abs_ddg", "mean_ddg", "frac_destabilizing",
                       "frac_neutral", "std_ddg", "max_ddg"]:
            if extra in rob_df.columns:
                rob_merge_cols.append(extra)
        merged = rob_df[rob_merge_cols].copy()
        merged = merged.merge(rmsf_df[["position", "rmsf_avg"]], on="position", how="inner")
        if plddt_df is not None:
            merged = merged.merge(plddt_df[["position", "plddt"]], on="position", how="left")
        if bfactor_df is not None:
            merged = merged.merge(bfactor_df[["position", "bfactor"]], on="position", how="left")
        if sasa_df is not None:
            merged = merged.merge(sasa_df[["position", "sasa"]], on="position", how="left")
        if conservation_df is not None:
            merged = merged.merge(conservation_df[["position", "conservation"]],
                                  on="position", how="left")

        # Secondary structure and burial (if enabled)
        if use_dssp:
            pdb_files = list(Path(protein_dir).glob("*.pdb"))
            if pdb_files:
                try:
                    ss = assign_secondary_structure(str(pdb_files[0]))
                    if ss:
                        n_merged = len(merged)
                        n_ss = len(ss)
                        if n_ss == n_merged:
                            merged["ss"] = ss
                        elif n_ss > n_merged:
                            # DSSP sees more residues (e.g. HETATM
                            # modified residues); truncate to match.
                            merged["ss"] = ss[:n_merged]
                        else:
                            # DSSP sees fewer residues; assign what we
                            # can, leave the rest as NaN (excluded from
                            # stratification).
                            merged["ss"] = pd.Series(
                                ss + [None] * (n_merged - n_ss),
                                index=merged.index,
                            )
                    elif idx == 0:
                        # Log once if SS assignment returned None
                        print(f"  Note: SS assignment returned None for "
                              f"first protein {pid} (mkdssp may not be "
                              f"installed)")
                except Exception as e:
                    print(f"  Warning: SS assignment failed for {pid}: {e}")
                try:
                    burial = compute_burial(str(pdb_files[0]))
                    if burial and len(burial) == len(merged):
                        merged["rsa"] = burial
                        merged["burial_class"] = pd.cut(
                            merged["rsa"], bins=[0, 0.2, 0.5, 1.0],
                            labels=["core", "boundary", "surface"]
                        )
                    elif "sasa" in merged.columns and merged["sasa"].notna().sum() >= 10:
                        # Fallback: SASA-based burial (quantile terciles)
                        terciles = merged["sasa"].quantile([1/3, 2/3])
                        merged["burial_class"] = pd.cut(
                            merged["sasa"],
                            bins=[-np.inf, terciles.iloc[0], terciles.iloc[1], np.inf],
                            labels=["core", "boundary", "surface"]
                        )
                except Exception as e:
                    print(f"  Warning: Burial assignment failed for {pid}: {e}")

        per_protein_data.append((merged, pid))

        if (idx + 1) % 200 == 0:
            print(f"  [{idx+1}/{len(protein_ids)}] {len(per_protein_results)} processed")

    print(f"\nProcessed: {len(per_protein_results)} proteins")
    print(f"Skipped: {n_skip_no_robustness} no robustness, "
          f"{n_skip_no_rmsf} no RMSF, {n_skip_too_short} too short, "
          f"{n_skip_too_long} too long (>={max_seq_length})")

    if not per_protein_results:
        print("No proteins to analyze!")
        return

    # --- File suffix for robustness column ---
    rob_suffix = f"_{rob_col}" if rob_col != "mean_abs_ddg" else ""

    # --- Save per-protein results ---
    results_df = pd.DataFrame([asdict(r) for r in per_protein_results])
    pp_fname = f"per_protein_correlations{rob_suffix}.tsv"
    results_df.to_csv(out_dir / pp_fname, sep="\t", index=False)
    print(f"Per-protein results: {out_dir / pp_fname}")

    # --- Pooled analysis ---
    pooled = run_pooled_analysis(per_protein_data, per_protein_results, scorer,
                                 rob_col=rob_col, transform=transform)
    pooled_fname = f"pooled_results{rob_suffix}.json"
    with open(out_dir / pooled_fname, "w") as f:
        json.dump(asdict(pooled), f, indent=2, default=_json_default)
    print(f"Pooled results: {out_dir / pooled_fname}")

    # --- Stratified analysis ---
    strat_ss = run_stratified_analysis(per_protein_data, "ss", rob_col=rob_col)
    strat_burial = run_stratified_analysis(per_protein_data, "burial_class",
                                            rob_col=rob_col)
    stratified = {"secondary_structure": strat_ss, "burial": strat_burial}
    strat_fname = f"stratified_results{rob_suffix}.json"
    with open(out_dir / strat_fname, "w") as f:
        json.dump(stratified, f, indent=2, default=_json_default)
    print(f"Stratified results: {out_dir / strat_fname}")

    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f"SUMMARY ({scorer})")
    print(f"{'='*60}")
    print(f"Proteins analyzed:  {pooled.n_proteins}")
    print(f"Residues pooled:    {pooled.n_residues:,}")
    print(f"")
    print(f"Per-protein median rho (robustness vs RMSF): "
          f"{pooled.median_rho_robustness_rmsf:.3f}")
    print(f"Per-protein median rho (pLDDT vs RMSF):      "
          f"{pooled.median_rho_plddt_rmsf:.3f}")
    print(f"Per-protein median rho (SASA vs RMSF):       "
          f"{pooled.median_rho_sasa_rmsf:.3f}")
    if not np.isnan(pooled.median_rho_bfactor_rmsf):
        print(f"Per-protein median rho (Bfactor vs RMSF):   "
              f"{pooled.median_rho_bfactor_rmsf:.3f}")
        print(f"Per-protein median rho (rob vs Bfactor):    "
              f"{pooled.median_rho_robustness_bfactor:.3f}")
    print(f"Per-protein median partial rho (rob|SASA):   "
          f"{pooled.median_rho_partial_sasa:.3f}")
    if not np.isnan(pooled.median_rho_partial_plddt):
        print(f"Per-protein median partial rho (rob|pLDDT):  "
              f"{pooled.median_rho_partial_plddt:.3f}")
    print(f"Frac where |rho_robustness| > |rho_pLDDT|:   "
          f"{pooled.frac_robustness_beats_plddt:.3f}")

    # Alternative robustness measures summary
    alt_measures = {
        "frac_destab": "rho_frac_destab_rmsf",
        "frac_neutral": "rho_frac_neutral_rmsf",
        "std_ddg": "rho_std_ddg_rmsf",
        "max_ddg": "rho_max_ddg_rmsf",
        "mean_ddg": "rho_mean_ddg_rmsf",
    }
    alt_medians = {}
    for label, attr in alt_measures.items():
        vals = [getattr(r, attr) for r in per_protein_results
                if not np.isnan(getattr(r, attr))]
        if vals:
            alt_medians[label] = np.median(vals)
    if alt_medians:
        print(f"\nAlternative robustness measures (median rho vs RMSF):")
        for label, med in sorted(alt_medians.items(), key=lambda x: -abs(x[1])):
            print(f"  {label:20s}: {med:.3f}")
    print(f"")
    print(f"Pooled rho (robustness vs RMSF):  {pooled.pooled_rho_robustness_rmsf:.3f} "
          f"(p={pooled.pooled_pval_robustness_rmsf:.2e})")
    print(f"Pooled rho (pLDDT vs RMSF):       {pooled.pooled_rho_plddt_rmsf:.3f}")
    print(f"Pooled rho (SASA vs RMSF):        {pooled.pooled_rho_sasa_rmsf:.3f}")
    print(f"Pooled partial rho (rob|SASA):    {pooled.pooled_rho_robustness_rmsf_partial_sasa:.3f}")
    if not np.isnan(pooled.pooled_rho_robustness_rmsf_partial_plddt):
        print(f"Pooled partial rho (rob|pLDDT):   {pooled.pooled_rho_robustness_rmsf_partial_plddt:.3f}")
    print(f"Pooled R^2 (robustness):           {pooled.pooled_r2_robustness_rmsf:.3f}")
    print(f"Pooled R^2 (pLDDT):                {pooled.pooled_r2_plddt_rmsf:.3f}")
    print(f"Pooled R^2 (SASA):                 {pooled.pooled_r2_sasa_rmsf:.3f}")
    print(f"Pooled R^2 (rob+pLDDT):            {pooled.pooled_r2_joint:.3f}")
    print(f"Pooled R^2 (rob+SASA):             {pooled.pooled_r2_joint_sasa:.3f}")
    print(f"Delta R^2 (rob+pLDDT - pLDDT):     {pooled.pooled_delta_r2:.3f}")
    print(f"Delta R^2 (rob+SASA - SASA):       {pooled.pooled_delta_r2_over_sasa:.3f}")
    if not np.isnan(pooled.pooled_rho_bfactor_rmsf):
        print(f"Pooled rho (Bfactor vs RMSF):      {pooled.pooled_rho_bfactor_rmsf:.3f}")
        print(f"Pooled rho (rob vs Bfactor):       {pooled.pooled_rho_robustness_bfactor:.3f}")
        print(f"Pooled R^2 (Bfactor):               {pooled.pooled_r2_bfactor_rmsf:.3f}")
        print(f"Pooled R^2 (rob+Bfactor→RMSF):     {pooled.pooled_r2_joint_bfactor:.3f}")
        print(f"Delta R^2 (rob+Bfactor - Bfactor): {pooled.pooled_delta_r2_over_bfactor:.3f}")

    # === B-FACTOR AS TARGET ===
    if not np.isnan(pooled.pooled_rho_robustness_bfactor_target):
        print(f"\n{'='*60}")
        print(f"B-FACTOR AS TARGET (predicting experimental dynamics)")
        print(f"{'='*60}")
        print(f"Per-protein median rho (rob vs Bfactor):     "
              f"{pooled.median_rho_robustness_bfactor_target:.3f}")
        print(f"Per-protein median rho (pLDDT vs Bfactor):   "
              f"{pooled.median_rho_plddt_bfactor:.3f}")
        print(f"Per-protein median rho (SASA vs Bfactor):    "
              f"{pooled.median_rho_sasa_bfactor:.3f}")
        print(f"Per-protein median partial (rob|SASA→Bf):    "
              f"{pooled.median_rho_robustness_bfactor_partial_sasa:.3f}")
        print(f"Per-protein median partial (rob|pLDDT→Bf):   "
              f"{pooled.median_rho_robustness_bfactor_partial_plddt:.3f}")
        print(f"Frac |rho_rob| > |rho_pLDDT| (Bf target):   "
              f"{pooled.frac_robustness_beats_plddt_bfactor:.3f}")
        print(f"")
        print(f"Pooled rho (rob → Bfactor):         {pooled.pooled_rho_robustness_bfactor_target:.3f}")
        print(f"Pooled R^2 (rob → Bfactor):         {pooled.pooled_r2_robustness_bfactor_target:.3f}")
        print(f"Pooled rho (pLDDT → Bfactor):       {pooled.pooled_rho_plddt_bfactor:.3f}")
        print(f"Pooled R^2 (pLDDT → Bfactor):       {pooled.pooled_r2_plddt_bfactor:.3f}")
        print(f"Pooled rho (SASA → Bfactor):        {pooled.pooled_rho_sasa_bfactor:.3f}")
        print(f"Pooled R^2 (SASA → Bfactor):        {pooled.pooled_r2_sasa_bfactor:.3f}")
        print(f"Pooled partial (rob|SASA → Bf):     {pooled.pooled_rho_robustness_bfactor_partial_sasa:.3f}")
        print(f"Pooled partial (rob|pLDDT → Bf):    {pooled.pooled_rho_robustness_bfactor_partial_plddt:.3f}")
        print(f"Pooled R^2 (rob+pLDDT → Bfactor):   {pooled.pooled_r2_bfactor_joint_plddt:.3f}")
        print(f"Delta R^2 over pLDDT (Bf target):    {pooled.pooled_delta_r2_bfactor_over_plddt:.3f}")
        print(f"Pooled R^2 (rob+SASA → Bfactor):    {pooled.pooled_r2_bfactor_joint_sasa:.3f}")
        print(f"Delta R^2 over SASA (Bf target):     {pooled.pooled_delta_r2_bfactor_over_sasa:.3f}")

    if strat_ss:
        print(f"\nBy secondary structure (RMSF target):")
        for cat in sorted(strat_ss.keys()):
            d = strat_ss[cat]
            label = {"H": "Helix", "E": "Sheet", "C": "Coil"}.get(cat, cat)
            print(f"  {label:8s}: rho_rob={d.get('rho_robustness_rmsf', float('nan')):.3f}  "
                  f"rho_plddt={d.get('rho_plddt_rmsf', float('nan')):.3f}  "
                  f"n={d.get('n_residues', 0):,}")

        # Check if B-factor stratification data exists
        has_bf_strat = any("rho_robustness_bfactor" in d for d in strat_ss.values())
        if has_bf_strat:
            print(f"\nBy secondary structure (B-factor target):")
            for cat in sorted(strat_ss.keys()):
                d = strat_ss[cat]
                label = {"H": "Helix", "E": "Sheet", "C": "Coil"}.get(cat, cat)
                print(f"  {label:8s}: rho_rob={d.get('rho_robustness_bfactor', float('nan')):.3f}  "
                      f"rho_plddt={d.get('rho_plddt_bfactor', float('nan')):.3f}  "
                      f"n={d.get('n_residues', 0):,}")

    if strat_burial:
        has_bf_burial = any("rho_robustness_bfactor" in d for d in strat_burial.values())
        if has_bf_burial:
            print(f"\nBy burial (B-factor target):")
            for cat in sorted(strat_burial.keys()):
                d = strat_burial[cat]
                print(f"  {cat:10s}: rho_rob={d.get('rho_robustness_bfactor', float('nan')):.3f}  "
                      f"rho_plddt={d.get('rho_plddt_bfactor', float('nan')):.3f}  "
                      f"n={d.get('n_residues', 0):,}")

    # --- Figures ---
    if make_figures:
        generate_figures(
            per_protein_results, per_protein_data, pooled,
            strat_ss, strat_burial, output_dir, scorer,
            rob_col=rob_col, dataset_name=dataset_name,
        )


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    raise TypeError(f"Not JSON serializable: {type(obj)}")


if __name__ == "__main__":
    main()
