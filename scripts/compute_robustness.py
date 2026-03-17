#!/usr/bin/env python3
"""
Compute per-residue and global mutational robustness for protein sequences.

Modular design: any scorer that implements the DDGScorer interface can be
plugged in. Built-in scorers:
  - esm1v:      ESM-1v masked marginals (sequence-only, no structure needed)
  - thermompnn: ThermoMPNN (structure-conditioned, requires PDB)

Works with:
  - ATLAS proteins (reads PDB + sequence from ATLAS download directory)
  - Arbitrary PDB files
  - Arbitrary sequences (for sequence-only scorers)

Usage examples:
  # Single protein, ESM-1v (sequence-only):
  python compute_robustness.py --scorer esm1v --pdb_file 1ubq.pdb --output_dir results/

  # Single protein, ThermoMPNN (structure-conditioned):
  python compute_robustness.py --scorer thermompnn --pdb_file 1ubq.pdb --output_dir results/ \
      --thermompnn_dir /path/to/ThermoMPNN

  # Batch: all ATLAS proteins, ESM-1v:
  python compute_robustness.py --scorer esm1v \
      --atlas_dir /sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas \
      --output_dir /sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas_robustness --batch

  # Batch: all ATLAS proteins, ThermoMPNN (submit as SLURM array):
  python compute_robustness.py --scorer thermompnn \
      --atlas_dir /sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas \
      --output_dir /sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas_robustness --batch \
      --thermompnn_dir /sci/labs/orzuk/orzuk/github/ThermoMPNN --batch_start 0 --batch_end 100

  # From sequence string (no PDB needed, sequence-only scorers):
  python compute_robustness.py --scorer esm1v --sequence "MKTAYIAKQRQISFVK..." \
      --protein_id test_protein --output_dir results/

Output per protein:
  {protein_id}_ddg_matrix.npy     - L x 19 DDG matrix (NaN at WT positions removed)
  {protein_id}_robustness.json    - per-residue and global robustness metrics
  {protein_id}_robustness.tsv     - per-residue robustness profile (easy to merge with ATLAS)
"""

import os
import sys
import json
import argparse
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Standard amino acids (alphabetical, 20)
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
N_AA = len(AA_LIST)


# ==========================================================================
# SCORER INTERFACE
# ==========================================================================

class DDGScorer(ABC):
    """Abstract base class for DDG scoring methods.

    A scorer computes ddG(i, b) = score(mutant_with_b_at_i) - score(wildtype)
    for each position i and each substitution b != wildtype[i].

    Subclasses must implement:
      - name: str property
      - requires_structure: bool property
      - load_model(): initialize model/weights (called once)
      - score_sequence(seq): return a scalar score for the full sequence
        (lower = more stable, by convention)

    Optionally override:
      - compute_ddg_matrix(seq, pdb_path): for scorers that can compute the
        full L x 19 matrix more efficiently than L*19 individual calls
        (e.g., masked marginals, batch mode)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name for this scorer (e.g., 'esm1v', 'thermompnn')."""
        pass

    @property
    @abstractmethod
    def requires_structure(self) -> bool:
        """Whether this scorer needs a PDB structure (vs. sequence-only)."""
        pass

    @abstractmethod
    def load_model(self, device: str = "cuda"):
        """Load model weights. Called once before scoring."""
        pass

    @abstractmethod
    def score_sequence(self, seq: str, pdb_path: Optional[str] = None) -> float:
        """Score a full sequence. Lower = more stable (convention).
        For structure-conditioned scorers, pdb_path is required.
        """
        pass

    def compute_ddg_matrix(self, seq: str, pdb_path: Optional[str] = None,
                           chain_id: Optional[str] = None) -> np.ndarray:
        """Compute L x 19 DDG matrix by exhaustive single-point mutations.

        Returns:
            ddg_matrix: np.ndarray of shape (L, 19), where column j is the
                substitution to AA_LIST[j'] (skipping the wildtype AA,
                so columns are the 19 non-WT amino acids in AA_LIST order
                with the WT position removed).

        Override this method for scorers that have a faster approach
        (e.g., masked marginals don't need L*19 forward passes).
        """
        L = len(seq)
        # Use full 20-column matrix initially, then compress
        ddg_full = np.full((L, N_AA), np.nan, dtype=np.float32)

        wt_score = self.score_sequence(seq, pdb_path)

        for i in range(L):
            wt_aa = seq[i]
            for j, aa in enumerate(AA_LIST):
                if aa == wt_aa:
                    continue
                mut_seq = seq[:i] + aa + seq[i + 1:]
                try:
                    mut_score = self.score_sequence(mut_seq, pdb_path)
                    ddg_full[i, j] = mut_score - wt_score
                except Exception:
                    ddg_full[i, j] = np.nan

        # Compress to L x 19 (drop the WT column at each row)
        ddg_matrix = np.zeros((L, N_AA - 1), dtype=np.float32)
        for i in range(L):
            wt_idx = AA_TO_IDX.get(seq[i])
            row = ddg_full[i]
            # Remove the WT entry
            ddg_matrix[i] = np.concatenate([row[:wt_idx], row[wt_idx + 1:]])

        return ddg_matrix


# ==========================================================================
# BUILT-IN SCORERS
# ==========================================================================

class ESM1vScorer(DDGScorer):
    """ESM-1v masked marginals scorer (sequence-only).

    Computes DG as negative log-likelihood of the sequence under ESM-1v.
    DDG = DG(mutant) - DG(wildtype).

    Fast mode (default): uses masked marginals for the full L x 19 matrix
    in L forward passes (instead of L*19). This is the "free robustness"
    approach from the thesis.
    """

    def __init__(self, use_masked_marginals: bool = True):
        self._model = None
        self._alphabet = None
        self._batch_converter = None
        self._device = "cuda"
        self.use_masked_marginals = use_masked_marginals

    @property
    def name(self) -> str:
        return "esm1v"

    @property
    def requires_structure(self) -> bool:
        return False

    def load_model(self, device: str = "cuda"):
        import torch
        from esm import pretrained
        self._device = device
        self._model, self._alphabet = pretrained.esm1v_t33_650M_UR90S_1()
        self._model = self._model.eval().to(device)
        self._batch_converter = self._alphabet.get_batch_converter()

    def score_sequence(self, seq: str, pdb_path: Optional[str] = None) -> float:
        """Negative log-likelihood of sequence under ESM-1v (DG proxy)."""
        import torch
        data = [("protein", seq)]
        _, _, tokens = self._batch_converter(data)
        tokens = tokens.to(self._device)
        with torch.no_grad():
            logits = self._model(tokens)["logits"]
            log_probs = torch.log_softmax(logits, dim=-1)
            # Tokens layout: [CLS] aa1 aa2 ... aaL [EOS]
            target = tokens[:, 1:len(seq) + 1]
            nll = -log_probs[:, 1:len(seq) + 1].gather(
                -1, target.unsqueeze(-1)).squeeze(-1)
            return nll.sum().item()

    def compute_ddg_matrix(self, seq: str, pdb_path: Optional[str] = None,
                           chain_id: Optional[str] = None) -> np.ndarray:
        """Compute DDG matrix using masked marginals (L forward passes).

        For each position i, mask it and get p(b | seq_\\i) for all b.
        DDG(i, b) = log p(wt_aa | context) - log p(b | context)
        This is the "free robustness" approximation (fast, no L*19 calls).
        """
        if not self.use_masked_marginals:
            return super().compute_ddg_matrix(seq, pdb_path)

        import torch
        L = len(seq)
        mask_idx = self._alphabet.mask_idx

        # Get token indices for all amino acids
        aa_token_indices = [self._alphabet.get_idx(aa) for aa in AA_LIST]

        # Build all L masked sequences at once
        data = [("protein", seq)]
        _, _, base_tokens = self._batch_converter(data)
        # base_tokens shape: (1, L+2) with [CLS] seq [EOS]

        ddg_full = np.full((L, N_AA), np.nan, dtype=np.float32)

        for i in range(L):
            tokens = base_tokens.clone().to(self._device)
            tokens[0, i + 1] = mask_idx  # +1 for CLS token

            with torch.no_grad():
                logits = self._model(tokens)["logits"]
                log_probs = torch.log_softmax(logits[0, i + 1], dim=-1)

            wt_aa = seq[i]
            wt_token = self._alphabet.get_idx(wt_aa)
            wt_logp = log_probs[wt_token].item()

            for j, aa in enumerate(AA_LIST):
                if aa == wt_aa:
                    continue
                mut_logp = log_probs[aa_token_indices[j]].item()
                # DDG = -log p(wt) - (-log p(mut)) = log p(mut) - log p(wt)
                # Convention: positive DDG = mutant less stable
                ddg_full[i, j] = -(mut_logp - wt_logp)

        # Compress to L x 19
        ddg_matrix = np.zeros((L, N_AA - 1), dtype=np.float32)
        for i in range(L):
            wt_idx = AA_TO_IDX.get(seq[i])
            row = ddg_full[i]
            ddg_matrix[i] = np.concatenate([row[:wt_idx], row[wt_idx + 1:]])

        return ddg_matrix


class ThermoMPNNScorer(DDGScorer):
    """ThermoMPNN scorer (structure-conditioned).

    Uses ThermoMPNN to predict DDG for single-point mutations given
    a backbone structure. Requires a PDB file.

    ThermoMPNN predicts DDG directly (not absolute DG), so score_sequence()
    is not meaningful — use compute_ddg_matrix() instead.

    Install:
        git clone https://github.com/Kuhlman-Lab/ThermoMPNN.git
        cd ThermoMPNN
        mamba env create -f environment.yaml  # or conda
        mamba activate thermompnn

    Required files (inside the cloned repo):
        vanilla_model_weights/v_48_020.pt     - ProteinMPNN backbone weights
        models/thermoMPNN_default.pt          - Fine-tuned ThermoMPNN checkpoint

    Set THERMOMPNN_DIR env var to the cloned repo root, or pass --thermompnn_dir.

    Ref: Diaz et al. 2024, https://github.com/Kuhlman-Lab/ThermoMPNN
    """

    def __init__(self, thermompnn_dir: Optional[str] = None,
                 checkpoint: str = "thermoMPNN_default.pt",
                 chain_id: str = "A",
                 batch_size: int = 1000):
        self._model = None
        self._device = "cuda"
        self._thermompnn_dir = thermompnn_dir or os.environ.get(
            "THERMOMPNN_DIR", None)
        self._checkpoint = checkpoint
        self._chain_id = chain_id
        self._batch_size = batch_size  # mutations per forward pass

    @property
    def name(self) -> str:
        return "thermompnn"

    @property
    def requires_structure(self) -> bool:
        return True

    def load_model(self, device: str = "cuda"):
        self._device = device

        if self._thermompnn_dir is None:
            raise ValueError(
                "ThermoMPNN directory not set. Either:\n"
                "  1. Set THERMOMPNN_DIR environment variable, or\n"
                "  2. Pass --thermompnn_dir /path/to/ThermoMPNN\n"
                "The directory should contain vanilla_model_weights/ and models/.\n"
                "Clone from: https://github.com/Kuhlman-Lab/ThermoMPNN"
            )

        thermompnn_dir = Path(self._thermompnn_dir)

        # Add ThermoMPNN repo to sys.path so we can import its modules
        repo_str = str(thermompnn_dir)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        try:
            import torch
            from omegaconf import OmegaConf
            from train_thermompnn import TransferModelPL
            from protein_mpnn_utils import alt_parse_PDB
        except ImportError as e:
            raise ImportError(
                f"Failed to import ThermoMPNN modules: {e}\n"
                "Make sure you have the ThermoMPNN environment activated:\n"
                "  mamba activate thermompnn\n"
                "And THERMOMPNN_DIR points to the cloned repo root."
            )

        self._alt_parse_PDB = alt_parse_PDB

        # Build config (mirrors ThermoMPNN's config.yaml defaults)
        config = {
            "training": {
                "num_workers": 1,
                "learn_rate": 0.001,
                "epochs": 100,
                "lr_schedule": True,
            },
            "model": {
                "hidden_dims": [64, 32],
                "subtract_mut": True,
                "num_final_layers": 2,
                "freeze_weights": True,
                "load_pretrained": True,
                "lightattn": True,
            },
            "platform": {
                "thermompnn_dir": str(thermompnn_dir),
            },
        }
        # Load local.yaml from ThermoMPNN repo (has model architecture defaults)
        # but our config takes priority (overrides hardcoded paths from original lab)
        local_yaml = thermompnn_dir / "local.yaml"
        if local_yaml.exists():
            cfg = OmegaConf.merge(OmegaConf.load(str(local_yaml)), config)
        else:
            cfg = OmegaConf.create(config)

        # Load checkpoint
        checkpoint_path = thermompnn_dir / "models" / self._checkpoint
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"ThermoMPNN checkpoint not found: {checkpoint_path}\n"
                "Download from the ThermoMPNN repository or check the path."
            )

        self._model = TransferModelPL.load_from_checkpoint(
            str(checkpoint_path), cfg=cfg
        ).model  # extract TransferModel from the Lightning wrapper
        self._model = self._model.eval().to(device)

        print(f"  ThermoMPNN loaded from {checkpoint_path}")

    def score_sequence(self, seq: str, pdb_path: Optional[str] = None) -> float:
        """Not applicable for ThermoMPNN (it predicts DDG directly).
        Returns 0.0 for compatibility with the base class interface."""
        return 0.0

    def compute_ddg_matrix(self, seq: str, pdb_path: Optional[str] = None,
                           chain_id: Optional[str] = None) -> np.ndarray:
        """Compute L x 19 DDG matrix using ThermoMPNN.

        ThermoMPNN predicts DDG directly for each (position, wt->mut) pair.
        We parse the PDB once, build all L*19 Mutation objects, and run
        them through the model. The model handles ProteinMPNN encoding
        internally via tied_featurize.
        """
        if pdb_path is None:
            raise ValueError("ThermoMPNN requires a PDB file (pdb_path)")

        import torch
        from datasets import Mutation  # ThermoMPNN's Mutation dataclass

        L = len(seq)
        ddg_full = np.full((L, N_AA), np.nan, dtype=np.float32)

        # Parse PDB using ThermoMPNN's parser
        # Use per-protein chain_id if provided, otherwise fall back to default
        parse_chain = chain_id or self._chain_id
        pdb_data = self._alt_parse_PDB(pdb_path,
                                        input_chain_list=parse_chain)
        # Fallback: if parsing returned empty, try without specifying chain
        if not pdb_data or not pdb_data[0].get("seq", ""):
            pdb_data = self._alt_parse_PDB(pdb_path)
        # Fallback 2: ATLAS/GROMACS PDBs have no chain ID (column 22 is blank).
        # Add chain "A" to all ATOM/HETATM records and retry.
        if not pdb_data or not pdb_data[0].get("seq", ""):
            import tempfile
            with open(pdb_path) as f:
                lines = f.readlines()
            fixed_lines = []
            for line in lines:
                if (line.startswith("ATOM") or line.startswith("HETATM")) and len(line) > 21:
                    # PDB column 22 (0-indexed: 21) is the chain ID
                    if line[21] == " ":
                        line = line[:21] + "A" + line[22:]
                fixed_lines.append(line)
            tmp = tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False)
            tmp.writelines(fixed_lines)
            tmp.close()
            pdb_data = self._alt_parse_PDB(tmp.name, input_chain_list="A")
            os.unlink(tmp.name)

        # Verify sequence matches
        parsed_seq = pdb_data[0].get("seq", "")
        if len(parsed_seq) != L:
            print(f"  Warning: PDB sequence length ({len(parsed_seq)}) != "
                  f"input sequence length ({L}). Using PDB sequence.")
            seq = parsed_seq
            L = len(seq)
            ddg_full = np.full((L, N_AA), np.nan, dtype=np.float32)

        # Store the sequence that will be used for DDG computation, so
        # the caller can reconcile any metadata vs. PDB discrepancy.
        self._last_parsed_seq = seq

        # Build all mutation objects
        all_mutations = []
        all_indices = []  # (position_i, aa_j) for tracking
        for i in range(L):
            wt_aa = seq[i]
            if wt_aa not in AA_TO_IDX:
                continue  # skip non-standard
            for j, aa in enumerate(AA_LIST):
                if aa == wt_aa:
                    continue
                mut = Mutation(
                    position=i,
                    wildtype=wt_aa,
                    mutation=aa,
                    ddG=None,
                    pdb=pdb_data[0].get("name", "protein"),
                )
                all_mutations.append(mut)
                all_indices.append((i, j))

        # Run in batches to avoid OOM
        with torch.no_grad():
            for batch_start in range(0, len(all_mutations), self._batch_size):
                batch_end = min(batch_start + self._batch_size,
                                len(all_mutations))
                batch_muts = all_mutations[batch_start:batch_end]
                batch_idx = all_indices[batch_start:batch_end]

                predictions, _ = self._model(pdb_data, batch_muts)

                for (i, j), pred in zip(batch_idx, predictions):
                    if pred is not None and "ddG" in pred:
                        ddg_full[i, j] = pred["ddG"].cpu().item()

        # Compress to L x 19 (drop WT column at each row)
        ddg_matrix = np.zeros((L, N_AA - 1), dtype=np.float32)
        for i in range(L):
            wt_idx = AA_TO_IDX.get(seq[i])
            if wt_idx is not None:
                row = ddg_full[i]
                ddg_matrix[i] = np.concatenate([row[:wt_idx], row[wt_idx + 1:]])

        return ddg_matrix


# ==========================================================================
# SCORER REGISTRY
# ==========================================================================

SCORER_REGISTRY: Dict[str, type] = {
    "esm1v": ESM1vScorer,
    "thermompnn": ThermoMPNNScorer,
}


def get_scorer(name: str, **kwargs) -> DDGScorer:
    """Get a scorer instance by name."""
    if name not in SCORER_REGISTRY:
        raise ValueError(
            f"Unknown scorer '{name}'. Available: {list(SCORER_REGISTRY.keys())}"
        )
    return SCORER_REGISTRY[name](**kwargs)


def register_scorer(name: str, scorer_class: type):
    """Register a custom scorer class."""
    SCORER_REGISTRY[name] = scorer_class


# ==========================================================================
# ROBUSTNESS METRICS
# ==========================================================================

def compute_robustness_metrics(ddg_matrix: np.ndarray, seq: str
                               ) -> Dict[str, Any]:
    """Compute per-residue and global robustness metrics from a DDG matrix.

    Args:
        ddg_matrix: np.ndarray of shape (L, 19), DDG values for all
            non-wildtype substitutions at each position.
        seq: wildtype sequence string (for annotation).

    Returns:
        Dictionary with:
          - per_residue: list of dicts with per-position metrics
          - global: dict with whole-protein aggregate metrics
    """
    L = len(seq)
    assert ddg_matrix.shape[0] == L, (
        f"DDG matrix has {ddg_matrix.shape[0]} rows but sequence has {L} residues"
    )

    per_residue = []
    for i in range(L):
        row = ddg_matrix[i]
        valid = row[~np.isnan(row)]

        if len(valid) == 0:
            metrics = {
                "position": i + 1,  # 1-based
                "wt_aa": seq[i],
                "mean_abs_ddg": np.nan,
                "mean_ddg": np.nan,
                "std_ddg": np.nan,
                "max_ddg": np.nan,
                "min_ddg": np.nan,
                "frac_destabilizing": np.nan,
                "frac_neutral": np.nan,
                "n_valid": 0,
            }
        else:
            metrics = {
                "position": i + 1,
                "wt_aa": seq[i],
                # Type 1 robustness measures (intrinsic, DG-independent)
                "mean_abs_ddg": float(np.mean(np.abs(valid))),
                "mean_ddg": float(np.mean(valid)),
                "std_ddg": float(np.std(valid)),
                "max_ddg": float(np.max(valid)),
                "min_ddg": float(np.min(valid)),
                "frac_destabilizing": float(np.mean(valid > 1.0)),
                "frac_neutral": float(np.mean(np.abs(valid) < 0.5)),
                "n_valid": int(len(valid)),
            }

        per_residue.append(metrics)

    # Global metrics (aggregated across all positions)
    all_valid = ddg_matrix[~np.isnan(ddg_matrix)]
    per_res_mean_abs = np.array([r["mean_abs_ddg"] for r in per_residue])
    per_res_mean_abs_valid = per_res_mean_abs[~np.isnan(per_res_mean_abs)]

    global_metrics = {
        # Whole-protein robustness (Type 1)
        "global_mean_abs_ddg": float(np.mean(np.abs(all_valid))) if len(all_valid) > 0 else np.nan,
        "global_mean_ddg": float(np.mean(all_valid)) if len(all_valid) > 0 else np.nan,
        "global_std_ddg": float(np.std(all_valid)) if len(all_valid) > 0 else np.nan,
        "global_frac_destabilizing": float(np.mean(all_valid > 1.0)) if len(all_valid) > 0 else np.nan,
        "global_frac_neutral": float(np.mean(np.abs(all_valid) < 0.5)) if len(all_valid) > 0 else np.nan,
        # Per-residue robustness landscape statistics
        "robustness_mean_of_means": float(np.mean(per_res_mean_abs_valid)) if len(per_res_mean_abs_valid) > 0 else np.nan,
        "robustness_std_of_means": float(np.std(per_res_mean_abs_valid)) if len(per_res_mean_abs_valid) > 0 else np.nan,
        "robustness_cv": float(np.std(per_res_mean_abs_valid) / np.mean(per_res_mean_abs_valid))
            if len(per_res_mean_abs_valid) > 0 and np.mean(per_res_mean_abs_valid) != 0 else np.nan,
        # Spatial smoothness: autocorrelation of per-residue robustness
        "robustness_autocorr_lag1": float(_autocorr(per_res_mean_abs_valid, lag=1)),
        # Protein info
        "sequence_length": L,
        "total_mutations_scored": int(np.sum(~np.isnan(ddg_matrix))),
    }

    return {"per_residue": per_residue, "global": global_metrics}


def _autocorr(x: np.ndarray, lag: int = 1) -> float:
    """Compute autocorrelation at a given lag."""
    if len(x) <= lag:
        return np.nan
    x = x - np.mean(x)
    denom = np.sum(x ** 2)
    if denom == 0:
        return np.nan
    return float(np.sum(x[:-lag] * x[lag:]) / denom)


# ==========================================================================
# I/O HELPERS
# ==========================================================================

def extract_sequence_from_pdb(pdb_path: str, chain_id: str = "A") -> str:
    """Extract amino acid sequence from a PDB file.

    If the requested chain_id is not found, falls back to the first
    available chain (ATLAS minimized PDBs sometimes relabel chains).
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import protein_letters_3to1

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]

    # Try requested chain first, then fall back to first available chain
    if chain_id in model:
        chain = model[chain_id]
    else:
        available = [c.id for c in model]
        if not available:
            raise ValueError(f"No chains found in {pdb_path}")
        chain = model[available[0]]

    seq = []
    for residue in chain:
        if residue.id[0] != " ":  # skip HETATMs
            continue
        resname = residue.get_resname()
        try:
            seq.append(protein_letters_3to1.get(resname, "X"))
        except Exception:
            seq.append("X")

    return "".join(seq)


def find_atlas_proteins(atlas_dir: str) -> List[Dict[str, str]]:
    """Find all downloaded ATLAS proteins and their files."""
    proteins_dir = Path(atlas_dir) / "proteins"
    if not proteins_dir.exists():
        raise FileNotFoundError(f"ATLAS proteins directory not found: {proteins_dir}")

    proteins = []
    for protein_dir in sorted(proteins_dir.iterdir()):
        if not protein_dir.is_dir():
            continue
        done_marker = protein_dir / ".done"
        if not done_marker.exists():
            continue  # incomplete download

        pdb_chain = protein_dir.name
        # Find PDB file
        pdb_files = list(protein_dir.glob("*.pdb"))
        if not pdb_files:
            continue

        proteins.append({
            "protein_id": pdb_chain,
            "pdb_path": str(pdb_files[0]),
            "dir": str(protein_dir),
        })

    return proteins


def save_results(protein_id: str, seq: str, ddg_matrix: np.ndarray,
                 metrics: Dict, scorer_name: str, output_dir: str):
    """Save robustness results for one protein."""
    out_dir = Path(output_dir) / scorer_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. DDG matrix (numpy binary, compact)
    np.save(str(out_dir / f"{protein_id}_ddg_matrix.npy"), ddg_matrix)

    # 2. Full metrics as JSON
    result = {
        "protein_id": protein_id,
        "scorer": scorer_name,
        "sequence": seq,
        "sequence_length": len(seq),
        "global_metrics": metrics["global"],
        "per_residue_metrics": metrics["per_residue"],
    }
    with open(out_dir / f"{protein_id}_robustness.json", "w") as f:
        json.dump(result, f, indent=2, default=_json_default)

    # 3. Per-residue TSV (easy to merge with ATLAS RMSF/pLDDT files)
    import csv
    tsv_path = out_dir / f"{protein_id}_robustness.tsv"
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "position", "wt_aa", "mean_abs_ddg", "mean_ddg", "std_ddg",
            "max_ddg", "min_ddg", "frac_destabilizing", "frac_neutral", "n_valid"
        ], delimiter="\t")
        writer.writeheader()
        for row in metrics["per_residue"]:
            writer.writerow(row)


def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if np.isnan(obj) if isinstance(obj, float) else False:
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ==========================================================================
# MAIN
# ==========================================================================

def process_single_protein(protein_id: str, seq: str, pdb_path: Optional[str],
                           scorer: DDGScorer, output_dir: str,
                           skip_existing: bool = True,
                           chain_id: Optional[str] = None) -> bool:
    """Process one protein: compute DDG matrix, robustness metrics, save."""
    out_dir = Path(output_dir) / scorer.name
    json_path = out_dir / f"{protein_id}_robustness.json"
    if skip_existing and json_path.exists():
        return True  # already done

    # Validate
    if scorer.requires_structure and pdb_path is None:
        print(f"  SKIP {protein_id}: scorer '{scorer.name}' requires structure "
              f"but no PDB provided")
        return False

    # Filter out non-standard amino acids
    standard = set(AA_LIST)
    if any(aa not in standard for aa in seq):
        original_len = len(seq)
        # Replace non-standard with X (will get NaN in DDG matrix)
        seq_clean = "".join(aa if aa in standard else "X" for aa in seq)
        if seq_clean.count("X") > 0.1 * original_len:
            print(f"  SKIP {protein_id}: >10% non-standard amino acids")
            return False
        seq = seq_clean

    # Compute DDG matrix
    try:
        ddg_matrix = scorer.compute_ddg_matrix(seq, pdb_path, chain_id=chain_id)
    except Exception as e:
        print(f"  FAIL {protein_id}: {e}")
        return False

    # The scorer may use the PDB-derived sequence (e.g. when the metadata
    # canonical sequence differs from the resolved structure).  Reconcile
    # by reading back the actual sequence the scorer used when the matrix
    # shape differs from the input sequence.
    if ddg_matrix.shape[0] != len(seq):
        # The scorer's PDB parser may include modified residues (e.g. MSE)
        # that differ from the canonical metadata sequence.  Use the exact
        # sequence the scorer parsed, stored as scorer._last_parsed_seq.
        parsed_seq = getattr(scorer, "_last_parsed_seq", None)
        if parsed_seq is not None and len(parsed_seq) == ddg_matrix.shape[0]:
            seq = parsed_seq
        else:
            print(f"  FAIL {protein_id}: DDG matrix ({ddg_matrix.shape[0]}) vs "
                  f"sequence ({len(seq)}) mismatch, could not reconcile")
            return False

    # Compute robustness metrics
    try:
        metrics = compute_robustness_metrics(ddg_matrix, seq)
    except Exception as e:
        print(f"  FAIL {protein_id} (metrics): {e}")
        return False

    # Save
    save_results(protein_id, seq, ddg_matrix, metrics, scorer.name, output_dir)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-residue and global mutational robustness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Scorer selection
    parser.add_argument("--scorer", type=str, default="esm1v",
                        choices=list(SCORER_REGISTRY.keys()),
                        help="DDG scoring method (default: esm1v)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for model (cuda/cpu, default: cuda)")

    # Input: single protein
    parser.add_argument("--pdb_file", type=str, default=None,
                        help="Path to a single PDB file")
    parser.add_argument("--sequence", type=str, default=None,
                        help="Amino acid sequence string (for seq-only scorers)")
    parser.add_argument("--protein_id", type=str, default=None,
                        help="Protein identifier (default: derived from filename)")
    parser.add_argument("--chain_id", type=str, default="A",
                        help="Chain ID to extract from PDB (default: A)")

    # Input: batch (ATLAS)
    parser.add_argument("--atlas_dir", type=str, default=None,
                        help="Path to ATLAS download directory")
    parser.add_argument("--batch", action="store_true",
                        help="Process all proteins in atlas_dir")
    parser.add_argument("--batch_start", type=int, default=0,
                        help="Start index for batch processing (for SLURM arrays)")
    parser.add_argument("--batch_end", type=int, default=-1,
                        help="End index for batch processing (-1 = all)")

    # Input: list of PDB files
    parser.add_argument("--pdb_list", type=str, default=None,
                        help="Text file with one PDB path per line")

    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for output files")

    # Options
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip proteins that already have output (default: True)")
    parser.add_argument("--no_skip_existing", action="store_false",
                        dest="skip_existing",
                        help="Recompute even if output exists")
    parser.add_argument("--esm_masked_marginals", action="store_true",
                        default=True,
                        help="Use masked marginals for ESM-1v (fast, default)")
    parser.add_argument("--no_masked_marginals", action="store_false",
                        dest="esm_masked_marginals",
                        help="Use exhaustive scoring for ESM-1v (slow, L*19 calls)")

    # ThermoMPNN-specific options
    parser.add_argument("--thermompnn_dir", type=str, default=None,
                        help="Path to cloned ThermoMPNN repo (or set THERMOMPNN_DIR)")
    parser.add_argument("--thermompnn_checkpoint", type=str,
                        default="thermoMPNN_default.pt",
                        help="ThermoMPNN checkpoint filename in models/ dir")

    args = parser.parse_args()

    # Build scorer
    scorer_kwargs = {}
    if args.scorer == "esm1v":
        scorer_kwargs["use_masked_marginals"] = args.esm_masked_marginals
    elif args.scorer == "thermompnn":
        scorer_kwargs["thermompnn_dir"] = args.thermompnn_dir
        scorer_kwargs["checkpoint"] = args.thermompnn_checkpoint
        scorer_kwargs["chain_id"] = args.chain_id
    scorer = get_scorer(args.scorer, **scorer_kwargs)

    print(f"Loading scorer: {scorer.name} (device={args.device})")
    scorer.load_model(device=args.device)

    # Determine input mode
    proteins_to_process = []  # list of (protein_id, seq, pdb_path, chain_id)

    if args.sequence is not None:
        # Sequence mode (no PDB)
        pid = args.protein_id or "input_sequence"
        proteins_to_process.append((pid, args.sequence, args.pdb_file, args.chain_id))

    elif args.pdb_file is not None:
        # Single PDB mode
        pdb_path = args.pdb_file
        pid = args.protein_id or Path(pdb_path).stem
        seq = extract_sequence_from_pdb(pdb_path, args.chain_id)
        proteins_to_process.append((pid, seq, pdb_path, args.chain_id))

    elif args.pdb_list is not None:
        # List of PDB files
        with open(args.pdb_list) as f:
            pdb_paths = [line.strip() for line in f if line.strip()]
        for pdb_path in pdb_paths:
            pid = Path(pdb_path).stem
            try:
                seq = extract_sequence_from_pdb(pdb_path, args.chain_id)
                proteins_to_process.append((pid, seq, pdb_path, args.chain_id))
            except Exception as e:
                print(f"SKIP {pdb_path}: {e}")

    elif args.batch and args.atlas_dir:
        # ATLAS batch mode
        atlas_proteins = find_atlas_proteins(args.atlas_dir)
        print(f"Found {len(atlas_proteins)} ATLAS proteins")

        # Apply batch range
        end = args.batch_end if args.batch_end >= 0 else len(atlas_proteins)
        atlas_proteins = atlas_proteins[args.batch_start:end]
        print(f"Processing indices {args.batch_start} to "
              f"{args.batch_start + len(atlas_proteins) - 1}")

        for p in atlas_proteins:
            try:
                # ATLAS protein IDs are "{pdb}_{chain}" e.g. "2fb5_B"
                # Extract the chain ID from the protein name
                parts = p["protein_id"].rsplit("_", 1)
                chain_id = parts[1] if len(parts) == 2 else "A"
                seq = extract_sequence_from_pdb(p["pdb_path"], chain_id)
                proteins_to_process.append((p["protein_id"], seq, p["pdb_path"], chain_id))
            except Exception as e:
                print(f"SKIP {p['protein_id']}: {e}")

    else:
        parser.error("Provide one of: --pdb_file, --sequence, --pdb_list, "
                     "or --batch with --atlas_dir")

    # Process all proteins
    import time as _time
    print(f"\nProcessing {len(proteins_to_process)} proteins with {scorer.name}")
    print(f"Output: {args.output_dir}/{scorer.name}/")
    print("-" * 60, flush=True)

    n_ok, n_skip, n_fail = 0, 0, 0
    t_start = _time.time()
    for idx, (pid, seq, pdb_path, prot_chain) in enumerate(proteins_to_process):
        # Check if already done
        out_json = (Path(args.output_dir) / scorer.name /
                    f"{pid}_robustness.json")
        if args.skip_existing and out_json.exists():
            n_skip += 1
            if (idx + 1) % 100 == 0:
                print(f"[{idx+1}/{len(proteins_to_process)}] ... "
                      f"({n_ok} ok, {n_skip} skipped, {n_fail} failed)",
                      flush=True)
            continue

        t0 = _time.time()
        print(f"[{idx+1}/{len(proteins_to_process)}] {pid} (L={len(seq)}) ...",
              end=" ", flush=True)
        try:
            success = process_single_protein(
                pid, seq, pdb_path, scorer, args.output_dir,
                skip_existing=args.skip_existing, chain_id=prot_chain
            )
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            success = False
        elapsed = _time.time() - t0
        if success:
            n_ok += 1
            print(f"OK ({elapsed:.1f}s)", flush=True)
        else:
            n_fail += 1
            print(f"FAILED ({elapsed:.1f}s)", flush=True)

    total_elapsed = _time.time() - t_start
    print("-" * 60)
    print(f"Done! {n_ok} computed, {n_skip} skipped, {n_fail} failed")
    print(f"Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    if n_ok > 0:
        print(f"Avg per protein: {total_elapsed/n_ok:.1f}s")
    print(f"Results in: {args.output_dir}/{scorer.name}/", flush=True)


if __name__ == "__main__":
    main()
