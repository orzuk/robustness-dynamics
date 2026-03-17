#!/usr/bin/env python3
"""
Generate case study figure (Fig 5) for the robustness-dynamics paper.

Creates a multi-panel figure showing per-residue robustness vs dynamics
on individual proteins, with 3D structure renders colored by each metric
and an aligned line plot.

Panels A-C: PyMOL renders (robustness, RMSF, pLDDT) — generated via
  headless PyMOL scripting (pymol -cq).
Panel D: matplotlib line plot of z-scored metrics along the sequence.

Usage (on cluster):
  # Generate everything for Syntaxin-1A:
  python scripts/generate_case_study_figure.py \
      --protein_id 1ez3_B \
      --atlas_dir /sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas \
      --robustness_dir /sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas_robustness \
      --scorer thermompnn \
      --output_dir figures/case_study

  # Skip PyMOL (line plot only):
  python scripts/generate_case_study_figure.py ... --line-plot-only

  # Run PyMOL script separately (if pymol not in current venv):
  pymol -cq figures/case_study/1ez3_B_pymol_script.pml
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats


# ============================================================================
# DATA LOADING (reuses conventions from correlate_robustness_dynamics.py)
# ============================================================================

def load_atlas_tsv(protein_dir: Path, suffix: str) -> Optional[pd.DataFrame]:
    """Load a per-residue TSV from an ATLAS protein directory."""
    matches = list(protein_dir.glob(f"*{suffix}"))
    if not matches:
        return None
    return pd.read_csv(matches[0], sep="\t")


def load_rmsf(protein_dir: Path) -> Optional[pd.DataFrame]:
    """Load RMSF, averaging across replicates."""
    df = load_atlas_tsv(protein_dir, "_RMSF.tsv")
    if df is None:
        return None
    rmsf_cols = [c for c in df.columns if "rmsf" in c.lower()]
    if not rmsf_cols:
        numeric = [c for c in df.columns if df[c].dtype in (np.float64, np.float32)]
        rmsf_cols = numeric
    result = pd.DataFrame()
    result["position"] = range(1, len(df) + 1)
    result["rmsf"] = df[rmsf_cols].mean(axis=1).values
    return result


def load_plddt(protein_dir: Path) -> Optional[pd.DataFrame]:
    """Load pLDDT data."""
    df = load_atlas_tsv(protein_dir, "_pLDDT.tsv")
    if df is None:
        return None
    plddt_cols = [c for c in df.columns if "plddt" in c.lower()]
    if not plddt_cols:
        numeric = [c for c in df.columns if df[c].dtype in (np.float64, np.float32)]
        plddt_cols = numeric[:1]
    result = pd.DataFrame()
    result["position"] = range(1, len(df) + 1)
    result["plddt"] = df[plddt_cols[0]].values
    return result


def load_bfactor(protein_dir: Path) -> Optional[pd.DataFrame]:
    """Load B-factor data."""
    df = load_atlas_tsv(protein_dir, "_Bfactor.tsv")
    if df is None:
        return None
    bfac_cols = [c for c in df.columns if "bfactor" in c.lower() or "b_factor" in c.lower()]
    if not bfac_cols:
        numeric = [c for c in df.columns if df[c].dtype in (np.float64, np.float32)]
        bfac_cols = numeric[:1]
    result = pd.DataFrame()
    result["position"] = range(1, len(df) + 1)
    result["bfactor"] = df[bfac_cols[0]].values
    return result


def load_robustness(robustness_dir: Path, scorer: str, protein_id: str) -> Optional[pd.DataFrame]:
    """Load per-residue robustness TSV."""
    tsv_path = robustness_dir / scorer / f"{protein_id}_robustness.tsv"
    if not tsv_path.exists():
        return None
    return pd.read_csv(tsv_path, sep="\t")


def find_pdb(protein_dir: Path) -> Optional[Path]:
    """Find PDB file in ATLAS protein directory."""
    for pattern in ("*.pdb", "*.ent"):
        matches = list(protein_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def zscore(x):
    """Z-score normalization, handling NaN."""
    x = np.array(x, dtype=float)
    mask = ~np.isnan(x)
    if mask.sum() < 2:
        return x
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sd


# ============================================================================
# PYMOL SCRIPT GENERATION
# ============================================================================

def generate_pymol_script(pdb_path: str, protein_id: str, chain: str,
                          robustness_df: pd.DataFrame, rmsf_df: pd.DataFrame,
                          plddt_df: pd.DataFrame, output_dir: str,
                          width: int = 2400, height: int = 1800) -> str:
    """Generate a PyMOL .pml script that renders 3 panels."""

    out = Path(output_dir)
    chain_upper = chain.upper()

    # Prepare value mappings (residue index -> value)
    # Robustness: high std_ddg = sensitive/rigid -> blue; low = robust/flexible -> red
    # RMSF: high = flexible -> red; low = rigid -> blue
    # pLDDT: high = confident/rigid -> blue; low = flexible -> red

    rob_vals = robustness_df.set_index("position")["std_ddg"].to_dict()
    rmsf_vals = rmsf_df.set_index("position")["rmsf"].to_dict()
    plddt_vals = plddt_df.set_index("position")["plddt"].to_dict()

    # Build the alter commands for each metric
    def make_alter_cmds(vals_dict, obj_name):
        cmds = []
        for pos, val in sorted(vals_dict.items()):
            if np.isfinite(val):
                cmds.append(f'cmd.alter("{obj_name} and resi {pos}", "b={val:.4f}")')
        return cmds

    rob_min = np.nanpercentile(list(rob_vals.values()), 2)
    rob_max = np.nanpercentile(list(rob_vals.values()), 98)
    rmsf_min = np.nanpercentile(list(rmsf_vals.values()), 2)
    rmsf_max = np.nanpercentile(list(rmsf_vals.values()), 98)
    plddt_min = np.nanpercentile(list(plddt_vals.values()), 2)
    plddt_max = np.nanpercentile(list(plddt_vals.values()), 98)

    pml = f"""# Auto-generated PyMOL script for case study figure
# Protein: {protein_id}
# Run: pymol -cq {protein_id}_pymol_script.pml

import pymol
from pymol import cmd

# Settings for publication quality
cmd.set("ray_opaque_background", 1)
cmd.set("ray_shadows", 0)
cmd.set("antialias", 2)
cmd.set("cartoon_fancy_helices", 1)
cmd.set("cartoon_smooth_loops", 1)
cmd.set("spec_reflect", 0.3)
cmd.bg_color("white")

# Load structure
cmd.load("{pdb_path}", "protein")
# ATLAS PDBs may have blank chain IDs — only filter chain if present
if cmd.count_atoms("chain {chain_upper}") > 0:
    cmd.remove("not chain {chain_upper}")
cmd.remove("resn HOH")
cmd.remove("not polymer.protein")
# ATLAS PDBs from MD may have multiple models — keep only first
cmd.split_states("protein", 1, 1)
if cmd.count_atoms("protein_0001") > 0:
    cmd.delete("protein")
    cmd.set_name("protein_0001", "protein")

# Store the view after orienting so all panels match
cmd.orient("protein")
cmd.turn("y", 0)  # adjust rotation if needed
stored_view = cmd.get_view()

# ========== Panel A: Robustness index ==========
cmd.create("rob", "protein")
cmd.alter("rob", "b=0")
"""
    for pos, val in sorted(rob_vals.items()):
        if np.isfinite(val):
            pml += f'cmd.alter("rob and resi {pos}", "b={val:.4f}")\n'

    pml += f"""
cmd.rebuild("rob")
cmd.show("cartoon", "rob")
cmd.cartoon("putty", "rob")
cmd.set("cartoon_putty_scale_min", 0.4, "rob")
cmd.set("cartoon_putty_scale_max", 2.5, "rob")
cmd.set("cartoon_putty_scale_power", 1.0, "rob")
# High std_ddg (sensitive) = blue, low (robust) = red
cmd.spectrum("b", "red_white_blue", "rob", minimum={rob_min:.3f}, maximum={rob_max:.3f})
cmd.set_view(stored_view)
cmd.ray({width}, {height})
cmd.png("{out / f'{protein_id}_panel_A_robustness.png'}", width={width}, height={height})
cmd.disable("rob")

# ========== Panel B: RMSF ==========
cmd.create("rmsf_obj", "protein")
cmd.alter("rmsf_obj", "b=0")
"""
    for pos, val in sorted(rmsf_vals.items()):
        if np.isfinite(val):
            pml += f'cmd.alter("rmsf_obj and resi {pos}", "b={val:.4f}")\n'

    pml += f"""
cmd.rebuild("rmsf_obj")
cmd.show("cartoon", "rmsf_obj")
cmd.cartoon("putty", "rmsf_obj")
cmd.set("cartoon_putty_scale_min", 0.4, "rmsf_obj")
cmd.set("cartoon_putty_scale_max", 2.5, "rmsf_obj")
cmd.set("cartoon_putty_scale_power", 1.0, "rmsf_obj")
# High RMSF (flexible) = red, low (rigid) = blue
cmd.spectrum("b", "blue_white_red", "rmsf_obj", minimum={rmsf_min:.3f}, maximum={rmsf_max:.3f})
cmd.set_view(stored_view)
cmd.ray({width}, {height})
cmd.png("{out / f'{protein_id}_panel_B_rmsf.png'}", width={width}, height={height})
cmd.disable("rmsf_obj")

# ========== Panel C: pLDDT ==========
cmd.create("plddt_obj", "protein")
cmd.alter("plddt_obj", "b=0")
"""
    for pos, val in sorted(plddt_vals.items()):
        if np.isfinite(val):
            pml += f'cmd.alter("plddt_obj and resi {pos}", "b={val:.4f}")\n'

    pml += f"""
cmd.rebuild("plddt_obj")
cmd.show("cartoon", "plddt_obj")
cmd.cartoon("putty", "plddt_obj")
cmd.set("cartoon_putty_scale_min", 0.4, "plddt_obj")
cmd.set("cartoon_putty_scale_max", 2.5, "plddt_obj")
cmd.set("cartoon_putty_scale_power", 1.0, "plddt_obj")
# High pLDDT (confident/rigid) = blue, low (flexible) = red
cmd.spectrum("b", "red_white_blue", "plddt_obj", minimum={plddt_min:.3f}, maximum={plddt_max:.3f})
cmd.set_view(stored_view)
cmd.ray({width}, {height})
cmd.png("{out / f'{protein_id}_panel_C_plddt.png'}", width={width}, height={height})

cmd.quit()
"""
    return pml


# ============================================================================
# LINE PLOT (Panel D)
# ============================================================================

def generate_line_plot(protein_id: str, robustness_df: pd.DataFrame,
                       rmsf_df: pd.DataFrame, plddt_df: pd.DataFrame,
                       bfactor_df: Optional[pd.DataFrame],
                       output_dir: str, rho_rmsf: float = None):
    """Generate per-residue line plot of z-scored metrics."""

    positions = robustness_df["position"].values
    std_ddg = robustness_df["std_ddg"].values

    # Z-score all metrics
    z_rob = zscore(std_ddg)
    z_rmsf = zscore(rmsf_df["rmsf"].values) if rmsf_df is not None else None
    z_plddt = zscore(plddt_df["plddt"].values) if plddt_df is not None else None
    z_bfac = zscore(bfactor_df["bfactor"].values) if bfactor_df is not None else None

    fig, ax = plt.subplots(figsize=(10, 3.5))

    # Plot -std_ddg so that "up" = flexible/robust (matches RMSF direction)
    ax.plot(positions, -z_rob, color="#2166AC", linewidth=1.5, alpha=0.85,
            label=r"$-\mathrm{std}(\Delta\Delta G)$ (robustness)")
    if z_rmsf is not None:
        ax.plot(positions, z_rmsf, color="#D6604D", linewidth=1.5, alpha=0.85,
                label="RMSF")
    if z_plddt is not None:
        ax.plot(positions, -z_plddt, color="#4DAF4A", linewidth=1.2, alpha=0.7,
                linestyle="--", label=r"$-$pLDDT")
    if z_bfac is not None:
        ax.plot(positions, z_bfac, color="#984EA3", linewidth=1.0, alpha=0.6,
                linestyle=":", label="B-factor")

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="-")
    ax.set_xlabel("Residue position", fontsize=11)
    ax.set_ylabel("Z-scored value", fontsize=11)

    title = protein_id.upper().replace("_", " chain ")
    if rho_rmsf is not None:
        title += f"  ($\\rho$ = {rho_rmsf:.3f})"
    ax.set_title(title, fontsize=12)

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_xlim(positions[0], positions[-1])
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    out = Path(output_dir)
    fig.savefig(out / f"{protein_id}_panel_D_lineplot.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{protein_id}_panel_D_lineplot.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Line plot saved: {out / f'{protein_id}_panel_D_lineplot.png'}")


# ============================================================================
# COMPOSITE FIGURE
# ============================================================================

def composite_figure(protein_id: str, output_dir: str):
    """Combine panels A-D into a single figure (if all PNGs exist)."""
    out = Path(output_dir)
    panel_a = out / f"{protein_id}_panel_A_robustness.png"
    panel_b = out / f"{protein_id}_panel_B_rmsf.png"
    panel_c = out / f"{protein_id}_panel_C_plddt.png"
    panel_d = out / f"{protein_id}_panel_D_lineplot.png"

    if not all(p.exists() for p in [panel_a, panel_b, panel_d]):
        print("  Skipping composite: not all panels available yet.")
        print(f"  Missing: {[str(p) for p in [panel_a, panel_b, panel_c, panel_d] if not p.exists()]}")
        return

    from PIL import Image

    img_a = Image.open(panel_a)
    img_b = Image.open(panel_b)
    img_c = Image.open(panel_c) if panel_c.exists() else None
    img_d = Image.open(panel_d)

    # Layout: top row = 3 structure panels, bottom row = line plot
    struct_panels = [img_a, img_b]
    if img_c is not None:
        struct_panels.append(img_c)

    # Resize structure panels to same height
    min_h = min(im.height for im in struct_panels)
    resized = []
    for im in struct_panels:
        ratio = min_h / im.height
        resized.append(im.resize((int(im.width * ratio), min_h), Image.LANCZOS))

    top_width = sum(im.width for im in resized)
    top_height = min_h

    # Scale line plot to match top row width
    ratio = top_width / img_d.width
    img_d_resized = img_d.resize((top_width, int(img_d.height * ratio)), Image.LANCZOS)

    # Add labels
    total_h = top_height + img_d_resized.height + 20
    composite = Image.new("RGB", (top_width, total_h), "white")

    x = 0
    for im in resized:
        composite.paste(im, (x, 0))
        x += im.width

    composite.paste(img_d_resized, (0, top_height + 20))

    composite.save(out / f"{protein_id}_fig5_composite.png", dpi=(300, 300))
    print(f"  Composite saved: {out / f'{protein_id}_fig5_composite.png'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate case study figure for robustness-dynamics paper")
    parser.add_argument("--protein_id", required=True,
                        help="ATLAS protein ID (e.g., 1ez3_B)")
    parser.add_argument("--atlas_dir", required=True,
                        help="Path to ATLAS data directory")
    parser.add_argument("--robustness_dir", required=True,
                        help="Path to robustness output directory")
    parser.add_argument("--scorer", default="thermompnn",
                        help="DDG scorer (default: thermompnn)")
    parser.add_argument("--output_dir", default="figures/case_study",
                        help="Output directory for figure panels")
    parser.add_argument("--line-plot-only", action="store_true",
                        help="Skip PyMOL rendering, generate line plot only")
    parser.add_argument("--run-pymol", action="store_true",
                        help="Also execute the PyMOL script (requires pymol in PATH)")
    parser.add_argument("--no-composite", action="store_true",
                        help="Skip composite figure generation")
    parser.add_argument("--width", type=int, default=2400,
                        help="PyMOL render width in pixels")
    parser.add_argument("--height", type=int, default=1800,
                        help="PyMOL render height in pixels")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    protein_id = args.protein_id
    chain = protein_id.split("_")[-1] if "_" in protein_id else "A"
    protein_dir = Path(args.atlas_dir) / "proteins" / protein_id

    print(f"=== Case study figure for {protein_id} ===")

    # --- Load data ---
    print("Loading data...")
    robustness_df = load_robustness(Path(args.robustness_dir), args.scorer, protein_id)
    if robustness_df is None:
        print(f"ERROR: No robustness data found for {protein_id}")
        sys.exit(1)
    print(f"  Robustness: {len(robustness_df)} residues")

    rmsf_df = load_rmsf(protein_dir)
    if rmsf_df is not None:
        print(f"  RMSF: {len(rmsf_df)} residues")
    else:
        print("  WARNING: No RMSF data found")

    plddt_df = load_plddt(protein_dir)
    if plddt_df is not None:
        print(f"  pLDDT: {len(plddt_df)} residues")
    else:
        print("  WARNING: No pLDDT data found")

    bfactor_df = load_bfactor(protein_dir)
    if bfactor_df is not None:
        print(f"  B-factor: {len(bfactor_df)} residues")

    pdb_path = find_pdb(protein_dir)
    if pdb_path:
        print(f"  PDB: {pdb_path}")
    else:
        print("  WARNING: No PDB file found (PyMOL panels will fail)")

    # --- Align lengths (use shortest) ---
    n = len(robustness_df)
    if rmsf_df is not None:
        n = min(n, len(rmsf_df))
    if plddt_df is not None:
        n = min(n, len(plddt_df))
    robustness_df = robustness_df.iloc[:n].copy()
    if rmsf_df is not None:
        rmsf_df = rmsf_df.iloc[:n].copy()
    if plddt_df is not None:
        plddt_df = plddt_df.iloc[:n].copy()
    if bfactor_df is not None:
        bfactor_df = bfactor_df.iloc[:n].copy()

    # --- Compute correlation ---
    rho_rmsf = None
    if rmsf_df is not None:
        rho, pval = stats.spearmanr(robustness_df["std_ddg"].values,
                                     rmsf_df["rmsf"].values,
                                     nan_policy="omit")
        rho_rmsf = rho
        print(f"  Spearman rho (std_ddg vs RMSF): {rho:.3f} (p={pval:.2e})")

    # --- Save extracted data as TSV (for reference) ---
    merged = robustness_df[["position", "wt_aa", "std_ddg", "mean_abs_ddg"]].copy()
    if rmsf_df is not None:
        merged["rmsf"] = rmsf_df["rmsf"].values
    if plddt_df is not None:
        merged["plddt"] = plddt_df["plddt"].values
    if bfactor_df is not None:
        merged["bfactor"] = bfactor_df["bfactor"].values
    merged.to_csv(out / f"{protein_id}_per_residue_data.tsv", sep="\t", index=False)
    print(f"  Per-residue data saved: {out / f'{protein_id}_per_residue_data.tsv'}")

    # --- Generate line plot (Panel D) ---
    print("Generating line plot (Panel D)...")
    generate_line_plot(protein_id, robustness_df, rmsf_df, plddt_df,
                       bfactor_df, args.output_dir, rho_rmsf=rho_rmsf)

    # --- Generate PyMOL script (Panels A-C) ---
    if not args.line_plot_only:
        if pdb_path is None:
            print("ERROR: Cannot generate PyMOL panels without PDB file")
        else:
            print("Generating PyMOL script (Panels A-C)...")
            pml_script = generate_pymol_script(
                str(pdb_path), protein_id, chain,
                robustness_df, rmsf_df, plddt_df,
                args.output_dir, args.width, args.height)

            pml_path = out / f"{protein_id}_pymol_script.pml"
            with open(pml_path, "w") as f:
                f.write(pml_script)
            print(f"  PyMOL script saved: {pml_path}")
            print(f"  To render: pymol -cq {pml_path}")

            if args.run_pymol:
                print("Running PyMOL...")
                result = subprocess.run(
                    ["pymol", "-cq", str(pml_path)],
                    capture_output=True, text=True)
                if result.returncode == 0:
                    print("  PyMOL rendering complete!")
                else:
                    print(f"  PyMOL failed (exit {result.returncode}):")
                    print(f"  stderr: {result.stderr[:500]}")

    # --- Composite ---
    if not args.no_composite and not args.line_plot_only:
        print("Generating composite figure...")
        try:
            composite_figure(protein_id, args.output_dir)
        except ImportError:
            print("  Pillow not installed, skipping composite (pip install Pillow)")
        except Exception as e:
            print(f"  Composite failed: {e}")

    print("Done!")


if __name__ == "__main__":
    main()
