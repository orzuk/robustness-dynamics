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
    if "position" in df.columns:
        result["position"] = df["position"].values
    else:
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
    if "position" in df.columns:
        result["position"] = df["position"].values
    else:
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
    if "position" in df.columns:
        result["position"] = df["position"].values
    else:
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
                          plddt_df: pd.DataFrame,
                          bfactor_df: Optional[pd.DataFrame],
                          output_dir: str,
                          width: int = 2400, height: int = 1800,
                          clip_pct: float = 10.0,
                          trim_termini: int = 0) -> str:
    """Generate a PyMOL .py script that renders 4 panels (rob, pLDDT, RMSF, B-factor).

    Parameters
    ----------
    clip_pct : float
        Percentile for robust color clipping (e.g. 10 means 10th-90th percentile).
    trim_termini : int
        Exclude first/last N residues from color range computation.
    """

    out = Path(output_dir)
    chain_upper = chain.upper()

    rob_vals = robustness_df.set_index("position")["std_ddg"].to_dict()
    rmsf_vals = rmsf_df.set_index("position")["rmsf"].to_dict()
    plddt_vals = plddt_df.set_index("position")["plddt"].to_dict()
    bfac_vals = bfactor_df.set_index("position")["bfactor"].to_dict() if bfactor_df is not None else {}

    def robust_range(vals_dict):
        """Compute color range using percentile clipping, optionally trimming termini."""
        positions = sorted(vals_dict.keys())
        if trim_termini > 0 and len(positions) > 2 * trim_termini:
            interior = positions[trim_termini:-trim_termini]
            v = [vals_dict[p] for p in interior if np.isfinite(vals_dict[p])]
        else:
            v = [x for x in vals_dict.values() if np.isfinite(x)]
        return (np.percentile(v, clip_pct), np.percentile(v, 100 - clip_pct))

    rob_min, rob_max = robust_range(rob_vals)
    rmsf_min, rmsf_max = robust_range(rmsf_vals)
    plddt_min, plddt_max = robust_range(plddt_vals)
    bfac_min, bfac_max = robust_range(bfac_vals) if bfac_vals else (0, 1)

    pml = f"""#!/usr/bin/env python3
# Auto-generated PyMOL render script for case study figure
# Protein: {protein_id}
# Run: python {protein_id}_pymol_render.py

import pymol
pymol.finish_launching(["pymol", "-cq"])
from pymol import cmd

def render_panel(obj_name, vals_dict, color_ramp, vmin, vmax, view, out_path, width, height):
    \"\"\"Render one structure panel: regular cartoon colored by B-factor spectrum.\"\"\"
    cmd.alter(obj_name, "b=0")
    for pos, val in sorted(vals_dict.items()):
        if val == val:  # skip NaN
            cmd.alter(f"{{obj_name}} and resi {{pos}}", f"b={{val:.4f}}")
    cmd.rebuild(obj_name)
    cmd.hide("everything", obj_name)
    cmd.show("cartoon", obj_name)
    # Use regular cartoon (not putty) — single clean color per residue
    cmd.set("cartoon_discrete_colors", 1, obj_name)
    cmd.spectrum("b", color_ramp, obj_name, minimum=vmin, maximum=vmax)
    cmd.set_view(view)
    cmd.ray(width, height)
    cmd.png(out_path, width=width, height=height)
    cmd.disable(obj_name)

def run():
    # Settings for publication quality
    cmd.set("ray_opaque_background", 1)
    cmd.set("ray_shadows", 0)
    cmd.set("antialias", 2)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.set("spec_reflect", 0.3)
    cmd.bg_color("white")

    # Load structure
    cmd.load("{pdb_path}", "prot")
    # ATLAS PDBs may have blank chain IDs — only filter chain if present
    if cmd.count_atoms("chain {chain_upper}") > 0:
        cmd.remove("not chain {chain_upper}")
    cmd.remove("resn HOH")
    cmd.remove("not polymer.protein")
    # ATLAS PDBs from MD may have multiple models — keep only first
    n_states = cmd.count_states("prot")
    if n_states > 1:
        cmd.split_states("prot", 1, 1)
        cmd.delete("prot")
        cmd.set_name("prot_0001", "prot")

    # Store the view after orienting so all panels match
    cmd.orient("prot")
    stored_view = cmd.get_view()

    # ========== Panel B: pLDDT (prediction) ==========
    cmd.create("plddt_obj", "prot")
"""
    # Build vals dicts as Python literals in the generated script
    pml += f"""
    plddt_vals = {dict((k, round(v, 4)) for k, v in plddt_vals.items() if np.isfinite(v))}
    render_panel("plddt_obj", plddt_vals, "red_white_blue", {plddt_min:.3f}, {plddt_max:.3f},
                 stored_view, "{out / f'{protein_id}_panel_B_plddt.png'}", {width}, {height})

    # ========== Panel C: Robustness index (std DDG, prediction) ==========
    cmd.create("rob", "prot")
    rob_vals = {dict((k, round(v, 4)) for k, v in rob_vals.items() if np.isfinite(v))}
    render_panel("rob", rob_vals, "red_white_blue", {rob_min:.3f}, {rob_max:.3f},
                 stored_view, "{out / f'{protein_id}_panel_C_robustness.png'}", {width}, {height})

    # ========== Panel D: RMSF ==========
    cmd.create("rmsf_obj", "prot")
    rmsf_vals = {dict((k, round(v, 4)) for k, v in rmsf_vals.items() if np.isfinite(v))}
    render_panel("rmsf_obj", rmsf_vals, "blue_white_red", {rmsf_min:.3f}, {rmsf_max:.3f},
                 stored_view, "{out / f'{protein_id}_panel_D_rmsf.png'}", {width}, {height})

    # ========== Panel E: B-factor ==========
"""
    if bfac_vals:
        pml += f"""    cmd.create("bfac_obj", "prot")
    bfac_vals = {dict((k, round(v, 4)) for k, v in bfac_vals.items() if np.isfinite(v))}
    render_panel("bfac_obj", bfac_vals, "blue_white_red", {bfac_min:.3f}, {bfac_max:.3f},
                 stored_view, "{out / f'{protein_id}_panel_E_bfactor.png'}", {width}, {height})
"""
    else:
        pml += f"""    print("No B-factor data — skipping panel E")
"""

    pml += f"""
    cmd.quit()

run()
"""
    return pml


# ============================================================================
# LINE PLOT (Panel D)
# ============================================================================

def load_domains(domains_path: Optional[str]) -> list:
    """Load domain annotations from a JSON file.

    Expected format: list of dicts with keys "name", "start", "end",
    and optionally "color" (default cycles through a palette).

    Example JSON:
      [
        {"name": "Habc domain", "start": 1, "end": 37},
        {"name": "Linker",      "start": 38, "end": 50, "color": "#FDDBC7"},
        {"name": "SNARE (H3)",  "start": 51, "end": 127}
      ]
    """
    if domains_path is None:
        return []
    with open(domains_path) as f:
        domains = json.load(f)
    return domains


# Default pastel palette for domain shading (light-medium saturation)
_DOMAIN_COLORS = [
    "#C8DEF5",  # light-medium blue
    "#FDD0B4",  # light-medium salmon
    "#CCEBCC",  # light-medium green
    "#E0D6EE",  # light-medium lavender
    "#FFF0A0",  # light-medium yellow
    "#FBCCE0",  # light-medium pink
]

# Metric colors: predictors = blue family, responses = red family
METRIC_COLORS = {
    "std_ddg": "#053061",   # very dark blue  (predictor)
    "plddt":   "#92C5DE",   # light blue      (predictor)
    "rmsf":    "#F4A582",   # light red/salmon (response)
    "bfactor": "#8B0000",   # very dark red    (response)
}


def generate_line_plot(protein_id: str, robustness_df: pd.DataFrame,
                       rmsf_df: pd.DataFrame, plddt_df: pd.DataFrame,
                       bfactor_df: Optional[pd.DataFrame],
                       output_dir: str, rho_rmsf: float = None,
                       domains: Optional[list] = None,
                       smooth_window: int = 0):
    """Generate per-residue line plot of z-scored metrics.

    Parameters
    ----------
    smooth_window : int
        If > 0, overlay a rolling-average smoothed version of the
        robustness trace (useful to see the dynamics-correlated component
        without the ~3.6-residue helix-face oscillation).
    """

    positions = robustness_df["position"].values
    std_ddg = robustness_df["std_ddg"].values

    # Z-score all metrics
    z_rob = zscore(std_ddg)
    z_rmsf = zscore(rmsf_df["rmsf"].values) if rmsf_df is not None else None
    z_plddt = zscore(plddt_df["plddt"].values) if plddt_df is not None else None
    z_bfac = zscore(bfactor_df["bfactor"].values) if bfactor_df is not None else None

    fig, ax = plt.subplots(figsize=(10, 3.5))

    # Draw domain annotations as shaded regions (behind data)
    if domains:
        for i, dom in enumerate(domains):
            color = dom.get("color", _DOMAIN_COLORS[i % len(_DOMAIN_COLORS)])
            ax.axvspan(dom["start"], dom["end"], alpha=0.35, color=color,
                       zorder=0)
            mid = (dom["start"] + dom["end"]) / 2
            # Optional text offset for crowded labels (in residue units)
            text_x = mid + dom.get("text_offset", 0)
            # Place domain labels above the plot area (above the box)
            ax.text(text_x, 1.06, dom["name"], ha="center", va="bottom",
                    fontsize=11, fontstyle="italic", color="#333333",
                    transform=ax.get_xaxis_transform(), clip_on=False,
                    zorder=1)

    # Predictors: solid lines (blue family)
    ax.plot(positions, -z_rob, color=METRIC_COLORS["std_ddg"], linewidth=1.5, alpha=0.85,
            label=r"$-\mathrm{std}(\Delta\Delta G)$")
    if z_plddt is not None:
        ax.plot(positions, -z_plddt, color=METRIC_COLORS["plddt"], linewidth=1.5, alpha=0.85,
                label=r"$-$pLDDT")

    # Optional smoothed robustness trace
    if smooth_window > 0:
        smoothed = pd.Series(-z_rob).rolling(
            window=smooth_window, center=True, min_periods=1).mean().values
        ax.plot(positions, smoothed, color=METRIC_COLORS["std_ddg"], linewidth=2.5, alpha=0.9,
                label=f"smoothed (w={smooth_window})")

    # Targets: dashed lines (warm family)
    if z_rmsf is not None:
        ax.plot(positions, z_rmsf, color=METRIC_COLORS["rmsf"], linewidth=1.5, alpha=0.85,
                linestyle="--", label="RMSF")
    if z_bfac is not None:
        ax.plot(positions, z_bfac, color=METRIC_COLORS["bfactor"], linewidth=1.5, alpha=0.85,
                linestyle="--", label="B-factor")

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="-")
    # Combine protein name with x-axis label to avoid overlap with domain text
    prot_label = protein_id.upper().replace("_", " chain ")
    ax.set_xlabel(f"{prot_label} - residue position", fontsize=11)
    ax.set_ylabel("Z-scored value", fontsize=11)

    ax.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.85,
              edgecolor="none", bbox_to_anchor=(1.0, 1.0))
    ax.set_xlim(positions[0], positions[-1])
    ax.tick_params(labelsize=9)

    # Panel label (a) is added by LaTeX \panelimg, not embedded here

    plt.tight_layout()
    out = Path(output_dir)
    fig.savefig(out / f"{protein_id}_panel_A_lineplot.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{protein_id}_panel_A_lineplot.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Line plot saved: {out / f'{protein_id}_panel_A_lineplot.png'}")


# ============================================================================
# STANDALONE PANEL (single structure panel with label, title, colorbar)
# ============================================================================

def save_standalone_panel(img_path: str, output_path: str, label: str,
                          title: str, cmap, vmin: float, vmax: float,
                          vmin_label: str, vmax_label: str,
                          figsize=(6, 5), title_color: str = "black",
                          units: str = ""):
    """Wrap a rendered structure image into a standalone figure with
    colorbar showing numeric range and units.  Label/title handled by LaTeX."""
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.image as mpimg

    img = mpimg.imread(str(img_path))
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis("off")

    # Horizontal colorbar below the structure image with numeric ticks
    cbar_ax = ax.inset_axes([0.12, 0.02, 0.76, 0.03])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=10)
    # Show min/max numeric values plus units label on right
    cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
    cbar.set_ticklabels([f"{vmin:.1f}", f"{(vmin + vmax) / 2:.1f}", f"{vmax:.1f}"])
    if units:
        cbar.ax.text(1.04, 0.5, units, transform=cbar.ax.transAxes,
                     fontsize=10, va="center", ha="left")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    # Also save PDF version
    pdf_path = str(output_path).rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Standalone panel saved: {output_path}")


def generate_separate_panels(protein_id: str, output_dir: str,
                              robustness_df: pd.DataFrame = None,
                              rmsf_df: pd.DataFrame = None,
                              plddt_df: pd.DataFrame = None,
                              bfactor_df: pd.DataFrame = None,
                              rho_rmsf: float = None,
                              rho_plddt_rmsf: float = None,
                              rho_rob_bfac: float = None,
                              rho_plddt_bfac: float = None,
                              clip_pct: float = 10.0,
                              trim_termini: int = 0):
    """Generate each structure panel as a standalone figure with label,
    title, and colorbar. Panel (a) line plot is already standalone."""
    from matplotlib.colors import LinearSegmentedColormap

    out = Path(output_dir)
    panel_plddt = out / f"{protein_id}_panel_B_plddt.png"
    panel_rob = out / f"{protein_id}_panel_C_robustness.png"
    panel_rmsf = out / f"{protein_id}_panel_D_rmsf.png"
    panel_bfac = out / f"{protein_id}_panel_E_bfactor.png"

    rwb = LinearSegmentedColormap.from_list("rwb", ["red", "white", "blue"])
    bwr = LinearSegmentedColormap.from_list("bwr", ["blue", "white", "red"])

    def robust_range(vals, pct):
        v = vals[~np.isnan(vals)] if vals is not None else np.array([0, 1])
        if trim_termini > 0 and len(v) > 2 * trim_termini:
            v = v[trim_termini:-trim_termini]
        return (np.percentile(v, pct), np.percentile(v, 100 - pct))

    def rho_str(rho_rmsf_val, rho_bfac_val):
        parts = []
        if rho_rmsf_val is not None:
            parts.append(f"$\\rho_{{RMSF}}$ = {rho_rmsf_val:.2f}")
        if rho_bfac_val is not None:
            parts.append(f"$\\rho_{{Bfac}}$ = {rho_bfac_val:.2f}")
        return "\n" + ", ".join(parts) if parts else ""

    rob_vals = robustness_df["std_ddg"].values if robustness_df is not None else None
    rmsf_vals = rmsf_df["rmsf"].values if rmsf_df is not None else None
    plddt_vals = plddt_df["plddt"].values if plddt_df is not None else None
    bfac_vals = bfactor_df["bfactor"].values if bfactor_df is not None else None

    # Row 1: predictions (b) pLDDT, (c) std(DDG)
    # Row 2: responses   (d) RMSF,  (e) B-factor
    # Each tuple: (img, out, label, title, cmap, vmin, vmax, vmin_l, vmax_l, color, units)
    panels = []
    if panel_plddt.exists():
        r = robust_range(plddt_vals, clip_pct)
        panels.append((panel_plddt, out / f"{protein_id}_fig_panel_b.png",
                        "(b)", "pLDDT" + rho_str(rho_plddt_rmsf, rho_plddt_bfac),
                        rwb, r[0], r[1], "flexible", "rigid",
                        METRIC_COLORS["plddt"], ""))
    if panel_rob.exists():
        r = robust_range(rob_vals, clip_pct)
        panels.append((panel_rob, out / f"{protein_id}_fig_panel_c.png",
                        "(c)", r"std($\Delta\Delta G$)" + rho_str(rho_rmsf, rho_rob_bfac),
                        rwb, r[0], r[1], "flexible", "rigid",
                        METRIC_COLORS["std_ddg"], "kcal/mol"))
    if panel_rmsf.exists():
        r = robust_range(rmsf_vals, clip_pct)
        panels.append((panel_rmsf, out / f"{protein_id}_fig_panel_d.png",
                        "(d)", "RMSF",
                        bwr, r[0], r[1], "rigid", "flexible",
                        METRIC_COLORS["rmsf"], r"$\AA$"))
    if panel_bfac.exists():
        r = robust_range(bfac_vals, clip_pct)
        panels.append((panel_bfac, out / f"{protein_id}_fig_panel_e.png",
                        "(e)", "B-factor",
                        bwr, r[0], r[1], "rigid", "flexible",
                        METRIC_COLORS["bfactor"], r"$\AA^2$"))

    for img_path, out_path, label, title, cmap, vmin, vmax, vmin_l, vmax_l, title_color, units in panels:
        save_standalone_panel(str(img_path), str(out_path), label, title,
                              cmap, vmin, vmax, vmin_l, vmax_l,
                              title_color=title_color, units=units)

    # Also copy line plot as panel_a with consistent naming (always overwrite)
    panel_a_src = out / f"{protein_id}_panel_A_lineplot.png"
    panel_a_dst = out / f"{protein_id}_fig_panel_a.png"
    if panel_a_src.exists():
        import shutil
        shutil.copy2(str(panel_a_src), str(panel_a_dst))
        pdf_src = out / f"{protein_id}_panel_A_lineplot.pdf"
        pdf_dst = out / f"{protein_id}_fig_panel_a.pdf"
        if pdf_src.exists():
            shutil.copy2(str(pdf_src), str(pdf_dst))
        print(f"  Panel (a) copied: {panel_a_dst}")


# ============================================================================
# COMPOSITE FIGURE
# ============================================================================

def composite_figure(protein_id: str, output_dir: str,
                     robustness_df: pd.DataFrame = None,
                     rmsf_df: pd.DataFrame = None,
                     plddt_df: pd.DataFrame = None,
                     bfactor_df: pd.DataFrame = None,
                     rho_rmsf: float = None,
                     rho_plddt_rmsf: float = None,
                     rho_rob_bfac: float = None,
                     rho_plddt_bfac: float = None,
                     clip_pct: float = 10.0,
                     trim_termini: int = 0):
    """Combine panels into a single matplotlib figure.

    Layout:  (a) line plot on top (full width)
             (b) pLDDT      (c) std(DDG)   2x2 grid below (predictions)
             (d) RMSF       (e) B-factor                   (responses)
    """
    out = Path(output_dir)
    panel_lineplot = out / f"{protein_id}_panel_A_lineplot.png"
    panel_plddt = out / f"{protein_id}_panel_B_plddt.png"
    panel_rob = out / f"{protein_id}_panel_C_robustness.png"
    panel_rmsf = out / f"{protein_id}_panel_D_rmsf.png"
    panel_bfac = out / f"{protein_id}_panel_E_bfactor.png"

    required = [panel_lineplot, panel_rob, panel_rmsf]
    if not all(p.exists() for p in required):
        print("  Skipping composite: not all panels available yet.")
        all_panels = [panel_lineplot, panel_plddt, panel_rob, panel_rmsf, panel_bfac]
        print(f"  Missing: {[str(p) for p in all_panels if not p.exists()]}")
        return

    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.image as mpimg

    img_lineplot = mpimg.imread(str(panel_lineplot))
    img_plddt = mpimg.imread(str(panel_plddt)) if panel_plddt.exists() else None
    img_rob = mpimg.imread(str(panel_rob))
    img_rmsf = mpimg.imread(str(panel_rmsf))
    img_bfac = mpimg.imread(str(panel_bfac)) if panel_bfac.exists() else None

    has_bfac = img_bfac is not None
    has_plddt = img_plddt is not None

    # Build subtitles: each predictor shows rho with both targets
    def rho_str(rho_rmsf_val, rho_bfac_val):
        parts = []
        if rho_rmsf_val is not None:
            parts.append(f"$\\rho_{{RMSF}}$ = {rho_rmsf_val:.2f}")
        if rho_bfac_val is not None:
            parts.append(f"$\\rho_{{Bfac}}$ = {rho_bfac_val:.2f}")
        return "\n" + ", ".join(parts) if parts else ""

    rob_title = r"std($\Delta\Delta G$)" + rho_str(rho_rmsf, rho_rob_bfac)
    plddt_title = "pLDDT" + rho_str(rho_plddt_rmsf, rho_plddt_bfac)
    rmsf_title = "RMSF"
    bfac_title = "B-factor"

    # Helper for robust color ranges
    def robust_range(vals, pct):
        v = vals[~np.isnan(vals)] if vals is not None else np.array([0, 1])
        if trim_termini > 0 and len(v) > 2 * trim_termini:
            v = v[trim_termini:-trim_termini]
        return (np.percentile(v, pct), np.percentile(v, 100 - pct))

    rob_vals = robustness_df["std_ddg"].values if robustness_df is not None else None
    rmsf_vals = rmsf_df["rmsf"].values if rmsf_df is not None else None
    plddt_vals = plddt_df["plddt"].values if plddt_df is not None else None
    bfac_vals = bfactor_df["bfactor"].values if bfactor_df is not None else None

    # 2x2 panel layout: top = predictions, bottom = responses
    #   (b) pLDDT      (c) std(DDG)
    #   (d) RMSF        (e) B-factor
    rwb = LinearSegmentedColormap.from_list("rwb", ["red", "white", "blue"])
    bwr = LinearSegmentedColormap.from_list("bwr", ["blue", "white", "red"])

    panels_2x2 = []
    if has_plddt:
        panels_2x2.append(
            # row 0, col 0: pLDDT (prediction)
            {"img": img_plddt, "label": "(b)", "title": plddt_title,
             "cmap": rwb, "vmin_label": "flexible", "vmax_label": "rigid",
             "range": robust_range(plddt_vals, clip_pct), "row": 0, "col": 0,
             "title_color": METRIC_COLORS["plddt"]})
    panels_2x2.append(
        # row 0, col 1: robustness (prediction)
        {"img": img_rob, "label": "(c)", "title": rob_title,
         "cmap": rwb, "vmin_label": "flexible", "vmax_label": "rigid",
         "range": robust_range(rob_vals, clip_pct), "row": 0, "col": 1,
         "title_color": METRIC_COLORS["std_ddg"]})
    panels_2x2.append(
        # row 1, col 0: RMSF (response)
        {"img": img_rmsf, "label": "(d)", "title": rmsf_title,
         "cmap": bwr, "vmin_label": "rigid", "vmax_label": "flexible",
         "range": robust_range(rmsf_vals, clip_pct), "row": 1, "col": 0,
         "title_color": METRIC_COLORS["rmsf"]})
    if has_bfac:
        panels_2x2.append(
            {"img": img_bfac, "label": "(e)", "title": bfac_title,
             "cmap": bwr, "vmin_label": "rigid", "vmax_label": "flexible",
             "range": robust_range(bfac_vals, clip_pct), "row": 1, "col": 1,
             "title_color": METRIC_COLORS["bfactor"]})

    # Figure layout: line plot on top, 2x2 structure grid below (tight)
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.55, 1.0, 1.0],
                  hspace=0.02, wspace=0.04)

    # --- Top row: line plot spanning both columns ---
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.imshow(img_lineplot)
    ax_a.axis("off")

    # --- Bottom 2x2: structure panels ---
    for info in panels_2x2:
        ax = fig.add_subplot(gs[1 + info["row"], info["col"]])
        ax.imshow(info["img"])
        ax.axis("off")
        # Panel label and title — top-left corner, same height
        ax.text(0.02, 0.98, info["label"], transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left")
        ax.text(0.10, 0.98, info["title"], transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top", ha="left",
                color=info.get("title_color", "black"))

        # Horizontal colorbar below the structure image
        cbar_ax = ax.inset_axes([0.15, 0.02, 0.7, 0.025])
        vmin, vmax = info["range"]
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=info["cmap"], norm=norm)
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.text(-0.02, 0.5, info["vmin_label"], transform=cbar.ax.transAxes,
                     fontsize=6.5, va="center", ha="right", fontstyle="italic")
        cbar.ax.text(1.02, 0.5, info["vmax_label"], transform=cbar.ax.transAxes,
                     fontsize=6.5, va="center", ha="left", fontstyle="italic")

    plt.savefig(out / f"{protein_id}_fig5_composite.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.savefig(out / f"{protein_id}_fig5_composite.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Composite saved: {out / f'{protein_id}_fig5_composite.png'}")


# ============================================================================
# MAIN
# ============================================================================

def process_protein(protein_id: str, args):
    """Process a single protein: load data, generate line plot, PyMOL script,
    and composite figure."""
    out = Path(args.output_dir)
    chain = protein_id.split("_")[-1] if "_" in protein_id else "A"
    protein_dir = Path(args.atlas_dir) / "proteins" / protein_id

    print(f"\n=== Case study figure for {protein_id} ===")

    # --- Load data ---
    print("Loading data...")
    robustness_df = load_robustness(Path(args.robustness_dir), args.scorer, protein_id)
    if robustness_df is None:
        print(f"ERROR: No robustness data found for {protein_id}, skipping.")
        return
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

    # --- Compute correlations ---
    rho_rmsf = None
    if rmsf_df is not None:
        rho, pval = stats.spearmanr(robustness_df["std_ddg"].values,
                                     rmsf_df["rmsf"].values,
                                     nan_policy="omit")
        rho_rmsf = rho
        print(f"  Spearman rho (std_ddg vs RMSF): {rho:.3f} (p={pval:.2e})")

    rho_plddt_rmsf = None
    if plddt_df is not None and rmsf_df is not None:
        rho, pval = stats.spearmanr(plddt_df["plddt"].values,
                                     rmsf_df["rmsf"].values,
                                     nan_policy="omit")
        rho_plddt_rmsf = rho
        print(f"  Spearman rho (pLDDT vs RMSF): {rho:.3f} (p={pval:.2e})")

    rho_rob_bfac = None
    if bfactor_df is not None:
        mask = bfactor_df["bfactor"].notna()
        if mask.sum() > 10:
            rho, pval = stats.spearmanr(robustness_df.loc[mask, "std_ddg"].values,
                                         bfactor_df.loc[mask, "bfactor"].values,
                                         nan_policy="omit")
            rho_rob_bfac = rho
            print(f"  Spearman rho (std_ddg vs B-factor): {rho:.3f} (p={pval:.2e})")

    rho_plddt_bfac = None
    if plddt_df is not None and bfactor_df is not None:
        mask = bfactor_df["bfactor"].notna()
        if mask.sum() > 10:
            rho, pval = stats.spearmanr(plddt_df.loc[mask, "plddt"].values,
                                         bfactor_df.loc[mask, "bfactor"].values,
                                         nan_policy="omit")
            rho_plddt_bfac = rho
            print(f"  Spearman rho (pLDDT vs B-factor): {rho:.3f} (p={pval:.2e})")

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

    # --- Load domain annotations ---
    # Try per-protein domains file first, then fall back to --domains arg
    domains_path = args.domains
    if domains_path is None:
        # Auto-discover: look for data/domains/<protein_id>_domains.json
        script_dir = Path(__file__).resolve().parent.parent
        auto_path = script_dir / "data" / "domains" / f"{protein_id}_domains.json"
        if auto_path.exists():
            domains_path = str(auto_path)
            print(f"  Auto-discovered domains: {auto_path}")
    domains = load_domains(domains_path)
    if domains:
        print(f"  Domains: {len(domains)} regions loaded")

    # --- Generate line plot (Panel A) ---
    print("Generating line plot (Panel A)...")
    generate_line_plot(protein_id, robustness_df, rmsf_df, plddt_df,
                       bfactor_df, args.output_dir, rho_rmsf=rho_rmsf,
                       domains=domains,
                       smooth_window=args.smooth_window)

    # --- Generate PyMOL script (Panels B-D) ---
    if not args.line_plot_only:
        if pdb_path is None:
            print("ERROR: Cannot generate PyMOL panels without PDB file")
        else:
            print("Generating PyMOL script (Panels B-E)...")
            pml_script = generate_pymol_script(
                str(pdb_path), protein_id, chain,
                robustness_df, rmsf_df, plddt_df, bfactor_df,
                args.output_dir, args.width, args.height,
                clip_pct=args.clip_pct,
                trim_termini=args.trim_termini)

            pml_path = out / f"{protein_id}_pymol_render.py"
            with open(pml_path, "w") as f:
                f.write(pml_script)
            print(f"  PyMOL script saved: {pml_path}")
            print(f"  To render: python {pml_path}")

            if args.run_pymol:
                print("Running PyMOL...")
                result = subprocess.run(
                    ["python", str(pml_path)],
                    capture_output=True, text=True)
                if result.returncode == 0:
                    print("  PyMOL rendering complete!")
                    if result.stdout.strip():
                        print(f"  stdout: {result.stdout[:500]}")
                else:
                    print(f"  PyMOL failed (exit {result.returncode}):")
                    print(f"  stderr: {result.stderr[:500]}")
                    print(f"  stdout: {result.stdout[:500]}")

    # --- Separate panels (standalone figures for each panel) ---
    if args.separate_panels and not args.line_plot_only:
        print("Generating separate standalone panels...")
        try:
            generate_separate_panels(protein_id, args.output_dir,
                                     robustness_df=robustness_df,
                                     rmsf_df=rmsf_df,
                                     plddt_df=plddt_df,
                                     bfactor_df=bfactor_df,
                                     rho_rmsf=rho_rmsf,
                                     rho_plddt_rmsf=rho_plddt_rmsf,
                                     rho_rob_bfac=rho_rob_bfac,
                                     rho_plddt_bfac=rho_plddt_bfac,
                                     clip_pct=args.clip_pct,
                                     trim_termini=args.trim_termini)
        except Exception as e:
            print(f"  Separate panels failed: {e}")

    # --- Composite ---
    if not args.no_composite and not args.line_plot_only:
        print("Generating composite figure...")
        try:
            composite_figure(protein_id, args.output_dir,
                            robustness_df=robustness_df,
                            rmsf_df=rmsf_df,
                            plddt_df=plddt_df,
                            bfactor_df=bfactor_df,
                            rho_rmsf=rho_rmsf,
                            rho_plddt_rmsf=rho_plddt_rmsf,
                            rho_rob_bfac=rho_rob_bfac,
                            rho_plddt_bfac=rho_plddt_bfac,
                            clip_pct=args.clip_pct,
                            trim_termini=args.trim_termini)
        except ImportError:
            print("  Pillow not installed, skipping composite (pip install Pillow)")
        except Exception as e:
            print(f"  Composite failed: {e}")

    print(f"Done with {protein_id}!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate case study figure for robustness-dynamics paper")
    parser.add_argument("--protein_id", required=True, nargs="+",
                        help="ATLAS protein ID(s) (e.g., 1ez3_B 1qcs_A 2vfx_C)")
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
    parser.add_argument("--domains", default=None,
                        help="Path to JSON file with domain annotations "
                             "(list of {name, start, end, [color]}). "
                             "If not provided, auto-discovers "
                             "data/domains/<protein_id>_domains.json")
    parser.add_argument("--separate-panels", action="store_true",
                        help="Generate each panel (a-e) as a standalone figure "
                             "with label, title, and colorbar")
    parser.add_argument("--no-composite", action="store_true",
                        help="Skip composite figure generation")
    parser.add_argument("--smooth-window", type=int, default=0,
                        help="Rolling average window for robustness trace "
                             "(0 = no smoothing)")
    parser.add_argument("--clip-pct", type=float, default=10.0,
                        help="Percentile for robust color clipping in PyMOL "
                             "renders (default: 10 = 10th-90th percentile)")
    parser.add_argument("--trim-termini", type=int, default=0,
                        help="Exclude N first/last residues from color range "
                             "computation (0 = use all residues)")
    parser.add_argument("--width", type=int, default=2400,
                        help="PyMOL render width in pixels")
    parser.add_argument("--height", type=int, default=1800,
                        help="PyMOL render height in pixels")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for protein_id in args.protein_id:
        process_protein(protein_id, args)

    print(f"\n=== All done ({len(args.protein_id)} proteins) ===")


if __name__ == "__main__":
    main()
