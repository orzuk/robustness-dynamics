#!/usr/bin/env python3
"""
Generate LaTeX tables for the robustness-dynamics paper from unified results.

Reads unified_results.json (produced by collect_results.py) and outputs
LaTeX table fragments that can be \\input{} into the paper or copy-pasted.

Usage:
  python generate_latex_tables.py  # uses default paths under data/paper_results/
  python generate_latex_tables.py --results unified_results.json --table table1
"""

import json
import argparse
from pathlib import Path

from paper_config import (
    TABLE1_COLUMNS_ALL as TABLE1_COLUMNS, TABLE1_PREDICTORS,
    TABLE2_SS_STRATA, TABLE2_BURIAL_STRATA,
    TABLE3_MODEL_ORDER, ALT_ROBUSTNESS_MEASURES,
    DATASETS,
)


def _fmt(val, decimals=3, sign=False):
    """Format a number for LaTeX, handling None/missing."""
    if val is None:
        return "---"
    if sign and val > 0:
        return f"$+${abs(val):.{decimals}f}"
    if sign and val < 0:
        return f"$-${abs(val):.{decimals}f}"
    return f"{val:.{decimals}f}"


def _neg_sign(val, decimals=3):
    """Format with $-$ for negatives, no sign for positives."""
    if val is None:
        return "---"
    if val < 0:
        return f"$-${abs(val):.{decimals}f}"
    return f"{val:.{decimals}f}"


def _signed(val, decimals=3):
    """Format with explicit sign ($-$0.xxx or $+$0.xxx). Use only for Delta R^2."""
    if val is None:
        return "---"
    if val < 0:
        return f"$-${abs(val):.{decimals}f}"
    return f"$+${val:.{decimals}f}"


def _bold(text):
    """Wrap LaTeX text in bold."""
    return r"\textbf{" + text + "}"


def _highlight_best_in_row(cells, higher_is_better=True, use_abs=False):
    """Bold the best (highest |value| or highest/lowest value) among cells.

    cells: list of (formatted_string, raw_value_or_None).
    Returns list of formatted strings with the best one bolded.
    """
    valid = [(i, abs(v) if use_abs else v)
             for i, (_, v) in enumerate(cells) if v is not None]
    if not valid:
        return [c[0] for c in cells]
    if higher_is_better:
        best_idx = max(valid, key=lambda x: x[1])[0]
    else:
        best_idx = min(valid, key=lambda x: x[1])[0]
    result = []
    for i, (fmt_str, _) in enumerate(cells):
        if i == best_idx and fmt_str != "---":
            result.append(_bold(fmt_str))
        else:
            result.append(fmt_str)
    return result


def _highlight_best_in_columns(grid):
    """Bold the best |rho| within each column across rows.

    grid: list of (row_label, [(cell_str, raw_value), ...])
    Returns list of (row_label, [cell_str, ...]) with best per column bolded.
    """
    n_rows = len(grid)
    if n_rows == 0:
        return []
    n_cols = len(grid[0][1])
    # Find best row index for each column
    best_per_col = {}
    for col in range(n_cols):
        valid = [(row_idx, abs(grid[row_idx][1][col][1]))
                 for row_idx in range(n_rows)
                 if grid[row_idx][1][col][1] is not None]
        if valid:
            best_per_col[col] = max(valid, key=lambda x: x[1])[0]
    # Build output with bolding
    result = []
    for row_idx, (label, cells) in enumerate(grid):
        out_cells = []
        for col, (fmt_str, _) in enumerate(cells):
            if col in best_per_col and best_per_col[col] == row_idx and fmt_str != "---":
                out_cells.append(_bold(fmt_str))
            else:
                out_cells.append(fmt_str)
        result.append((label, out_cells))
    return result


def _get_run(results, dataset, scorer, target):
    """Get a run from the unified results, or empty dict."""
    key = f"{dataset}_{scorer}_{target}"
    return results.get("runs", {}).get(key, {})


def _get_corr(run, field):
    """Safely get a correlation field."""
    return run.get("correlation", {}).get("pooled", {}).get(field)


def _get_pp(run, field):
    """Safely get a per-protein summary field."""
    return run.get("correlation", {}).get("per_protein_summary", {}).get(field)


def _get_strat(run, strat_type, category, field):
    """Safely get a stratified field."""
    return run.get("stratified", {}).get(strat_type, {}).get(category, {}).get(field)


# ============================================================================
# TABLE 1: Main bivariate results
# ============================================================================

def _table1_header(lines, n_cols):
    """Shared column headers for Table 1 and Supp Table S1."""
    headers1 = [""]
    headers2 = [""]
    for ds_name, target in TABLE1_COLUMNS:
        ds = DATASETS[ds_name]
        headers1.append(r"\textbf{" + ds.display_name + "}")
        tgt_label = "RMSF" if target == "rmsf" else "B-factor"
        if ds_name == "rci_s2":
            tgt_label = r"$1{-}S^2_\mathrm{RCI}$"
        headers2.append(r"\textbf{" + tgt_label + "}")
    lines.append(" & ".join(headers1) + r" \\")
    lines.append(" & ".join(headers2) + r" \\")
    lines.append(r"\midrule")


def generate_table1(results: dict) -> str:
    """Generate Table 1: simple bivariate correlations and R^2."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Bivariate correlations between per-residue predictors and dynamics")
    lines.append(r"targets across all dataset--target combinations.")
    lines.append(r"\emph{Median per-protein $\rho$}: median across proteins of the")
    lines.append(r"within-protein Spearman rank correlation.")
    lines.append(r"\emph{Pooled $\rho$}: Spearman correlation on all residues after")
    lines.append(r"within-protein z-scoring.")
    lines.append(r"\emph{Pooled $R^2$}: OLS on z-scored residues.")
    lines.append(r"Conservation scores from ConSurf Rate4Site (ATLAS only).")
    lines.append(r"Best value in each predictor comparison is shown in \textbf{bold}.")
    lines.append(r"All pooled correlations significant at $p < 10^{-10}$.")
    lines.append(r"Partial correlations and incremental $R^2$ are in Supplementary Table~\ref{tab:supp_partial}.}")
    lines.append(r"\label{tab:pooled}")
    lines.append(r"\small")
    n_cols = len(TABLE1_COLUMNS)
    col_spec = "r" * n_cols
    lines.append(r"\begin{tabular}{@{}l " + col_spec + r"@{}}")
    lines.append(r"\toprule")

    _table1_header(lines, n_cols)

    # n proteins / n residues
    row_np = ["$n$ proteins"]
    row_nr = ["$n$ residues"]
    for ds_name, target in TABLE1_COLUMNS:
        run = _get_run(results, ds_name, "thermompnn", target)
        np_val = run.get("n_proteins")
        nr_val = run.get("n_residues")
        row_np.append(f"{np_val:,}" if np_val else "---")
        row_nr.append(f"{nr_val:,}" if nr_val else "---")
    lines.append(" & ".join(row_np) + r" \\")
    lines.append(" & ".join(row_nr) + r" \\")
    lines.append(r"\midrule")

    # --- Median per-protein rho (highlight best |rho| per column) ---
    lines.append(r"\multicolumn{" + str(n_cols + 1) + r"}{l}{\textit{Median per-protein Spearman $\rho$ (predictor, target)}} \\")

    # Predictors: ESM-1v, ThermoMPNN, pLDDT, SASA, Conservation
    pred_list = ["esm1v", "thermompnn", "plddt", "sasa", "conservation"]
    pred_labels = {"esm1v": "ESM-1v", "thermompnn": "ThermoMPNN",
                   "plddt": "pLDDT", "sasa": "SASA",
                   "conservation": "Conservation"}
    med_rho_grid = []
    for pred in pred_list:
        row_cells = []
        for ds_name, target in TABLE1_COLUMNS:
            if pred == "conservation":
                run = _get_run(results, ds_name, "thermompnn", target)
                val = _get_pp(run, "median_rho_conservation")
            elif pred in ("plddt", "sasa"):
                run = _get_run(results, ds_name, "thermompnn", target)
                val = _get_pp(run, f"median_rho_{pred}")
            else:
                run = _get_run(results, ds_name, pred, target)
                val = _get_pp(run, "median_rho_robustness")
            row_cells.append((_neg_sign(val), val))
        med_rho_grid.append(row_cells)

    # Highlight best |rho| per column
    n_cols = len(TABLE1_COLUMNS)
    for col_idx in range(n_cols):
        col_cells = [(med_rho_grid[pred_idx][col_idx][0],
                       med_rho_grid[pred_idx][col_idx][1])
                      for pred_idx in range(len(pred_list))]
        highlighted = _highlight_best_in_row(col_cells, higher_is_better=True, use_abs=True)
        for pred_idx in range(len(pred_list)):
            old_fmt, old_val = med_rho_grid[pred_idx][col_idx]
            med_rho_grid[pred_idx][col_idx] = (highlighted[pred_idx], old_val)

    for pred_idx, pred in enumerate(pred_list):
        row = [r"\quad " + pred_labels[pred]]
        for col_idx in range(n_cols):
            row.append(med_rho_grid[pred_idx][col_idx][0])
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")

    # --- Pooled rho (highlight best |rho| per column) ---
    lines.append(r"\multicolumn{" + str(n_cols + 1) + r"}{l}{\textit{Pooled Spearman $\rho$ (z-scored residues)}} \\")
    pooled_rho_grid = []
    for pred in pred_list:
        row_cells = []
        for ds_name, target in TABLE1_COLUMNS:
            if pred == "conservation":
                run = _get_run(results, ds_name, "thermompnn", target)
                val = _get_corr(run, "pooled_rho_conservation")
            elif pred in ("plddt", "sasa"):
                run = _get_run(results, ds_name, "thermompnn", target)
                val = _get_corr(run, f"pooled_rho_{pred}")
            else:
                run = _get_run(results, ds_name, pred, target)
                val = _get_corr(run, "pooled_rho_robustness")
            row_cells.append((_neg_sign(val), val))
        pooled_rho_grid.append(row_cells)

    for col_idx in range(n_cols):
        col_cells = [(pooled_rho_grid[p][col_idx][0],
                       pooled_rho_grid[p][col_idx][1])
                      for p in range(len(pred_list))]
        highlighted = _highlight_best_in_row(col_cells, higher_is_better=True, use_abs=True)
        for p in range(len(pred_list)):
            pooled_rho_grid[p][col_idx] = (highlighted[p], pooled_rho_grid[p][col_idx][1])

    for pred_idx, pred in enumerate(pred_list):
        row = [r"\quad " + pred_labels[pred]]
        for col_idx in range(n_cols):
            row.append(pooled_rho_grid[pred_idx][col_idx][0])
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")

    # --- Pooled R² (highlight best per column) ---
    lines.append(r"\multicolumn{" + str(n_cols + 1) + r"}{l}{\textit{Pooled $R^2$ (OLS on z-scored residues)}} \\")
    r2_preds = ["esm1v", "thermompnn", "plddt"]
    r2_grid = []
    for pred in r2_preds:
        row_cells = []
        for ds_name, target in TABLE1_COLUMNS:
            if pred == "plddt":
                run = _get_run(results, ds_name, "thermompnn", target)
                val = _get_corr(run, "pooled_r2_plddt")
            else:
                run = _get_run(results, ds_name, pred, target)
                val = _get_corr(run, "pooled_r2_robustness")
            row_cells.append((_fmt(val), val))
        r2_grid.append(row_cells)

    for col_idx in range(n_cols):
        col_cells = [(r2_grid[p][col_idx][0], r2_grid[p][col_idx][1])
                      for p in range(len(r2_preds))]
        highlighted = _highlight_best_in_row(col_cells, higher_is_better=True)
        for p in range(len(r2_preds)):
            r2_grid[p][col_idx] = (highlighted[p], r2_grid[p][col_idx][1])

    for pred_idx, pred in enumerate(r2_preds):
        label = {"esm1v": "ESM-1v", "thermompnn": "ThermoMPNN", "plddt": "pLDDT"}[pred]
        row = [r"\quad " + label]
        for col_idx in range(n_cols):
            row.append(r2_grid[pred_idx][col_idx][0])
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")

    # --- Frac beats pLDDT ---
    row = [r"Frac $|\rho_\text{rob}| > |\rho_\text{pLDDT}|$"]
    for ds_name, target in TABLE1_COLUMNS:
        run = _get_run(results, ds_name, "thermompnn", target)
        val = _get_pp(run, "frac_robustness_beats_plddt")
        if val is not None:
            row.append(f"{val*100:.1f}\\%")
        else:
            row.append("---")
    lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ============================================================================
# TABLE 2: Stratified correlations
# ============================================================================

def generate_table2(results: dict) -> str:
    """Generate Table 2: stratified pooled rho across all datasets."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Stratified pooled Spearman $\rho$ between ThermoMPNN")
    lines.append(r"$\operatorname{std}(\Delta\Delta G)$ and dynamics target,")
    lines.append(r"grouped by secondary structure (DSSP classification: $\alpha$-helix H,")
    lines.append(r"$\beta$-sheet E, coil C) and by burial class based on relative solvent accessibility (RSA):")
    lines.append(r"core (RSA $<$ 20\%), boundary (20--50\%), surface ($>$ 50\%).")
    lines.append(r"pLDDT values shown in parentheses where available.")
    lines.append(r"Best $|\rho|$ within each column (across structural classes) is shown in \textbf{bold}.}")
    lines.append(r"\label{tab:stratified}")
    lines.append(r"\small")
    n_cols = len(TABLE1_COLUMNS)
    col_spec = "r" * n_cols
    lines.append(r"\begin{tabular}{@{}l " + col_spec + r"@{}}")
    lines.append(r"\toprule")

    # Headers (same as Table 1)
    headers1 = [""]
    headers2 = [""]
    for ds_name, target in TABLE1_COLUMNS:
        ds = DATASETS[ds_name]
        headers1.append(r"\textbf{" + ds.display_name + "}")
        tgt_label = "RMSF" if target == "rmsf" else "B-factor"
        if ds_name == "rci_s2":
            tgt_label = r"$1{-}S^2_\mathrm{RCI}$"
        headers2.append(r"\textbf{" + tgt_label + "}")
    lines.append(" & ".join(headers1) + r" \\")
    lines.append(" & ".join(headers2) + r" \\")
    lines.append(r"\midrule")

    n_cols = len(TABLE1_COLUMNS)

    # Secondary structure
    lines.append(r"\multicolumn{" + str(n_cols + 1) + r"}{l}{\textit{Secondary structure}} \\")
    ss_labels = {"H": r"$\alpha$-helix", "E": r"$\beta$-sheet", "C": "Coil"}
    # Collect all rows first, then bold best |rho| within each column
    ss_grid = []  # list of (label, [(cell_str, rho_rob), ...])
    for ss in TABLE2_SS_STRATA:
        row_cells = []
        for ds_name, target in TABLE1_COLUMNS:
            run = _get_run(results, ds_name, "thermompnn", target)
            rho_rob = _get_strat(run, "secondary_structure", ss, "rho_robustness")
            rho_plddt = _get_strat(run, "secondary_structure", ss, "rho_plddt")
            if rho_rob is not None:
                cell = _neg_sign(rho_rob)
                if rho_plddt is not None:
                    cell += r"\,(" + _neg_sign(rho_plddt) + ")"
            else:
                cell = "---"
            row_cells.append((cell, rho_rob))
        ss_grid.append((r"\quad " + ss_labels[ss], row_cells))
    # Bold the best |rho| within each column (across strata)
    for row_label, row_cells in _highlight_best_in_columns(ss_grid):
        lines.append(" & ".join([row_label] + row_cells) + r" \\")
    lines.append(r"\midrule")

    # Burial
    lines.append(r"\multicolumn{" + str(n_cols + 1) + r"}{l}{\textit{Burial (RSA cutoffs)}} \\")
    burial_labels = {"core": "Core", "boundary": "Boundary", "surface": "Surface"}
    burial_grid = []
    for burial in TABLE2_BURIAL_STRATA:
        row_cells = []
        for ds_name, target in TABLE1_COLUMNS:
            run = _get_run(results, ds_name, "thermompnn", target)
            rho_rob = _get_strat(run, "burial", burial, "rho_robustness")
            rho_plddt = _get_strat(run, "burial", burial, "rho_plddt")
            if rho_rob is not None:
                cell = _neg_sign(rho_rob)
                if rho_plddt is not None:
                    cell += r"\,(" + _neg_sign(rho_plddt) + ")"
            else:
                cell = "---"
            row_cells.append((cell, rho_rob))
        burial_grid.append((r"\quad " + burial_labels[burial], row_cells))
    for row_label, row_cells in _highlight_best_in_columns(burial_grid):
        lines.append(" & ".join([row_label] + row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ============================================================================
# TABLE 3: Alternative measures + Multi-DDG regression (united)
# ============================================================================

def generate_table3(results: dict) -> str:
    """Generate Table 3: alternative robustness measures + multi-DDG regression."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comparison of robustness summary statistics and")
    lines.append(r"multi-$\Delta\Delta G$ regression models (ThermoMPNN scorer).")
    lines.append(r"\emph{Top panel}: median per-protein Spearman $\rho$ for scalar")
    lines.append(r"robustness summaries.")
    lines.append(r"\emph{Bottom panel}: 5-fold protein-level cross-validated $R^2$")
    lines.append(r"for regression models predicting dynamics from $\Delta\Delta G$")
    lines.append(r"features; proteins are held out as entire units so the model")
    lines.append(r"must generalize to unseen proteins.")
    lines.append(r"``Feats'' = number of input features.")
    lines.append(r"Best value in each column section is shown in \textbf{bold}.}")
    lines.append(r"\label{tab:alt_and_multi}")
    lines.append(r"\small")
    n_cols = len(TABLE1_COLUMNS)
    col_spec = "r" * n_cols
    lines.append(r"\begin{tabular}{@{}l r " + col_spec + r"@{}}")
    lines.append(r"\toprule")

    # Headers - build dynamically from TABLE1_COLUMNS
    ds_short = {"atlas": "ATLAS", "bbflow": "BBFlow", "pdb_designs": "PDB des.",
                "rci_s2": "NMR"}
    tgt_short = {"rmsf": "RMSF", "bfactor": "B-fac"}
    h1 = ["", ""]
    h2 = ["", "Feats"]
    for ds_name, target in TABLE1_COLUMNS:
        h1.append(r"\textbf{" + ds_short.get(ds_name, ds_name) + "}")
        tgt_label = tgt_short.get(target, target)
        if ds_name == "rci_s2":
            tgt_label = r"$1{-}S^2_\mathrm{RCI}$"
        h2.append(r"\textbf{" + tgt_label + "}")
    lines.append(" & ".join(h1) + r" \\")
    lines.append(" & ".join(h2) + r" \\")
    lines.append(r"\midrule")

    # --- Top half: scalar robustness summaries (median per-protein rho) ---
    multicolspan = n_cols + 2
    lines.append(r"\multicolumn{" + str(multicolspan) + r"}{l}{\textit{Scalar robustness summaries (med.\ per-protein $\rho$)}} \\")

    # Collect all alt robustness values for per-column highlighting
    alt_grid = []  # [measure_idx][col_idx] = (formatted, raw_val)
    for measure_key, measure_label in ALT_ROBUSTNESS_MEASURES:
        row_cells = []
        for ds_name, target in TABLE1_COLUMNS:
            run = _get_run(results, ds_name, "thermompnn", target)
            alt = run.get("alt_robustness_medians", {})
            val = alt.get(measure_key)
            if val is not None:
                row_cells.append((_neg_sign(val), val))
            else:
                row_cells.append(("---", None))
        alt_grid.append(row_cells)

    # Add pLDDT baseline row
    plddt_cells = []
    for ds_name, target in TABLE1_COLUMNS:
        run = _get_run(results, ds_name, "thermompnn", target)
        val = _get_pp(run, "median_rho_plddt")
        if val is not None:
            plddt_cells.append((_neg_sign(val), val))
        else:
            plddt_cells.append(("---", None))
    alt_grid.append(plddt_cells)

    # Highlight best |rho| per column across all scalar measures + pLDDT
    for col_idx in range(n_cols):
        col_cells = [(alt_grid[m][col_idx][0], alt_grid[m][col_idx][1])
                      for m in range(len(alt_grid))]
        highlighted = _highlight_best_in_row(col_cells, higher_is_better=True, use_abs=True)
        for m in range(len(alt_grid)):
            alt_grid[m][col_idx] = (highlighted[m], alt_grid[m][col_idx][1])

    # Output alt robustness rows
    for m_idx, (measure_key, measure_label) in enumerate(ALT_ROBUSTNESS_MEASURES):
        is_primary = measure_key == "std_ddg"
        label = measure_label
        if is_primary:
            label += r" \textbf{(primary)}"
        row = [r"\quad " + label, "1"]
        for col_idx in range(n_cols):
            row.append(alt_grid[m_idx][col_idx][0])
        lines.append(" & ".join(row) + r" \\")

    # pLDDT baseline row
    plddt_idx = len(ALT_ROBUSTNESS_MEASURES)
    row = [r"\quad pLDDT (baseline)", "1"]
    for col_idx in range(n_cols):
        row.append(alt_grid[plddt_idx][col_idx][0])
    lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")

    # --- Bottom half: multi-DDG regression (CV R²) ---
    lines.append(r"\multicolumn{" + str(multicolspan) + r"}{l}{\textit{Regression models (5-fold protein-level CV $R^2$)}} \\")

    # Model display names and feature counts
    model_info = {
        "ols_std_ddg": (r"OLS $\operatorname{std}(\Delta\Delta G)$", 1),
        "ols_mean_ddg": (r"OLS mean $\Delta\Delta G$", 1),
        "ols_mean_abs_ddg": (r"OLS mean$|\Delta\Delta G|$", 1),
        "ols_sasa": ("OLS SASA", 1),
        "ols_plddt": ("OLS pLDDT", 1),
        "ols_std_plddt": (r"OLS std $+$ pLDDT", 2),
        "ridge_20ddg": (r"Ridge: 20 $\Delta\Delta G$", 20),
        "ridge_nonlinear_only": ("Ridge: 4 NL only", 4),
        "ridge_20ddg_nonlinear": (r"Ridge: 20 $\Delta\Delta G$ + 4 NL", 24),
        "ridge_20ddg_plddt": (r"Ridge: 20 $\Delta\Delta G$ + pLDDT", 21),
        "ridge_20ddg_nonlinear_plddt": (r"Ridge: 20 $\Delta\Delta G$ + NL + pLDDT", 25),
    }

    # Collect all model values for highlighting
    model_grid = []  # [model_idx][col_idx] = (formatted, raw_val)
    model_keys_used = []
    for model_key in TABLE3_MODEL_ORDER:
        if model_key not in model_info:
            continue
        model_keys_used.append(model_key)
        row_cells = []
        for ds_name, target in TABLE1_COLUMNS:
            run = _get_run(results, ds_name, "thermompnn", target)
            multi = run.get("multi_ddg", {})
            models = multi.get("models", {})
            model = models.get(model_key, {})
            val = model.get("cv_r2_mean")
            row_cells.append((_fmt(val) if val is not None else "---", val))
        model_grid.append(row_cells)

    # Highlight best R² per column
    for col_idx in range(n_cols):
        col_cells = [(model_grid[m][col_idx][0], model_grid[m][col_idx][1])
                      for m in range(len(model_grid))]
        highlighted = _highlight_best_in_row(col_cells, higher_is_better=True)
        for m in range(len(model_grid)):
            model_grid[m][col_idx] = (highlighted[m], model_grid[m][col_idx][1])

    for m_idx, model_key in enumerate(model_keys_used):
        label, n_feats = model_info[model_key]
        row = [r"\quad " + label, str(n_feats)]
        for col_idx in range(n_cols):
            row.append(model_grid[m_idx][col_idx][0])
        lines.append(" & ".join(row) + r" \\")

        # Add midrule after baselines
        if model_key == "ols_std_plddt":
            lines.append(r"\midrule")

    # Delta R² row
    lines.append(r"\midrule")
    row = [r"\quad $\Delta R^2$ (best $-$ pLDDT)", ""]
    for ds_name, target in TABLE1_COLUMNS:
        run = _get_run(results, ds_name, "thermompnn", target)
        multi = run.get("multi_ddg", {})
        val = multi.get("delta_r2_over_plddt")
        row.append(_signed(val) if val is not None else "---")
    lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ============================================================================
# SUPP TABLE S1: Partial correlations and incremental R²
# ============================================================================

def generate_table_s1(results: dict) -> str:
    """Generate Supp Table S1: partial correlations, Delta R², and conservation collinearity."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Partial correlations and incremental variance explained.")
    lines.append(r"\emph{$\Delta R^2$}: increase in $R^2$ when adding ThermoMPNN")
    lines.append(r"$\operatorname{std}(\Delta\Delta G)$ to the baseline predictor.")
    lines.append(r"\emph{Partial $\rho$}: Spearman correlation of robustness vs.")
    lines.append(r"target after controlling for the indicated confounder.")
    lines.append(r"Conservation scores from ConSurf Rate4Site (ATLAS only).")
    lines.append(r"All partial correlations significant at $p < 10^{-6}$.}")
    lines.append(r"\label{tab:supp_partial}")
    lines.append(r"\small")
    n_cols = len(TABLE1_COLUMNS)
    col_spec = "r" * n_cols
    lines.append(r"\begin{tabular}{@{}l " + col_spec + r"@{}}")
    lines.append(r"\toprule")

    _table1_header(lines, n_cols)

    # --- Delta R² (ThermoMPNN over baselines) ---
    lines.append(r"\multicolumn{" + str(n_cols + 1) + r"}{l}{\textit{$\Delta R^2$ (adding ThermoMPNN to baseline)}} \\")
    for baseline, field_name, label in [
        ("plddt", "delta_r2_over_plddt", r"$+$ pLDDT"),
        ("sasa", "delta_r2_over_sasa", r"$+$ SASA"),
        ("conservation", "pooled_delta_r2_over_conservation", r"$+$ Conservation"),
    ]:
        row = [r"\quad " + label]
        for ds_name, target in TABLE1_COLUMNS:
            run = _get_run(results, ds_name, "thermompnn", target)
            val = _get_corr(run, field_name)
            row.append(_signed(val) if val is not None else "---")
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")

    # --- Partial rho ---
    lines.append(r"\multicolumn{" + str(n_cols + 1) + r"}{l}{\textit{Partial $\rho$ (ThermoMPNN $|$ confounder)}} \\")
    for conf, label in [("plddt", r"$|$\,pLDDT"), ("sasa", r"$|$\,SASA"),
                         ("conservation", r"$|$\,Conservation")]:
        row = [r"\quad " + label]
        for ds_name, target in TABLE1_COLUMNS:
            run = _get_run(results, ds_name, "thermompnn", target)
            field = f"pooled_partial_rho_{conf}"
            val = _get_corr(run, field)
            row.append(_neg_sign(val))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")

    # --- Robustness vs conservation (collinearity) ---
    row_col = [r"$\rho$(robustness, conservation)"]
    for ds_name, target in TABLE1_COLUMNS:
        run = _get_run(results, ds_name, "thermompnn", target)
        val = _get_corr(run, "pooled_rho_robustness_conservation")
        row_col.append(_neg_sign(val))
    lines.append(" & ".join(row_col) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

TABLE_GENERATORS = {
    "table1": ("Table 1 (bivariate results)", generate_table1),
    "table_s1": ("Supp Table S1 (partial correlations & incremental R^2)", generate_table_s1),
    "table2": ("Table 2 (stratified)", generate_table2),
    "table3": ("Table 3 (alt measures + multi-DDG)", generate_table3),
}


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from unified results")
    from paper_config import CLUSTER
    default_base = CLUSTER.paper_results_dir
    parser.add_argument("--results", type=str,
                        default=f"{default_base}/unified_results.json",
                        help="Path to unified_results.json")
    parser.add_argument("--output-dir", type=str, default=default_base,
                        help="Base output directory (tables go into Tables/ subdirectory)")
    parser.add_argument("--table", type=str, default=None,
                        help="Generate only this table (e.g., 'table1')")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    out_dir = Path(args.output_dir) / "Tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    tables_to_gen = TABLE_GENERATORS
    if args.table:
        tables_to_gen = {args.table: TABLE_GENERATORS[args.table]}

    for table_id, (description, generator) in tables_to_gen.items():
        print(f"Generating {description}...")
        latex = generator(results)
        out_path = out_dir / f"{table_id}.tex"
        with open(out_path, "w") as f:
            f.write(latex)
        print(f"  -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
