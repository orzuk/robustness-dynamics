# Cluster run plan — Packing control (already coded) + FrustraMPNN (new)

**Author aid for Or.** Written 2026-07 while cluster was unavailable. Two independent
tracks. **Track A** (packing / contact-number control) needs *no new code* — it was
implemented in May 2026 (commits `4a2020f`, `1fa59f9`, `ba481c0`) but never run.
**Track B** (FrustraMPNN frustration) is the new code added in this commit.

All paths come from `scripts/slurm/config.sh` (edit the 3 paths there if they moved).
Assumed cluster layout (Hebrew U / SLURM):
- `REPO_DIR=/sci/labs/orzuk/orzuk/github/robustness-dynamics`
- `PROJECT_DIR=/sci/labs/orzuk/orzuk/projects/ProteinStability`
- `VENV_DIR=$PROJECT_DIR/envs/robustness` (existing ThermoMPNN/analysis env)
- GPU partition `catfish`, `--gres=gpu:l4:1`; CPU partition `glacier`; `CHUNK_SIZE=50`.

> **Sanity first:** `cd $REPO_DIR && git pull` so the cluster has this commit.

---

## TRACK A — Packing / contact-number control (RE-RUN ONLY, no new compute)

Packing (CN@6Å, CN@8Å, WCN) is computed *inside* `correlate_robustness_dynamics.py`
from each PDB, using ΔΔG matrices you already have. So "running packing" = re-running
the correlation + regression + postprocess steps with `--force` so the existing
analysis JSONs are regenerated *with* the packing fields.

```bash
cd $REPO_DIR
source scripts/slurm/config.sh
source $VENV_DIR/bin/activate

# 0. Preview what will be resubmitted (no jobs launched):
python scripts/run_all_analyses.py --dry-run --force

# 1. Resubmit ALL correlations + multi-DDG regressions (now emit packing fields).
#    Dispatches one SLURM job per (dataset, scorer, target) in CORRELATION_RUNS.
python scripts/run_all_analyses.py --force

# 2. WAIT for all jobs to finish:  squeue -u $USER   (until empty)

# 3. Collect + regenerate tables (packing row is already in TABLE1_PREDICTORS):
python scripts/run_all_analyses.py --postprocess-only
#    (equivalently: python scripts/collect_results.py --output $PROJECT_DIR/data/paper_results/unified_results.json
#                    python scripts/generate_latex_tables.py --output-dir $PROJECT_DIR/data/paper_results)
```

**What to verify afterwards** (the R1 answer):
- In `unified_results.json`, each (dataset, thermompnn) run now has
  `pooled_rho_robustness_*_partial_packing` and `delta_r2_over_packing` fields.
- Table 1 has a **packing** row; Table 3 has `ols_packing`, `ols_std_packing`,
  `ridge_20ddg_nonlinear_packing`, `ridge_20ddg_nonlinear_plddt_packing`.
- The decisive number: **partial ρ(sd(ΔΔG), dynamics | WCN)**. If it stays
  substantial → sd(ΔΔG) carries info beyond packing (defensible). If it collapses →
  report honestly; that is the result.

Only ThermoMPNN needs this to answer R1, but `--force` refreshes every scorer.
To limit scope/time: `python scripts/run_all_analyses.py --force --only-dataset atlas`.

---

## TRACK B — FrustraMPNN frustration (NEW)

New files in this commit:
- `scripts/compute_frustration.py` — driver; writes two scorer views per protein:
  `frustrampnn` (std of the frustration profile — the sd(ΔΔG) analogue) and
  `frustrampnn_native` (F_i(native) — the canonical single-residue frustration).
  Both expose their quantity in the `std_ddg` column, so the existing correlation
  pipeline consumes them unchanged.
- `scripts/slurm/0b_setup_frustrampnn_env.sh` — one-time env setup.
- `scripts/slurm/14_compute_robustness_frustrampnn.sh` — batch compute job.
- `config.sh` gained `FRUSTRAMPNN_ENV`, `FRUSTRAMPNN_CHECKPOINT`.

### B0. One-time environment (dedicated env — torch-geometric conflicts with ThermoMPNN)
```bash
cd $REPO_DIR
bash scripts/slurm/0b_setup_frustrampnn_env.sh
```

### B1. Checkpoint
Download the FrustraMPNN model checkpoint from its Zenodo record
(**doi 10.5281/zenodo.17978321**) and place/symlink it at `$FRUSTRAMPNN_CHECKPOINT`
(`$PROJECT_DIR/models/frustrampnn/checkpoint.ckpt`).

### B2. SMOKE TEST — do this before anything else ⚠️
`compute_frustration.py` parses FrustraMPNN's `model.predict()` DataFrame
(`position, wildtype, mutation, frustration_pred`). That schema is inferred from
the FrustraMPNN docs, **not yet verified against the real package.** Run one protein
and check the warnings + outputs before launching arrays:
```bash
source $FRUSTRAMPNN_ENV/bin/activate
cd $REPO_DIR
# pick any ATLAS pdb:
PDB=$(ls $ATLAS_DIR/proteins/*/*.pdb | head -1)
python scripts/compute_frustration.py \
    --pdb_file "$PDB" --protein_id smoke_test \
    --output_dir /tmp/frust_smoke \
    --frustrampnn_checkpoint $FRUSTRAMPNN_CHECKPOINT --device cuda --no_skip_existing

# Inspect:
head /tmp/frust_smoke/frustrampnn/smoke_test_robustness.tsv
head /tmp/frust_smoke/frustrampnn_native/smoke_test_robustness.tsv
```
**PASS criteria:**
- No `WARN: ... wildtype mismatches` line (means position indexing aligns with the
  parsed sequence). If it warns, the `position` column is not 0-indexed vs our
  sequence — tell Claude; a one-line fix in `predict_profile` handles 1-indexing /
  residue-numbering.
- `frustrampnn_native/...tsv` `std_ddg` column holds real numbers (F_native), most
  between roughly −4 and +2; not all-NaN. (All-NaN ⇒ the model doesn't emit a
  native-identity row; then F_native needs a different extraction — tell Claude.)
- `frustrampnn/...tsv` `std_ddg` column is the spread of the profile (small positive).

### B3. Register the scorers in paper_config (so correlate/collect/tables see them)
Edit `scripts/paper_config.py`:
1. Add `"frustrampnn", "frustrampnn_native"` to `available_scorers=[...]` for the
   datasets you will run: **atlas, bbflow, pdb_designs, rci_s2, relaxdb**.
   (atlas & bbflow currently use the default factory — add an explicit
   `available_scorers=["esm1v","thermompnn","proteinmpnn","frustrampnn","frustrampnn_native"]`
   line to each.)
2. Add them to the Table-1 predictor rows:
   `TABLE1_PREDICTORS = [..., "packing", "frustrampnn", "frustrampnn_native"]`.

Do B3 only *after* B2 passes, and run frustration compute (B4) on every dataset you
add, so `collect_results` doesn't look for missing frustration JSONs.

### B4. Batch compute frustration (GPU array jobs)
Array size = ceil(n_proteins / 50) − 1. Submit per dataset:
```bash
cd $REPO_DIR && source scripts/slurm/config.sh

# ATLAS (1938 proteins) -> 0-38
sbatch --array=0-38 scripts/slurm/14_compute_robustness_frustrampnn.sh

# BBFlow (100) -> 0-1
FRUST_INPUT_DIR=$BBFLOW_PROCESSED FRUST_OUTPUT_DIR=$BBFLOW_ROBUSTNESS \
  sbatch --array=0-1 scripts/slurm/14_compute_robustness_frustrampnn.sh

# PDB designs (306) -> 0-6
FRUST_INPUT_DIR=$PDB_DESIGNS_DIR FRUST_OUTPUT_DIR=$PDB_DESIGNS_ROBUSTNESS \
  sbatch --array=0-6 scripts/slurm/14_compute_robustness_frustrampnn.sh

# RCI-S2 (759) -> 0-15
FRUST_INPUT_DIR=$PROJECT_DIR/data/rci_s2_processed FRUST_OUTPUT_DIR=$PROJECT_DIR/data/rci_s2_robustness \
  sbatch --array=0-15 scripts/slurm/14_compute_robustness_frustrampnn.sh

# RelaxDB (126) -> 0-2
FRUST_INPUT_DIR=$PROJECT_DIR/data/relaxdb_processed FRUST_OUTPUT_DIR=$PROJECT_DIR/data/relaxdb_robustness \
  sbatch --array=0-2 scripts/slurm/14_compute_robustness_frustrampnn.sh
```
Tip: launch ATLAS `--array=0` first as a live 50-protein check before the full sweep.
Each protein writes `frustrampnn/{id}_*` and `frustrampnn_native/{id}_*` under the
dataset's robustness dir.

### B5 + B6. Correlate + postprocess (reuses everything)
Once B3 is applied and B4 jobs are done, the new scorers are just more entries in
`CORRELATION_RUNS`:
```bash
source $VENV_DIR/bin/activate     # analysis env (correlate step is CPU)
cd $REPO_DIR
python scripts/run_all_analyses.py --dry-run          # confirm frustrampnn* runs appear
python scripts/run_all_analyses.py                    # submits only the missing (frustration) runs
# wait for squeue to clear, then:
python scripts/run_all_analyses.py --postprocess-only
```

**What to verify (the science):**
- ρ(frustration, dynamics) vs ρ(sd(ΔΔG), dynamics) — head-to-head (is frustration
  better, as hypothesised?). Compare rows `thermompnn` vs `frustrampnn_native` vs
  `frustrampnn` in Table 1.
- **partial ρ(F_native, dynamics | packing)** vs the same for sd(ΔΔG): does frustration
  survive the contact-number control better? (The one place frustration might beat
  sd(ΔΔG) on the circularity charge.)
- Sign check: `frustrampnn_native` ρ vs RMSF should be **negative** (minimally
  frustrated = rigid). If it's positive, the sign convention is flipped — flag it.
- On RelaxDB `R2` / `R2R1` (slow μs–ms exchange): does frustration correlate better
  than sd(ΔΔG)? (Frustration theory predicts it should — potential NMR win.)

---

## Anything else worth running while you're at it
- **MegaScale experimental-ΔΔG × B-factor** (Tier-2 #7, the real circularity-breaker)
  is already scaffolded (`preprocess_megascale_natural.py`, dataset `megascale`,
  scorer `experimental`, commits `a17f98b`/`f46ed50`). Populate `$PROJECT_DIR/data/megascale`
  then include it in the `run_all_analyses.py --force` sweep. Separate mini-plan if wanted.

## Honesty / what I could NOT verify without the cluster
1. The FrustraMPNN `predict()` DataFrame schema (B2 validates it). If it differs,
   `compute_frustration.py::predict_profile` is the only place to adjust.
2. Whether `correlate_robustness_dynamics.py` accepts an arbitrary `--scorer` string
   (it should — it just reads `{robustness_dir}/{scorer}/`). The B5 `--dry-run`
   confirms the runs are built before submitting.
3. Exact `torch`/CUDA wheel for the cluster (0b script notes where to pin it).
```
