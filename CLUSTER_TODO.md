# CLUSTER TODO — robustness-dynamics revision runs

Status tracker for the Genetics-revision compute. Detailed commands live in
**`FRUSTRAMPNN_AND_PACKING_RUN_PLAN.md`** — this file is the scannable checklist.
Last updated 2026-07 (cluster was unavailable; nothing below has been run yet).

`cd $REPO_DIR && git pull` first so the cluster has the latest code.

---

## A. Packing / contact-number control  (R1's #1 fix — CODE DONE, just run)
Already implemented May 2026 (CN@6Å, CN@8Å, WCN in correlate + multi-DDG Ridge);
never executed. No new compute — packing is computed inside the correlation job.

- [ ] `python scripts/run_all_analyses.py --dry-run --force`   (preview)
- [ ] `python scripts/run_all_analyses.py --force`   (resubmit all correlations + Ridge)
- [ ] wait for `squeue -u $USER` to clear
- [ ] `python scripts/run_all_analyses.py --postprocess-only`   (collect + tables)
- [ ] **VERIFY:** partial ρ(sd(ΔΔG), dynamics | WCN) in unified_results.json + packing
      row in Table 1. Survives → real signal beyond packing. Collapses → report it.

## B. FrustraMPNN frustration  (NEW code, committed 488b013)
Mechanistic reframe + complementary predictor. Does NOT fix circularity.

- [ ] **B0** env: `bash scripts/slurm/0b_setup_frustrampnn_env.sh`
- [ ] **B1** checkpoint → `$FRUSTRAMPNN_CHECKPOINT` (Zenodo 10.5281/zenodo.17978321)
- [ ] **B2** ⚠️ SMOKE TEST on 1 PDB (validates the inferred predict() schema).
      Check: no wildtype-mismatch WARN; `frustrampnn_native` std_ddg = real F_native
      (not all-NaN). If either fails → ping Claude (1-line fix in predict_profile).
- [ ] **B3** paper_config.py: add `"frustrampnn","frustrampnn_native"` to
      available_scorers of atlas/bbflow/pdb_designs/rci_s2/relaxdb + to TABLE1_PREDICTORS
- [ ] **B4** compute (GPU arrays): atlas `--array=0-38`; bbflow `0-1`; pdb_designs `0-6`;
      rci_s2 `0-15`; relaxdb `0-2`  (see plan for the FRUST_INPUT_DIR/OUTPUT_DIR env vars)
- [ ] **B5** `python scripts/run_all_analyses.py` (submits only the new frustration runs)
- [ ] **B6** wait, then `python scripts/run_all_analyses.py --postprocess-only`
- [ ] **VERIFY:** ρ(F_native, dynamics) vs ρ(sd(ΔΔG), dynamics) head-to-head; partial
      vs packing; sign should be NEGATIVE vs RMSF; check R2/R2R1 (slow-exchange) win.

## C. MegaScale experimental ΔΔG  (INVENTORY-FIRST gate — target is B-factor, NOT RMSF)
MegaScale has no MD → dynamics target is crystal B-factor, only for natural proteins
with an X-ray structure. Count usable proteins BEFORE investing. (Plan Track C.)
- [ ] **C0** run the 2 inventory commands (no GPU, minutes): `preprocess_megascale.py`
      + `megascale_atlas_overlap.py`
- [ ] report back: Route-2 ATLAS∩MegaScale overlap count; Route-1 #natural proteins ≥100 variants
- [ ] GO/NO-GO: overlap ≳30–40 → RMSF version; else Route-1 ≥20 → B-factor version; else shelve
- [ ] (HELD until GO) code unification + full preprocess→correlate→table run

---

## Paper-side tasks (NO cluster needed — can do anytime)
- [x] **A1** safe TeX surgery on `Revision/robustness_dynamics_revision_nmr.tex`
      (dropped conservation-partialing → descriptive; cut n=9 NMR-APP; retitled §3.5;
      softened design claim) — done 2026-07
- [ ] **A2** Option-C reframe of abstract/intro/discussion, built around frustration
      theory (drop "biophysical not evolutionary" + "complements pLDDT" as theses)
- [ ] wire `frust_native` as a first-class Table-1 predictor row (after B passes)
- [ ] case studies: pre-register selection + add failure cases (R2.2)
- [ ] Ridge coefficients vs side-chain properties (R2.4)

Venue after fixes: Protein Science → PLOS Comp Bio → Biophysical Journal.
See `ref_reports/genetics_revision_plan.md` (§7 = FrustraMPNN) in the paper dir.
