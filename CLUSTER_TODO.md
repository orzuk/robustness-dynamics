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

- [x] **B0** env: `bash scripts/slurm/0b_setup_frustrampnn_env.sh` — done
- [x] **B1** checkpoint → `$FRUSTRAMPNN_CHECKPOINT` — done. NOTE: also needed the base
      ProteinMPNN vanilla weights at `$(dirname checkpoint)/vanilla_model_weights/v_48_020.pt`
      (6.4 MB, wget from dauparas/ProteinMPNN) — not bundled in the Zenodo checkpoint.
- [x] **B2** ⚠️ SMOKE TEST — PASSED 2026-07-10 (job 45454934). Fixed two bugs first:
      (a) ls|head pipefail (50eabc9), (b) blank-chain PDBs → empty predict() frame
      (c60206a: relabel blank chain in compute_frustration._resolve_chain_and_path).
- [x] **B3** paper_config.py: added `frustrampnn`,`frustrampnn_native` to
      atlas/bbflow/pdb_designs/rci_s2/relaxdb available_scorers + TABLE1_PREDICTORS (596a962).
- [~] **B4** compute (GPU arrays) — LAUNCHED 2026-07-10. Canary (atlas --array=0,
      50 proteins) COMPLETED clean: processed=50 failed=0. Full sweep submitted:
      atlas 45455298_[1-38], bbflow 45455299, pdb_designs 45455300, rci_s2 45455301,
      relaxdb 45455302. ~53 s/protein → ~40 min/task. Spot-check non-ATLAS dataset
      counts climb >0 (different PDB chain conventions). Resume-safe (--skip_existing).
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
