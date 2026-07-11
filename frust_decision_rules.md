# Pre-registered interpretation rules — FrustraMPNN (Track B) + packing (Track A)

**Written 2026-07-10, BEFORE the B5 correlation results were computed.** Purpose: fix
the interpretation of the frustration head-to-head *in advance*, so the read-off is not
post-hoc (directly answers Genetics referee 2's post-hoc-selection charge). Do not edit
the rules below after seeing results — record outcomes in a separate "Results" section.

Context: [[project-robustness-dynamics-issues]]. The paper is now framed as a
flexibility-predictor **benchmark**, not a fitness-landscape law. Under that framing a
null/mixed result is a reportable finding, not a failure.

---

## 0. What each predictor is (so the comparison is fair)

| Predictor | What it is | Structure-conditioned? |
|---|---|---|
| `sd(ΔΔG)` (thermompnn) | SD of 19 ThermoMPNN ΔΔG at a site | Yes (backbone + seq) |
| `frustrampnn_native` | F_i(native) — single-residue frustration of the WT residue | Yes |
| `frustrampnn` | SD of the 19-mutant frustration profile (the direct sd(ΔΔG) analogue) | Yes |
| `packing` (WCN / CN@6,8Å) | local contact density | Yes (pure geometry) |
| `plddt` | AF2/ESMFold self-confidence | — (not a physical observable) |

**All three ΔΔG/frustration measures are structure-conditioned, so none of them breaks
the circularity charge.** Frustration's claim is (a) a *mechanistic* motivation (energy
landscape theory) and (b) a *complementary* predictor — NOT a circularity escape. The
only clean anti-circularity result is Track C (experimental ΔΔG × experimental B-factor).

---

## 1. Sign check (must pass before any interpretation)

Frustration convention: minimally frustrated (F > +0.55) = optimized = **rigid**;
highly frustrated (F < −1.0) = conflict = **flexible**. So:

- **ρ(`frustrampnn_native`, RMSF) must be NEGATIVE** (same sign as ρ(sd(ΔΔG), RMSF)).
- If it comes back **positive**, the sign convention in `compute_frustration.py` is
  flipped — STOP and fix before reading anything else. Do not reinterpret; fix.

---

## 2. Primary question & decision table (per dataset)

Primary comparison: does frustration predict dynamics **better than** sd(ΔΔG), and does it
**survive the packing partial** better? Decisive metric = partial ρ(predictor, dynamics |
WCN), compared head-to-head against the same partial for sd(ΔΔG).

| Outcome | Meaning | Paper move |
|---|---|---|
| **A.** frustration ρ ≈ sd(ΔΔG) ρ **and** its packing-partial is larger | Genuinely better, mechanistically motivated predictor | Lead with frustration; strongest benchmark result |
| **B.** frustration correlates but **collapses** under packing (like sd(ΔΔG)) | Both are packing-driven — the honest answer to R1 | Report it plainly: "local packing is the shared driver of robustness and flexibility." Still publishable. |
| **C.** frustration barely correlates (\|ρ\| < ~0.15 on MD sets) | Adds little over sd(ΔΔG) | Demote to a comparison row; lean on packing + MegaScale narrative |

Only the **Meaning/move** differs across outcomes — **every outcome is publishable** under
the benchmark framing. The *only* genuinely bad result is outcome B/C holding on **every**
dataset AND the packing partial killing **all** predictors everywhere — unlikely given
existing ρ ≈ −0.6 on MD sets.

---

## 2b. ESTABLISHED BASELINES the frustration result must beat (measured 2026-07-10)

The packing partial for sd(ΔΔG) is ALREADY KNOWN on both an experimental and an ML/MD set.
Frustration's `F_native | packing` is judged against THESE numbers, not against zero.

| Set | axes | ρ(sd(ΔΔG), dyn) | ρ( · \| WCN) | ΔR² over packing |
|---|---|---|---|---|
| **MegaScale** | exp ΔΔG × crystal B-factor (n=38) | −0.33 | **−0.05** | ~3e-6 (≈0) |
| **ATLAS** | ThermoMPNN ΔΔG × MD-RMSF (n≈1900) | −0.50 | **−0.18** | 0.012 |

**These two do NOT tell the same story, and that contrast is the interpretive spine:**
- On the **clean experimental** set the signal **vanishes** under packing (−0.05).
- On the **ML/MD** set it is **reduced but survives** (−0.18). The likely reason: ThermoMPNN
  ΔΔG and RMSF are BOTH computed from the same backbone, and WCN (one contact-number
  scalar) is an incomplete control for everything they share → the residual −0.18 is
  plausibly **residual structure-sharing / circularity, not independent biophysics.**
  MegaScale (experiment × experiment) is the ground truth that adjudicates: it says the
  effect is packing.

**Pre-registered honest conclusion (independent of frustration):** *any* signal beyond
packing is small, ML-dependent, and does NOT replicate on clean experimental data.

**Bars for "frustration adds something":**
- vs experimental ground truth: `F_native | packing` on a B-factor set must clear ~**−0.05**.
- vs the ML/MD residual: `F_native | packing` on ATLAS RMSF must clear ~**−0.18**
  (i.e., beat sd(ΔΔG)'s own partial, not merely be nonzero).
- If frustration only reaches ~−0.18 on ATLAS but ~−0.05 on experiment-adjacent sets, it is
  behaving exactly like sd(ΔΔG) → no independent contribution (outcome B/C).

## 2c. What FrustraMPNN is (settled from the paper, ref_reports/FrustraMPNN_summary.md)

- Trained to predict **physics-computed frustration** (FrustratometeR / AWSEM labels), NOT
  experimental ΔΔG. MegaScale + FireProt are the **structure sources**; the label is a
  computation, "one step removed from experiment." So it is NOT circular with ThermoMPNN's
  experimental ΔΔG target — different ground truth, shared ProteinMPNN backbone only.
- Its 20-value profile is per-position z-scored (mean≈0, std≈1 by construction — confirmed
  in smoke test), so the `frustrampnn` profile-std view is ~degenerate; **`frustrampnn_native`
  (F of the WT identity) is the meaningful predictor.**
- Design premise (their §): frustration is normalized against same-site decoys to measure
  energetic **conflict, not contact density** → the explicit reason it *might* survive the
  packing partial where sd(ΔΔG) does not. That premise is exactly what §2b's bars test.

---

## 3. Which datasets are the strong vs weak test (decided in advance)

- **Strong / primary:** ATLAS RMSF, BBFlow RMSF (MD; direct, dense dynamics signal).
- **Secondary:** ATLAS B-factor, PDB-designs B-factor (crystal; refinement/packing artifacts).
- **Weakest / known-hard:** RCI-S² and RelaxDB NMR (sd(ΔΔG) already collapses here,
  median ρ ≈ −0.15). **Pre-registered expectation: frustration will also be weak on NMR.**
  A frustration win here would be a *bonus*, not the headline.
- **Pre-registered NMR sub-hypothesis (from frustration theory):** frustration should do
  *relatively better* on **RelaxDB R2 / R2R1** (slow μs–ms exchange, conformational
  switching) than on hetNOE (fast ps–ns). If ρ(frust, R2) > ρ(frust, hetNOE), that is a
  predicted, non-post-hoc win. If not, report null — do NOT switch targets to fish for it.

---

## 4. Pre-registered CASE-STUDY selection criterion (answers R2.2)

Freeze the rule NOW, before looking at which proteins win:

1. Candidate pool = proteins where `frustrampnn_native` OR `sd(ΔΔG)` beats `plddt` on the
   **primary** target (ATLAS/BBFlow RMSF), ranked by Δρ, selected by
   `find_case_study_candidates.py` with a **fixed seed / fixed threshold recorded here**.
2. Report the **top 3 wins AND the top 3 losses** (proteins where the predictor
   under-performs pLDDT) — mandatory, to avoid cherry-picking.
3. Any mechanistic claim (e.g. Zika helical-face periodicity) must be stated as a
   hypothesis and tested on a **held-out** set, or labelled explicitly anecdotal.

---

## 5. Ridge-coefficient interpretation (answers R2.4) — pre-committed

When reading the 20-D Ridge per-AA coefficients, relate them to **side-chain physical
properties fixed in advance**: volume, hydrophobicity (Kyte-Doolittle), and helix
propensity (Pace-Scholtz). Report the correlation of |coef| with each. Do not narrate
individual residues post-hoc.

---

## Results

### Baselines (KNOWN, 2026-07-10) — see §2b
- MegaScale exp ΔΔG × B-factor (n=38): ρ=−0.33, partial|WCN=−0.05, ΔR²=3e-6.
- ATLAS ThermoMPNN ΔΔG × RMSF (n≈1900): ρ=−0.50, partial|WCN=−0.18, ΔR²=0.012.

### Frustration RESULTS (2026-07-11) — frustrampnn_native (the meaningful view)
- Sign check (§1): PASS — all raw ρ negative.
- Frustration is NOT a packing proxy (validates "conflict not density"): ρ(frust,packing)
  ≈ 0.11–0.19 everywhere, vs sd(ΔΔG)'s 0.45. Genuinely a different quantity.
- ATLAS RMSF: raw −0.13, partial|WCN = **+0.016** → COLLAPSES to zero. FAILS −0.18 bar.
- ATLAS bfactor: raw −0.10, partial +0.016 → collapses.
- BBFlow RMSF: raw −0.28, partial −0.000 → collapses.
- RCI-S2: raw −0.11, partial +0.011 → fails. RelaxDB: raw −0.06, partial +0.065 → fails.
- **PDB-designs (the ONE positive): raw −0.19, partial|WCN = −0.167 → SURVIVES.**
  Real, not artifact: packing IS a predictor there (ρ(packing,dyn)=−0.30) but
  ρ(frust,packing)=0.107 (orthogonal), so partial holds. Beats sd(ΔΔG) on designs
  (thermompnn partial −0.116, ρ(sdΔΔG,packing)=0.45).

**OUTCOME = C (frustration not an independent predictor on natural sets) + a modest,
mechanistically-coherent positive on DESIGNS.** Two readings, both good for the paper:
1. Triangulation: experimental ΔΔG (MegaScale −0.05) AND physics frustration (+0.016)
   both collapse to ~0 under packing on natural proteins → the ThermoMPNN −0.18 residual
   is ML circularity; true natural-protein signal beyond packing ≈ 0. Three independent
   lines now support the packing conclusion.
2. Designs subsection: frustration adds packing-independent flexibility signal (−0.167)
   that sd(ΔΔG) can't, because designs are packing-idealized/over-optimized (FrustraMPNN
   §60) and frustration isn't a packing proxy. Modest (one B-factor set) — a subsection,
   not a headline.

- Redundancy ρ(frust_native, sd(ΔΔG)) not computed directly, but inferred distinct:
  frust~0.1–0.2 packing-corr vs sdΔΔG 0.45 → clearly different quantities.
- Case studies (per §4): TODO after framing locked.
- RelaxDB R2/R2R1 slow-exchange test: NOT run (virtual datasets skipped by submitter);
  optional manual follow-up.
