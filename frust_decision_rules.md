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

## Results (fill in AFTER B5/B6 — leave blank until then)

- Sign check (§1): _pending_
- ATLAS RMSF: ρ(frust_native) = __ , ρ(sd ΔΔG) = __ , partial|WCN = __ / __
- BBFlow RMSF: __
- B-factor sets: __
- NMR (RCI-S², RelaxDB hetNOE / R2 / R2R1): __
- Outcome (A/B/C per §2): __
- Case studies (per §4): __
