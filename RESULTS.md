# Results — literature-only BH C–N feasibility transfer to JnJ

> **Scope note (2026-07-29).** This page documents the **JnJ/BH** line of work. It is retained as
> methodological context — the domain-shortcut diagnosis below is the reason the rest of the project
> is built the way it is — but it is **no longer the headline result.** The JnJ gain is *transductive*
> and collapses to 0.40 under an inductive test, and JnJ is adversarially OOD by construction.
> The project's anchor result is the **Chemlex amide** experiment, which holds inductively and beats
> the in-domain supervised ceiling: see `CLAUDE.md` §2 and `LAB_JOURNAL.md` (2026-07-10).
> Project charter and current direction: `CLAUDE.md`; the pivot is recorded in
> `docs/decisions/008-refocus-to-publication.md`.

> Predicting **Buchwald–Hartwig C–N coupling feasibility** on the held-out **JnJ HTE 2024** plate,
> training on **literature/external data only** (no JnJ labels), via domain-adapted Positive-Unlabeled
> learning. All numbers are under the frozen anti-cheat protocol (`src/bhpu/eval/protocol.py`):
> **substrate-disjoint VAL/TEST** (share 0 aryls & 0 amines), models **selected on VAL**, **TEST
> touched once**. Feasibility = assay yield ≥ 10 %.

## Headline

| | naive PU | **PAYN baseline (spy/SCAR)** | **better-PAYN (IW-PU)** |
|---|---|---|---|
| ECFP, literature-only | 0.561 | 0.554 | **0.710** |
| ECFP, literature + ORD | 0.604 | 0.607 | **0.720** |

(JnJ **TEST ROC-AUC**, model selected on VAL.) Naive literature→JnJ transfer is ≈ random
(historically ~0.52); **covariate-shift importance weighting lifts it to 0.72**, beating the
PAYN-parity baseline by **~0.15 AUC**. Winner (VAL-selected): **importance-weighted PU-bagging on
ECFP substrate features, trained on literature+ORD positives vs the transductive JnJ grid** —
**VAL ROC-AUC 0.773 / TEST ROC-AUC 0.720, TEST AUPRC 0.720 (base rate 0.436)**.

> **Scope of the 0.72 (read this).** The method is **transductive**: the unlabeled pool U is the
> candidate substrate library you want to score, and the gain **requires the scored substrates'
> structures to be present in U**. Removing the TEST substrates from U (a purely *inductive* test)
> collapses TEST to **0.40** (below random) while VAL stays 0.75. So 0.72 is the honest number for
> the **plate-design use case** — pre-ranking feasibility across a *known* aryl×amine matrix — which
> is exactly this project's goal; it is **not** a claim of generalization to novel substrates. That
> remains unsolved (§ Robustness).

## The diagnosis (why naive transfer fails)

The **domain-shortcut discriminator AUC = 1.000** in every configuration: literature reactions and
the JnJ grid are *trivially separable*, so a naive P-vs-U classifier learns "was this reaction
*reported*?" instead of "does it *work*?". This — not class-prior shift (to which ROC-AUC is
invariant) — is why naive transfer ranks ≈ randomly. See `docs/decisions/003-*` and
`docs/literature/notes/theory-crossdomain-pu.md`.

## Full arm matrix (ROC-AUC / AUPRC)

Arms: `spy` = PAYN-parity (spy reliable-negatives, SCAR); `naive` = plain PU-bagging;
`iw` = covariate-shift importance-weighted PU (the fix); `coral` = CORAL feature alignment.
Family `pu` = positives + transductive grid U; `supervised` = literature positives + measured
negatives.

### DRFP (reagent-inclusive) — literature-only  · domain AUC 1.000
| arm | VAL AUC | TEST AUC | | arm | VAL AUC | TEST AUC |
|---|---|---|---|---|---|---|
| sup_coral ◄VAL | 0.605 | 0.534 | | pu_iw | 0.278 | 0.264 |
| sup_naive | 0.474 | 0.327 | | pu_naive | 0.271 | 0.263 |
| | | | | pu_spy | 0.266 | 0.265 |

**DRFP is catastrophic** — PU arms score *below random* (~0.26). The reaction fingerprint encodes
the reagents/conditions, which are the strongest domain-shortcut axis (literature has diverse
conditions; the grid U has 4 fixed general condition sets). **Lesson: use substrate features.**

### ECFP (substrate) — literature-only  · domain AUC 1.000
| arm | VAL AUC | TEST AUC | TEST AP |
|---|---|---|---|
| **pu_iw** ◄VAL | **0.669** | **0.710** | 0.718 |
| pu_naive | 0.619 | 0.561 | 0.532 |
| pu_spy (PAYN) | 0.616 | 0.554 | 0.518 |
| sup_coral | 0.595 | 0.702 | 0.601 |
| pu_coral | 0.408 | 0.531 | 0.453 |
| pu_coral_iw | 0.383 | 0.505 | 0.435 |

### ECFP (substrate) — literature + ORD (40,072 positives)  · domain AUC 1.000
| arm | VAL AUC | TEST AUC | TEST AP |
|---|---|---|---|
| **pu_iw** ◄VAL | **0.773** | **0.720** | 0.720 |
| pu_naive | 0.695 | 0.604 | 0.538 |
| pu_spy (PAYN) | 0.679 | 0.607 | 0.512 |
| sup_iw | 0.611 | 0.592 | 0.483 |
| pu_coral_iw | 0.465 | 0.436 | 0.389 |

## What worked, what didn't

- **Importance weighting (covariate-shift) — the win.** Reweighting literature positives by
  `w(x)=d(x)/(1−d(x))` toward the JnJ marginal lifts PU TEST AUC 0.56→0.71 (lit) and 0.60→0.72
  (lit+ORD). This is the core of the "better PAYN".
- **Data expansion (ORD, 35.9k mined pairs) — helps selection.** It raises VAL 0.669→0.773 (much
  more confident model selection) and holds TEST at 0.72. The corpus is **provably JnJ-leak-free**
  (0 exact test pairs in ORD; 2,221 substrate-overlapping pairs dropped — `docs/decisions/004-*`).
- **Substrate ECFP ≫ reaction DRFP** for cross-domain transfer here (DRFP's condition bits *are*
  the domain shortcut).
- **CORAL — an honest negative result.** Second-order feature alignment *hurts* on 4096-dim sparse
  binary fingerprints (worst arms, ~0.37–0.46 VAL): the covariance whiten/recolor is numerically
  unstable and smears the class signal. It helps only on dense low-dim features (see the unit test).
  → Disabled in the final model.
- **Transductive U matters.** U = the JnJ combinatorial grid (JnJ substrate structures, unlabeled) —
  "U must live where the test lives". Legitimate for plate design (you know your candidate substrates)
  and label-free.

## Final model + plate-design demo (`scripts/08`, SLURM 55122)

Winner refit on literature+ORD (IW-only, no CORAL, 80 bags):

- **JnJ TEST: ROC-AUC 0.722, AUPRC 0.723** (base rate 0.436); VAL ROC-AUC 0.771.
- **Plate design (budget 96):** a model-ranked plate is **100 % feasible vs 43.1 % for a random
  plate → 2.32× hit-rate uplift** (2.29× enrichment over the base rate). Demo plate written to
  `artifacts/demo_plate.csv`; top picks are aryl chlorides × diverse amines.
- **Effective sample size 6,627** of 40,072 positives — importance weighting concentrates on the
  ~6.6k most JnJ-like literature positives (healthy, not collapsed).
- **Calibration caveat:** raw PU-bagging scores are compressed low (ECE 0.42, no high-confidence
  bin) — *ranking* is strong (what plate design uses) but absolute probabilities need isotonic +
  prior-shift calibration (the natural next refinement; does not affect AUC/AUPRC/uplift).

## Robustness: transductive vs inductive (`scripts/11`, SLURM 55125)

The winning IW-PU model, ECFP, literature+ORD:

| grid U | VAL ROC-AUC | **TEST ROC-AUC** |
|---|---|---|
| transductive (U = full JnJ grid) | 0.772 | **0.720** |
| inductive (U excludes TEST substrates) | 0.752 | **0.403** |

VAL is unchanged (VAL substrates stay in U) but **TEST collapses below random** when its substrates
are removed from U. The gain is therefore **transductive** — the model ranks feasibility over a
*known* substrate library but does not extrapolate to unseen substrates. This is the honest boundary
of the method: it is a plate-design pre-scorer (the stated goal), not a novel-substrate predictor.

## WS4: do GFN2-xTB quantum descriptors help? (`scripts/11`) — **no**

Per-substrate GFN2-xTB electronics (HOMO/LUMO/gap/dipole/partial-charges; computed for all 24,525
substrates in 45 s via `tblite`) added to or replacing ECFP, winning IW-PU config, transductive:

| features | VAL ROC-AUC | TEST ROC-AUC |
|---|---|---|
| ECFP (substrate fp) | 0.772 | **0.720** |
| ECFP ⊕ xTB | 0.744 | 0.669 |
| xTB only (18 dims) | 0.626 | 0.614 |

xTB-only carries real (if weaker) signal, but **adding xTB to ECFP *hurts*** — the 18 continuous
global descriptors dilute the 4096-bit fingerprint. **Verdict: connectivity fingerprints suffice;
global semiempirical descriptors do not add advantage for feasibility here.** (This left open the
*site-specific* question — answered next.)

## WS5: do *site-specific* descriptors help? (`scripts/12`–`15`) — **no (robust)**

The WS4 caveat was that whole-molecule descriptors average away local reactivity. So we built
**reactive-site** features at the atoms that actually couple: **morfeus sterics** (Sterimol/%V_bur/
SASA at the aryl C–X carbon along the N-approach axis; %V_bur/pyramidalisation/SASA at the amine N)
and **g-xTB electronics** (Grimme 2025, ≈ωB97M-V: gap/IP/EA/dipole + per-atom EEQ_BC charge & Wiberg
BO *at the reacting atom*), over **20-conformer** ensembles with three aggregation modes. Reacting
atoms were **product-resolved** (`scripts/14`: the template's canonical site is only 36.7% correct
on ambiguous multi-site substrates, so we take the site that reproduces the observed product).
Subset-first eval (JnJ+lit+grid, IW-PU, frozen protocol), best TEST ROC-AUC per feature:

| feature | transductive | inductive |
|---|---|---|
| **ECFP (baseline)** | **0.709** | **0.448** |
| morfeus (site sterics) | 0.680 | 0.471 |
| g-xTB (site electronics) | 0.598 | 0.381 |
| morfeus ⊕ g-xTB | 0.626 | 0.421 |
| ECFP ⊕ site | 0.702 | 0.380 |

- **ECFP wins transductively; nothing beats ~0.47 inductively** (ECFP 0.448). Adding site
  descriptors to ECFP *hurts* the inductive case (0.35–0.38).
- **Conformer aggregation: `mean` ≥ `boltzmann` > `lowest`** — the single lowest-energy conformer is
  the worst choice; averaging over the ensemble is best.
- **Domain-shortcut AUC = 1.000 for all 26 arms** — physical descriptors are *not* more
  domain-invariant than fingerprints; the shortcut is not a representation artifact.

**Verdict:** even proper site-specific physics (correct atoms, sterics + g-xTB electronics, conformer
ensembles) does **not** beat ECFP and does **not** solve novel-substrate generalisation. The
inductive gap needs a different lever (substrate-aware domain adaptation, or condition/context
features), not richer per-substrate descriptors. Descriptors were therefore **not** scaled beyond
the subset (per the "no wasted compute on a negative" rule).

> **Caveat (open — re-test with conditions).** This negative was measured **substrate-only**.
> Substrate descriptors may carry electronic/steric information that is only *usable once the
> reaction conditions are present* (a bulky ligand matters more for a hindered substrate, etc.). So
> the descriptor verdict is **conditional on substrate-only learning**; it must be re-evaluated as
> `substrate-descriptors ⊕ conditions` before treating descriptors as settled-useless
> (`docs/decisions/006`).

## WS6: reaction conditions — the biggest untapped signal (`scripts/17`,`18`)

We had been encoding **only the substrates** (aryl⊕amine ECFP) — no catalyst/ligand/base/solvent.
But JnJ is a **condition screen**: 1,137 condition sets (32 cat × 8 base × 6 solv), and **conditions
drive 43% of the label variance**, flipping outcome for **35%** of pairs. So the 0.72 is a
substrate-pair number blind to ~43% of the signal.

| | TEST ROC-AUC |
|---|---|
| **In-domain** (supervised JnJ VAL→TEST, conditions *shared*), substrate | 0.609 |
| **In-domain, substrate ⊕ condition-FP** | **0.769  (+0.16)** |
| Transfer (lit→JnJ), substrate | 0.69 |
| Transfer, substrate ⊕ condition-FP | **0.26 (crash)** |

- **Conditions are real, learnable, and large** (+0.16 AUC in-domain) — encoding them is the
  highest-value direction left.
- **But structural condition fingerprints crash in transfer** (0.69→0.26): literature and JnJ share
  only **8 of 81 catalysts**, so condition FPs become a literature-vs-JnJ **domain tag** (the same
  mechanism that sank reagent-inclusive DRFP). Rebuilding U to span JnJ conditions was necessary but
  not sufficient — the *positives* (literature) still don't use JnJ's conditions.
- **→ The fix is condition *descriptors*** (shared physical space: ligand %V_bur/electronics, base
  pKa, solvent parameters), so a bulky electron-rich literature ligand and a different-but-similar
  JnJ ligand map close. That is the clear next build.

## Data expansion: Pistachio (`scripts/16`) — a modest, real gain

Mining Pistachio added **201,340** product-confirmed, **JnJ-leak-free** C–N pairs (0 exact test
pairs; 54,937 with Pd). On ECFP IW-PU it lifts transfer **TEST 0.721 → 0.736** (VAL 0.771→0.779) and
raises the importance-weighting effective sample size **6,627 → 28,149** (4× more JnJ-like
positives). Pistachio is **licensed** (corpus gitignored, never published) — so the **publishable
headline stays ORD-only (0.721)**; 0.736 is an internal enhancement.

## Honest caveats / next

- **The headline is transductive** (see Robustness). Inductive generalization to novel substrates is
  the open problem — candidate directions: substrate-aware domain adaptation, site-specific QM
  descriptors, or MLIP embeddings.
- **Calibration:** PU-bagging scores are compressed (ECE 0.42); ranking is strong but probabilities
  need isotonic + prior-shift calibration for interpretable thresholds.
- **BH specificity:** only 2,724/35,929 mined pairs have detectable Pd; the corpus is C–N-general. A
  Pd-only ablation is a natural next step.
- **Compute note:** cap CatBoost `thread_count` to the SLURM allocation — unset, it grabs all node
  cores and oversubscribes array tasks ~10–100× (fixed in `models/boosting.py`).
- Reproduce: `sbatch slurm/transfer.sbatch` (arms) → `scripts/09_summarize_transfer.py` (table) →
  `sbatch slurm/demo.sbatch` (final model + plate uplift) → `scripts/10_precompute_xtb.py` +
  `scripts/11_qm_and_inductive.py` (QM + inductive). Artifacts in `artifacts/`.
