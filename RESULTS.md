# Results — literature-only BH C–N feasibility transfer to JnJ

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

## Honest caveats / next

- **Transduction:** U contains the TEST substrate *structures* (unlabeled, no labels). This is the
  intended plate-design setup, but a purely *inductive* robustness check (U excluding TEST substrates)
  is a good next validation.
- **BH specificity:** only 2,724/35,929 mined pairs have detectable Pd; the corpus is C–N-general.
  A Pd-only ablation is a natural next step.
- **Features:** substrate ECFP already gives 0.72; QM/xtb/UMA descriptors (WS4) are a candidate lift
  but need environment setup — only worth it if this ceiling is limiting.
- Reproduce: `sbatch slurm/transfer.sbatch` (arms) → `scripts/09_summarize_transfer.py` (table) →
  `sbatch slurm/demo.sbatch` (final model + plate uplift). Artifacts in `artifacts/transfer_*.json`.
