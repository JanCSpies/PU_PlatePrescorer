# ADR 003 — Cross-domain literature→JnJ transfer: diagnosis and the "better PAYN" design

- **Status:** accepted (autonomous run, 2026-07-07)
- **Context:** naive literature→JnJ PU transfer scores ≈ **0.52 ROC-AUC** (random). This is the
  central research problem of the week. A research sub-agent (see
  `docs/literature/notes/theory-crossdomain-pu.md`) produced the diagnosis below; this ADR records
  the decision it drives.

## Diagnosis (why 0.52)

The four distribution shifts between the literature corpus and the JnJ HTE plate do **not** hurt
the same metric — that is the key insight:

1. **Class-prior shift** (literature is positive-rich; JnJ base rate ≈ 0.32–0.44). ROC-AUC is
   *invariant* to prior shift and to any monotone score transform. So the prior gap **cannot**
   by itself produce 0.52 AUC — it is purely a *calibration* problem.
2. **Domain shortcut (the actual culprit).** With P = literature and U = an unlabeled pool, a
   P-vs-U classifier's cheapest error-minimising direction is "literature manifold vs the other
   manifold" — near-zero Bayes error for *domain*, and orthogonal to *feasibility*. The substrates
   are near-disjoint, so the model learns *where a reaction was reported*, not *whether it works*.
   A 0.52 AUC means the ranking is wrong, which only the shortcut (not prior/calibration) explains.
3. **SAR/SNAR reporting bias** (selection-at-random violated: chemists publish likely successes;
   negatives under-reported). A second-order amplifier — only hurts AUC where reporting features
   overlap reactivity features.
4. **Covariate shift** — the literature decision boundary is extrapolated into JnJ regions with no
   training support.

**Smoking-gun diagnostic (run first, publish the number):** train a literature-vs-JnJ *domain
discriminator* on DRFP. Its held-out AUC ≈ 1.0 *is* the explanation of the 0.52 baseline, and the
same `d(x)` gives us importance weights for the fix.

## Decision — the "better PAYN"

A **literature-only, transductive, domain-debiased PU** pipeline (never uses JnJ labels for
training). Ordered by payoff:

1. **Transductive U = the JnJ combinatorial grid** (`datasets/processed/.../unlabeled_grid`, 26 368
   rows over JnJ substrate space, measured/test pairs removed) — *U must live where the test lives*.
   This alone attacks the covariate-support problem. We already have this grid + its DRFP features.
2. **Kill the domain shortcut** via feature alignment: **CORAL** (closed-form, ~10 lines) →
   escalate to DANN + nnPU label loss if VAL holds. Target: domain-AUC → 0.5 while held-out-literature
   feasibility AUC stays high.
3. **Importance-weighted PU** (cheapest, drops straight into CatBoost `sample_weight`):
   `w(x) = d(x) / (1 − d(x))`, self-normalised + clipped. Grounded in Sakai–Shimizu (AAAI 2019),
   PU under covariate shift. Report effective sample size (fragile under disjoint support).
4. **SAR-debias the positives** (v2): a propensity `e(x_e)` on *bibliometric* features only
   (source stratum, aryl/amine literature-popularity counts, year, ligand commonness) — reactivity-
   orthogonal; never DFT/DRFP, never JnJ membership.
5. **Label-free prior + calibration:** estimate `π_jnj` (DEDPUL ∥ KM2 ∥ TIcE, cross-checked),
   isotonic-calibrate on held-out literature, prior-shift-correct (Saerens EM). Needed for
   plate ranking + precision@conf + an honest demo metric; independent of AUC so it can't hurt.

Why this is "better than PAYN": PAYN (spy-PU, SCAR) and bagging-PU are **in-domain** methods that
assume P and U share `p(x | y=1)` and a stable prior. Cross-domain, every spy/bag learns the *same*
domain direction, so bagging *stabilises the wrong signal* and PAYN's spy-tolerance threshold
(calibrated in the literature density) extrapolates meaninglessly onto JnJ. We keep PAYN as the
in-domain baseline; its failure is the story.

## Guardrails (unchanged, non-negotiable)

- Evaluated **only** through `src/bhpu/eval/protocol.py` (`FrozenEvalProtocol`): literature-only →
  JnJ, substrate-disjoint VAL/TEST, `assert_training_clean` rejects any JnJ substrate/pair leak.
  TEST is touched once, at the very end; model selection on VAL only.
- **Leakage across sources** (ORD/Pistachio may *contain* the JnJ reactions): every mined source is
  audited at substrate + pair + reaction-hash level against JnJ before it can enter training.
- **Cross-source dedup** when merging corpora (InChIKey reaction hash).

## Consequences

- WS2 (domain adaptation) is reclassified from "nice to have" to **the primary score lever**; WS1
  (more C–N data) supplies substrate diversity + positives but will *not* alone fix 0.52.
- New package `src/bhpu/adapt/` (domain discriminator, CORAL, importance weighting, prior/calibration).
- Implementation libraries available offline: `adapt` (CORAL/DANN/KMM/KLIEP), `densratio`, `pywsl`
  (uPU/nnPU), DEDPUL, `sklearn.calibration`. Prefer re-implementing the small closed-form pieces
  (CORAL, self-normalised weights) to keep the dependency surface and typing under our control.
