# 003 — First PU model: amide coupling under HATU / DIPEA / DMF, PAYN-style, lab inventory as U

- **Status:** Accepted, 2026-10-09
- **Supersedes:** — · **Superseded by:** 006 in part (P, features, evaluation rule)

## Context

I take one narrow reaction type to a validated model before adding sources or classes: amide coupling
(acid + true amine) under a condition containing HATU, DIPEA and DMF. Positives come from ORDerly
(first round); the plate will be built from the group's inventory (529 acids × 538 amines); Chemlex is
the benchmark (validation 1,725 / test 1,313 reactions of the HATU family). Details and sources:
`docs/design/modeling.md`.

What the literature gives and does not give:

- PAYN (our group, *JACS* 2026) shows spy-based PU with CatBoost recovers much of the gap to fully
  labelled models, but only with P and U simulated at random inside one HTE set (SCAR by construction,
  P:U 1:1, random CV, yield MAE). It never tests real literature positives, a different U, or transfer
  to another source.
- Chemlex's own supervised model reaches AUC 0.79–0.81 on both-unseen (all conditions), a rough
  ceiling.
- Literature-trained models often only learn substrate popularity (Beker et al. 2022).
- With P from the literature and U from an inventory, a P-vs-U classifier may learn "literature-like
  vs inventory-like" molecules instead of reactivity.

## Decision

1. **Models.** Baselines first: random, substrate popularity, nearest neighbour to P, additive
   acid + amine, chemist rule. Then PU:
   - a P-vs-U CatBoost classifier,
   - PAYN (spy classifier → reliable negatives → CatBoost yield regressor on P + reliable negatives),
   - bagging PU.
2. **Output.** A ranking. The PAYN regressor's predicted yield is used only to rank; it is never
   compared on one scale with Chemlex conversions.
3. **U.** The lab inventory pairs; on validation also U = inventory + literature acids × literature
   amines, and the P:U ratio (1:1 subsample vs full).
4. **Features.** Morgan r=2, 2048 bits per substrate, concatenated; reacting-centre descriptors only if
   the shift checks show the model learns single substrates.
5. **Evaluation.**
   - On Chemlex validation: ROC-AUC, PR-AUC, hit rate and enrichment at k = 24 / 48 / 96, with a
     bootstrap over whole substrates.
   - Results are grouped by whether the substrates are in P.
   - Shift checks: adversarial validation, correlation with literature-likeness, pair shuffle.
   - Test once, with the configuration committed beforehand.

## Consequences

- The first result answers the question PAYN left open: does literature-PU transfer to a different
  substrate space and to measured HTE outcomes?
- Every PU model must beat the popularity and nearest-neighbour baselines to count.
- Chemlex numbers are on a harder-than-typical set (~5,600 designed negatives, not flagged); absolute
  values are reported with that caveat.
- Chemlex-derived tables (CC BY-NC-ND) and inventory-derived tables (internal) are never shared; only
  code (CLAUDE.md §9).

## Alternatives considered

- **Ranking-only classifier without PAYN's regression step:** simpler; kept as model M1.1 for
  comparison.
- **Prior-based PU (Elkan–Noto, nnPU):** needs a class prior that is not identifiable from biased PU data
  (Bekker & Davis 2020); left until a prior can be taken from validation.
- **SAR / propensity weighting with usage statistics** (literature frequency, availability): a later
  step, once the shift checks show how large the bias is.
