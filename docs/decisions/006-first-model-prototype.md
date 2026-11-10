# 006 — First model prototype: P at two widths, reacting-centre descriptors, success rule fixed first

- **Status:** Accepted, 2026-10-20
- **Supersedes:** 003 in part (P, features, evaluation rule) · **Superseded by:** 007 in part
  (item 2, the feature list; item 3, the success rule)

## Context

ADR 003 plans baselines, then PU models. Counted on 2026-10-20 on the Chemlex validation set
(1,725 reactions, 32.8 % positive; test side not opened, `scratch/count_m0_anchors.py`):

| baseline | ROC-AUC | hits in top 96 |
|---|---|---|
| random | 0.50 | 0.33 |
| popularity | 0.54 | 0.69 |
| amine class rule | 0.71 | 0.59 |

A PU model is only worth building if it beats both non-random baselines. P is small: 1,177 HATU
pairs, 1,176 reported once. The whole class has 28,158 pairs. The literature notes
(`docs/literature/notes/2026-10-20-first-model-findings.md`) agree that popularity must be beaten,
that ranking needs no class prior, and that our labelling is SAR, not SCAR.

## Decision

1. **P at two widths.** Every model is trained on P of the HATU family and on P of the whole class.
   The comparison is reported; the better one is used.
2. **Features.** Morgan counts (r = 2, 2,048 bits) per substrate, plus about 15 descriptors at the
   reacting atom marked `:1` in the class SMARTS (amine class, branching at the alpha atoms, heavy
   atom count). Descriptors are computed from the class YAML, not hard-coded.
3. **Success rule, fixed before the first PU run.** A PU model counts as better than a baseline only
   if the paired bootstrap difference (resampling whole acids and amines, same samples for both
   models) has a 95 % interval above 0 for ROC-AUC, on Chemlex validation. It must beat both the
   popularity and the amine-class baseline, and it must also rank above the amine-class baseline
   within each amine class (primary, secondary, aniline, heteroaryl).
4. **Split by time block.** ORDerly has no patent field. Counted 2026-10-20 on the full file
   (625,697 reactions): the nearest key, `extracted_from_file`, has 489 values, a median of 707
   reactions each, and every file holds a single grant year. It is a time block of the patent corpus,
   not a patent; the 1,178 HATU positives fall in 123 files, 5 files hold 29 % of them. Stage 1 keeps
   it as `source_document`. Splits of P keep reactions of one block together, which tests
   generalisation over time (Bradshaw 2025) but does not separate analogues from one patent. The
   Chemlex validation set has no patents and is split by substrate.
5. **Order.** Baselines B0 to B4 first, then bagging PU, then the P-vs-U classifier and PAYN spy.
   The test side is read once, after the configuration is frozen and committed.

## Alternatives considered

- **Follow ADR 003 unchanged:** no rule for what counts as success.
- **Diagnostic first (adversarial validation):** kept as a step inside M2, not a gate.

## Consequences

- scikit-learn is added (Luka, 2026-10-20). catboost is decided when the PU models start.
- The stages in `docs/design/modeling.md` ("Stages of the prototype") must exist before code (CLAUDE.md §2).
