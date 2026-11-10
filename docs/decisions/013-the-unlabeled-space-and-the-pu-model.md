# 013 — How the unlabeled space is built, and how the PU model is fitted

- **Status:** Accepted, 2026-10-23, **before stage S2 was written**
- **Supersedes:** 012 in part (item 4, which proposed B8) · **Superseded by:** —

ADR 008 decided *that* the unlabeled space comes from the literature, and pre-registered how a model
will be judged. It did not say how U is enumerated or how the model is fitted. This record settles
both, and it is written before any of it exists, for the same reason ADR 008 item 6 was.

## Context

`scratch/count_u_space.py` counted the space U would be drawn from. Two numbers decide the design.

**The cross-product is unusable.** Class-wide P is 28,158 pairs over 11,543 acids and 9,959 amines,
so the full product is **114,956,737** pairs. Sampling is not a convenience; it is forced.

**And sampling naively would open a shortcut.** **67.4 %** of acids and **67.3 %** of amines appear in
exactly **one** reported pair. A substrate reported 321 times contributes 321 rows to P; one reported
once contributes a single row. In a uniformly sampled U both contribute about the same. So a
classifier can separate P from U by **how popular a substrate is**, never looking at whether the two
molecules fit each other. That is the provenance problem of ADR 008 Finding 1, relocated into a
different feature — and under the spy technique it is worse than elsewhere, because the "reliable
negatives" would come out as the *unpopular* substrates rather than the incompatible pairs.

A third number is not a problem to solve but a fact to state. Only **50 of the 2,615** validation
pairs have both substrates present in the class-wide literature, and by substrate only **30 of 95**
validation acids and **11 of 113** validation amines appear at all. The model therefore extrapolates
onto a nearly disjoint set of molecules. That is a property of the benchmark — Chemlex deliberately
explored substrates the literature does not cover — and it makes a good score there evidence of
generalisation rather than of memory.

## Decision

### 1. U is built by frequency-matched negative sampling

For each positive pair, draw **k** counter-pairs such that the **marginal distribution of acids and
of amines in U matches their distribution in P**. Partners are drawn from P's own empirical marginals,
so a substrate appears in U about as often as it appears in P.

The reasoning is the same move ADR 008 item 4 already made once: **remove a nuisance variable by
making it identical in both classes.** There, literature-versus-inventory provenance was removed by
drawing U from the literature. Here, substrate popularity is removed by matching the marginals. What
is left to separate P from U is the **joint** distribution — which pairs go together — and that is
the only thing we want the model to learn.

It also sharpens a question we could not answer. B2 sees the pair, B3 sees the two substrates
separately, and the bootstrap could not distinguish them. With the marginals matched, a model that
learns only the marginals learns nothing, so anything it does learn is what B3 cannot represent.

**k = 10.** 5 and 20 are run as a sensitivity check on the primary configuration only, and the result's
dependence on k is reported. k is a free parameter of the *training* set; it touches no part of the
pre-registered evaluation, so moving it is not selective.

### 2. P is subtracted from U on **flat** InChIKeys

Our fingerprints are computed without chirality and the first InChI block ignores isotopes, so two
stereo variants, and an isotopologue, are **one input** to the model. Subtracting on stereo keys left
five known HATU-family positives sitting in the inventory U, each carrying a feature vector identical
to a positive.

The error is asymmetric, and that is the whole argument: removing one pair too many costs one
candidate out of hundreds of thousands, while leaving a known positive in U teaches the model that a
published reaction is unknown. **No chemistry rule is needed** — subtract on the key the featuriser
actually uses, and the two stay consistent by construction.

The counted cost: at the class width this merges 352 of 28,158 pair-groups, 200 of them safely; in
the HATU family it is 7 groups, and the family goes from 1,177 pairs to 1,170.

### 3. The model is fitted with the spy technique, as PAYN used it

A random subset of known positives is hidden inside U. A classifier is trained on P against U, and the
threshold is set so that about **95 %** of the spies score positive. Everything in U below that
threshold becomes a **reliable negative**, and the final model is trained on P plus those negatives.

Chosen over the two alternatives for stated reasons. Estimating the class prior π would put a number
in the paper that rests on a separability assumption which reporting bias is precisely what casts
doubt on; π is not honestly estimable here. Ranking-only would need no π at all, but it gives up
calibrated scores and it makes the group's own prior method unavailable as a comparison. The spy
threshold is a free parameter and is reported as one.

**The classifier is scikit-learn's `HistGradientBoostingClassifier`**, not CatBoost. PAYN used
CatBoost; adding it would be a new dependency, which CLAUDE.md §5 forbids without agreement, and the
two are the same family of gradient-boosted trees. If the difference ever turns out to matter, that
is a measurement to make and then a dependency to ask for, in that order.

### 4. Both feature blocks are fitted, and both are reported

`morgan` and `morgan_centre`, separately, each with its adversarial AUC beside it (ADR 008 item 7).
ADR 012 measured a real trade: `morgan_centre` carries less provenance and finds analogues better,
while `morgan` has the higher supervised ceiling. Picking one now, before any model is fitted, would
be choosing on a guess; running both costs one extra fit and makes the trade a result.

The ten hand-written descriptors are **not** model inputs (ADR 008 item 1).

### 5. B8 is declared, before anything is run

**B8: nearest positive pair on `morgan_centre`**, measured at **0.711** on the validation set.

It sits below B4 (0.743) and well below B6 (0.822), so it is not the bar. It exists to isolate one
thing nothing else can: the same fingerprint, with fitting and without. If a fitted PU model on
`morgan_centre` reaches 0.75, then fitting bought 0.04 — a decomposition a referee will ask for, and
one that cannot be constructed after the fact without choosing it to look good.

This supersedes ADR 012 item 4, which proposed it.

## Consequences

- A new stage writes U; `src/platepu/library/unlabeled.py` keeps building the **inventory** pairs,
  which under ADR 008 item 4 are the *scoring* set and not training data. The two must not be confused
  in the tables or in the report.
- U's construction needs its own verification, because a mis-built U is invisible downstream. The one
  that must exist: a classifier given **only** each side's popularity must score about **0.5** on P
  against U. If it scores higher, the marginals are not matched and item 1 has failed.
- k, the spy fraction and the 95 % threshold live in `configs/models/`, not in code.
- The result's dependence on k is reported, not hidden behind a chosen value.
- Nothing here touches ADR 008 item 6. The evaluation was fixed before the model existed and stays
  fixed.

## Alternatives rejected

- **The full cross-product**, or the HATU-family product. 115 M pairs cannot be materialised; the
  family product (537,200) can, but 73–81 % of its substrates are single-partner, so the popularity
  shortcut stays wide open, and only 2 validation pairs are placeable inside it.
- **Restricting U to the inventory's heavy-atom range.** It would narrow the domain gap, but it keeps
  only 63 of the 1,177 HATU positives and 16 validation pairs: it removes the training data to fix
  the test.
- **Estimating the class prior**, and **ranking only**: see item 3.
- **CatBoost**: see item 3.
