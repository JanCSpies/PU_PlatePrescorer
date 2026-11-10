# 008 — Hand-written chemistry belongs in the opponent, not in the features

- **Status:** Accepted, 2026-10-22 (item 6 rewritten from counts and accepted the same day)
- **Supersedes:** 007 in part (item 3, acid subclasses; item 5, the success rule) · **Superseded by:** —

ADR 007 is Accepted and is not rewritten (CLAUDE.md §8). This record supersedes the two parts of it
that today's measurements overturned, and settles the questions stage S2 cannot start without.

## Context

On 2026-10-22 I built stage F (substrate features) and stage S1 (the eight baselines B0 to B7), ran
them, and then measured three things that decide how stage S2 has to be built. All counts are on the
Chemlex validation side and the group inventory; the Chemlex test side was not opened.

    uv run python scripts/benchmark_split.py chemlex amide_coupling HATU_DIPEA_DMF ~/University/Phd/Dataset/Datasets/derived_samples
    uv run python scratch/count_subclass_evidence.py ~/University/Phd/Dataset/Datasets
    uv run python scratch/count_adversarial.py ~/University/Phd/Dataset/Datasets
    uv run python scratch/count_b2_vs_b3.py ~/University/Phd/Dataset/Datasets

**Finding 1 — the features can tell a literature molecule from a lab molecule.** Adversarial
validation (the proxy A-distance of Ben-David et al., Mach. Learn. 2010, 79, 151): train a classifier
to say which source a substrate came from, and report its 5-fold cross-validated ROC-AUC. On the
Morgan fingerprint it reaches **0.984**. On the ten hand-written descriptors it reaches only
**0.559**. A fingerprint model can therefore learn "looks like a molecule somebody patented" instead
of "reacts", and nothing in its validation AUC would show the difference, because our positives come
from the literature and our unlabeled space does not.

**Finding 2 — the hand-written baseline is strong, and it is strong for a reason I put there.**
B6 (amine subclass + sterics + electronics, no fitted parameter) reaches **0.822** against B1
popularity at 0.524 and B2 nearest pair at 0.632. If the same hand-written chemistry also sits in the
model's feature vector, a PU model that beats B6 has not shown that it learned anything: it has shown
that my rule works, twice.

**Finding 3 — the comparison resolves about 0.05 AUC.** Owen's pigeonhole bootstrap over whole acids
and whole amines (2,000 resamples, seed 20261022, units are **flat** InChIKeys) gives B6 − B4 =
+0.080 with a 95 % interval of [+0.037, +0.134]; B2 − B3 = +0.010 with [−0.070, +0.084]. Paired
differences are far narrower than each AUC on its own, because the scores are correlated. The amines
carry most of the uncertainty: resampling amines alone gives a half-width of 0.083 for B6 − B2,
resampling acids alone 0.029.

A claim I made earlier in the day and now retract: that B2 beats B3 by about 0.012 and so the pairing
adds almost nothing. The interval covers zero in both directions. I cannot say whether knowing which
acid went with which amine helps; I can only say the data here cannot resolve it.

## Decision

### 1. Hand-written chemistry lives in the opponent and in the reporting, never in the model's features

Any rule I write by hand — a subclass taxonomy, a SMARTS, a steric or electronic term — may be used
in three places: a baseline the model has to beat, a stratum the results are reported in, and the
definition of a positive. It may **not** be a column of the feature matrix a PU model fits.

The reason is Finding 2. A benchmark is only informative if the thing being tested and the thing it is
tested against are not the same thing. This is the single rule the rest of this record follows from.

### 2. The acid subclasses are removed; a fingerprint rooted at the reacting atom replaces them

ADR 007 item 3 added nine acid subclasses. They are deleted from
`configs/classes/amide_coupling.yml` as of today. Three reasons, in order of weight:

1. Item 1 forbids them as features, and no baseline uses them: B4 to B6 are defined on the amine
   alone, because the within-class evaluation of ADR 007 needs amine classes and nothing needs acid
   classes.
2. A Morgan count fingerprint **rooted at the reacting atom** (`morgan_centre`, radius 2) recovers
   the same families generically, at 99.9 % purity over 11,787 acids, with no SMARTS to write and no
   taxonomy to defend to a referee.
3. The counts that motivated them survive without them: alpha-quaternary acids succeed at 0.136
   against 0.470 for heteroaryl pi-donors. That contrast is real and worth reporting; it does not
   require a hand-written column to be real.

The nine definitions and what they showed stay reproducible in `scratch/count_subclass_evidence.py`,
so this is a reversal with its evidence kept, not a deletion.

The amine subclasses stay, for the two permitted uses only: baseline B4 and within-class reporting.

A consequence to watch: `subclass_of` returns `None` for a role with no subclasses, so the `subclass`
column of the feature table is null for all 11,796 acids rather than `"other"`. That is deliberate —
"this role has no taxonomy" is not the same statement as "this acid fell outside every class" — but it
means `a_subclass` must never be one-hot encoded in stage S2, where a silently encoded null column
would add a constant feature that looks informative.

### 3. The validation set is every labelled family row disjoint from test on flat keys

The validation set used to be Chemlex's own `both_unseen == "train"` side. That threw away the rows
Chemlex discarded to keep *its* train and test clean — a reason that does not bind us, because our
model never trains on Chemlex. It is now every labelled row in the condition family whose acid **and**
whose amine appear nowhere in the test side.

The exclusion is on the **flat**, stereo-agnostic InChIKey, not the stereo key, because our
fingerprints are computed without chirality: two stereoisomers carry the identical feature vector, so
a stereoisomer of a test substrate sitting in validation is the same molecule to every model we build.
Chemlex's own split had exactly that leak (5 acids, 1 amine).

Result: **2,615** validation reactions over **94** acid and **112** amine flat keys, 29.14 % positive,
against 1,725 over 53 amines before; the test side is unchanged at 1,313; the two sides share **0**
acids and **0** amines, which the split report now records as `shared_acids` and `shared_amines`. The
old validation set is **not** a subset of the new one: 150 of its rows carried a stereoisomer of a
test substrate and are now excluded.

Flat keys, not stereo keys, are also the resampling unit of every bootstrap, for the same reason.

### 4. The unlabeled space for stage S2 is built from literature substrates

U is enumerated from the substrates that appear in the literature corpus, not from the group
inventory. Finding 1 is why: an inventory-based U differs from P by provenance (0.984 separable) and
by size (inventory pairs average 23.73 heavy atoms against 38.66 for validation pairs), so a classifier
separating P from U would be rewarded for learning the catalogue rather than the chemistry.

**The transfer to our inventory comes later, and as a shift test, not as training data.** The
inventory is what the plate is actually built from, so it cannot be avoided; the question is where it
enters. It enters at scoring time, on a model trained on literature P and literature U, and the
adversarial AUC between the literature substrates and the inventory substrates is reported beside the
score as the honest statement of how far out of domain the prediction is. If that number stays near
1.0 on the feature block the model uses, the model is not transferable and the plate design has to
fall back on the baseline. This is a deliberate ordering: train in the domain where the labels are,
measure the distance to the domain where the decision is.

### 5. A substrate-level model competes with the pair model

Finding 3 cannot separate B2 from B3, so I do not assume the pair matters. Stage S2 fits both a model
on the pair and a model on each substrate independently, and reports them side by side. If the
substrate-level model is not worse, it is the one we take forward, because it needs no pairing and
generalises to a substrate the corpus has never paired.

### 6. The success rule — **Accepted 2026-10-22**, before stage S2 was run

The first draft of this item asked for a statistical win against B1 and B4 and descriptive reporting
against the rest. That is kept, but it is no longer the primary test, because it is not where a
provable result lives. Counted with `scratch/count_within_amine_and_widening.py`.

**Primary test, pre-registered: can the model rank acids within one amine?**

B4, B5 and B6 read only amine columns (`b_subclass`, `b_n_heavy_at_2_bonds`, `b_n_ewg_on_ring`). Fix
an amine and every acid gets the same score, so their ROC-AUC across acids within that amine is
**exactly 0.5 by construction, not by measurement**. Any discrimination there is something no
hand-written rule in this project can have, and it is the half of the grid a plate needs: a scorer
that ranks only amines picks rows, not cells.

It is also the statistically strongest test available, because the opponent is a **constant**. Every
other comparison pits one noisy estimate against another.

- Statistic: pooled over amines — Mann-Whitney over all (positive acid, negative acid) pairs that
  share an amine, not a mean of per-amine AUCs. **Pooling is required, not a convenience:** 70 of the
  112 validation amines are usable (at least one positive and one negative acid partner, 70.5 % of
  rows), but the median usable amine has only **one** positive, so a per-amine AUC takes three
  values and is noise. The top 15 amines alone give over 3,000 comparable acid pairs.
- Success: the bootstrap interval excludes 0.5, resampling acid and amine **flat** keys (Owen).
- Reported beside it: how many amines carried it, and the same statistic restricted to the 41 amines
  with at least 3 positives and 3 negatives, as a robustness check.

**The mirror test is not a substitute.** Fixing an *acid* and ranking amines looks denser (72 of 94
acids usable, 94 % of rows) but B4, B5 and B6 are **not** constant there — the amine varies, so they
carry real signal and the opponent is no longer 0.5. Recorded so that nobody later "strengthens" the
test by swapping the axes, which would quietly throw the argument away.

**Secondary test, statistical:** pooled AUC against **B1 (popularity)** and **B4 (amine subclass)**,
the two baselines a referee will raise; the paired bootstrap interval for the difference must exclude
zero.

**Descriptive against the rest** (B0, B2, B3, B5, B6, B7): report the difference and its interval and
claim nothing the interval does not support. **B6 is named as the headline comparison in the paper,
with its interval, whether we beat it or not** — a referee rebuilds B6 in half an hour, so leaving it
out is a hole, while reporting "comparable to the full hand rule, with no hand-written chemistry in
the features" is a defence.

**Widening the validation set is not part of this plan.** It was the obvious lever on power and it
does not work: pooling every condition family, after the same seal, leaves **94 acid and 112 amine**
flat keys — identical to HATU alone, because every substrate the other families add is already there.
It buys 1,445 more reactions (+55 %) and cross-condition contrasts, not substrate diversity, and
power depends on substrates. Any cross-condition work is a separate question.

Reported within amine class throughout (ADR 007 item 1), weighted by pairs, with Owen's bootstrap on
flat keys.

**Known residual risk.** The counts say the primary test is *measurable*; they cannot say it will be
*distinguishable from 0.5*. That needs the model's own variance and is not knowable before stage S2
runs. If it comes out indistinguishable, that is the result and it gets reported as such.

**This is a pre-registration.** It was accepted before a single PU model existed, so the bar was
chosen without knowing whether we clear it. That is the only state in which a bar means anything, and
it is why this item was not written as "whatever the model turns out to be good at". It is not
revised after stage S2 runs; if the result falls short of it, the result is what gets reported.

### 7. Three things are reported beside every model, always

1. The **adversarial AUC** of the feature block the model used, per role, so the provenance risk is on
   the page next to the result rather than in a methods appendix.
2. The **stratified** result: per amine class, not only pooled, and with the class sizes.
3. **B7 (size only)** and the mean heavy-atom count of each scored set, so a reader can see how much
   of any result is explained by molecular size alone.

### 8. P is taken at the class width

B2 and B3 at the HATU width (1,177 pairs) score *worse* than at the class width (28,158 pairs): B2
drops from 0.632 to 0.590, and at the HATU width B2 falls behind B3 (−0.031, 95 % interval
[−0.109, +0.044], B2 ahead in only 18.6 % of resamples). Narrowing P to the one condition family costs more in data than it gains in specificity,
so P is the class-wide positives unless a later measurement says otherwise.

## Consequences

- One config file changes: the acid role of `configs/classes/amide_coupling.yml` loses its
  `subclasses` block. `morgan_centre` is already in `configs/features.yml`.
- Stage S2 needs a new table, the literature-built U, before any model is fitted.
- Every S2 run has to emit the adversarial AUC, so that measurement moves out of `scratch/` and into
  the stage.
- The provenance finding (0.984 against 0.559) may be worth reporting as a result of the paper in its
  own right. **Open**, Luka's decision.
- How the prospective plate arms are designed (model-picked against random, and against which
  baseline) is **open**, Luka's decision, and has to be settled before the plate is frozen.

## Alternatives rejected

- **Keep the acid subclasses as features and accept the circularity.** Rejected by item 1. The cost
  is a benchmark that cannot be interpreted, which is worse than a weaker model.
- **Build U from the inventory, because that is what the plate is made of.** Rejected by Finding 1.
  The inventory still enters, at scoring time, under item 4.
- **Drop the fingerprint and model only the ten hand-written descriptors**, which separate the sources
  at just 0.559. Rejected: that makes the model a refit of my own rule, which item 1 exists to
  prevent. The fingerprint stays, and its provenance risk is reported rather than hidden.
- **Require a significant win against all eight baselines.** Rejected by Finding 3 as an impossible
  bar on 94 acids and 112 amines; replaced by the two tiers of item 6.
