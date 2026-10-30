# 023 — The information axis is measured under a Setting-4 split, on a reweighted U, with a Kronecker–Tanimoto kernel

- **Status:** Accepted, 2026-10-30, **before any selection number exists**
- **Supersedes:** 018 in part (item 3, the fold rule only) · **Answers:** 022 item 4 (the estimator) · **Superseded by:** —

ADRs 018, 019, 020 and 022 are Accepted and are not rewritten (CLAUDE.md §8). This record replaces one
sentence of ADR 018 item 3, the fold rule, and supplies the instrument ADR 022 item 4 left open.

How this sits beside ADR 019. ADR 019 already supersedes 018 items 3 and 4 in part, but what it takes
from item 3 is the missing interval, and what it takes from item 4 is the rule. It does not touch the
fold rule: ADR 019 says nothing about how the folds are cut. So the fold rule is still in force through
018 item 3, and this record replaces it there. ADR 019 items 1, 2, 3 and 5 stand. Item 4 stands in
whether it is used (a pigeonhole bootstrap) and is narrowed here in what it resamples (Consequences).
ADR 020 item 1 (interaction capability is an elimination criterion) is what forces item 3 below. The
ADR 018's `Superseded by` field now also names this record for item 3; its body is untouched, which is
the mechanism the project already uses on ADRs 001, 003, 006 and 018. ADR 019's field is **not** edited,
because nothing of ADR 019 is superseded here.

## Context

Four measurements of 2026-10-29 say the axis as written cannot be built as written. The scripts are
`scratch/count_setting4_split.py`, `scratch/count_shortcut_probe.py` and
`scratch/count_shortcut_probe_extra.py`. They are not published, so every number is written out here
(ADR 021 item 2).

**1. The acid-grouped fold rule tests the wrong setting.** ADR 018 item 3 groups the 5 folds on the acid
flat key alone. Measured on P at `family = "all"` against `unlabeled_literature_k10`: **100 % of test
pairs have an amine that appears in training**, in all 5 folds, for P and for U, at all 5 CV seeds, and
**zero** test pairs have both sides unseen. 4,658 distinct amines carry 113,596 pairs, so no amine can
stay unseen. The rule cannot produce an unseen amine, whatever the seed. On the stricter reading, an
amine counted as seen only when it appears in a training *positive*, it is 68 % for P and 93 % for U,
still with nothing unseen on both sides. In the taxonomy of Viljanen, Airola and Pahikkala (note:
`docs/literature/notes/viljanen-2022-pairwise-kernels-vec-trick.md`, Table 1 p. 7) that is **Setting 3**,
"new acid, known amine". Our deployment, a new reaction class with new substrates on both sides scored
on an inventory grid, is their **Setting 4**. An axis that picks the block is being read on a setting
the method will never be deployed in.

**2. Their Setting-4 split is affordable here.** Split the acid flat keys and the amine flat keys
independently. Train only on pairs whose both members are train-side. Test only on pairs whose both
members are test-side. Discard every mixed pair. At 3 × 3 (their 9-fold version): **44 %** of pairs are
discarded, the training block holds about **51,100** pairs and a test cell about **12,400**. Over 20
partition seeds the smallest test cell held **1,025 P and 9,776 U pairs, over 1,821 acids and 1,517
amines**, so the pigeonhole bootstrap of ADR 019 item 4 has thousands of resampling units in the worst
cell. 2 × 2 discards 50 %. 4 × 4 discards 37.5 % but shrinks the test cell to about 7,000 pairs.

**3. Blocking introduces a marginal mismatch, and it is above the stage gate's tolerance.** U's per-role
marginals are matched to the **whole** of P (ADR 013 item 1, ADR 015), so an in-block P is a subsample
and the match degrades inside a block. Ranking P above U by **in-block** frequency reads
**0.538–0.570**. By **full-P** frequency it reads 0.495–0.516, which is the stage gate's own 0.5049 and
0.5095. So the shortcut the sampler was built to close reopens at 0.04–0.07 above chance, against the
gate's 0.02 tolerance, in every cell and at every seed. ADR 018 item 3 foresaw this ("Holding positives
out leaves the rest slightly off U's match"), proposed rebuilding U per fold, rejected it as not worth
the cost, and said the mismatch would be reported beside the result. The measurement says reporting it
is not enough. It is three times the gate's tolerance, and it is the kind of signal a fitted instrument
reads instead of chemistry.

**4. A remedy was tested and works; the probe that was to size the bias failed instructively.**
Reweighting U's `n_draws` inside the block by raking (iterative proportional fitting, 200 passes, target
= each flat key's in-block P count scaled to U's mass) drives the marginal total-variation distance from
0.17–0.24 to **≤ 0.0009** and the frequency rank-AUC to **0.4999–0.5000**. The cost is real. **23–29 %
of block U pairs are given weight 0**, and U's effective sample size falls to about **0.68** of its
nominal size. These were measured on the training-block arrangement of the probe. Whether they hold for
a 12,400-pair test cell has not been counted (Consequences, number 2).

Evidence that the raking removes the one-sided shortcut: an in-sample fit at `d = all` (10,943 distinct
fingerprints over 11,345 substrates, near-perfect memorisation capacity) reads **0.5001** on the
reweighted arm. This is earned **only for the one-sided marginal shortcut**, which is what the probe
fits, per role, on unique molecule rows with summed weights. It does not show that U is clean. It must
not, because the pairing is what the axis is meant to read, and ADR 015 already says nothing can make
U bias-free. And there is a matching gap that nothing yet closes: zeroing 23–29 % of U is not neutral on
the pair distribution, so the raking may destroy real pair signal together with the marginal signal. That
is why the planted-interaction control of item 4 is a **gate** and not a diagnostic.

The out-of-fold probe I had designed to size the bias **failed by its own pre-registered rule**, and the
cause is the reason this record fixes the split before anything else. Its folds ran over *pairs*, so
holding a pair out removes its own contribution from its molecule's training counts: a held-out P looks
less P-like and a held-out U more so. An identity-only oracle using no fingerprint at all reads
**0.189** on the acid side, where shuffled labels read 0.498. That is the artefact in pure form, an AUC
far below chance made by the fold geometry alone. **The artefact needs a molecule to appear in training
and in test at once.** Under a Setting-4 split no test-cell molecule appears in the training block, so
it has no channel. That is a deduction from the split rule, not a measurement, and control 2 of item 4
measures it anyway.

**5. RLScore, the authors' implementation, was tried on 2026-10-30.** I installed it in an isolated
throwaway environment, outside the project, to see whether it can be the estimator. It cannot, for
measured reasons.

| | |
|---|---|
| licence | MIT (Pahikkala and Airola, 2008–2016), standard text, no added clauses |
| PyPI 0.8.1 on Python 3.13 | does not build: C generated by a pre-3.0 Cython dies on `longintrepr.h`, which CPython 3.12 made private |
| GitHub master on Python 3.13 | installs |
| `CGKronRLS` on SciPy 1.18.1 (ours) | fails at runtime: `PairwiseKernelOperator` subclasses `scipy.sparse.linalg.LinearOperator` without setting `_xp`, which SciPy 1.18 requires |
| `CGKronRLS` on SciPy 1.14.1 | runs |

The project is on SciPy 1.18.1, scikit-learn 1.9.1 and numpy 2.5.3. The first version of this question,
asked before the licence was read, was a blocker on the licence. The licence turned out to be fine. The
blocker is the install path and the SciPy contract.

The same environment gave the first known-answer control. A synthetic pure-interaction task: label =
sign of the product of one feature on each side, so both sides' marginals are symmetric by construction
and there is no marginal signal at all. 3,000 possible pairs, 1,200 observed, 600 train and 600 test,
seed 0.

| | AUC |
|---|---|
| `CGKronRLS`, Kronecker kernel (linear per-side kernels, `maxiter` 200, `regparam` 1.0) | **0.9575** |
| logistic regression on the concatenated sides | **0.5658** |

It reproduces ADR 020's central claim in miniature: the Kronecker route finds signal that lives only in
the pairing, and the concatenated route does not. It is a **synthetic control**, not a result about our
data.

The script is `scratch/probe_kronecker_reference.py`, and its docstring carries every step of building
the reference environment, which was the part not written down the first time. It runs in both stacks
and computes whichever half the stack allows:

    $PROBE/.venv/bin/python scratch/probe_kronecker_reference.py   # the Kronecker half
    uv run python scratch/probe_kronecker_reference.py             # the additive half

**The additive figure was checked in both.** In the reference environment scikit-learn 1.9.1 sits
beside SciPy 1.14.1, which are not tuned to each other, and the additive half emits divide-by-zero and
overflow warnings there — so its number was re-read in the project's own stack (SciPy 1.18.1), where it
is clean. Both give **0.5658**, so the warnings did not move it. Checking that was the point; the two
stacks agreeing is itself the control on the control.

## Decision

### 1. The information axis is cross-validated by the Setting-4 rule, at 3 × 3

This replaces ADR 018 item 3's "5 folds grouped on the acid flat key". Everything else in that item
stands: PAUC as the primary reading, cross-fitting because the deliverable is the input to a fitted
model, and a model-free reading beside it whose disagreement is a finding.

- **What is split.** The distinct **acid flat keys** of P ∪ U into 3 parts, and the distinct **amine flat
  keys** of P ∪ U into 3 parts, independently, each by a seeded random permutation into near-equal parts.
  Flat keys, because the fingerprints are chirality-blind, so two stereoisomers are one input and
  splitting on stereo keys would leak a molecule across the boundary.
- **What trains and what is tested.** For each of the 9 cells `(i, j)`: the **training block** is every
  pair whose acid is outside part `i` and whose amine is outside part `j`. The **test cell** is every pair
  whose acid is in part `i` and whose amine is in part `j`. Every other pair, train acid with test amine
  or the reverse, is **discarded**: used neither for fitting nor for scoring. This is their Table 1, p. 7,
  Setting 4.
- **3 × 3, fixed.** Not 2 × 2 and not 4 × 4. I fix it now, before any selection number exists, so it
  cannot move after a number is seen. It is a judgement on the counted trade-off and not a measurement:
  4 × 4 discards less (37.5 % against 44 %) but leaves about 7,000 pairs per test cell, and 2 × 2 keeps a
  larger cell at a 50 % discard with only 4 cells to average. 3 × 3 is also the authors' own shape.
- **The reading.** One PAUC per cell over P against weighted U in that cell. The axis value per candidate
  `d` is the mean over the 9 cells. IPM beside it, per ADR 019 item 5.
- **Hyperparameters.** Anything chosen is chosen on an inner split of the training block formed **by the
  same rule**: the block's own acid parts and amine parts split 75/25, train on both-inner-train pairs,
  validate on both-inner-validation pairs, discard the mixed ones. That is the authors' nested protocol
  (p. 24). The inner split never touches the test cell. Item 3 leaves very little to choose.
- **Seeds.** One partition seed, `20261030`, for all seven candidates, so every `d` is read on the
  identical 9 cells and the comparison is paired. Sensitivity is reported and not selected on: the same
  protocol is rerun at `20261030 + 1 … + 4` and the spread of each candidate's axis value is a reported
  number. The inner 75/25 seed is derived from the partition seed and the cell index.
- **The estimand changes, and this is the headline of the record.** ADR 018 item 3 says "every positive
  is scored out of fold". Under this rule that is false and cannot be made true. A positive is scored only
  if its acid and its amine land in matching parts, so about **44 % of P is never scored**. PAUC is
  measured on the **Setting-4 subpopulation of P**, not on P. The axis no longer answers "how well does
  this block separate P from U" but "how well does it separate the positives that can be scored in
  this way". I do not treat this as a cost to be paid. It changes what the number means. The paper has
  to say so in a sentence. The report carries the share of P scored, and a comparison of the scored
  against the discarded positives on literature frequency and heavy-atom count, so a reader can see
  whether the subpopulation is special. Whether it is a sentence or a problem is decided by that
  comparison (Consequences, number 3).

### 2. U is reweighted, in the training block and in each test cell, and the cost is reported beside every result

- **The method.** Raking (iterative proportional fitting) on U's `n_draws`, 200 passes, for each role
  targeting each flat key's P count in the **same set of rows**, scaled to that set's U mass. Fixed before
  any selection number exists: 200 passes, the same for every candidate and every cell.
- **Both the training block and each test cell are raked, each on its own rows.** If only the block were
  raked, the test cell would still carry the 0.538–0.570 in-block frequency mismatch and the reported
  figure would be partly a frequency reading. PAUC is therefore a **weighted AUC on the raked weights**
  of the test cell. The residual total-variation distance is reported for **both**.
- **Where it belongs in the code.** In the selection protocol, computed per set of rows from those rows.
  It is **not** a change to stage U's stored table. Stage U writes one table of draws whose marginals
  match the whole of P. A block weight is a property of a *split*, not of U, and it depends on the
  partition seed. Writing it into `<root>/unlabeled/...` would make a stage table depend on a fold seed,
  would give stage S2 a column it must not use, and would break "one stage reads one table and writes
  one table" (CLAUDE.md §5).
- **Reported beside every result, per cell:** the share of U pairs given weight 0, U's effective sample
  size as a fraction of its nominal size, the residual marginal total-variation distance per role (block
  and test cell), and the in-block frequency rank-AUC before and after. A result whose residual TVD is
  above a tolerance is not read as a selection number. The tolerance is fixed from the counts of
  Consequences number 2, not from the single 0.0009 observation.
- **What the raking is not.** It removes the one-sided marginal shortcut. It does not make U clean, and
  it may remove pair signal (Context, number 4). Control 3 of item 4 is the check on the second.
- **Why reweighting and not rebuilding U per fold.** ADR 018 item 3 rejected rebuilding on cost. The
  measurement adds a second reason: rebuilding would draw a different U per cell, so the nine cells
  would no longer be nine readings of one unlabeled space.

### 3. The estimator is the Kronecker product pairwise kernel with a binarised per-side Tanimoto base kernel, in kernel ridge regression, written here

This answers ADR 022 item 4 and respects ADR 020 item 1: the Kronecker feature map is the outer product
of the two sides' maps, so the model represents the interaction by construction, and `make_model` is not
used on pair-level data.

Here the estimator is an **instrument comparing seven descriptors**, not a product. Its sensitivity must
not vary with what it measures, and the ladder it measures spans 3.0 to 34.2 bits per molecule and 615
to 10,943 distinct fingerprints (ADR 018 item 1). Of the four admissible kernels (linear, Poly2D,
Gaussian, Kronecker; the others need homogeneous domains and an acid is not an amine):

- **A boosted tree is rejected as the instrument** on the one thing ADR 020 does carry: it underfits
  4,096 sparse columns at 120 iterations. Its reading therefore depends on a candidate's sparsity as much
  as on its information. I do not claim more. ADR 020's recorded inversion of `morgan` (0.4780) below
  `morgan_centre` (0.4896) is the **linear** model. The tree appears there only as a 0.58–0.61 range with
  no ordering per block. The per-candidate ordering of the tree is a number to be printed before this
  clause is published (Consequences, number 4).
- **Poly2D** adds the acid-acid and amine-amine second-order blocks, capacity in exactly the per-side
  direction the raking of item 2 nulls. It would put back what the effective sample size was spent to
  remove.
- **A Gaussian pairwise kernel** is algebraically the Kronecker form with a Gaussian base kernel (squared
  distance is additive over the two blocks; p. 12 of the paper). It carries a bandwidth `γ` whose sensible
  value scales with the number of set bits, so it would have to be rescaled per candidate: seven
  instruments, not one.
- **Tanimoto is a ratio**, so it reads the same way at `d = 0` and at `d = all`. Normalised per side, the
  Kronecker kernel has unit diagonal, so there is no scale parameter to set per candidate. The authors
  use this on binary fingerprints (Heterodimer, p. 21).
- **Binarised, not MinMax on counts.** The blocks are counts (`uint8`, max 51). `score/baselines` binarises
  (`> 0`), and B2, B8 and `pair_knn` all read the binary form. MinMax on counts would be a second
  similarity in the project, against CLAUDE.md §5, and it leaves the positive-semidefinite case the
  literature establishes (binary Tanimoto, p. 21).
- **`λ` is fixed across all seven candidates.** My argument against `γ` applies to kernel ridge's `λ` and
  to the iteration budget word for word: a per-candidate choice makes seven instruments. They are chosen
  **once**, on the inner 75/25 split, and the choice is recorded before any candidate's PAUC is read. The
  per-candidate values on the inner split are reported as a diagnostic, so a mis-tuning at the ends of the
  ladder stays visible.
- **Two claims are verification items, not assertions.** That a product of PSD kernels is PSD (Schur
  product theorem) and that Tanimoto on binary vectors is a kernel are both standard, and neither is
  established in our record. Both are checked by the minimum eigenvalue of our own Gram matrices before
  the axis is read (Consequences, number 5). If it is negative beyond numerical noise, the whole clause
  fails and the estimator question reopens.
- **Fitting.** Weighted kernel ridge regression, P at 1 and U at 0, weights `n_draws × rake weight`,
  solved by conjugate gradient with the generalised vec trick, because the pair set is not a complete
  matrix (113,596 observed of 6,138 × 5,015 possible) and the closed form does not apply.
- **The implementation is our own. RLScore is a reference implementation and never a project
  dependency.** About one module and one test module. It is checked against RLScore, run in an isolated
  throwaway environment on SciPy 1.14.1, and against a naive dense kernel on a small hand-made case. The
  reason is measured (Context, number 5). Adopting RLScore would mean running it from GitHub master and
  downgrading SciPy for the whole project because of one alpha library's subclassing, and the next SciPy
  change would break it again. As a reference it gives the value without the cost. The licence is not the
  reason.

### 4. A known-answer control, fixed before any selection number exists

The instrument is void unless it reads what we already know. Controls, all run on the same cells and the
same path as the real reading:

1. **Shuffled labels inside the test cell.** Permute the P/U label with its weight inside the cell. Must
   read **0.50 ± 0.02**. If not, the reading is fold geometry and not data.
2. **The identity-only oracle under the Setting-4 split.** The oracle that read 0.189 under pair-folds:
   a per-flat-key `P / (P + U)` ratio from the training block, no fingerprint, scored on the test cell.
   Must read **0.50 ± 0.02**, because no test-cell molecule occurs in the training block. It is the
   empirical form of the deduction in Context number 4, and it catches a split implemented differently
   from how it is written.
3. **A planted interaction, positive control.** A synthetic label on the real pairs, determined by a
   genuine cross-side interaction of two bits that are actually set at the candidate `d` (the authors'
   chessboard, our fingerprints). The instrument must read **well above 0.5** at every candidate. The
   threshold and the bit pair are fixed with the counts before the axis is read. Without it, a reading of
   0.5 on the real data cannot be told from a dead instrument, and it is the check on whether the
   raking destroyed pair signal. The synthetic pure-interaction probe of Context number 5 (0.9575
   against 0.5658) is the first control of this kind, on toy data; it does not replace this one.
4. **The shuffled-P pairing control.** Permute P's amine column inside the training block to get P′,
   which keeps P's in-block marginals and destroys the pairing, and read the axis on it. This is stage U's
   verification item 4 moved to the axis. It must fall relative to the real reading. If it does not, the
   candidate is not being read on the pairing.

All four are **gates on the instrument**, not on the data, so they do not contradict ADR 017 ("a probe
is a diagnostic, not a gate"): ADR 017 governs probes that judge our U, and these judge our measuring
device. If any fails, the axis is void and no block is picked from it.

### 5. The model-free reading gets a product beside the minimum

`score/pair_knn.py` uses the **minimum** of the two sides' Tanimoto. Whether the minimum is positive
semidefinite is unknown, and the vec trick does not cover it. I add the **product** of the two sides'
Tanimoto as a **new reading beside the minimum** and do not redefine the minimum. This is the pattern B8
set beside B2. ADR 020's bar of **0.6843** does not move. The difference between minimum and product
becomes a measurement and not a choice. Both are read under the same split and the same weights, and a
disagreement with the primary reading is a finding (ADR 018 item 3).

### 6. Chemlex is still not read

ADR 018 items 5 and 6 stand unchanged: the validation split is opened only after the pick is made, the
test side never, and stage F gains a column only for the block that wins.

## Consequences

- ADR 018 item 3's fold rule no longer applies. Items 1, 2, 5 and 6 stand, and ADR 019 stands with item 4
  narrowed: the pigeonhole bootstrap resamples the acid and amine flat keys **of the test cell**, since
  cells share no substrates by construction, and the per-candidate interval brackets the mean over cells.
  Size stays 5 repeats, 500 resamples, seed 20261028.
- **The estimand is the Setting-4 subpopulation of P.** About 44 % of P is never scored. The axis remains
  a comparison between candidates on identical rows, which is all ADR 019 item 2 claims for it, but it no
  longer reads "every positive". The paper carries the sentence.
- The axis costs 7 candidates × 9 cells fits, plus the inner splits, four controls and the seed
  sensitivity. It is paid once.
- U's effective sample size inside a block falls to about 0.68, and 23–29 % of block U gets weight 0.
  Intervals widen accordingly.
- The raking is validated against the one-sided shortcut only. Whether it removes pair signal is open
  until control 3 is read.
- The RLScore comparison figures are in the session scratchpad and not in the repository. They are not
  citable until the probe is a script in `scratch/` and, if kept, a count script under ADR 021.
- Where the code lives, whether the protocol writes a table through `io`, whether the winner's stage F
  column is added before or after the Chemlex agreement check of ADR 018 item 5, and whether the three
  scratch scripts are published (each hard-codes a path) are **not decided here**.
- `docs/design/modeling.md` gains the subsection of 2026-10-30.

**Numbers needed first (CLAUDE.md §8), before any code. This is the next piece of work.**

1. **Per-side counts.** Distinct acids `m` and amines `q` in the training block and in each test cell at
   the fixed partition seed. The vec-trick cost is in `(m + q)·n`, not `n`, and the record has no per-side
   split.
2. **The block and the cell after raking.** For the smallest of the 9 cells, the training block and a
   typical cell, over the 5 reported partition seeds: P and U pairs with non-zero weight, distinct acids
   and amines with non-zero weight, effective sample size, residual TVD per role. The 1,025 / 9,776 /
   1,821 / 1,517 figures are pre-raking. The TVD tolerance for the report is fixed from these numbers.
   The test-cell raking has not been measured at all.
3. **The discarded subpopulation.** The share of P never scored at 3 × 3 (and at 4 × 4, once, for the
   record), and scored against discarded on literature frequency and heavy-atom count.
4. **The boosted tree's ordering of the seven candidates**, from `scripts/counts/count_block_ladder.py`
   as it stands, printed before the tree clause of item 3 is published.
5. **The PSD checks.** The minimum eigenvalue of the normalised per-side Tanimoto Gram matrix for acids
   and for amines on the block substrates at `d = 0` and `d = all`, and of the pair Gram matrix on a
   sample for the Kronecker product and, for item 5, for the minimum.
6. **One iteration's cost at our scale.** Wall time and peak memory for one vec-trick product and for a
   full fit at n ≈ 51,100. The note's 1.29e9 operations and under 1 GB are arithmetic and not measured
   here.
7. **Distinct fingerprints per candidate, restricted to the block substrates.** ADR 018 item 4's last
   tie-break counts them over all 11,345 substrates. Under a block split the count may differ.
8. **The base rate per test cell**, P against weighted U with non-zero weight, because PAUC and its
   interval are read on it and the cells are not equal.
9. **The RLScore check** reproducible from the repository: the synthetic probe as a `scratch/` script,
   and our own fit against RLScore on SciPy 1.14.1 and a naive dense kernel.

## Alternatives rejected

- **Keep the acid-grouped folds and report the limitation.** Rejected on measurement: the rule cannot
  produce an unseen amine at any seed, so it reads Setting 3 while the paper claims Setting 4.
- **Setting 2 or 3 as the axis, with Setting 4 beside it.** Rejected. It would choose the block on an
  easier setting than the deployment. Reporting all four settings as *diagnostics* is not rejected and
  costs one extra readout per cell.
- **2 × 2 and 4 × 4.** Rejected on the counted trade-off, 4 × 4 for the 7,000-pair cell, 2 × 2 for having
  only four cells to average.
- **Rebuild U per fold** (ADR 018 item 3's own proposal). Rejected for cost and, newly, because it makes
  the nine cells nine different unlabeled spaces.
- **Report the mismatch instead of fixing it.** Rejected: 0.04–0.07 is three times the gate's tolerance,
  and the raked arm shows the fix is available at a measured price.
- **Rake the training block only.** Rejected: the test cell would keep the in-block frequency mismatch.
- **`λ` chosen per candidate on the inner split.** Rejected: seven instruments. Reported as a diagnostic.
- **MinMax on counts.** Rejected: a second similarity in the project.
- **Redefine the model-free reading as the product.** Rejected: moves ADR 020's 0.6843 bar.
- **Poly2D, a Gaussian pairwise kernel, a boosted tree.** Rejected as instruments, each for its own
  reason in item 3. None is rejected as a *model* for stage S2, which is a different decision.
- **RLScore as a dependency.** Rejected for the measured reasons in Context number 5, not for its
  licence.
- **Pin SciPy below 1.15 to use RLScore directly.** Rejected: it downgrades a core numeric library for
  the whole project because of one alpha library, and nothing else here asks for it.
- **The one-line `_xp` shim on RLScore's operator.** Rejected: it is a patch to an installed copy of a
  library installable only from GitHub master. It breaks again at the next SciPy change, and I would be
  maintaining a fork under another name.
- **Drop the information axis and choose on Chemlex.** Rejected in ADR 018 and still rejected: a new
  reaction class has no Chemlex.

## References

- `docs/literature/notes/viljanen-2022-pairwise-kernels-vec-trick.md` (Table 1 p. 7, the Setting-4 split;
  p. 8 and §4.4 p. 13, the Kronecker kernel; p. 12, Gaussian as Kronecker; p. 21, Tanimoto per side;
  Theorem 1 p. 10 and p. 19, the vec-trick cost; p. 24, the nested 75/25; Table 5 p. 20, Metz 93,356 and
  Merget 167,995 pairs; p. 23 and p. 33, RLScore).
- ADR 013 item 1, ADR 015, ADR 017, ADR 018, ADR 019, ADR 020, ADR 021, ADR 022.
- `docs/design/modeling.md` (stage U, stage F, stage S2, and the information-axis subsection).
