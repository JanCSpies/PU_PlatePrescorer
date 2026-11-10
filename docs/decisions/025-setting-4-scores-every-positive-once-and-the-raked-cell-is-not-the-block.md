# 025 — The Setting-4 split scores every positive once, and the raked test cell is not the training block

- **Status:** Accepted, 2026-10-30, **before any selection number exists**
- **Supersedes:** 023 in part (Decision item 1, the estimand sentence;
  Context 2, the bootstrap-unit figures;
  Context 4, the three raking figures;
  Decision item 2, the tolerance;
  Decision item 3, the boosted-tree sentence and the two verification items;
  the Consequences bullet on the subpopulation;
  Consequences "Numbers needed first", number 6 reclassified) · **Superseded by:** —

ADR 023 is Accepted and its body is not edited (CLAUDE.md §8). This record corrects it. Everything in
ADR 023 not named in the Supersedes line stands: the 3 × 3 Setting-4 rule, the raking in the block and
in each test cell, the Kronecker–Tanimoto kernel ridge estimator written here, the four controls, the
product beside the minimum, and Chemlex not read.

**This record corrects an error of mine.** ADR 023 says, as its headline, that "about 44 % of P is never
scored", that ADR 018 item 3's "every positive is scored out of fold" becomes false, and that PAUC
therefore measures a subpopulation, "a change to the estimand, not a cost". That is wrong. I wrote it,
and when I explained the record I made it the headline, which went beyond what the drafting note said.
It was caught by the counts ADR 023 itself demanded (its number 3).

The scripts behind this record are `scratch/count_setting4_adr023.py`, `scratch/count_setting4_adr023_extra.py`,
`scratch/count_tree_budget.py` and `scratch/count_psd_kernels.py`, with outputs under `scratch/private/`.
They are not published, so every number is written out here (ADR 021 item 2). The partition seeds are
20261030–20261034, nine cells each, so 45 cell comparisons.

## Context

**1. The 44 % is the per-cell discard, not a loss of population.**

| | measured |
|---|---|
| positives scored exactly once, 3 × 3 | 11,396 of 11,396 |
| positives never scored | 0 |
| same, at five partition seeds | 0 |
| same, at 4 × 4 | 0 |
| share of pairs discarded per cell | 0.4444 (= 4/9) |
| per-cell range | 0.4366 to 0.4593 |

Commands: `uv run python scratch/count_setting4_adr023.py`; a second, independent fifteen-line
computation of mine gave the same zero.

Why I got it wrong. In ordinary *k*-fold cross-validation nothing is discarded: you train on (k−1)/k and
test on 1/k. Under the Setting-4 rule one fold trains on 4/9 of the pairs, tests on 1/9 and discards the
4/9 mixed pairs **for that fold**. I read "discarded in a fold" as "discarded". But the nine cells
partition P into nine disjoint parts, because a pair's acid lies in exactly one acid part and its amine
in exactly one amine part. Run all nine as test cells in turn and every pair is tested once. What is
reduced is the **training set per fold**, 4/9 instead of 8/9. The tested population is not reduced.

The lesson is that a claim about a *procedure* needs checking as hard as a measured number does. I had
counted the 44 % and not asked what it was a fraction of.

**2. The split is clean on the two things the discarded-subpopulation worry was about.** Per cell,
scored P against discarded P:

| | median | range |
|---|---|---|
| CLES, P(scored > discarded) | 0.49 to 0.503 | 0.44 to 0.575 |
| KS | 0.04 to 0.07 | maximum 0.13 |
| random-pair-subset null (1/9 against 4/9 of P), CLES | | 0.478 to 0.519 |
| same null, KS median | 0.02 | |

Cell scatter is about twice the null, because pairs cluster on substrates (one amine carries 462
pairs), but it is zero-mean and sign-symmetric. I read it as partition noise: a limitation, not a bias.

Pooled distributions (five-value summary as recorded; mean):

| | recorded | mean |
|---|---|---|
| acid literature frequency | 1, 1, 3, 8, 138 | 9.47 |
| amine literature frequency | 1, 1, 4, 24, 462 | 40.51 |
| acid heavy atoms | 3, 12, 17, 23, 95 | 18.43 |
| amine heavy atoms | 1, 7, 13, 18, 74 | 13.56 |

Cell-equal weighting (weight 1/nP(cell), P per cell 1,043 to 1,524, so pair weights span 1.18 to 1.46)
moves the mean amine frequency from 40.51 to 39.15 (−3.4 %) and everything else by 2 % or less.

**3. The real selection effect is the raking, and it has a direction.** In a raked test cell **3.2 %** of
P pairs have no non-zero-weight U partner on a role (1,864 acid-side and 1,732 amine-side, of about
58,980 cell-P over 45 cells). They are almost all frequency-1 singletons (maximum 3 or 4) and they are
bigger:

| mean heavy atoms | excluded P pairs | all P |
|---|---|---|
| acid side | 19.6 | 18.4 |
| amine side | 17.7 | 13.4 |

This is a limitation the split does not have, and it goes in the paper.

**4. Three figures in ADR 023 Context 4 are wrong, all in the optimistic direction.** They were measured
on a training-block arrangement and carried over to the test cell as if the two were alike. They are
different objects.

| | ADR 023 | training block, measured | raked test cell, measured |
|---|---|---|---|
| U effective sample size, fraction of nominal | "about 0.68" | 0.511 / 0.577 / 0.636 (about 0.58) | 0.176 / 0.218 / 0.338 |
| U rows given weight 0 | "23 to 29 %" | 21.2 / 25.7 / 30.5 % of rows, 19.9 / 24.4 / 29.3 % of mass | 49.4 / 57.3 / 63.6 % |
| residual TVD | "≤ 0.0009" | ≤ 0.0011 (90 values, 95th percentile 0.00074) | 0.011–0.063 acid, 0.013–0.051 amine |

(min / median / max over the 45 sets.) The 0.68 was computed with P in the numerator and only U in the
denominator. U alone is the figure in the table.

The cell TVD has a **floor**: the share of P whose key has no non-zero-weight U cannot be matched by any
reweighting of U. TVD divided by that floor has median 1.00 (amine) and 1.05 (acid). The excess over the
floor is at most 0.0074 (acid) and 0 (amine). Raising the raking from 200 to 2,000 passes leaves the
nine-cell TVDs unchanged to four decimals, so the floor is structural and not a failure to converge.

**5. The bootstrap has fewer units than ADR 023 assumed.** Context 2 of ADR 023 gives 1,821 acids and
1,517 amines for the worst cell. Those are pre-raking. After raking, distinct keys with non-zero weight
equal the P keys exactly in every set:

| | acids | amines |
|---|---|---|
| test cell (min / median / max) | 711 / 816 / 962 | 594 / 683 / 750 |
| training block (min / median / max) | 2,637 / 2,843 / 2,999 | 2,159 / 2,287 / 2,448 |

Hundreds per side in a cell, not thousands. I had called the earlier figure reassuring, and it was not
the right figure.

**6. Sizes for the cost expression** (ADR 023 count 1), min / median / max over 45 sets:

| | n | m | q | (m+q)·n |
|---|---|---|---|---|
| training block | 46,343 / 50,690 / 54,932 | 3,800 / 3,803 / 3,804 | 3,104 / 3,105 / 3,106 | 3.20e8 / 3.50e8 / 3.80e8 |
| test cell | 10,446 / 12,492 / 14,768 | 1,831 / 1,880 / 1,892 | 1,515 / 1,532 / 1,544 | 3.5e7 / 4.26e7 / 5.05e7 |

Counting only non-zero-weight pairs, (m+q)·n is 1.63e8 / 2.00e8 / 2.41e8 for blocks and
6.0e6 / 9.0e6 / 1.38e7 for cells.

**7. Base rate per test cell** (ADR 023 count 8). P / (P + Σ raked U weight), min / median / max:
0.0882 / 0.0934 / 0.0970. Raking preserves U's mass, so this equals the pre-raking value. Global 0.0934.
Within a seed, max/min across cells is 1.06 to 1.09. On the effective-size reading,
P / (P + U ESS) is 0.315 / 0.370 / 0.429, about four times the nominal.

**8. Distinct fingerprints per candidate** (ADR 023 count 7), medians, with min–max over 45 sets where
recorded:

| d | 0 | 1 | 2 | 3 | 4 | 6 | all |
|---|---|---|---|---|---|---|---|
| inside the block | 438 (420–456) | 1,866 | 3,573 | 4,928 | 5,740 | 6,517 | 6,879 (6,870–6,885) |

Per test cell: 327 (304–347) up to 3,400 (3,348–3,427). The ordering d = 0 < 1 < 2 < 3 < 4 < 6 < all is
strictly increasing in all 90 sets, with no ties, under both the all-pairs and the non-zero-weight
variants. ADR 018 item 4's "fewest distinct fingerprints" tie-break is therefore well defined inside the
protocol and always favours the smaller d. No change is needed.

**9. The boosted tree's ordering of `morgan` and `morgan_centre`** (ADR 023 count 4). Setup:
`HistGradientBoostingClassifier(random_state=0, early_stopping=False)`, 4,096 columns, balanced 3,000
against 3,000, `StratifiedKFold(5, shuffle=True)`, read at 120 / 200 / 500 / 1,500 iterations through
`staged_predict_proba`. Command: `uv run python scratch/count_tree_budget.py <seed> <stem>`, seeds
20261029–20261031 on the literature U and 20261029 on the both-sides control (four runs).

| iterations | `morgan` against `morgan_centre`, four runs |
|---|---|
| 120 | `morgan` below, by 0.008–0.017 |
| 200 | `morgan` below |
| 500 | mixed |
| 1,500 | `morgan` ahead in all four, by 0.005 to 0.026 |

Our committed `make_model` uses 200 iterations, inside the inverting range. Caveats: the 120-iteration
gaps are of the order of the fold standard deviation (0.005–0.023); no paired fold differences were
computed; `morgan` is still rising at 1,500, so even that is not converged. For reference, on the same
pool logistic regression gives `morgan` 0.4780 and `morgan_centre` 0.4896, which reproduces ADR 020's
inversion. That inversion is the **linear** model's.

**10. The positive-semidefiniteness checks** (ADR 023 count 5). Command:
`uv run python scratch/count_psd_kernels.py ~/University/Phd/Dataset/Datasets`. `numpy.linalg.eigvalsh`
in float64, matrices symmetrised first (measured asymmetry 0.0), noise floor n·eps·λmax, about 3e-9 at
the largest sizes. Known-answer control passed:

| matrix | minimum eigenvalue | verdict |
|---|---|---|
| identity 500 | +1.000 | PSD |
| `[[1,2],[2,1]]` | −1.000 | flagged |
| random symmetric unit-diagonal 500×500 | −16.98, 233 negatives | flagged |

Per-side Tanimoto on the **full** distinct sets, no sampling:

| | minimum | maximum |
|---|---|---|
| acid d=0 | −6.4e-12 | 2.2e3 |
| amine d=0 | −3.1e-12 | 1.26e3 |
| acid d=all | −6.8e-15 | 9.4e2 |
| amine d=all | −3.2e-15 | 5.3e2 |

All are PSD at the floor. Sub-samples at 1,000 and 3,000 agree.

The Kronecker product on 12 pair samples (seeds 20261030 and 7; n = 1,000 / 2,000 / 4,000; both d) has
minimum eigenvalues between −3e-13 and +2.6e-2, none below the floor. At d=all, n=1,000, seed 20261030
the matrix is strictly positive definite (minimum +2.56e-2, condition number 638).

`min(k_A, k_B)` is **not** PSD. Minimum eigenvalue −0.21 to −0.28 at d=0, n=1,000, down to −7.83 to −7.97
at d=all, n=4,000. That is −2.5 % of λmax, with 260–270 negative eigenvalues; 10⁹ to 10¹¹ times the
floor, growing with n, identical at both seeds.

The two conclusions are not equally strong. A negative eigenvalue on a sample **proves** non-PSD, since a
principal submatrix of a PSD matrix is PSD. The absence of negatives on a sample proves nothing. So the
Kronecker form stands on the Schur product **theorem**, and the sample only confirms the code. The
minimum is **refuted by counterexample**.

A trap in our own code, found on the way: `score/baselines._side_similarity` works in float32, whose
rounding of about 6e-8 per entry would itself appear as a negative eigenvalue. A PSD check run through
our production similarity returns a false non-PSD verdict. The check above built its Gram matrices in
float64 from the raw bits.

Also kept: at d=0 the acid Gram matrix has 5,731 zero eigenvalues of 5,989, so its rank is about 258. At
d=all about 32 are zero, so the rank is about 5,957. That is the dimension the kernel sees, and it
describes the poverty of the d=0 candidate better than "615 distinct fingerprints" does.

## Decision

### 1. The estimand is the whole of P, scored once, out of fold

This replaces the estimand sentence of ADR 023 Decision item 1, the "44 % never scored" claim, and the
Consequences bullet on the subpopulation. ADR 018 item 3's "every positive is scored out of fold" is
**true** under the 3 × 3 rule, in the form: **each positive is scored once, by a model trained without
its acid and without its amine.** PAUC is measured on P, not on a subpopulation. The paper says this
sentence.

The report carries the share of P scored (it must read 100 %, so a different number is a bug) and the
cell-level comparison of Context 2 as a limitation: partition noise, not bias. The discard of 44 % per
cell is reported as what it is, a cut in the training set per fold.

### 2. The raking figures are stated for the block and for the cell, separately

The three figures of ADR 023 Context 4 are replaced by the table in Context 4 above, and the bootstrap
units of ADR 023 Context 2 by Context 5. The 3.2 % of P with no non-zero-weight U partner on a role, and
its direction (bigger molecules), is a reported limitation (Context 3). The base rate is recorded both
ways (Context 7). **The axis reads the nominal one**, P / (P + Σ raked U weight), because PAUC is a
weighted AUC and its prior is the weight share. The effective-size value is reported beside it as the
reason the intervals are wide.

### 3. TVD tolerances (decided by Luka, 2026-10-30)

- **Training block:** raw residual TVD at most **0.002**, about twice the observed maximum of 0.0011.
- **Test cell:** report the **floor** (the share of P whose key has no non-zero-weight U) as its own
  quantity, and set the tolerance on the **excess over the floor** at **0.01**, against an observed
  maximum of 0.0074.

A result outside its tolerance is not read as a selection number (ADR 023 item 2). The reason for the
shape in the cell: a raw threshold would mostly measure how many singletons a cell holds, not how well
the raking worked, and a raking that did nothing would pass it.

### 4. The boosted-tree sentence is reworded

ADR 023 item 3 says the tree "underfits 4,096 sparse columns at 120 iterations". It is replaced by: **the
tree's ordering of `morgan` and `morgan_centre` depends on its budget. It inverts them at 120 to 200
iterations and reverses by 1,500.** With the caveats of Context 9. The tree stays rejected as the
instrument; the argument now stands on a printed number and not on a docstring.

### 5. The PSD items are discharged

The Kronecker–Tanimoto form is positive semidefinite by theorem, confirmed on code (Context 10). The
minimum stays a **K-NN-only** similarity, usable because neighbours need a similarity and not a kernel,
and it is not carried into a kernel model. ADR 020's bar of **0.6843** does not move, and the product is
added as a reading beside it, as ADR 023 item 5 says. Any future PSD check builds its Gram matrices in
float64.

### 6. The cost of one vec-trick product is measured with the implementation

ADR 023 lists it among the counts required before code. It cannot be: nothing can be timed before the
product exists. It becomes a measurement belonging **with** the implementation, computed on the sizes of
Context 6. Feasibility is not in question: the literature's own experiments bracket our n (Viljanen note,
Table 5 p. 20).

## Consequences

- Of the nine items under ADR 023's "Numbers needed first", 1, 2, 3, 4, 5, 7 and 8 are answered here,
  and 6 is reclassified (Decision 6). Number 9 has **two halves and they part company.** Its first half,
  the synthetic probe as a `scratch/` script, is **done**: `scratch/probe_kronecker_reference.py` carries
  the whole reference-environment recipe in its docstring, runs in either stack and computes whichever
  half that stack allows, and gives 0.9575 for the Kronecker kernel against 0.5658 for logistic
  regression on the concatenated sides — the additive figure checked in both stacks, because in the
  reference environment scikit-learn 1.9.1 sits beside SciPy 1.14.1 and that half emits overflow
  warnings there. Its second half, **our own** fit against RLScore and against a naive dense kernel,
  cannot be done before our own fit exists, so it rides with the implementation exactly as number 6
  does. The next piece of work is therefore the implementation, not more counting.
- ADR 023's `Superseded by` field now names this record and the items it replaces. That field is the only
  line of ADR 023 that changed, and it goes in the same commit as this record, so the pointer and the
  thing it points at never exist apart. Its body is untouched, which is the mechanism this project uses
  on ADRs 001, 003, 006, 018 and 023.
- The three scratch scripts behind this record, and the PSD script, are not published, so every number is
  written out above (ADR 021 item 2).
- `docs/design/modeling.md` carries the Setting-4 protocol subsection and needs the corrected figures. That
  is **outstanding**, not done in this piece of work.
- Intervals are wider than ADR 023 said: U's effective sample size in a cell is 0.18 to 0.34 of nominal,
  and the bootstrap has hundreds of units per side.
- No new file or folder is added, so CLAUDE.md §10 does not change.

## Alternatives rejected

- **A single raw cell TVD tolerance of 0.07.** Rejected: observed cell values span 0.011 to 0.063, so it
  would mostly measure how many singletons a cell holds, and a raking that did nothing would pass it.
- **Not raking the test cells at all.** Rejected: the cell would keep the 0.538–0.570 in-block frequency
  mismatch and the reported figure would be partly a frequency reading. This is the hole data-guardian
  found in the first version of ADR 023.
- **Keep ADR 023's estimand sentence and add a note.** Rejected: it is false, and the paper would carry a
  limitation that does not exist.
- **Carry the training-block raking figures to the cell.** Rejected: Context 4 shows they differ by a
  factor of two to three.
- **Replace the minimum with the product in the model-free reading.** Rejected again: it moves ADR 020's
  0.6843 bar. The product is added beside it.
- **Check PSD through the production similarity.** Rejected: float32 gives a false verdict (Context 10).

## References

- ADR 018 item 3, ADR 019 item 4, ADR 020, ADR 021 item 2, ADR 023 (corrected in part).
- `docs/literature/notes/viljanen-2022-pairwise-kernels-vec-trick.md` (Table 1 p. 7, Setting 4; Table 5
  p. 20, sizes).
- `docs/design/modeling.md` (information-axis subsection, outstanding).
