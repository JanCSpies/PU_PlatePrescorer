# 019 — The information axis is PAUC, and two assumptions of its proof fail here

- **Status:** Accepted, 2026-10-28, **before any selection number was measured**
- **Supersedes:** 018 in part (item 3, the information axis without an interval, and item 4, the rule) · **Superseded by:** —

This is a new record and not an amendment of ADR 018, because it is **new knowledge, not a correction**.
ADR 018 stated what I knew before I read the literature on evaluating PU models without negatives. It
stays as the honest snapshot of that point. What I found since does not show 018 was wrong. It shows the
information axis it pre-registered already exists, with a proof, and that the proof does not cover our
setting. ADR 018 is not rewritten (CLAUDE.md §8). Only its `Superseded by` line points here.

## Context

**The axis in ADR 018 item 3 is a known quantity.** It is the ROC-AUC of ranking the positive validation
set above the unlabeled one, ties counted as one half. I had not seen it named.

**The primary source is older than the one I first found.** Wei, Wang, Tu and Li, *Robust model
selection for positive and unlabeled learning with constraints*, Science China Information Sciences
2022, 65(11), 212101, DOI 10.1007/s11432-020-3167-1, received 17 July 2020. Their Theorem 1, "Equivalent
PU model evaluator":

    AUC_PN = AUC_PU / (1 - pi)  -  pi / (2 (1 - pi))

Both coefficients are constants, so `AUC_PU` preserves the ordering of candidate models with respect to
the supervised `AUC_PN`. The class prior pi is needed to convert the number, not to choose between
models.

**The confirmation.** Wang, Wu, Li, Zhang, Niu and Sugiyama, *Accessible, Realistic, and Fair
Evaluation of Positive-Unlabeled Learning Algorithms*, ICLR 2026, arXiv:2509.24228. They restate the
quantity as the **proxy AUC score (PAUC)**, Definition 2 and Proposition 2, and recommend it: "We
recommend using PA and PAUC in future PU learning experiments, especially in real-world applications
where negative data cannot be obtained." Their reasoning treats the unlabeled validation data as
"corrupted negative data", and AUC is robust to corrupted labels (Charoenphakdee, Lee, Sugiyama, ICML
2019).

Notes: `docs/literature/notes/wei-2022-robust-model-selection-pu.md` and
`docs/literature/notes/wang-2026-fair-evaluation-pu-learning.md`. Both are agent summaries I have not yet
checked against the PDFs line by line. Locators: Wei et al. Theorem 1 p. 5, Section 3.5 p. 5, IPM
sentence and Table 2 p. 8; Wang et al. Definition 2 p. 4, Proposition 2 and the recommendation p. 5,
proof in Appendix B.2.

## Decision

### 1. The axis is PAUC, and Wei et al. is the source

The design, the code and the paper name it PAUC. The paper cites Wei et al. 2022 first, and Wang et al.
2026 as the confirmation and the recommendation.

### 2. Two assumptions of the proof fail here; I keep the statistic without the guarantee

- **The unlabeled set must follow the true marginal.** The proof rewrites the unlabeled density as the
  pi-weighted mixture of the two class-conditionals. Our U does not: its substrate marginals are matched
  to P's on purpose (ADR 013 item 1, ADR 015). The unlabeled distribution the plate actually faces is the
  inventory grid, whose pairs average 23.73 heavy atoms against 38.66 for the validation pairs
  (`docs/design/modeling.md`, line 366).
- **P must be a completely-at-random sample of the positives (SCAR).** Wei et al. say it in the proof,
  where Wang et al. leave it in the algebra: their Term I is estimable through P "since every x in P is
  obtained from the underlying distribution of positive samples completely at random". Reporting bias is
  exactly the violation of this, and it is the reason the project exists.

The ordering guarantee is therefore **not inherited**. I use PAUC as a comparison between candidates
measured the same way, and the paper says so instead of citing the theorem as cover.

In the working session of 2026-10-28 I first read this proof as not requiring SCAR. That reading was wrong:
Wei et al. state the assumption in one line where Wang et al. leave it inside the algebra. This record
corrects it.

### 3. One cause, two consequences

ADR 014 item 1 records that the spy threshold is calibrated on **reported** positives while the positives
hidden in U are **unreported** ones. That is the same SCAR violation, at a different step. It is one
defect with two consequences, and the paper carries it as one sentence, not as two unrelated caveats.

### 4. The information axis gets a bootstrap interval too

This tightens ADR 018 item 4, which asked for an interval on provenance and a point estimate on
information. Three reasons:

- Neither paper analyses the bias or the variance of PAUC, or says how much validation data it needs.
- Wei et al. found the variance a large enough problem to add a **control-variate** reduction step. It
  uses a classifier that discriminates P from U, with the density ratio as the control variate, and cuts
  the variance by a factor of (1 - rho^2).
- The gaps between neighbouring candidates on the `d` grid are likely to be small. Without an interval I
  could select a difference that is noise.

The eligibility and tie rules of ADR 018 item 4 now apply on **both** axes: among candidates whose
provenance interval overlaps the lowest candidate's, take the highest primary information; where the
information intervals overlap too, take the block with the fewest distinct fingerprints.

**The bootstrap size is the same on both axes; what is resampled is not.** Size: 5 repeats, 500
resamples, seed 20261028 (ADR 018 item 2). Two bootstrap configurations inside one protocol would be a
trap. On the provenance axis a row is one substrate and substrates are independent, so rows are
resampled. On the information axis a row is a **pair**, and pairs are not independent: the same acid
appears in many of them. Resampling pairs would treat each as fresh evidence and give an interval far too
narrow. The information axis therefore uses **Owen's pigeonhole bootstrap**: resample the acid and amine
flat keys and take the pairs they induce. The project already does this in
`scripts/counts/count_b2_vs_b3.py` (see its docstring). An interval on the wrong unit is worse than no
interval, because it looks rigorous.

A change to a pre-registration is allowed here **only** because no selection number has been measured, and
because it makes the criterion stricter, not looser.

### 5. The IPM evaluator is reported beside PAUC

Wei et al.'s other evaluator, with the identity loss, is the mean score on P minus the mean score on U.
Its transform is pure scale: `IPM_PN = IPM_PU / (1 - pi)`. They found that "using the IPM as the model
evaluator typically led to more robust performance than the AUC, therefore we report results of the IPM
in following experiments". Wang et al. recommend PAUC without mentioning this.

PAUC stays primary. I compare **different descriptors, each with its own fitted model**. IPM depends on
the score scale and AUC does not. Their comparison was over hyperparameters of one algorithm, at class
frequency 0.01. Reporting both costs nothing, and two papers disagreeing is a reason to show both
numbers and not pick one.

### 6. What neither paper established

Neither reports a rank-agreement or correlation statistic between PAUC-based selection and oracle
selection. Wang et al.'s own Table 1 is mixed:

| method | test AUC, selected by PAUC | test AUC, selected by oracle |
|---|---|---|
| PUbN | 93.44 | 94.33 |
| VPU-c | 93.51 | 91.79 |

Source: Wang et al. 2026, Table 1 (page `[n]`). No command: these are the authors' figures, not mine.

Neither paper applies these criteria to choosing a **feature representation**. Both select
hyperparameters and checkpoints. My use is an extension beyond what either validated, and the paper says
so.

## Alternatives considered

- **Amending ADR 018.** Rejected. This is new knowledge, not a correction, and 018 is the honest record
  of what I knew first.
- **Dropping the criterion because its assumptions fail.** Rejected. Every other label-free criterion
  rests on the same assumptions or stronger ones. The only label-bearing alternative is Chemlex, which is
  what I am trying not to need.

## Consequences

- ADR 018 items 3 and 4 are superseded in part. The candidate grid (item 1), the provenance axis (item
  2), the cross-fitting and the model-free report (item 3, those parts), Chemlex read afterwards (item 5)
  and stage F untouched until a `d` wins (item 6) stand.
- From ADR 018 onward the `Superseded by` field is filled when a later record narrows an earlier one, so
  a reader who opens 018 alone learns that 019 exists. ADRs 008, 012, 013 and 014 keep "—" although later
  records supersede them in part, because they are committed and CLAUDE.md §8 forbids rewriting them. The
  inconsistency is recorded here and is not carelessness.
- The selection numbers still do not exist. This record contains none of mine.
- The paper carries three sentences from this record: PAUC with its sources; the two failed assumptions,
  stated as one SCAR defect (item 3); and that using it to choose a representation goes beyond what
  either paper validated.
- A test for SCAR exists, and I record it as a pointer, not a commitment: Teisseyre, Furmanczyk and
  Mielniczuk, *Verifying the Selected Completely at Random Assumption in Positive-Unlabeled Learning*,
  arXiv:2404.00145. Its idea is that SCAR holds exactly when the feature distribution of the positives
  equals that of the labelled ones. Its motivating example, reported against unreported drug side
  effects, is structurally the same as reported against unreported reactions. It assumes a single-sample
  iid scenario, so it does not apply off the shelf to a two-sample setting with a constructed U. I have
  not read it beyond this; no note exists.

## References

- `docs/design/modeling.md` (stage U, stage F; heavy-atom means at line 366).
- ADRs 013, 014, 015, 018.
- `docs/literature/notes/wei-2022-robust-model-selection-pu.md`.
- `docs/literature/notes/wang-2026-fair-evaluation-pu-learning.md`.
