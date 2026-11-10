# 027 — What the 1,121 literature negatives can and cannot measure

- **Status:** Accepted, 2026-11-02, **before any Chemlex validation read**
- **Supersedes:** 024 item 1 (in part) and 026 item 2 (in part) · **Superseded by:** —

This record adds one use of the negatives and limits one claim about them. ADR 024 and ADR 026 stay as
written. Roles 2 and 3 of ADR 024 stand. ADR 014 item 1 (the spy technique is primary) and ADR 014 item 2
(`morgan_centre` is the primary block) stand.

## Context

**1. A diagnostic was run, and it answered a question I had got wrong.** On 2026-10-30 I measured that the
P-against-U model scores the 1,121 known negatives like hidden positives, and cannot separate them from the
positives (journal, 2026-11-02, first entry, result (c)):

| set | out-of-fold quartiles, morgan_centre (25 % / median / 75 %) |
|---|---|
| U | 0.054 / 0.070 / 0.081 |
| spies | 0.069 / 0.081 / 0.106 |
| the 1,121 known negatives | 0.069 / 0.081 / 0.114 |

`auc_n_vs_ptrain`, out of fold, every spy fraction, both blocks: 0.4596 to 0.4802.
Command: `bash scratch/run_spy_fractions.sh ~/University/Phd/Dataset/Datasets 4`.

The same runs carry two more AUCs that belong next to it, because on their own the quartiles above are
easy to misread. Out of fold, spy fraction 0.20, both seeds:

| contrast | morgan_centre | morgan |
|---|---|---|
| published positives against U (`auc_p_vs_u`) | 0.650, 0.656 | 0.707, 0.704 |
| the 1,121 negatives against U, **negatives as the positive class** | 0.683, 0.678 | 0.755, 0.743 |

**Read the direction carefully.** `scratch/count_spy_threshold.py:209` computes the second row with the
*unlabeled* pair as the positive class, so the logs print 0.317 / 0.322 and 0.245 / 0.257. The table above
states one minus that, which is the form the sentence "the known negatives score above U" requires. The
quartiles in the table higher up confirm the direction independently: N sits at 0.069 / 0.081 / 0.114 and U
at 0.054 / 0.070 / 0.081. I had this direction the wrong way round once and the quartiles are what caught it.

Together the three contrasts say what the objective learned: it separates the literature from the
constructed unlabeled space (0.65 to 0.71), it ranks reported failures **above** that space rather than
below it (0.68 to 0.76), and it cannot separate those failures from reported successes at all (0.46 to
0.48). That is a selection signal, not a yield signal.

No shuffled-label control was run. Every line in these logs is `shuffle=False`, so the project has no
permutation null for these three numbers, and none should be quoted until one is run.

I read that as "the features carry no yield information". That does not follow. It only shows that a model
trained on a different question (P against U) cannot see it. On 2026-11-02 I asked what a plain
classifier trained on the question itself would do. Asking it exposed the error.

**2. The measurement.** 11,396 reported positives against the 1,121 reported negatives,
`HistGradientBoostingClassifier(max_iter=200, early_stopping=False)`, 5 folds, two seeds, both blocks.
Run on 2026-11-02:

    uv run python scratch/count_pos_vs_neg.py ~/University/Phd/Dataset/Datasets <block> <seed>

ROC-AUC, positive against negative:

| block | protocol | seed 20261102 | seed 20261103 |
|---|---|---|---|
| morgan | acid-grouped (Setting 3) | 0.7006 | 0.7077 |
| morgan | both sides unseen (Setting 4) | 0.6846 | 0.6872 |
| morgan_centre | acid-grouped | 0.6354 | 0.6121 |
| morgan_centre | both sides unseen | 0.5967 | 0.5684 |

Controls, same command:

| control | ROC-AUC |
|---|---|
| permuted labels, four runs | 0.5002 / 0.5022 / 0.5122 / 0.5160 |
| permuted labels, mean | 0.508 (null standard error about 0.009; the floor is 0.5) |
| summed heavy atoms alone, no model | 0.4760 |
| random folds (upper bound only) | 0.653 to 0.738 |

The heavy-atom control says the signal is not size. Random folds are an upper bound and nothing more: 410
acid flat keys and 356 amine flat keys occur in both classes, so an acid seen only among the negatives can be
memorised. The Setting-4 subset holds 3,900 pairs at seed 20261102 and 3,901 at seed 20261103, of which 3,539 are
positive in both.

**Coverage, and why it matters.** Only 333 of the 1,121 negative pairs have both substrates in the stage F
table. The 792 missing substrates were featurised in the script with `features.substrate.substrate_features`,
under the same class rule and features config, exactly as `scratch/count_spy_threshold.py` does. I did not
restrict to the 333. Those are the negatives whose substrates happen to be popular, and that selection would
make the answer meaningless.

**3. The limitation I cannot measure away.** A low-yield entry sits in the same scope table as the successes
beside it. Setting 4 removes shared substrates, not shared papers. ADR 022 already recorded that the document
level is unmeasurable, because ORDerly carries no patent identity. So part of the margin above 0.5 may be "I
recognise this scope table" and not "I recognise a bad reaction". I cannot separate the two with the data I
have.

## Decision

1. **The diagnostic use is allowed. The method use is not.** A model may be fitted on the 1,121 as labels
   when it scores nothing, is discarded, and its only output is a number in a record. ADR 024 item 1 ("no
   model in this project is fitted with the 1,121 as labels, as reliable negatives, or as targets") is
   narrowed to the models that produce scores. Roles 2 and 3 of ADR 024 stand untouched, because they are
   measured with the PU model and not with this one.
2. **ADR 026 item 2 overstates the check.** It calls the 1,121 "the only available check" on the
   reliable-negative set. The quartiles in Context 1 show the check cannot tell "the threshold cut the wrong
   subspace" from "the known negatives look like published positives, which they are". Both halves hold: it
   remains the only check available, and it cannot vouch for the step.
3. **Nothing else changes.** This is my decision of 2026-11-02. I weighed three options and chose the
   first. The evidence above was on the table and the choice was made **before** any validation read. That is
   the only thing that separates it from choosing a result (ADR 008 item 6).
   - *Chosen:* keep the pre-registered plan. The spy technique stays the primary method (ADR 014 item 1),
     `morgan_centre` stays the primary block (ADR 014 item 2), the negatives stay an instrument. Today's
     finding goes into the paper as a result about the method: a classifier trained on the real literature
     negatives reaches 0.686 (morgan, seed 20261102) under a both-sides-unseen split, while the PU objective
     reaches 0.46 to 0.48 on the same contrast. So the P-against-U objective discards yield information that
     is present in the features.
   - *Rejected: switch the primary block to `morgan`.* It is the better block on this question by about
     0.10, far above the seed spread. But it also separates literature substrates from lab ones at 0.859
     (acids) and 0.879 (amines) (ADR 012 item 2, ADR 018), and the plate scores lab substrates. It carries
     the larger provenance shortcut on the task that counts. Buying 0.10 on a literature question with that
     risk was the wrong trade.
   - *Rejected: let the negatives into training.* It would destroy the only instrument there is, for 1,121
     rows that are biased reported near-misses. It is also the larger deviation from PAYN.

## Alternatives considered

The two rejected options in Decision 3. A third, not taken: restrict the diagnostic to the 333 pairs already
in the stage F table. Rejected in Context 2 (popular-substrate selection).

## Consequences

- `configs/models/pu_spy_morgan_centre.yml` and `configs/models/pu_spy_morgan.yml` now exist with
  `spy_fraction: 0.20`. The lift measurement at the four fractions is in the journal entry of 2026-11-02
  (result (b)).
- `scratch/count_pos_vs_neg.py` is the diagnostic. It is git-ignored. Whether it joins the published count
  scripts is open, for ADR 011's list.
- The "same scope table" confound is owed a sentence in the paper's limitations.
- ADR 024 and ADR 026 are not edited, including their `Superseded by:` lines.
- Open, not decided here: whether a yield-aware objective is worth designing, now that the information is
  known to be present in the features.

## References

- Journal, 2026-11-02 (two entries).
- ADR 008 item 6; ADR 011; ADR 012 item 2; ADR 014 items 1 and 2; ADR 018; ADR 022; ADR 024 items 1 and 2;
  ADR 026 item 2.
- `docs/design/modeling.md`.
