# 030 — The model's feature block is chosen by a label-free resolution rule, per reaction class

- **Status:** Accepted, 2026-11-02, **before any Chemlex validation read**
- **Supersedes:** 014 in part (item 2, `morgan_centre` as primary for everything) and 018 in part (item 6,
  stage F stays untouched until a `d` wins) · **Superseded by:** —

ADRs 014 and 018 are Accepted and are not rewritten (CLAUDE.md §8). Only the `Superseded by:` line of each
was edited, as ADR 018 already does for its own predecessors. This record replaces the criterion by which
the model's block is chosen. It renames nothing on an outcome.

## Context

1. **Stage S2b ran for the first time today and its primary configuration could not rank the plate.** The top
   96 inventory pairs all carried one score, 0.993091, shared by 1,300 pairs. Choosing 96 from 1,300 equally
   scored candidates is a tie-break, not a prediction.
2. **Cause: those 1,300 pairs have one feature vector between them.** No model can score identical rows
   differently.

       uv run python scratch/count_top_tie_cause.py ~/University/Phd/Dataset/Datasets

3. **My first hypothesis was wrong, and I record it as wrong.** I thought the near-balanced spy-arm class ratio
   saturated the probabilities. I reweighted to five effective class ratios. At 1:1.4, 1:2.8, 1:5.7, 1:10.0
   and 1:19.9 the top 96 never held more than two distinct scores, and at 1:2.8 the top tie grew to 5,292.
   Fixing the ratio does not help. The sweep is the second half of the same script, one run, so it shares a fit
   setup with the step-2 count above and the two are comparable:

       uv run python scratch/count_top_tie_cause.py ~/University/Phd/Dataset/Datasets
4. **The representation is the cause.** `morgan_centre` keeps only environments rooted on the reacting atom
   (d = 0). I measured, over the 284,600 laboratory candidate pairs, the share of pairs whose count vector is
   shared with no other pair:

       PYTHONPATH=scratch uv run python scratch/count_resolution_bar.py ~/University/Phd/Dataset/Datasets

   | d | inventory (largest tie) | validation | literature positives |
   |---|---|---|---|
   | 0 | 2.0 % (5,915) | 27.5 % | 10.1 % |
   | 1 | 13.7 % (1,116) | 81.1 % | 37.0 % |
   | 2 | 43.3 % (924) | 92.8 % | 57.3 % |
   | 3 | 68.3 % (120) | 99.7 % | 71.9 % |
   | 4 | 82.3 % (45) | 99.7 % | 80.0 % |
   | 6 | 89.1 % (24) | 99.7 % | 90.3 % |
   | all | 91.0 % (6) | 99.7 % | 97.7 % |

   The d series is not a ladder at its first step. From d = 1 upward each block contains the one before
   (checked at d = 1, 2, 3, 4, 6, 8 and 100 on all 14 hand-written fixtures). d = 0 is special: it can carry
   bits that no larger d has. RDKit drops an environment whose bond set lies inside one it has already
   recorded. The radius-2 environment of a terminal reacting atom is kept at d = 0, where it is the only
   root. At d = 1 the neighbour becomes a root and swallows it. On ethylamine (`CCN`) bit 317 is set at d = 0
   and absent at d = 1; the other two bits stay. This happened on 7 of 14 fixtures: pivalic acid,
   tert-butylamine, aniline, benzylamine, 2-aminopyridine, 4-chloroaniline, ethylamine. It is the quirk
   `configs/features.yml` already records in the `morgan_centre` comment, one step further along. Measured by
   chem-verifier, 2026-11-02, in `tests/test_features.py`
   (`test_near_loses_the_radius_two_bit_of_a_terminal_centre_at_the_first_step`).
   The rule below is untouched by this: it counts distinct feature vectors and never relied on nesting. That
   count does not fall as d grows. Two pieces of evidence, of unequal weight:

   - **On the real data (the premise of the rule):** the inventory shares in the table rise from 2.0 % to
     91.0 %. Command: `count_resolution_bar.py`, above. It is green.
   - **On made-up molecules (a sanity check only):** over seven hand-written amine fixtures the number of
     distinct vectors is 4, 5, 6, 7, 7, 7, 7, 7 at d = 0, 1, 2, 3, 4, 5, 6, 100. This is chem-verifier's RDKit
     reference check, 2026-11-02, pinned by `test_near_distinct_vectors_never_fall_as_the_distance_grows` in
     `tests/test_features.py`. That test is red until the block is built (`near_distance` is not yet in
     `FeatureConfig`). It will reproduce the figure once this passes:

           uv run pytest tests/test_features.py -k distinct_vectors

     I cannot reproduce it independently today, because the fixtures are the verifier's.

   At d = 0 only 2 % of the plate's candidates are distinguishable from every other candidate. I counted on
   raw counts, not binarised bits, because `features/blocks.block_matrix` never binarises and the learner
   sees counts (ADR 018).
5. **The binding constraint is the set the plate is drawn from.** Validation saturates by d = 3. The
   inventory does not until d = 6. A rule computed on training or validation data would have missed this.
6. **Resolution and literature-against-laboratory separability rise together; there is no sweet spot.** The
   acid AUC goes from 0.697 at d = 0 to 0.861 at d = all. Size-adjusted it goes from 0.796 at d = 3 to 0.811
   at d = all, inside the ±0.03 intervals, so most of the apparent cost of a wider block is molecular size.
   Command, run 2026-11-02, output in `scratch/count_centre_distance_provenance.out.txt`:

       uv run python scratch/count_centre_distance_provenance.py ~/University/Phd/Dataset/Datasets

   It uses `platepu.score.axes.provenance_auc` with the ADR 018 settings (5 repeats, 500 resamples, seed
   20261028), over 5,570 literature-only acids against 249 lab-only, and 4,536 against 283 amines. These
   counts supersede the "11,203 against 189" quoted in the docstring of `score/axes.py`, which is stale.
7. **The axis ADR 014 item 2 chose on is weaker than I thought.** An independent literature memo of today
   reports that "choose the representation with the lowest provenance AUC" has no theoretical support
   (Ben-David et al., NIPS 2007; Zhao et al., ICML 2019: when label marginals differ, domain invariance
   cannot also guarantee low error). A provenance difference is a shortcut only if provenance-like features
   correlate with the training label. Here they cannot: P and U are both built from literature substrates and
   are frequency-matched.

   **None of these citations is checked against a paper.** The memo is an independent AI literature search
   run from my brief, and it states its own access levels. Ben-David 2007 was confirmed only through
   Crossref, a bibliographic check and not a reading. Zhao et al. 2019, the load-bearing one, rests on a
   search-engine summary that nobody in this project has checked against the paper. See
   `docs/literature/notes/2026-11-02-independent-pu-memo.md`. This item is a reason to doubt ADR 014 item 2's
   axis, not the basis of the rule: the rule in Decision 1 stands on the resolution table (Context 4) alone.

## Decision

1. **The model's feature block is chosen by a rule, not by preference.** Take the smallest d whose share of
   uniquely resolved pairs, measured on the set the model will score, is within 5 percentage points of the
   best achievable over all candidate d. Smallest, because every increment costs extrapolation onto
   laboratory compounds. Within a margin of the best, because the achievable maximum differs per reaction
   class and an absolute threshold would not transfer.
2. **For amide coupling the rule selects d = 6.** The best achievable on the inventory is 91.0 % and d = 6
   gives 89.1 %. The choice is unchanged at a 2-point margin, so it does not depend on the 5. The free
   parameter is insensitive across a 2.5-fold range. That is the only honest defence a free parameter has.
3. **The rule is computable for a reaction class with no HTE dataset.** It needs the class's literature
   substrates and the laboratory inventory, and no label of any kind. That makes it a method and not a
   tuning step.
4. **The baselines keep their own blocks.** B2 stays on `morgan`, B8 on `morgan_centre`. The rule governs
   what a fitted model sees. A retrieval baseline is a different consumer. B8's 0.7115 on validation against
   B2's 0.6317 is evidence about retrieval, and this record does not touch it. Both figures are the
   pre-cut-off pair from ADR 015 item 1; after the rebuild of 2026-10-29 they are 0.7221 and 0.632. The
   comparison holds either way, because the two were measured together on one path.
5. **Stage F gains the block.** ADR 018 item 6 said stage F stays untouched until a d wins. One has, so
   `configs/features.yml` and `features/substrate.py` gain it, tests first, and the feature table is rebuilt.
   I keep default RDKit, not `includeRedundantEnvironments=True`. The redundant setting would restore strict
   nesting, but a large d would then no longer equal `morgan` bit for bit. That equality lets the whole d
   table be compared with every number already measured under the two existing columns. I judged
   comparability worth more than a tidier invariant. The far boundary is pinned by
   `test_near_beyond_the_diameter_equals_the_whole_molecule_fingerprint`.

## What I owe and what I am not claiming

- The rule has been applied once, to one reaction class, and I fixed its margin knowing that class's numbers.
  It is mechanical and label-free, so it is not fitted in the usual sense. It is not yet validated across
  classes, and the paper must say so.
- I expect `morgan` to beat `morgan_centre` for fitted models. I record the expectation before the read. The
  grounds are the supervised ceiling (0.937 against 0.892, ADR 012), the yield-signal diagnostic (0.686
  against 0.583, ADR 027) and the resolution table above.
- The seven-amine figure's test must be green before this ADR is committed. If the figure comes out
  differently when the block is built, the ADR gets a superseding record, not a quiet edit, as with the
  406/669 inventory counts earlier today.
  **Discharged, 2026-11-02.** The block is built and the test is green; the figure stands (4, 5, 6, 7, 7,
  7, 7, 7). Command: `uv run pytest tests/test_features.py -k distinct_vectors`. Whole suite after the
  block: 765 passed, 1 skipped, `ruff check .` and `mypy src` clean.
- The Zhao et al. claim in Context 7 must be checked against the paper before it appears in anything I
  publish. So must Ben-David 2007, which Crossref confirmed exists and nothing more.
- ADR 014 item 2 already anticipated this: "If `morgan` does better once fitted, that is reported, and it is
  a finding about what fitting extracts, not a reason to rename it primary." I do not rename anything on an
  outcome. I replace the criterion, on label-free grounds, before the read.

## Alternatives considered

- **Keep d = 0 and let the tier protocol absorb the tie.** The tiers rescue the plate claim. They do not
  rescue the model. A representation that collapses 98 % of the candidate grid discards chemistry the model
  could use, and the yield diagnostic measures that loss at 0.583 against 0.686.
- **Switch to `morgan` (d = all).** It is not the smallest block that clears the bar, and every increment
  costs extrapolation.
- **A universal fingerprint for every reaction class.** The diversity of a candidate library sits at
  different distances from the reacting centre in different classes. A fixed d is a guess, and for this class
  it is already measured wrong.
- **DRFP, the obvious universal reaction fingerprint.** By design it cancels substructure unchanged by the
  reaction. On a library whose members differ far from the reacting centre it would behave like d = 0 and hit
  the same wall (literature memo, 2026-11-02; licence MIT, verified at source).
- **Fixing the class ratio instead.** Measured and rejected in Context 3.

## Consequences

- `configs/features.yml` and `features/substrate.py` gain the new block. The feature table is rebuilt.
  Stages S2a and S2b are re-run for it. The existing `pu_spy_*` and `pu_score_*` configs gain a sibling.
- Every number measured under `morgan_centre` for a fitted model is superseded for the primary line. That
  includes the first S2a run's 15,250 reliable negatives and the four S2b score tables of today. They are
  kept and reported, not deleted: the comparison across blocks is part of the result.
- The validation read has still not been spent, and I took this decision without it.

## References

- `docs/design/modeling.md` (S2a, S2b; the stage F delta for the new block belongs in its "Stage F — features" section).
- ADR 012 (supervised ceiling), 014 (item 2), 018 (items 5, 6), 027 (yield-signal diagnostic).
- `docs/literature/notes/2026-11-02-independent-pu-memo.md`: third-party summary, not a paper (Ben-David et al., NIPS 2007; Zhao et al., ICML 2019; DRFP).
