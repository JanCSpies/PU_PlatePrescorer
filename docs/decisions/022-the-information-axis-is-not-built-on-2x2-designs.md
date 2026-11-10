# 022 — The information axis is not built on 2×2 interaction designs; the counts are the result

- **Status:** Accepted, 2026-10-29
- **Supersedes:** — (it answers a question ADR 018 item 3 left open, and supersedes nothing) · **Superseded by:** —

ADR 018 item 3 defines the information axis as a ROC-AUC of positives against U. ADR 019 item 4 only
adds an interval to it. This record changes neither. It records that a stricter measure, which I read
about today, cannot replace that axis, and it says what stays open.

## Context

Pahikkala et al. define the interaction concordance index (note:
`docs/literature/notes/pahikkala-2025-interaction-concordance-index.md`). Its unit is a complete 2×2
design: two acids, two amines, all four outcomes observed. It is invariant to the two per-side main
effects, so it depends on the interaction alone. Proposition 1 (proved) says an additively separable
predictor scores exactly 0.5. Our information axis is plain ROC-AUC, which has no such property. The
same paper shows kernel ridge regression at IC-index exactly 0.5 on all seven datasets, blind to the
pairing, still beating k-NN on C-index on three of them. On two others the same blindness showed as a
much lower C-index. So C-index exposes the defect unpredictably.

The question was whether IC-index can be computed on our data. I counted, in two rounds. Both scripts
are in `scratch/` and are not published, so every number is written out here (ADR 021 item 2).
Flat (stereo-agnostic) keys are primary, because the fingerprints are chirality-blind. Stereo keys move
every count by 4–7 % in the literature and 1–4 % in Chemlex, so the key space decides nothing. An
`unlabeled` row is a missing cell, not a zero.

### Round one

    uv run python scratch/count_complete_2x2_designs.py

| | literature | Chemlex validation |
|---|---|---|
| acids × amines | 6,138 × 5,015 | 94 × 112 |
| observed pairs | 12,365 | 2,611 |
| matrix density | 0.00040 | 0.248 |
| complete 2×2 designs | 10,495 | 505,865 |
| admissible, binary (contrast ≠ 0) | 1,573 | 151,670 |
| fully crossed (\|contrast\| = 2) | 42 | 5,945 |
| admissible, continuous | 10,445 | 393,663 |

The literature has 11,396 positive plus 1,121 negative pairs in `reported.parquet`, 12,365 after flat
keys collapse 152 duplicates. The method was checked against brute force on a hand-made example and on
70 random cells at seed 0.

A U pair has no observed outcome, so the constructed unlabeled space contributes no design at all.

### Round two: three levels of control

A complete literature design is not a controlled experiment. Its four cells can come from four papers
with different reagents, solvents and yield definitions. I measured how many designs survive control.

    uv run python scratch/count_2x2_control_levels.py

About 35 s.

| level of control | complete | admissible | fully crossed |
|---|---|---|---|
| none (baseline, reproduces round one) | 10,495 | 1,573 | 42 |
| same condition family (HATU/DIPEA/DMF) | 1,035 | 160 | 0 |
| same exact `condition_key` | 5,066 | 876 | 27 |
| same grant-week file (rigorous upper bound) | 3,395 | 597 | 21 |
| tightest proxy, index gap ≤ 10 | 2,977 | 543 | 17 |

Restriction removes designs and barely changes how often the contrast is non-zero. The admissible share
is 15.0 % at baseline, 15.5 % in the family, 17.3 % per `condition_key`, 17.6 % per file.

### Three findings

1. **The document level can only be bracketed.** ORDerly carries no patent identifier in our data.
   `source_document` is ORDerly's `extracted_from_file`, a USPTO grant-week block of many patents: 489
   values, one `grant_date` per file and one file per `grant_date`. `original_index` is a reaction id and
   `procedure_details` is free text with no patent number. So 597 is an upper bound and 543 an
   unvalidated proxy. Weak support for the proxy: two reports at index gap 1 share agents and solvents
   24 % of the time, against 3.3 % for a random pair in the same file (seed 0).
2. **The control is concentrated.** The 597 designs come from 29 files. The largest holds 227, the top
   two about 65 %, and 10 of the 29 files contribute one design each. Of the 21 fully crossed designs,
   12 come from one group. 209 acid pairs hold the 597, 98 of them one design each. These, and every
   other figure in this section, come from `scratch/count_2x2_control_levels.py`, the command above —
   it is the only one of the three scripts that reads `extracted_from_file`, `grant_date`, `yield_kind`
   or `temperature`. `scratch/count_complete_2x2_concentration.py` produced the *baseline* concentration
   instead: 937 acid pairs holding the 1,573 designs, 706 of them one design each, the largest 144, over
   485 acids and 286 amines, with the top ten acids carrying 22.5 % of design-incidences.
3. **In our own condition family there is no unambiguous design.** HATU/DIPEA/DMF has 160 admissible
   designs and 0 fully crossed.

Yield comparability cannot be checked. `yield_kind` reads "isolated" for every report that carries a
yield, so it cannot separate isolated from HPLC or crude yields. 68 % of the 28,633 kept reports have no
temperature and 63 % no time.

### A lead, not a result

The surviving designs sit in what look like single-patent library tables. One group is 5 acids × 43
amines (159 designs). Another is 23 acids × 11 amines over 93 cells at 37 % density, carrying 12 of the
crossed designs. If that reading holds, the patent literature contains small, dense, condition-controlled
substrate matrices, miniature HTE plates. I did not check this against the structures and these counts
cannot check it.

## Decision

### 1. The information axis is not built on 2×2 interaction designs

Two reasons, in this order.

1. U pairs have no observed outcome, so the P-versus-U axis has no designs at all. Not too few: none.
2. Inside the observed literature the controlled subset cannot be measured at the document level, is
   concentrated in two groups, and has no unambiguous design in our condition family.

ADR 018 item 3 and ADR 019 item 4 stand as written.

### 2. The counts are recorded as a negative result

They are the result of a measurement that was worth making, not a failed attempt. The numbers above are
the record.

### 3. Proposed, not decided: IC-index as an evaluation measure

I recommend, and I have not decided, that IC-index be computed at evaluation time on the Chemlex
validation split and on the prospective plate. There 151,670 admissible designs under one condition make
it well estimated. It would answer what AUC cannot: did the model learn the pairing, or only that some
acids are generally good?

This cannot be slipped in by this record. Adopting an evaluation measure changes the pre-registration,
and the rule is that model and scores are frozen before any well is run. It has to be decided
deliberately, before the plate. Until I decide, nothing in the evaluation changes.

### 4. Open: the estimator for the primary information axis

ADR 018 item 3 makes the primary reading a fitted model on pair-level data. ADR 020 item 1 eliminated
the default, so `make_model` is not used for pair-level data without the interaction check. That axis
has no instrument and nothing here supplies one. Candidates from Viljanen, Airola and Pahikkala 2022
(`docs/literature/notes/viljanen-2022-pairwise-kernels-vec-trick.md`): the Kronecker product pairwise
kernel, Poly2D, a Gaussian pairwise kernel, or a boosted tree with enough capacity. The symmetric,
anti-symmetric, ranking and MLPK kernels are ruled out, because they need both sides from the same
domain and an acid is not an amine. Choosing the instrument after seeing the candidates' numbers would
be choosing the result, so it is fixed first.

## Consequences

- No code changes. The information axis keeps ADR 018 item 3 and ADR 019 item 4.
- Item 3 waits for my decision and item 4 for an instrument. Both come before the plate.
- **Open: publish `scratch/count_2x2_control_levels.py`?** This negative result rests entirely on it, and
  ADR 021 records that ADR 011's forecast of which scripts would matter was wrong. It is my call.
  Publishing needs the §3 path fix that ADR 021 item 2 now requires: the script builds its root from a
  hard-coded `DS`, and must take the Datasets root from `sys.argv[1]`. The same holds for the other two
  scratch scripts.
- Until then the numbers live in this text only.

## Alternatives rejected

- **Use the 543 proxy-level designs anyway.** Rejected. The document level is unmeasured, and 12 of the
  21 crossed designs come from one group.
- **Use the continuous `max_yield` contrast to get 10,445 designs instead of 1,573.** Rejected. The
  yields come from different papers' conditions, so a continuous contrast is more condition-confounded,
  not less, and `yield_kind` cannot verify comparability.
- **Go to the raw ORD or USPTO sources for a real patent identifier.** Not rejected. It is unexplored
  work. Nobody has looked, so I cannot state its cost beyond this: it needs a new adapter pass over the
  raw files and a join back to ORDerly rows, estimated only after a first look at the raw schema.
- **Adopt IC-index as an evaluation measure in this record.** Not done, for the reason in item 3.

## References

- ADR 011 and ADR 021 (why the numbers are in the text and not in a published command).
- ADR 018 (item 3), ADR 019 (item 4), ADR 020 (item 1).
- `docs/literature/notes/pahikkala-2025-interaction-concordance-index.md`
- `docs/literature/notes/viljanen-2022-pairwise-kernels-vec-trick.md`
