# 010 — No benchmark-derived column in an open table

- **Status:** Accepted, 2026-10-22
- **Supersedes:** 009 in part (it left this open) · **Superseded by:** —

ADR 009 is Accepted and is not rewritten (CLAUDE.md §8). It settled where Chemlex-derived *tables*
live. It did not say what to do about a Chemlex-derived *column* inside an open table, and
data-guardian found one hours later.

## Context

`src/platepu/positives/build.py` added two per-row columns to the positives table: `leak_substrate`
(one of the pair's substrates also occurs in the benchmark) and `leak_pair` (both do). The positives
table built from ORDerly is licensed `open` and sits outside `private/`.

A boolean saying "this ORDerly substrate also occurs in Chemlex" is a fact derived from a
CC BY-NC-ND dataset. Row by row, 567 of them, it is a partial membership list of Chemlex keyed by
molecules we may publish — which is closer to redistributing Chemlex than anything else we write.
The licence guard could not see the problem, because the column rides inside a table whose own
licence is correctly `open`.

Against that, CLAUDE.md §6 requires the overlap to be **reported**: "evaluation sets never enter
training; check overlap by InChIKey and report it." Deleting the check outright would trade a
licence problem for a leakage problem.

## Decision

**The per-row columns go; the aggregate counts stay.**

`_count_leakage` replaces `_flag_leakage`: it returns `{"leak_substrate": n, "leak_pair": n}` and
writes nothing onto the rows. `build_positives` returns those counts as its third value, and
`scripts/positives.py` puts them in `positives.report.json`, where they already were.

Two counts are still derived from Chemlex, but they are aggregate and cannot be inverted to name a
substrate, so they are the kind of number a paper reports. §6 is satisfied, §9 is not strained, and
the open table carries nothing it should not.

**The counts now mean distinct pairs.** They are taken on the `all` family, which holds every kept
pair exactly once. Counting every row counted a pair once per condition family it belongs to, so the
previous figure was pair-rows and not pairs: on ORDerly it read 579 where the honest number is
**567**. `leak_pair` was and remains **1**.

## Consequences

- `positives.parquet` loses two columns, 18 to 16. Nothing read them — grepped across `src/`,
  `scripts/`, `scratch/` and `tests/` — and the table was regenerated: 29,335 rows and
  `{'all': 28158, 'HATU_DIPEA_DMF': 1177}`, both unchanged.
- `build_positives` returns three values instead of two. The two callers in `scripts/positives.py`
  are updated; `sizes()` takes `[0]`.
- The reported overlap drops from 579 to 567 for the reason above, not because anything changed in
  the data.
- A general rule to apply next time, which is the part worth remembering: **a table's licence
  describes its rows, not every column in them.** The guard checks the licence column, so a column
  derived from restricted data can ride into an open table unseen. When a stage joins against a
  restricted source, what comes back has to be an aggregate, or the table stops being open.

## Alternatives rejected

- **Keep the columns and move the positives table under `private/`.** Rejected: the positives are
  ORDerly, they are open, and they are what the paper has to ship (ADR 005). Hiding 29,335 open
  rows to protect 567 booleans is the wrong way round.
- **Drop the check entirely.** Rejected by §6. The overlap is exactly what has to be reported for
  anyone to believe the evaluation.
- **Keep the counts only in a private report.** Rejected as unnecessary: an aggregate of this kind
  is publishable, and a leakage number nobody can see is not a leakage check.
