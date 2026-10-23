# 009 — Chemlex is noncommercial, and the guard must see every table

- **Status:** Accepted, 2026-10-22
- **Supersedes:** 004 in part (the licence taxonomy and the name of the source list) · **Superseded by:** —

ADR 004 is Accepted and is not rewritten (CLAUDE.md §8). This record supersedes the parts of it that
today's audit overturned.

## Context

`src/platepu/ingest/chemlex.py:58` hard-codes `licence="open"` for every Chemlex record. Chemlex is
**CC BY-NC-ND 4.0**. CLAUDE.md §9 is explicit that no row of it and no table derived from it may be
shared or committed, so `open` is the wrong label, and every table downstream inherits it:
`derived/chemlex/1_records.parquet` and `2_rsmi.parquet` both carry `open` today.

That matters because of how the guard works. `write_table` in `src/platepu/io.py` does not read
chemistry or provenance; it reads the **label**. A wrong label is therefore not a cosmetic problem —
it is the guard being told to wave the rows through.

Found on 2026-10-22 by a pre-commit audit, together with three holes in the guard itself, two of
which are already fixed in the working tree:

1. The `private/` test was lexical, so `root/private/../library/x.parquet` passed while the bytes
   landed in `root/library/`, and a `private` symlink could point anywhere. **Fixed today:**
   `_under_private` resolves both paths before comparing.
2. The licence test was a deny-list naming `licensed` and `internal`, so a null licence, `"Internal"`,
   `"internal "` and any unrecognised value passed. **Fixed today:** it is now an allow-list, which
   asks that every value **be** `open`. A consequence worth stating, because it is what makes this
   record cheap: a new restricted value needs no entry anywhere to be protected.
3. A table with **no** licence column is still written anywhere. Not fixed; see item 6.

A ripple survey then found that the one-line change is not a one-line change, and that the obvious
fix would not actually hold.

## Decision

### 1. A third licence value, `noncommercial`

Chemlex records carry `licence="noncommercial"`. It is refused outside `private/` by the allow-list
already, so the guard needs no new rule.

It is a separate value and not `internal`, because the two say different things and we will have to
explain the difference in the paper: `internal` means "this is the group's own and nobody else's"
(ADR 004), while `noncommercial` means "this is published, and the licence forbids us to
redistribute it". Collapsing them would lose the distinction the moment a second CC BY-NC source
arrives.

### 2. `LICENSED_SOURCES` becomes `PRIVATE_SOURCES`

Every one of its four readers already treats membership as "withhold this source's rows from the
open outputs", not "licensed in the Pistachio sense": `io.py:58` (`source_dir`),
`scripts/positives.py:108`, and `scripts/map_missing.py:52` and `:54`. Chemlex is not licensed in
that sense, so keeping the old name would make the code state something untrue at four reading
points. `"chemlex"` joins `"pistachio"` in the renamed set, which routes stages 1 to 6, the
positives branch and the mapping-cache split in one edit.

### 3. The benchmark tables get a licence column, because a path is not a guard

`labels.parquet`, `validation.parquet` and `test.parquet` have **no** `licence` column, and their
paths are hard-coded in `scripts/benchmark.py:102` and `scripts/benchmark_split.py:54` rather than
chosen by `source_dir`. Moving them under `private/` alone would protect them by **convention**, and
a path convention is exactly what failed in hole 1 above: it looked like a guard for as long as
nobody wrote `..`.

So all three gain a `licence` column carrying the benchmark's licence, and they move under
`private/`. The guard then refuses them on their own content, and the protection survives the next
person who adds a path without reading this file.

### 4. The open RXNMapper cache keeps the Chemlex mappings it already holds

`derived/cache/rxnmapper-0.4.3/part-*.parquet` (63 parts) holds Chemlex reaction SMILES and atom
mappings in the **open** cache, with no `licence` column and no row-level source, so it cannot be
cleaned selectively. It is not purged and not re-run. Recorded as a known, accepted exception, which
ADR 004 already anticipated in its "still guarded by path only" note.

The reasoning: a cache entry is an atom mapping keyed by a reaction hash, which is a derived
computation rather than the Chemlex dataset, and re-mapping everything would cost a full RXNMapper
run for no gain in what is actually published. **Nothing in the paper is reproduced from this
cache**; it is a speed-up, and ADR 005 requires the published result to rebuild from open sources
alone. If that ever stops being true, this item has to be revisited.

### 5. Four hard-coded `open` stamps are corrected with the rest

These are the real risk, because each silently relabels:

- `src/platepu/positives/build.py:165` — `"licensed" if (s == "licensed").any() else "open"` would
  turn a `noncommercial` row into `open`. It needs a rule for a mixed table, and the rule is that a
  merged table carries the **most restrictive** licence of its inputs.
- `src/platepu/features/substrate.py:128` — `validation.assign(licence="open")` labels every
  validation substrate `open`, and its `LICENCES` tuple at `:34` would raise "unknown licence" on
  `noncommercial`.
- `scripts/score_baselines.py:56` — `LICENCES = {"validation": "open", ...}`.
- `src/platepu/ingest/record.py:79` — `Literal["open", "licensed"]` has to widen, or mypy rejects
  the adapter.

### 6. The licence-less tables stay a recorded gap, not a silent one

Nine writers produce tables with no `licence` column: the five `*.rejected.parquet` writers,
`scripts/positives.py:72`, `scripts/map_missing.py:32`, `scripts/benchmark.py:103` and
`scripts/benchmark_split.py:64`. Items 3 and 5 close the two that carry substrate information. The
rest keep the gap, and `tests/test_io.py` pins it with the list of writers in the docstring, so it is
not tidied into a refusal by someone who has not read this record.

## Implementation plan, in order

Tests first at every step (CLAUDE.md §7). Nothing here is done yet.

1. **Tests:** a `noncommercial` case in `tests/test_io.py`; `tests/test_ingest_chemlex.py:40` to
   expect `noncommercial`; a Chemlex routing assertion beside `tests/test_ingest_script.py:29`;
   the licence column asserted on all three benchmark tables; the most-restrictive merge rule in
   the positives tests. `tests/test_features.py:626-637` asserts the `open` stamp and must change;
   `test_unknown_licence_fails_loudly` must keep `cc_by` as its unknown value.
2. **Adapter:** `ingest/chemlex.py:58` and the `Literal` and docstring in `ingest/record.py:53,79`.
3. **Routing:** rename to `PRIVATE_SOURCES` in `io.py:24`, `positives.py:108`,
   `map_missing.py:52,54` and the test; add `"chemlex"`.
4. **Benchmark outputs:** the licence column and the `private/` path in `benchmark.py:102` and
   `benchmark_split.py:54`, and the usage lines in `positives.py:9`, `features.py:9`,
   `unlabeled.py:9` and `score_baselines.py:9`.
5. **Downstream stamps:** the four places in item 5.
6. **Docs:** CLAUDE.md §9, `docs/design/rebuild.md:24,66`, `docs/design/modeling.md:32,57,60,424`.
7. **Regenerate:** delete `derived/chemlex*`, `derived_samples/chemlex`,
   `derived_samples/benchmarks/chemlex` and the orphan `derived/chemlex_labels/` (whose
   `per_condition.csv` and `summary.txt` no current script writes), then re-run stages 1 to 6, the
   labels, the split, features and the scores. Nothing rewrites a licence in place, so the existing
   files keep their old label until they are deleted.

## Consequences

- The validation and test tables move under `private/`, so the Chemlex-derived evaluation data
  leaves the open paths. Every command in the design doc that names them changes.
- Stage F and stage S1 must be re-run after step 7. Both are deterministic from a fixed seed, so the
  numbers in ADR 008 and `docs/design/modeling.md` should reproduce exactly; if any of them moves,
  that is a finding and goes in the journal.
- A merged table now carries the most restrictive licence of its inputs (item 5), which is a rule
  this project did not have before.
- The paper is unaffected: ADR 005 already requires it to rebuild from open sources alone, and
  Chemlex is validation, never training data.

## Alternatives rejected

- **Label Chemlex `internal`.** Rejected by item 1: it is safe but says something false, and the
  distinction is one we have to explain in print.
- **Leave `open` and record the gap.** Rejected because the guard reads the label, so a wrong label
  disarms it for every path, present and future.
- **Move the benchmark tables under `private/` without a licence column.** Rejected by item 3: it is
  the same path-convention protection that hole 1 defeated this morning.
- **Purge and re-map the open cache.** Rejected by item 4: no row-level source makes selective
  cleaning impossible, and nothing published depends on the cache.
