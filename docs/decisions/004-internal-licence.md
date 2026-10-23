# 004 — Inventory-derived tables carry licence `internal` and are guarded like licensed ones

- **Status:** Accepted, 2026-10-20
- **Supersedes:** — · **Superseded by:** —

## Context

CLAUDE.md §9: the group's inventory is internal; tables derived from it live only under `private/` and
are never published. `io.write_table` is the one place that enforces where data may be written, but
it only checked a table's `licence` column for `"licensed"` (Pistachio). The lab library
(`library/inventory.py`) and the unlabeled space U (`library/unlabeled.py`) had no `licence` column,
so they stayed private only because `scripts/library.py` and `scripts/unlabeled.py` build their
paths under `private/`. One wrong path in a script would have written internal group data where it
may be shared, and nothing would have stopped it.

## Decision

1. A table's `licence` column takes one of three values: `open`; `licensed` (Pistachio and everything
   derived from it); `internal` (the group's inventory and everything derived from it).
2. `io.PRIVATE_LICENCES = {"licensed", "internal"}`. `write_table` refuses a table with any such row
   outside a `private/` folder below the output root, as it did for `licensed`.
3. `library_table` writes `licence = "internal"` on every compound and every rejected container;
   `unlabeled_pairs` writes it on every pair. The constant lives in `library/inventory.py`.
4. A table that mixes rows inherits the guard from any restricted row, because `write_table` checks
   every row. Model scores and plate picks built from U keep the column.

## Alternatives considered

- **A path helper in `io` (`private_dir`)** that every script must call: still a convention, the
  same weakness as before.
- **A separate `internal` flag column:** two mechanisms for one rule, and `write_table` would check
  both.

## Consequences

- Library and U tables gain a `licence` column. Re-run on 2026-10-20 (LAB_JOURNAL): otherwise
  identical, 529 acids, 538 amines, 284,595 pairs.
- Tests: `test_io.py` (internal refused outside `private/`, written under it), `test_library.py`,
  `test_unlabeled.py`.
- Still guarded by path only: the RXNMapper cache. Its rows carry no `licence`, and licensed
  reactions reach `private/cache/` only through `cache_dirs` and the routing in
  `scripts/map_missing.py`. Changing that changes the cache schema, so it waits until Pistachio is
  mapped.
