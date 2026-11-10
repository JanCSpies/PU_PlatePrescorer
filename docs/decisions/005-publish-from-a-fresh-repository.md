# 005 — The paper's code is published from a fresh repository; the prototype history stays private

- **Status:** Accepted, 2026-10-20
- **Supersedes:** — · **Superseded by:** —

## Context

`origin/main` (the archived prototype line, tag `v0-prototype`) still tracks 35 data files: 24
tables and 11 feature arrays under `artifacts/`, `datasets/processed/` and `final_datasets/`. Counted
on 2026-10-20 from their `Source` columns:
- ORD;
- HiTEA 2024;
- Merck 2025;
- Roche 2023;
- AZ ELN 2023;
- Dreher 2016;
- JNJ HTE 2024.

No row comes from Pistachio, Chemlex or the group inventory. Commit `016db58` added them as
"assembled open datasets"; the licence of each source has not been checked.

The repository is private. Merging PR #1 removes the files from `main`'s tree, not from its
history. Removing them from history (`git filter-repo`) needs a force push by the owner, breaks
every clone, and GitHub keeps old PR refs until support removes them.

## Decision

1. The history is not rewritten.
2. The code that goes with the paper is published from a new public repository. Its first commit is
   the rebuild tree, without the prototype history, after a data-guardian check of that tree.
3. The licence of each of the seven sources is recorded in `docs/literature/notes/` when the HTE sets
   enter the rebuild (`docs/design/rebuild.md`, step 10).
4. HiTEA rows (2,632) were in the prototype's training corpus (`literature_clean.parquet`). The
   rebuild reads no prototype table, so HiTEA stays a clean held-out set.

## Alternatives considered

- **Rewrite the history now:** costly for a private repository whose history never needs to be
  public under decision 2.
- **Make this repository public later, with its history:** would publish the prototype's data
  files and unverified results.

## Consequences

- Nothing changes on GitHub now.
- Before publication: create the public repository, push the rebuild tree as one first commit, and
  link it in the paper. CLAUDE.md §9 ("everything the paper ships must be reproducible from open
  sources alone") applies to that repository.
