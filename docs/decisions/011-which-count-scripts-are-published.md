# 011 — Which count scripts are published, and what the others are

- **Status:** Accepted, 2026-10-22
- **Supersedes:** — · **Superseded by:** —

## Context

Every number in the ADRs and in `LAB_JOURNAL.md` carries the command that regenerates it. That rule
(CLAUDE.md §8) is what makes a number a measurement rather than a claim.

A citation map built on 2026-10-22 found that **15 of the 19 files in `scratch/` are cited** that way,
and that `scratch/` is git-ignored. So every one of those commands is missing from a fresh clone. ADR
005 requires the published work to rebuild from open sources alone, and the sharpest case is sharper
still: CLAUDE.md §9 promises that for Chemlex "we publish only the code that rebuilds them from the
original file", and that builder sat in the ignored folder, so the promise could not be kept.

Three options were weighed: move all 15 into the repository, move only the ones publication depends
on, or leave them all local and correct the promise. Luka chose the middle one, because several of
the 15 were genuinely one-off and cleaning them to §5 would cost more than their answers are worth.

## Decision

### 1. Three scripts move into `scripts/counts/` and are published

| script | why it must be publishable |
|---|---|
| `build_chemlex_like_fixtures.py` | CLAUDE.md §9. It rebuilds `tests/data/chemlex_like_6rows.parquet` and the Chemlex rows of `mapped_12rows.parquet` from hand-written rows, which is the evidence that no CC BY-NC-ND row is in the repository. |
| `count_adversarial.py` | Produces the provenance result, 0.984 on the Morgan fingerprint against 0.559 on the descriptors (ADR 008 Finding 1). It may become a contribution of the paper in its own right. |
| `count_b2_vs_b3.py` | Holds Owen's pigeonhole bootstrap, which is the paper's **evaluation method**, and the 0.049 that the pre-registered success rule rests on (ADR 008 item 6). |

`scripts/counts/` is a new folder; CLAUDE.md §10 is updated in the same commit. All three pass
`ruff check`, `ruff format --check` and `mypy src scripts`; `sklearn.*` was added to the mypy
overrides in `pyproject.toml` for the adversarial script.

### 2. The rest stay in `scratch/`, and this record says so plainly

They are **local exploration records, not published code.** An ADR or journal entry that cites one is
citing a command that only runs on the machine it was written on. That is now stated rather than
implied, which is the whole point of this record: the previous state promised reproducibility it
could not deliver.

Two were considered for promotion and left out for a stated reason, so the reason is on the page
rather than in somebody's memory:

- `count_within_amine_and_widening.py` — produces the 70 usable amines and the 94/112 widening
  counts behind ADR 008 item 6. It uses single-letter names (`R`, `B`, `L`, `S`, `F`) throughout,
  against CLAUDE.md §5, so promoting it means rewriting it.
- `count_subclass_evidence.py` — the surviving record of the nine retired acid subclasses (ADR 008
  item 2). It fails `mypy` with five errors, one of them a real `list`/`ndarray` type confusion, so
  promoting it means fixing it.

**Where a local script produced a number the paper will use, the number is recorded in the ADR text
itself**, not only as a command. The record then survives the script. The figures that matter:

- 70 of 112 validation amines are usable for the within-amine test, 70.5 % of rows; 41 have ≥3
  positives and ≥3 negatives (ADR 008 item 6).
- Widening to all condition families leaves 94 acid and 112 amine flat keys, unchanged (ADR 008
  item 6).
- Validation pairs average 38.66 heavy atoms against 23.73 for inventory pairs (ADR 008, ADR 009).
- Nine acid subclasses, 99.9 % purity over 11,787 acids; alpha-quaternary succeeded at 0.136 against
  0.470 for heteroaryl pi-donors (ADR 008 item 2).

### 3. Paths cited before this record are redirected here, not rewritten

ADRs 001–010 are Accepted and `LAB_JOURNAL.md` is append-only (CLAUDE.md §8), so the paths inside
them are **not** edited. This table is the redirect:

| cited as | now at |
|---|---|
| `scratch/build_chemlex_like_fixtures.py` | `scripts/counts/build_chemlex_like_fixtures.py` |
| `scratch/count_adversarial.py` | `scripts/counts/count_adversarial.py` |
| `scratch/count_b2_vs_b3.py` | `scripts/counts/count_b2_vs_b3.py` |

Two citations in ADR 008 (lines 17–18) and eleven in `LAB_JOURNAL.md` point at the old paths and stay
that way. `docs/design/modeling.md` and `.claude/agents/chem-verifier.md` are living documents and
were updated.

This is how a rename works in an append-only system: you publish a redirect, you do not edit the
history. Rewriting those paths would have been the same mistake that was caught and reverted earlier
today, when a style preference was swept backwards through two 2026-10-02 journal entries.

### 4. What was deleted

`scratch/handcheck.py`, `scratch/handcheck2.py` (its one finding written into the journal first, as
the three diaryl amines and their counts, because deleting the producer means the record has to hold
the result and not a pointer), `scratch/export_explainer_data.py` (its page no longer exists),
`scratch/claude_code_plan_steps_1-3.md` (superseded by `docs/design/rebuild.md`),
`tests/data/uspto_4rows.parquet` (no test loads it) and `graphify-out/` (3.7 MB in 119 files, stale
since 2026-10-09, git-ignored, regenerable). Three published claude.ai pages were deleted too: all
three carried numbers that 2026-10-22 superseded and one pre-empted ADR 008 item 6.

## Consequences

- `scripts/` now holds a `counts/` subfolder, so CLAUDE.md §10 gains it.
- A number from a local script must appear **as a number** in the record that uses it. A bare command
  is no longer sufficient when the command is not published.
- Three figures produced on 2026-10-22 have **no script at all**: the 0-of-6 Chemlex fixture join,
  the 44 failing tests before the ADR 009 code, and the citation map itself. They stand on the
  journal entry alone. The first is the one worth fixing, because a licence argument rests on it;
  it should become a test.
- If `count_within_amine_and_widening.py` is ever needed by a referee, the pre-registration's basis
  is unpublished. That is a known weakness of this decision, not an oversight.

## Alternatives rejected

- **Move all 15.** Rejected as disproportionate: several were one-off dead ends, and §5 applies to
  everything in `scripts/`, so each would need rewriting.
- **Leave all 15 local and say nothing.** Rejected: §9's promise about Chemlex would stay unmet, and
  the ADRs would keep citing commands that do not exist.
- **Edit the old paths in ADR 008 and the journal.** Rejected by §8. A redirect costs one table; a
  rewritten record costs the ability to trust any record.
