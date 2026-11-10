# 021 — Seven more count scripts are published, and no count script hard-codes a path

- **Status:** Accepted, 2026-10-29
- **Supersedes:** 011 in part (item 2) · **Superseded by:** —

ADR 011 is Accepted and is not rewritten (CLAUDE.md §8). This record replaces one sentence of its
item 2, "the rest stay in `scratch/`", and nothing else. Item 1 (the three published scripts), item 3
(the redirect table) and item 4 (the deletions) stand as written.

## Context

ADR 011 drew a line at "what publication depends on" and left the rest in `scratch/` as local
exploration records. It listed what that costs: a number from a local script has to appear as a number
in the record that uses it, because the command is not published.

On 2026-10-22 that looked like enough. It was not. On 2026-10-29 I was shown
that the journal entry of that day and ADRs 018 and 020 cite eleven commands that exist only on this
machine. I answered with one word, "objavi" (publish), and the seven scripts moved into
`scripts/counts/`.

**This is a reversal of part of ADR 011, not a repeat of it.** The criterion in ADR 011 was right:
publish what the paper depends on. Its forecast of which scripts that would be was wrong. Within a week of
the record, the work behind these seven became load-bearing. ADR 018 is built on three of them and ADR
020 on three. The central claim of ADR 020, that a linear model on concatenated per-side features cannot
represent the pairing interaction, rests entirely on `count_block_ladder.py` and
`count_pipeline_control.py`. A reader cannot check an argument about an estimator from a number alone.
ADR 011's own consequence section named this failure: "A number from a local script must appear **as a
number** in the record that uses it." Quoting the number is not enough when the argument is about how
the number was produced.

The seven:

| script | what it produces |
|---|---|
| `count_baseline_aucs.py` | the validation ROC-AUC of every baseline at the 20 % cut-off: B4 0.7426, B5 0.8104, B6 0.8221, B7 0.5982, B8 0.7221, B2_all 0.6230, B3_all 0.5599, B1 0.5206 |
| `count_bit_frequency.py` | ADR 018's frequency-floor table: `morgan_centre` 615 distinct fingerprints against 10,943 for `morgan` |
| `count_fold_width.py` | ADR 018's re-folding cliff: 10,943 distinct at 2,048 bits, 62 at 8 bits |
| `count_centre_distance.py` | ADR 018's locality ladder, the candidate `d` values the block selection chooses between |
| `count_anchor_ablation.py` | the anchor ablation's nearest-neighbour probe: 0.6843 anchored against 0.6867 control (ADR 020) |
| `count_block_ladder.py` | ADR 020's feature-block ladder, the measurement that record is built on |
| `count_pipeline_control.py` | the known-answer control that reproduced 0.859 (acids) and 0.879 (amines), which showed the pipeline is sound and not broken |

The commands that regenerate them:

    uv run python scripts/counts/count_bit_frequency.py ~/University/Phd/Dataset/Datasets
    uv run python scripts/counts/count_fold_width.py ~/University/Phd/Dataset/Datasets
    uv run python scripts/counts/count_centre_distance.py ~/University/Phd/Dataset/Datasets
    uv run python scripts/counts/count_anchor_ablation.py ~/University/Phd/Dataset/Datasets
    uv run python scripts/counts/count_block_ladder.py ~/University/Phd/Dataset/Datasets
    uv run python scripts/counts/count_pipeline_control.py ~/University/Phd/Dataset/Datasets
    uv run python scripts/counts/count_baseline_aucs.py ~/University/Phd/Dataset/Datasets ~/University/Phd/Dataset/Datasets/derived_samples/private/benchmarks/chemlex/HATU_DIPEA_DMF/validation.parquet

**Publishing exposed a violation that `scratch/` had hidden.** All seven carried a module-level
`ROOT = os.path.expanduser("~/University/Phd/Dataset/Datasets/derived")`. That is one person's home
directory, about to be committed to a repository that ADR 005 says must rebuild from open sources alone,
and it breaks CLAUDE.md §3 ("paths come from config or arguments, never hard-coded"). data-guardian
blocked the commit on it. The three scripts ADR 011 published had a milder form of the same fault:
`sys.argv[1] if len(sys.argv) > 1 else "~/University/..."`. A fallback is still a hard-coded path.

## Decision

### 1. The seven are published; ADR 011 item 2 is superseded in part

The part superseded is the blanket sentence "The rest stay in `scratch/`" and the description of them
all as "local exploration records, not published code". The two scripts ADR 011 named and left out for a
stated reason are still out (item 3 below).

### 2. Every count script takes the dataset root as an argument and hard-codes no path

This is now the rule for `scripts/counts/`, so the next script is written that way and not fixed
afterwards.

- Each of the ten scripts that read data (`count_adversarial.py`, `count_b2_vs_b3.py`,
  `count_u_probes.py` and the seven above) reads the Datasets root from `sys.argv[1]` and exits with a
  `usage:` line if it is missing. The fallback default is removed from the three that ADR 011 published.
  `build_chemlex_like_fixtures.py` takes its own argument and is unchanged.
- No path to a home directory appears anywhere in `scripts/counts/`.

### 3. Chemlex: `count_baseline_aucs.py` takes the validation file by name, and refuses any other

It takes the validation parquet as `sys.argv[2]` and exits unless the file is called
`validation.parquet`. This replaces `assert "test" not in VALID`, which was a substring test on the whole
path and would have disappeared under `python -O`. The script reads only `key_a_stereo`, `key_b_stereo`
and `label`, from the **validation** split. It prints aggregates only, and nothing it prints can be
inverted to a Chemlex row (CC BY-NC-ND 4.0, ADR 009). The test side is never opened.

### 4. Three scripts stay unpublished, for three different reasons

| script | status | reason |
|---|---|---|
| `count_within_amine_and_widening.py` | out | single-letter names (`R`, `B`, `L`, `S`, `F`) against §5, as ADR 011 said. Publishing means rewriting it. |
| `count_subclass_evidence.py` | out | five `mypy` errors, one a real `list`/`ndarray` confusion, as ADR 011 said. Publishing means fixing it. |
| `count_block_two_axes.py` | **standing debt** | it holds the bootstrap and the size adjustment that ADR 018's consequences say must be published before the provenance axis can decide anything. |

The third is not a settled exclusion. It is work I owe, and ADR 018 already records it. Until it is
done, the provenance interval in ADR 012 and the selection in ADR 018 cite a command that runs only on
this machine.

## Consequences

- The command line of `count_adversarial.py`, `count_b2_vs_b3.py` and `count_u_probes.py` now
  **requires** an argument. The old journal entries that invoke them without one are stale in this new
  way as well, on top of the stale `scratch/` paths ADR 011 item 3 redirects (11 citations in
  `LAB_JOURNAL.md`, 2 in ADR 008). I do not edit them. ADR 011 item 3 settled that a redirect beats a
  rewrite, and this record is the second half of that redirect: where an old command shows no argument,
  pass the Datasets root.
- `scripts/counts/` now holds ten count scripts and the fixture builder, where ADR 011 left three files. CLAUDE.md §10 already lists the folder and needs no new line.
- The figures I quoted as numbers in ADR 011 item 2 stay in the text of ADR 011. Nothing about them
  changes.
- Each new script has to pass `ruff check`, `ruff format --check` and `mypy` like the first three. A
  script that cannot is not published and goes on the table in item 4.
- A published script is a promise. If one of these seven stops reproducing its number, that is recorded
  in a further ADR and the old figure is never edited.

## Alternatives rejected

- **Leave the seven in `scratch/` and quote their numbers in the text.** Rejected. ADR 020's argument is
  a measurement about an estimator: the same classifier, the same folds, the same features, a different
  answer. A reader cannot check that from a number. They need the code that made it.
- **Keep the home-directory fallback for convenience.** Rejected. It breaks §3 and the fresh-repository
  requirement of ADR 005, and a fallback that works on one machine hides the missing argument on every
  other. The cost of the rule is one argument on the command line.
- **Publish the seven and leave the three older fallbacks.** Rejected. Two patterns in one folder is how
  the next script copies the wrong one.
- **Edit the stale commands in the old journal entries.** Rejected by §8 and by ADR 011 item 3.

## References

- ADR 011 (the record this extends), ADR 005 (fresh repository), ADR 009 (Chemlex licence), ADR 018 and
  ADR 020 (the records built on the seven scripts).
- CLAUDE.md §3 (paths), §5 (names), §8 (append-only), §9 (Chemlex).
