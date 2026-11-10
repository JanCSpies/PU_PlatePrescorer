# 012 — A feature block is judged on two axes, and one earlier number does not reproduce

- **Status:** Accepted, 2026-10-23, except item 4, which is **Proposed**
- **Supersedes:** 008 in part (Finding 1, the provenance figures) · **Superseded by:** —

ADRs 008 and 009 are Accepted and are not rewritten (CLAUDE.md §8). This record corrects a number
they quote and replaces the way it should be read.

## Context

ADR 008 Finding 1 reports adversarial validation — can a classifier tell a literature substrate from
a lab one? — as **0.984** on the Morgan fingerprint against **0.559** on the ten hand-written
descriptors, and ADR 009 repeats it. That contrast carried a lot of weight: it is the argument for
building the unlabeled space from literature substrates (ADR 008 item 4) and for reporting provenance
beside every model (item 7).

**It does not reproduce.** Re-run on 2026-10-23 with
`uv run python scripts/counts/count_adversarial.py ~/University/Phd/Dataset/Datasets`:

| block | ADR 008 says | acid | amine |
|---|---|---|---|
| Morgan (ECFP4) | 0.984 | 0.864 | 0.870 |
| nine descriptors | 0.559 | 0.730 | 0.776 |

Two claims in the earlier record are wrong, not just imprecise. The fingerprint is not near-perfect
at 0.98, and the hand-written descriptors are **not** provenance-light: on amines they carry more
provenance than the rooted fingerprint does.

**Why it cannot be traced.** The feature table was rebuilt on 2026-10-22 under ADR 009 and the
earlier one was deleted; nothing was committed, so there is no version to compare against. Whether
the old figure came from a different table, a different definition, or a transcription error into the
ADR is unknown, and guessing would be worse than saying so. This is the third time in two days that
missing history has made a discrepancy unresolvable, and it is why the first commit followed.

## Decision

### 1. A feature block is judged on two axes, never one

Every block gets both numbers, always reported together:

- **Provenance (want low):** adversarial ROC-AUC, literature-only against lab-only substrates.
- **Reactivity (want high):** how well the block ranks the Chemlex validation reactions.

A block that reveals nothing about where a molecule came from is worthless if it also reveals nothing
about whether the reaction works. Every conclusion drawn before today rested on the first axis alone.

Measured 2026-10-23, `scratch/count_block_two_axes.py`, seed 20261022; 2,615 validation pairs with
762 successes, 28,158 class-wide positive pairs:

| block | provenance acid | provenance amine | reactivity, model-free | reactivity, supervised |
|---|---|---|---|---|
| size alone | 0.679 | 0.666 | 0.515 | 0.731 |
| nine descriptors | 0.707 | 0.724 | 0.696 | 0.862 |
| descriptors + size | 0.713 | 0.794 | 0.736 | 0.909 |
| **`morgan_centre`** | **0.707** | **0.752** | **0.711** | 0.892 |
| `morgan` (ECFP4) | 0.862 | 0.872 | 0.632 | 0.937 |

The reimplementation reproduces the real B2 (0.632) and B7 (0.598) exactly, which is what makes the
rest of the column trustworthy.

### 2. The floor is size, not chance, and provenance is reported adjusted for it

**Size alone separates the two sources at about 0.67.** Literature molecules genuinely are bigger, so
provenance is partly chemistry and 0.5 is not the reference point. Any descriptor correlated with
size inherits that much for free.

So provenance is also reported **size-adjusted**: the AUC computed within size quartiles and averaged.
Size alone scores 0.489 there, which is the sanity check that the adjustment works. The blocks score
0.709 / 0.726 (descriptors), 0.715 / 0.753 (`morgan_centre`) and 0.815 / 0.842 (ECFP4), so every
block carries provenance that size does not explain, and ECFP4 carries by far the most.

**The estimate is fragile and must be quoted with its interval.** The adversarial problem is 11,203
literature-only acids against **189** lab-only ones, and 9,623 against 202 for amines. Bootstrap
intervals are about ±0.04, so a gap under 0.04 is not a gap. Concretely: the differences among size,
the descriptors and `morgan_centre` on the acid side are **not** real. Only ECFP4 stands clear.

### 3. ECFP4 is dominated, and that is a result

On the model-free axis `morgan_centre` scores **0.711** against ECFP4's **0.632**, while carrying
**less** provenance on both roles. The whole-molecule fingerprint is beaten at finding analogues by
the same fingerprint rooted at the reacting atom, which is the first direct evidence for the
reasoning in ADR 008 item 2 — and independently matches what the condition-optimisation literature
reports, that features of the local environment of the reactive group beat bulk molecular properties.

ECFP4 wins one cell only: the supervised upper bound (0.937 against 0.892). That number is fitted on
the validation labels and is a **diagnostic, never a result**.

**No block sits in the useful corner.** The honest summary is that all of them are mediocre on both
axes, with a trade-off, and `morgan_centre` is the most defensible compromise rather than a win.

### 4. B2 is **not** redefined — **Proposed**: a new baseline instead

`morgan_centre` beats `morgan` at B2's own job. The tempting move is to swap B2's fingerprint. **That
is forbidden**, and the reason matters more than the baseline: B2 is named in ADR 008 item 6, which is
a pre-registration accepted before any model existed. Changing a pre-registered baseline's definition
*after* seeing that the alternative scores better is exactly the selective choice the pre-registration
exists to prevent.

If we want the rooted-fingerprint analogue score, it is a **new** baseline, declared now and before
stage S2 runs, not an edit to an old one. Proposed as **B8: nearest positive pair on `morgan_centre`,
0.711 on the validation set**, to be accepted or dropped before S2, never after.

## Consequences

- Wherever ADR 008's 0.984 / 0.559 appears — ADR 008 Findings and Alternatives, ADR 009 Context,
  `docs/design/modeling.md` and `LAB_JOURNAL.md` — the figures in item 1 above are the record. The
  older files keep their text (CLAUDE.md §8); this record is the correction.
- Reporting provenance beside every model (ADR 008 item 7) stands, with two changes: the number is
  quoted with its bootstrap interval, and the size-adjusted value is given beside it.
- `validation.parquet` carries `one_unseen == "test"` on 1,040 of its 2,615 rows. Our split uses
  `both_unseen` throughout and the pre-registration names it, so this is consistent — but it means
  the benchmark's *other* split is not available to us as a held-out set. Recorded so that nobody
  later reports a result on `one_unseen` believing it untouched.
- The supervised column of the table was fitted on the validation labels. It is a diagnostic and is
  never to be quoted as a model result; the single-use test side was not read.

## Alternatives rejected

- **Edit the 0.984 in ADR 008.** Rejected by §8, and the wrong number is itself part of the record.
- **Swap `morgan_centre` into B2.** Rejected by item 4.
- **Drop the fingerprint and model only the hand-written descriptors.** Rejected again, and now on
  evidence rather than principle: the descriptors carry *more* provenance than the rooted fingerprint
  on amines (0.724 against 0.752 is within the interval; size-adjusted 0.726 against 0.753 likewise),
  so the trade they offer is worse than it looked.
- **Adopt predicted DFT-level descriptors** (the released GNN library for carboxylic acids and alkyl
  amines). Rejected for now: it does not cover aryl amines, which are 1,238 of our 2,615 validation
  rows, and Haas et al. (PNAS 2022, 119, e2118451119) report that buried volumes, dipole moments and
  amine lone-pair descriptors gave "only marginal improvement or diminished performance" for exactly
  this reaction.
