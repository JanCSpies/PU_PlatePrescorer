# 018 — The feature block is chosen on label-free axes, fixed before any selection number is measured

- **Status:** Accepted, 2026-10-28, **before any selection number was measured**
- **Supersedes:** 012 in part (item 1, the reactivity axis measured on Chemlex) and 014 in part (item 2,
  `morgan_centre` as primary regardless of the selection) · **Superseded by:** 019 in part (items 3 and 4)
  · 023 in part (item 3, the fold rule) · 030 in part (item 6)

ADRs 008, 012, 014 and 017 are Accepted and are not rewritten (CLAUDE.md §8). This record replaces how
the block is chosen. It leaves intact ADR 008 item 1 (no hand-written chemistry in the features), ADR
012 item 2 (the floor is size, not chance, and a gap under 0.04 is not a gap) and ADR 017 (a probe is a
diagnostic, not a gate). It applies the pattern of ADR 017 to the choice of block: a block chosen after
the curve is seen is a hyperparameter search with a better name.

## Context

**The goal changed.** The method has to work for reaction classes with no HTE dataset, and be validated
by a prospective plate. ADR 012 judges a block on two axes, provenance and reactivity, but its
reactivity axis is measured on Chemlex. For a new class there is no Chemlex. A block chosen that way is
chosen with an instrument that will not exist where the method has to work. On 2026-10-28 I decided that
the selection uses label-free axes only. Chemlex is read afterwards, solely to report whether the
label-free criterion would have agreed.

**The incumbent block is far smaller than its name suggests.** `morgan_centre` passes `fromAtoms=[root]`,
so it yields exactly `radius + 1` environments. ADR 014 makes it the primary block, and nothing in the
design says how little it carries.

Every number in this section is measured on the stage F table rebuilt at the 20 % yield
cut-off, 11,345 substrates (journal, 2026-10-29). The first run of these three scripts
predated that rebuild and reported the same three shapes on 22,050 substrates; it is
superseded by the figures below, which are what the published commands print today.

    uv run python scripts/counts/count_bit_frequency.py ~/University/Phd/Dataset/Datasets

| block | bits set per molecule | distinct fingerprints among 11,345 substrates |
|---|---|---|
| `morgan_centre` | 3.0 | 615 |
| `morgan` | 34.2 | 10,943 |

**Two candidate families were measured and rejected.** Both tried to sit between "identifies the
molecule" and "describes shared chemistry" by compressing the whole-molecule fingerprint. Both failed
for the same reason: identity lives in the combination of set bits, and compression preserves
combinations until it destroys all of them at once.

*A frequency floor on `morgan`* (keep only bits set in at least `m` substrates):

    uv run python scripts/counts/count_bit_frequency.py ~/University/Phd/Dataset/Datasets

| floor m | bits kept | distinct fingerprints |
|---|---|---|
| 1 | 2,048 | 10,943 |
| 100 | 636 | 10,937 |
| 1,000 | 61 | 9,334 |
| 2,500 | 20 | 3,361 |

`morgan` is folded to 2,048 bits and every one is used. The median bit occurs in 64 substrates. Rare
bits are not where identity sits.

*Re-folding `morgan` to fewer bits.* This is exact, because RDKit folds as `hash % n_bits` and the widths
divide 2,048.

    uv run python scripts/counts/count_fold_width.py ~/University/Phd/Dataset/Datasets

| width | distinct fingerprints | share unique |
|---|---|---|
| 2,048 | 10,943 | 96.5 % |
| 64 | 10,911 | 96.2 % |
| 32 | 10,697 | 94.3 % |
| 16 | 1,839 | 16.2 % |
| 8 | 62 | 0.5 % |

This is a cliff, not a slope. The transition sits where the width falls below the number of bits a
molecule sets.

**The family that does work is locality.** Let `d` be the number of bonds from the reacting atom within
which an atom may contribute its own radius-2 environments. `d = 0` is today's `morgan_centre`; `d = all`
is today's `morgan`.

    uv run python scripts/counts/count_centre_distance.py ~/University/Phd/Dataset/Datasets

| d | mean roots | bits/molecule | distinct fingerprints | share unique |
|---|---|---|---|---|
| 0 | 1.0 | 3.0 | 615 | 5.4 % |
| 1 | 3.2 | 7.5 | 2,613 | 23.0 % |
| 2 | 5.1 | 12.0 | 5,229 | 46.1 % |
| 3 | 7.4 | 16.7 | 7,474 | 65.9 % |
| 4 | 9.6 | 20.7 | 8,862 | 78.1 % |
| 6 | 13.2 | 26.8 | 10,263 | 90.5 % |
| all | 18.6 | 34.2 | 10,943 | 96.5 % |

Every step is a real intermediate. The trade-off cannot be tuned with resolution, only with what the
descriptor looks at. Radius is not the lever either: an external ablation finds radius 2 best and radius
3 adding only conserved scaffold fragments and hash collisions (Mirji et al., arXiv 2609.22167, random
splits only; I use it as corroboration, and the measurement above is the basis).

## Decision

### 1. Candidates, fixed now

`d` in {0, 1, 2, 3, 4, 6, all}, radius 2 and 2,048 bits throughout, so only the root set varies.
Reference points are reported beside them and are never eligible to win: the nine hand-written
descriptors, and size alone.

### 2. Provenance axis, label-free

Size-adjusted adversarial ROC-AUC, literature-only against lab-only substrates, per role, averaged over
roles, with a bootstrap 95 % interval: 5 repeats, 500 resamples, seed 20261028. Low is good. **The floor
is about 0.67, not 0.5**, because literature molecules really are bigger (ADR 012 item 2).

### 3. Information axis, label-free

ROC-AUC of ranking positives above U. Every positive is scored out of fold, over 5 folds grouped on the
acid flat key, so a substrate never crosses a fold boundary.

- **Primary: cross-fitted**, because the deliverable is the input to a fitted model.
- **Model-free, reported beside it.** ADR 012 shows the two rank `morgan_centre` and `morgan` in
  opposite order (0.711 against 0.632 model-free, 0.892 against 0.937 supervised). If they disagree here,
  the disagreement is reported as a finding. It is not resolved by taking the convenient one.

This axis is label-free and also resists the popularity bias, which is why it can stand in for Chemlex.
U is drawn from P's own substrates with matched marginals (ADR 013 item 1, ADR 015), so the same
molecules sit on both sides and a descriptor cannot win by memorising which molecule has been published.
Stage U's gate measures exactly that. It reads **0.5049** for acids (`gate.acid.rank_auc`) and **0.5095**
for amines, in `<root>/unlabeled/amide_coupling/unlabeled_literature_k10.report.json`, from the U rebuilt
on 2026-10-28 on the post-cut-off P of 11,396 pairs. ADR 017's 0.497 is the earlier classifier-based
one-side popularity probe, on the earlier U, before the gate became a rank-AUC. It is a different
instrument on a different build, not a competing measurement of the same thing. The descriptor has to
use the pairing to win.

    uv run python scripts/unlabeled_literature.py amide_coupling ~/University/Phd/Dataset/Datasets/derived \
        --config configs/models/unlabeled_literature_k10.yml

**Limitation.** U's marginals are matched to the full P. Holding positives out leaves the rest slightly
off U's match. Rebuilding U per fold would fix it and is not worth the cost. The size of the mismatch is
reported beside the result.

### 4. The rule, fixed before the numbers

Among candidates whose provenance interval overlaps the lowest candidate's, take the highest primary
information. Where those intervals overlap too, take the block with the **fewest distinct fingerprints**.
That is parsimony, fixed in advance. ADR 012 established that a gap under 0.04 on the provenance axis is
not a gap.

### 5. Chemlex is not read until the pick is made

Then only the validation split, and only to report whether the label-free criterion chose the same
block. Agreement is the evidence that the protocol transfers. Disagreement is a result about what the
label-free criterion misses. The test side is never opened.

### 6. Stage F is not touched until a `d` wins

The candidates are computed by `scripts/counts/count_centre_distance.py` from the SMILES already in
the feature table, so the whole protocol runs without a stage F change. A tracked column is added only for the block that is chosen.

## Alternatives rejected

- **Choose on Chemlex and report the label-free axes beside it**, as ADR 012 did. It leaves unanswered
  how a new class would choose, which is the claim the paper makes.
- **Widen the radius instead of the sphere.** Rejected on measurement: `fromAtoms=[root]` gives
  `radius + 1` bits, so radius 5 would give six.
- **The two compression families above.** Rejected on measurement. They are recorded because the
  negative result is the useful part.

## Consequences

- ADR 014's choice of `morgan_centre` as primary stands only if `d = 0` wins under item 4. Until the
  selection is run, "primary" in ADR 014 is provisional.
- `scripts/counts/count_adversarial.py` has no bootstrap and no size adjustment today, though ADR 012
  quotes both. They exist only in `scratch/count_block_two_axes.py`. The provenance axis has to be
  published properly before it can decide anything.
- **ADR 012's figures no longer reproduce, and that is now measured rather than feared.** On the stage F
  table rebuilt at the 20 % cut-off, `count_adversarial.py` prints `morgan` at 0.859 (acid) and 0.879
  (amine) where ADR 012 has 0.864 and 0.870, and the nine descriptors at 0.721 and 0.700 where ADR 012
  has 0.730 and 0.776. The amine descriptor figure is the one that bites: ADR 012 argues the descriptors
  carry more provenance on amines than the rooted fingerprint, and on the rebuilt table they carry less
  (0.700 against `morgan_centre`'s 0.733). ADR 012 is **not** edited. This is recorded in a further ADR
  once `score/axes.py` is wired into the published script, which is the same outstanding work as above.
- The selection numbers (provenance interval, cross-fitted and model-free information, per `d`) do not
  exist yet. This record contains none of them.

## References

- `docs/design/modeling.md` (stage U, stage F), `docs/design/rebuild.md`.
- ADRs 008, 012, 013, 014, 015, 017.
- Mirji et al., arXiv 2609.22167 (radius ablation, random splits only; corroboration, not the basis):
  `docs/literature/notes/mirji-2026-role-aware-morgan-fingerprints.md`.
