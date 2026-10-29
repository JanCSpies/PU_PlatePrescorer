# 020 — A stage S2 model must be able to represent an interaction between the two sides

- **Status:** Accepted, 2026-10-29
- **Supersedes:** — · **Superseded by:** —

Both decisions below were derived from one measurement made on 2026-10-29. Neither was a preference.

## Context

**The measurement.** A balanced pool of 3,000 positives against 3,000 unlabeled pairs. The same
classifier and the same 5-fold cross-validation over several feature sets. ROC-AUC:

| feature set | anchored U | unanchored control |
|---|---|---|
| frequency, both sides | 0.5640 | 0.5547 |
| size alone | 0.5011 | 0.5329 |
| the nine descriptors | 0.5843 | 0.6080 |
| `morgan_centre`, both sides | 0.4896 | 0.4915 |
| `morgan`, both sides | 0.4780 | 0.4616 |

Command: `uv run python scripts/counts/count_block_ladder.py ~/University/Phd/Dataset/Datasets`.

**Why the fingerprint rows sit at chance.** U is drawn from P's own substrates and the per-role
marginals are matched by design (ADR 013, ADR 015). Each side's fingerprint distribution is therefore
nearly the same in P and in U. A linear model over the two sides' features concatenated can only form a
weighted sum of per-side terms. It cannot represent an interaction. The only thing that separates P from
U here is the pairing. So that model is structurally unable to do the task, and 0.5 is its correct
answer. It is not a defect.

**The three instruments agree once this is seen.** They were never in conflict:

| instrument | reading on `morgan` | why |
|---|---|---|
| linear model on the marginals | 0.4780 anchored, 0.4616 control (`morgan_centre`: 0.4896, 0.4915) | cannot represent the interaction |
| boosted tree | 0.58 to 0.61 | sees the interaction, underfits 4,096 sparse columns at 120 iterations |
| nearest neighbour on Tanimoto, K = 5 | 0.6843 anchored, 0.6867 control | a pair-to-pair similarity is itself an interaction measure |

The linear row is in the ladder above. The 0.58 to 0.61 range comes from an earlier run of the same
ladder, before the estimator was corrected. It is not a current measurement and I do not cite it as one.
Command for the nearest neighbour: `uv run python scripts/counts/count_anchor_ablation.py ~/University/Phd/Dataset/Datasets`;
the method is in the tracked module `src/platepu/score/pair_knn.py`.

**The pipeline is sound, not broken.** Through the identical path (`block_matrix`, `make_model`,
cross-validated ROC-AUC), the literature-only against lab-only task on `morgan` reproduces the published
count script exactly:

| | this path | `scripts/counts/count_adversarial.py` |
|---|---|---|
| acids | 0.859 | 0.859 |
| amines | 0.879 | 0.879 |

Shuffled folds give 0.858 and 0.882. Binarising the counts changes nothing (0.4896 stays 0.4896).
Command: `uv run python scripts/counts/count_pipeline_control.py ~/University/Phd/Dataset/Datasets`.

**`score/axes.make_model` picks logistic regression above 100 columns.** That rule is right for the
substrate-level adversarial task it came from, where it reproduces 0.859 and 0.879. It is wrong for
pair-level data, where it selects a model that cannot work. The rule is not at fault. Applying it outside
its origin is.

## Decision

### 1. Interaction capability is an elimination criterion for the S2 model

A candidate for stage S2 that cannot represent an interaction between the acid side and the amine side is
eliminated before any score is compared. It is not ranked low; it is out. A linear model on the two sides
concatenated is out. A model that sees the pair as one object (a pair-level similarity, a tree or
network over both sides) is in. `make_model` is not used for pair-level data without this check.

### 2. A linear model on the marginals is a transferable check on the sampler

It reads about 0.5 because of how U is built, not because of anything in our data. So it costs nothing on
a reaction class with no HTE set. If it is not near 0.5 there, that class's U construction failed. I
record it as a sampler check of the same standing as the stage gate (gate values 0.5049 acid and 0.5095
amine for amide coupling, from the stage U run).

What it does **not** say: everything learnable now lives in the pairing, and pairing is chemistry and
publishing habit at once. Closing the marginal shortcut did not separate the two.

## Alternatives considered

- **Keep `make_model` as it is for S2.** Rejected. At pair level it picks the one model family that
  cannot work, and reads chance.
- **Read the 0.48 as evidence that U has no signal.** Rejected. The nearest neighbour reads 0.6843 on the
  same data.
- **Tune the boosted tree until it matches the nearest neighbour.** Not done. It would tune an instrument
  to agree with another, and the result would say nothing about the data.

## Consequences

- The nearest-neighbour figure of **0.6843** on `morgan` is the bar a fitted PU model has to clear. It
  needs no labels and it transfers to a class without an HTE set.
- The anchor ablation found nothing under the rule fixed before it ran: every readout a comparison and
  not a level, and a difference below 0.05 not a difference (journal entry of 2026-10-29). This does not
  prove the sampler is clean. It says these probes did not find a leak.
- Stage S2 design (`docs/design/modeling.md`) must list interaction capability among its criteria. That
  edit waits for the stage-architect's delta and my approval.

## References

- ADRs 012, 013, 015, 017, 018, 019.
- `docs/design/modeling.md`.
- `docs/literature/notes/chen-2019-hidden-bias-dude.md`.
