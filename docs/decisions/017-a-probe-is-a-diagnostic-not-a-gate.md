# 017 — A probe is a diagnostic, not a gate

- **Status:** Accepted, 2026-10-28, **before stage U was written**
- **Supersedes:** 013 in part (its Consequences, the popularity probe as a pass criterion) ·
  **Superseded by:** —

ADR 013 is Accepted and is not rewritten (CLAUDE.md §8). This record corrects a pass criterion it
set, which measurement shows cannot be met by any correct sampler.

## Context

ADR 013 Consequences says: *"a classifier given only each side's popularity must score about 0.5 on P
against U. If it scores higher, the marginals are not matched and item 1 has failed."*

Built as designed — size-stratified, one side at a time, P subtracted on flat keys, k = 10 — the real
run gives:

| probe sees | AUC |
|---|---|
| **one** side's literature frequency (the marginal) | **0.497** |
| **both** sides' frequencies (the joint) | **0.597** |
| both sides' heavy-atom counts (the joint, size) | 0.536 |

**The marginal match is exact. Every signal the probe finds is in the joint**, and ADR 013's criterion
conflated the two: it named the marginals and then tested with a two-feature classifier.

## Decision

### 1. The criterion is split, and only the marginal one is a gate

- **Gate:** a probe seeing **one** side's frequency must score about 0.5, and the per-substrate U
  count must match k × its P count within a tolerance fixed before the run. This is what ADR 013
  item 1 actually promised, and it holds (0.497).
- **Diagnostics, reported and never pass/fail:** the joint popularity probe, the joint size probe, and
  the shuffled-P probe. Their values go in the report beside the model's own score.

### 2. Why no sampler can pass the old criterion

**A U whose joint structure matched P's in every respect would be P.** Something must differ, and any
difference is visible to a classifier given enough features. Size stratification closed the size axis
(0.640 → 0.510 on that probe) and left the degree axis visible; stratifying degree too would leave a
third. Each axis added also removes **real** chemistry — degree is bound up with amine class, since
the most-reported amines are small aliphatic primary ones — so the chase costs signal at every step
and never ends.

The residual is also partly caused by a step that cannot be dropped: subtracting known positives
takes proportionally more draws from popular substrates, because they have more known partners. Top-30
acids lose **5.8 %** of their draws and top-30 amines **9.1 %**, against 0.6–1.2 % for substrates with
a single partner. That deficit is a property of removing P, not a flaw in the draw.

### 3. What actually protects the result, and the one place it does not

**The evaluation does the filtering.** Chemlex is scored, not U, and on Chemlex popularity (B1) is
**0.524** and size (B7) is **0.598**. A model that learned nothing but these shortcuts scores about
0.5–0.6 there, not 0.822. So a shortcut that survives in P-against-U does not survive into the
pre-registered comparison.

**The exception is the spy step, and it is the reason the diagnostics are reported at all.** The
P-against-U classifier selects the reliable negatives. If it ranks partly by joint structure, the
negatives it keeps are biased, that bias enters the final model's training set, and **the evaluation
cannot filter it**, because it is baked into what the model was taught rather than into how it is
scored. So the S2a report carries the shift the selection introduces — the reliable negatives'
marginals, mean size and single-partner share against U's — and that shift is read beside every
result.

## Consequences

- `docs/design/modeling.md`'s stage U verification separates the gate from the diagnostics.
- The U report carries all three diagnostic values, so a reader sees what was exploitable.
- Nothing about the sampler changes: it was correct and the criterion was not.

## Alternatives rejected

- **Stratify by degree as well**, or **build a second U for the spy step alone.** Both were weighed
  against item 2: the first costs real chemistry and only moves the problem, the second doubles the
  tables and asks a reader to hold two U's in mind for a residual the evaluation already filters
  everywhere but one step.
- **Keep the old criterion and widen its tolerance until it passes.** Rejected outright. A gate tuned
  until the thing it guards gets through is not a gate, and writing one would be worse than having
  none.
