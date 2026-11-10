# 016 — The spy step is cross-fitted, and the learner hides nothing

- **Status:** Accepted, 2026-10-23, **before stage S2 was written**
- **Supersedes:** 013 in part (item 3, "as PAYN used it", on the scoring of spies) ·
  **Superseded by:** —

ADR 013 is Accepted and is not rewritten (CLAUDE.md §8). This record settles three details of its
item 3 that it did not specify, one of which turns out to be a deviation from PAYN.

## Context

The spy technique sets its threshold from the scores of known positives hidden in U. PAYN fits once
and reads those scores **in sample** — the spies are in the training set, labelled as unlabeled.

A flexible learner can partly memorise that label. The spy's score is then not the score a *fresh*
positive would get, and the threshold is calibrated on contaminated numbers.

**The direction of the contamination is not known, and that is the argument.** Spy scores are pushed
down by their own training label; but every U row is in the training set with the same label and is
pushed down too, so what happens to the overlap of the two distributions is not obvious. The spies are
5,632 rows among roughly 287,000, and a small hard class is not memorised less than a large easy one.
An earlier draft of this reasoning claimed the threshold comes out too low and therefore admits *too
many* reliable negatives; on checking, a lower threshold admits **fewer**. Neither claim can be made
without measuring it.

So the question is not which way the bias runs. It is that nobody can say, and a threshold whose bias
has no known sign and no known bound is a poor thing to build a training set on.

## Decision

### 1. Spy scores are taken **out of fold**, and in-sample is run beside it

**Primary: out of fold, 5 folds.** Each spy is scored by a model that did not train on it, so its own
label cannot move it. The question of direction disappears rather than being answered.

**Secondary: in sample, as PAYN does.** One more config file, the same code. The difference between
the two thresholds, and the resulting reliable-negative sets, is then a **measurement** rather than an
assumption — and it is one PAYN could not make, because its positives were carved synthetically out of
fully-labelled HTE sets where the question does not arise.

Cost: five fits instead of one, minutes at these sizes. ADR 003 already states that this project
**reuses** PAYN's method rather than reproducing its numbers, so the deviation is in scope; it is
recorded here rather than buried in code.

### 2. A plain P-against-U classifier runs as a secondary control

The spy classifier without a threshold: no spies, no reliable negatives, the raw P-vs-U score used
directly as a ranking. Nearly free — it is a step the primary pipeline already performs.

It answers the question a referee will certainly ask and we otherwise could not: **what did the spy
step buy?** Without it the spy result has nothing to be compared against, and "we used the spy
technique" would be a description rather than a justification.

Under ADR 014 item 1 this is a **secondary** method. It cannot become the headline, whatever it scores.

### 3. The learner's hidden validation split is turned off explicitly

`HistGradientBoostingClassifier` enables `early_stopping` by itself above 10,000 rows, and then carves
an internal validation split out of the training data using its own seed. Our training set is about
287,000 rows, so it would be on by default.

That is a tuning decision, taken by the library, on data we chose, with a seed we did not set — which
CLAUDE.md §5 calls hidden state. `early_stopping` is written out as `false` with a fixed `max_iter`,
and every learner parameter lives in the config file where the config hash records it.

`learner` is a recorded value and **not** a branch: any value but the implemented one raises. The day
a second learner is wanted, it arrives with its own measurement and its own record.

## Consequences

- Stage S2a gains a `fold` column and a `spy_scoring` config key with two values; the second value is
  a second config file, not a code path.
- The S2a report carries both thresholds and the size of both reliable-negative sets, so the cost of
  PAYN's choice is visible as a number.
- The result table gains one secondary row for the no-spy control.
- No new dependency: ADR 013 item 3's choice of scikit-learn stands.

## Alternatives rejected

- **In-sample only, for fidelity to PAYN.** Rejected: fidelity to a bias of unknown sign is not a
  virtue, and item 1 keeps the comparison anyway.
- **Out-of-fold only.** Rejected: it would leave the deviation from PAYN unmeasured, and that is the
  first thing a reader familiar with PAYN would ask about.
- **Leaving `early_stopping` at its default.** Rejected by §5. A default that changes behaviour
  depending on how many rows the data happens to have is exactly the kind of silent decision this
  project writes down.
