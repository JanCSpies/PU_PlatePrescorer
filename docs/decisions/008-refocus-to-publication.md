# ADR 008 — Refocus from hackathon deliverable to publication project

**Date:** 2026-07-29
**Status:** Accepted
**Decider:** Jan Christopher Spies (human)

## Context

The work in this repository began as a "Built with Claude: Life Sciences" hackathon entry
(Jul 7–13, 2026), targeting Buchwald–Hartwig C–N feasibility on the JnJ HTE 2024 (Neves) test set.
**The results were not submitted** and the hackathon is over.

What the week actually produced was more interesting than the deliverable it was aimed at. The JnJ
target turned out to be adversarially out-of-distribution by design (54.6 % of test rows use a base
that is <1 % frequent in the literature), and literature→JnJ transfer only worked *transductively* —
it collapsed to 0.40 ROC-AUC when test substrates were removed from U.

The **Chemlex amide** experiment is the result worth building on, because it holds **inductively**
(novel substrates) and clears a bar PAYN did not:

| ranking model | TEST ROC-AUC (inductive, n=2,234) |
|---|---|
| random | 0.500 |
| positives-only | 0.671 |
| **literature-PU (ours, zero target labels)** | **0.768** |
| in-domain supervised ceiling (real labels) | 0.733 |

## Decision

Refocus the repository as a **publication project targeting JACS or higher**, with the thesis:
*PU learning on real literature positives plus a virtual unlabeled substrate space raises HTE plate
hit rates, including for reaction classes with no existing HTE dataset.*

The paper's core experiment becomes a **prospective wet-lab HTE plate** — model-picked vs random
substrate selections under literature conditions — since retrospective public-dataset results, however
clean, are a benchmark rather than a demonstration.

The hackathon framing is retired: its rules, deadlines, submission format, and event logistics no
longer constrain anything.

## Consequences

**Deleted** (pure event logistics / superseded operational plans; recoverable from git history):
`rules.md`, `registration.md`, `SUBMISSION.md`, `AUTONOMOUS_RUN.md`, `BATTLEPLAN.md`, `JSP_WORKPLAN.md`.

**Rewritten:** `CLAUDE.md` (new charter), `README.md`.

**Retained and promoted:** `PROSPECTIVE_PLATE_PLAN.md` and
`PROSPECTIVE_REACTION_CONDITION_GROUPING.md` — written on 2026-07-11 as post-hackathon runway, these
are now the project's critical path rather than optional follow-ups.

**Retained unchanged:** `LAB_JOURNAL.md` (append-only; the lab notebook a paper's SI is reconstructed
from), `docs/decisions/001`–`007`, `docs/literature/notes/`, `RESULTS.md`, `Future_datasets.md`.

**Carried forward from the hackathon-era decisions:** everything in ADRs 001–007 still holds. In
particular the anti-leakage protocol (substrate-disjoint splits, test touched once, no tuning on test)
becomes *more* important, not less — it is what makes the numbers defensible under peer review.

**Deliberately not claimed:** generality. Chemlex is one reaction class; HiTEA BH reached only 0.709
with wide CIs, and the beyond-SCAR `prop_ips` variant that won on amide (+0.024) lost on BH (−0.033).
Establishing generality is the purpose of the prospective work, not an assumption behind it.
