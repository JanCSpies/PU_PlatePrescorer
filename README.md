# PU_PlatePrescorer

**Pre-scoring HTE plates from literature data alone, via Positive-Unlabeled learning.**

Research project of **Jan Christopher Spies**, Glorius group, Organisch-Chemisches Institut,
University of Münster.

## The idea

Exploratory high-throughput experimentation (HTE) — and especially **direct-to-biology** screening —
spends plate capacity on wells that simply fail. A failed well costs as much as a successful one and
produces no testable compound.

This project asks whether reaction **feasibility** can be predicted *before* the plate is laid out,
using only what the literature already contains:

- **P (positives)** — real reactions mined from the literature (ORD / Pistachio / USPTO). Chemists
  publish what worked, so literature data is positive-heavy and *biased*, not labelled.
- **U (unlabeled)** — a **virtual** substrate matrix, enumerated in silico from real building-block
  stock. Never run, so never labelled.
- **PU learning** turns that (P, U) pair into a feasibility ranking over the matrix, which is exactly
  what plate design needs.

Crucially, this requires **no HTE dataset for the target reaction** — which is the situation you are
actually in when you want the prediction.

## Result so far

On held-out Chemlex amide couplings (2,234 reactions, both substrates unseen, **zero target labels
used in training**):

| ranking model | ROC-AUC |
|---|---|
| random | 0.500 |
| positives-only (similarity to known positives) | 0.671 |
| **literature-PU (this work)** | **0.768** |
| in-domain supervised ceiling (trained on real labels) | 0.733 |

Top 25 % of the ranked matrix: **61.8 % hit rate vs 35.5 % random — 1.74×**.

The literature-PU model **beats the supervised model trained on real in-domain labels**: 258k diverse
literature positives out-signal 4,960 in-domain rows once the test substrates are novel.

**Scope, honestly:** this is one reaction class. The same pipeline reached 0.709 on HiTEA
Buchwald–Hartwig (n = 36, wide confidence intervals), and the beyond-SCAR variant that won on amide
lost on BH. Generality is not yet established — which is what the prospective work below is for.

## What's next

A **prospective wet-lab HTE plate**: model-picked vs random substrate selections, run under
literature conditions, on a reaction class with no existing HTE dataset. Model and scores are frozen
and committed before any well is run. See `PROSPECTIVE_PLATE_PLAN.md`.

## Relation to PAYN

Follows on from **PAYN** — *"Positivity is All You Need": Yield Prediction of Organic Reactions in
Biased Data Sets via Positive-Unlabeled Learning*, J. Am. Chem. Soc. 2026, 148, 15066–15075
([10.1021/jacs.6c00127](https://doi.org/10.1021/jacs.6c00127)) — which established PU learning on
**synthetic** P and U carved from existing HTE datasets. Here both P and U are real: literature
positives and a virtual substrate space, with real, unmodelled reporting bias.

## Repository

```bash
uv sync --extra dev      # Python 3.13, managed with uv
uv run ruff check . && uv run mypy src && uv run pytest
```

| Path | |
|---|---|
| `CLAUDE.md` | project charter — read first |
| `LAB_JOURNAL.md` | append-only experiment log |
| `RESULTS.md` | consolidated retrospective results |
| `PROSPECTIVE_PLATE_PLAN.md` | the wet-lab validation experiment |
| `src/pubench/` | PU models, features, mining, evaluation |
| `src/bhpu/` | earlier pipeline (chem, leakage, combinatorial space) |
| `artifacts/` | experiment outputs — every reported number traces here |

Large and licensed data (Pistachio-derived corpora, raw HTE sets) are **not** in this repository.

## Provenance and licence

Split from a shared Glorius-group working tree; foundational commits are by **Florian Boser** and
remain attributed in `git log`. MIT licensed — see `LICENSE`. Note that the licence covers the code;
bundled third-party datasets carry their own terms.
