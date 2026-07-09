# pubench — rundown (read this first)

*Overnight autonomous work by Jan's session (Claude), 2026-07-08. This is the entry point; details
in `PACKAGE.md` (what the package is), `EXPERIMENTS.md` (every run + result), `FRONTIER_METHODS.md`
(deferred methods + why).*

## TL;DR
- Built **`pubench`** (`src/pubench/`): a config-driven, registry-based PU-learning **benchmark** for
  BH feasibility classification, trained only on literature positives (P) + unlabeled pool (U),
  evaluated on the held-out JnJ HTE 2024 plate. 32 source files, mypy-strict, **24 tests**, all green.
- **6 PU methods** implemented (supervised, spy, nnPU, IW-PU, bagging-PU, Elkan–Noto), **3 featurizers**
  (ECFP subs-only, ECFP subs+cond, DRFP), density-ratio (uLSIF/RuLSIF) + prior utilities, a metric
  suite, and a YAML runner with provenance. Heavy runs go to SLURM (`scripts_JSP/pubench_run.sbatch`).
- **Headline result:** best **test ROC-AUC ≈ 0.68–0.71** — an **L2-regularised HistGradientBoosting**
  P-vs-U classifier on **substrate-only ECFP** (0.711 best config; 0.678 strictly val-selected),
  up from **0.594** for the naive CatBoost baseline. The gain is a **base-learner + regularisation**
  effect (HistGB curbs overfitting to the domain shortcut), not PU-specific machinery. **Conditions
  as fingerprints and DRFP both transfer worse** than substrate ECFP.
- **Why (the key finding, E-DIAG):** the **P-vs-U domain-discriminator AUC = 1.000** — literature
  positives and the JnJ target space are **non-overlapping in substrate feature space**. Transfer is
  capped because the model must extrapolate across (near) disjoint supports; PU/importance-weighting
  is structurally handicapped. **The lever is a shift-invariant physical *descriptor* representation,
  not the algorithm** — but Florian's g-xTB/morfeus descriptors cover 100% of test and only ~1% of the
  merged-P (Pistachio) substrates, so they can't train on P until computed for the P vocab (a GPU job).
- **No data leakage** (audited): P shares no substrate/pair with val/test; thresholds/prior from val
  only; test scored once. Transductive U (reuses JnJ substrate vocab, unlabeled) is the only overlap.

## Results so far (test ROC-AUC)
| experiment | best test ROC-AUC | notes |
|---|---|---|
| E1 (Phase 1) | supervised subs-only **0.594** | nnPU 0.505; conditions crash to 0.32 |
| E2 (Phase 2 batch 1) | supervised subs-only **0.594** | spy 0.566; nnPU/IW-PU 0.389 (linear + degenerate ratio) |
| E-DIAG | — | P-vs-U discriminator AUC **1.000** (both featurizers) = the domain gap |
| E4 (models × ECFP/DRFP) | bagging/elkan (HistGB) subs-only **0.689** | +0.10 vs CatBoost; DRFP 0.55, conditions 0.32–0.41 |
| E5 (base learner + reg) | **0.711** (HistGB + L2) / 0.678 val-selected | regularisation aids transfer; bagging didn't beat single HistGB |
| E3 | cancelled (redundant with E2+E4) | 10k-dim bagging too slow; superseded |

## What each artifact is
- `configs/pu_bench.*.yaml` — experiment configs (example, phase2, e3, e4).
- `artifacts/pubench/<exp>/comparison_table.{parquet,csv}` + `provenance.json` — results per run.
- `scripts_JSP/pubench_run.sbatch` — SLURM launcher (`sbatch [--partition=CPU] … <config>`).

## Recommendations (in priority order)
1. **Compute g-xTB/morfeus descriptors for the P (Pistachio) substrate vocabulary** and add a
   descriptor featurizer arm — the domain gap (AUC 1.0) says this is the only thing that can lift the
   ceiling. Biggest expected win; needs GPU (Florian's `gpuma`).
2. Condition **descriptors** (ligand %Vbur/cone-angle, base pKa, solvent polarity) via the canonicaliser
   keys, replacing the condition *fingerprints* that currently crash transfer.
3. Implement DETM (goodness-of-fit + double tilt) and the exact Kumagai ShiftAware-IW estimator once
   descriptors give some support overlap (with disjoint supports, even correct versions are limited).
4. Housekeeping: `bagging_pu` on 10k-dim ECFP is slow (HistGB bins every feature) — prefer subs-only
   (4k-dim) or add feature reduction; consider an inductive U variant (val-side substrates only).

## Reproduce
```bash
uv run pytest tests/pubench -q                                   # 24 tests
uv run python -m pubench.cli --config configs/pu_bench.phase2.yaml   # or sbatch for heavy runs
```
