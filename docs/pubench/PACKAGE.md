# `pubench` — PU-learning benchmark package (summary)

*Author: Jan's session (Claude). Location: `src/pubench/`. Installed via the `bhpu` wheel
(`pyproject.toml` `packages = ["src/bhpu", "src/pubench"]`). Python 3.13+, mypy-strict,
ruff (Google docstrings), pytest.*

## Purpose
Benchmark positive-unlabeled (PU) learning algorithms for Buchwald–Hartwig reaction **feasibility
classification** (works `y=1` vs not), trained **only on literature positives + an unlabeled pool**
and evaluated on the held-out, fully-labeled JnJ HTE 2024 plate — under a realistic setting where
SCAR is violated by reporting bias and distribution shift. Consumes the datasets assembled by
`scripts_JSP/assemble_datasets.py` into `final_datasets/`.

## The four splits (data contract — `pubench.data`)
| split | source file | label | role |
|---|---|---|---|
| **P** | `final_datasets/private/positives_merged_bh.parquet` (64,061) | all PU-positive | training positives (biased) |
| **U** | `final_datasets/unlabeled_grid.parquet` (52,736) | none | unlabeled pool (target-space) |
| **val** | `final_datasets/test.parquet` where `eval_split==val` (2,624) | `labels` 0/1 | model selection / thresholds |
| **test** | `final_datasets/test.parquet` where `eval_split==test` (1,920) | `labels` 0/1 | final eval only (never trained) |

`FinalDatasetsAdapter` loads them, projects to a canonical schema (substrates + `metal`/`ligand`/
`base`/`solvent` SMILES), and derives U's product SMILES. Target = binary `labels` (classification;
no yield/regression for now).

## Architecture (extension points, registry-based)
- **`registry.py`** — generic `Registry[T]`; featurizers/models/priors/density-ratios self-register
  by name via a decorator, so adding one touches a single file (runner unchanged).
- **`data/`** — `Split`, `SplitFrame`, `PUDataset`, `FinalDatasetsAdapter`, `schema.py` (the on-disk
  contract).
- **`features/`** — `Featurizer(ABC)` + `ECFPFeaturizer` (Morgan fps of substrates ⊕ conditions +
  metal one-hot; `use_conditions` toggle). *Pluggable* (descriptor/DRFP featurizers can be added.)
- **`models/`** — `PUModel(ABC)` with a `Requires` metadata block (`needs_prior`,
  `needs_propensity`, `assumes_fixed_negatives`, `handles_dd_shift`). Implemented:
  - `SupervisedBaseline` (CatBoost; treat U as negative — the naive lower bound).
  - `NonNegativePU` (Kiryo 2017 nnPU risk; self-contained numpy, **no torch**; float32 scorer).
  - `ImportanceWeightedPU` (density-ratio reweighting of the positives → target marginal; reuses
    the weighted nnPU risk; RuLSIF weights, clipped + self-normalised).
  - `SpyTwoStep` (Liu 2002 spy technique → reliable negatives → supervised; the PAYN baseline).
- **`priors/`** — `ClassPriorEstimator(ABC)`; `ElkanNotoPrior` (SCAR-based, Phase 1); `KM2`/`TIcE`
  stubs (Phase 2 — shift-robust).
- **`density_ratio/`** — `DensityRatioEstimator(ABC)`; `ULSIF`/`RuLSIF` (kernel LSIF, closed form);
  `KMM` stub. Used by the importance-weighting family.
- **`eval/`** — `metrics.py` (ROC-AUC, PR-AUC, balanced-acc, MCC, PAYN negative precision/recall,
  confusion) + `Evaluator` (val-selected thresholds: `default`/`val_youden`/`val_f1`).
- **`runner/`** — `ExperimentConfig` (YAML dataclasses), `Experiment` (builds the featurizer×model
  grid, writes `comparison_table.{parquet,csv}` + `provenance.json` [git hash + versions] +
  `resolved_config.yaml`), `provenance.py`. CLI: `python -m pubench.cli --config <yaml>`.

## How to run
```bash
uv run python -m pubench.cli --config configs/pu_bench.phase2.yaml   # heavy → prefer SLURM
uv run pytest tests/pubench -q                                       # 20 tests
```

## Key design choices / caveats
- **No test leakage**: models train on P + U only; val (labeled) is used for threshold/prior
  selection; test is scored once. The frozen substrate-disjoint val/test split comes from the
  dataset assembler (`FrozenEvalProtocol`).
- **Prior**: ElkanNoto degenerates under shift (→0.01); the risk models take a fixed prior = **val
  base rate (0.32)** in the Phase-2 config. Shift-robust KM2/TIcE are pending.
- **Licensing**: P (merged) is Pistachio-derived (licensed) → lives in gitignored `private/`;
  benchmark *metrics* are fine to share, raw P is not.

## Status
Phase 1 (skeleton + supervised + nnPU) and Phase 2 batch 1 (density-ratio, IW-PU, SpyTwoStep)
implemented + tested. Remaining Phase 2/3: real KM2/TIcE priors, `PropensityIPS_DR`,
`ShiftAwareIWPU` (Kumagai 2025), `DETM` (Liu 2025), `aPU`/`OSLS` (cautionary). See
`docs/pubench/EXPERIMENTS.md` for the running experiment log and results.
