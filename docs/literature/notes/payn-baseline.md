# PAYN — Running the Glorius PU-Learning Repo as a Baseline

- **Repo:** `GloriusGroup/PAYN` — "PAYN: Positivity Is All You Need". Cloned read-only to
  `external/PAYN/` (gitignored; upstream's own git history, not tracked in our repo).
- **Paper:** Boser, Spies, Glorius. *Yield Prediction of Organic Reactions in Biased Data Sets via
  Positive-Unlabeled Learning.* **JACS 2026, 148(14), 15066–15075**, doi:10.1021/jacs.6c00127
  (preprint: chemrxiv-2025-hq4rx). License: MIT.
- **Type:** method / baseline tooling.
- **TL;DR:** PAYN augments *negative* data for a **yield-regression** model. It uses the **spy variant of
  PU learning** under the **SCAR** assumption: a CatBoost *classifier* mines "reliable negatives" out of an
  unlabeled pool, then a CatBoost *regressor* is trained on `known positives (true yields) + reliable
  negatives (yield≈0)`. **It predicts yield, not binary feasibility**, and its stock entrypoint does
  **single-dataset K-fold CV**, not train-on-A / test-on-B — so using it for a *literature→JnJ* transfer
  needs a custom driver (details below). **Status: it runs** in our isolated env (130/130 tests pass; full
  `run_spy.py` completes end-to-end).

---

## 1. What PAYN does (precise method)

PAYN targets the "missing negatives" problem: literature/Reaxys yield data is almost all successful
reactions, so a model never learns what *fails*. PAYN manufactures trustworthy negatives via PU learning.

**Two-stage pipeline** (orchestrated by `run_spy.py`, per fold):

1. **Featurise + binarise.** Every SMILES column → ECFP, concatenated. The continuous `Yield` is binarised
   to `true_bin` with a threshold: `positive ⇔ Yield > yield_classification_threshold × yield_limit`
   (default `0.2 × 100 = 20 %`).
2. **Outer split.** Random **K-fold** CV (`KFold(shuffle=True)`) → `(true_train, true_val, true_test)`.
   The *test* fold keeps ground-truth yields and is what everything is scored on.
3. **Spy split (inner, on the training fold only) — the PU simulation.**
   - Partition the fold into a *labeled* chunk and an *unlabeled* chunk to hit a target
     `ratio_positives_to_unlabeled`. Keep only the **positives** of the labeled chunk as the clean
     positive set **P** (the labeled chunk's negatives are *discarded*). The whole unlabeled chunk **U**
     has its labels forced to 0 (`split_data_with_controlled_PU_ratio`). *(A no-discard variant,
     `split_data_with_original_PU_ratio`, instead builds U from all negatives + a slice of positives.)*
   - **Spy infiltration** (`spy_infiltration`): take `spy_rate` (20 %) of **P**, flip their label to 0,
     and mix them into **U** as "unlabeled spy". Spies are *known* positives now hiding among the zeros —
     they are the yardstick for how positives look when unlabeled.
4. **Spy classifier.** A `CatBoostClassifier` is trained with target `spy_inf_bin` (P=1, U∪spies=0),
   validated against `true_bin`. Hyper-params tuned by Optuna (`spy_model.eval_metric = MCC`).
5. **Reliable-negative threshold** (`AugmenNegativeIdentifier`): score `P(class=1)` for the spies, sort
   ascending, and set the threshold at the `spy_tolerance` quantile of spy scores (default 5 % → we allow
   only 5 % of spies to fall below it), **capped at 0.5**. Any *unlabeled* point scoring **≤ threshold** is
   declared a **"reliable negative"**; points above are **"undecisive"** and dropped. True positives + spies
   = "known positives".
6. **Recombine + regress** (`DataRecombination`, `RegModel`): build a training set of
   `known positives (keep real Yield) + reliable negatives (Yield := 0)` and fit a **`CatBoostRegressor`**
   (Optuna-tuned, `eval_metric = MAE`) — **this regressor is the actual PAYN deliverable**.
7. **Benchmarks per fold** (all scored on the ground-truth test fold): (a) *Augmented-negative* regressor =
   the PU model; (b) *True* regressor = trained on real labels incl. real negatives = **upper bound**;
   (c) *Positives-only* regressor = naive skyline baseline.

**SCAR assumption (baked in).** "Selected Completely At Random": the labeled positives are a uniform random
sample of *all* positives, i.e. label propensity `c = P(labeled | positive)` is **constant** over feature
space. The spy trick is exactly the classic Elkan–Noto estimator of `c`: spies emulate "a positive that
happens to be unlabeled", and their score distribution calibrates the negative threshold. **This holds when
train and test come from the same distribution; it is violated under domain shift** (literature→JnJ) — see
§5 caveats.

**What it predicts.** A **continuous yield**. Binary "feasibility" is only an *internal intermediate*
(the spy classifier, used to mine negatives). To produce a feasibility score you must either threshold the
predicted yield, or repurpose the spy classifier's `P(class=1)`.

---

## 2. Exact input schema

- **File formats:** `.csv`, `.xls`, `.xlsx` (`DataLoader` infers from extension; Excel needs `sheet_name`).
- **Required columns:**
  - **One column per reaction component you want featurised**, each holding a **SMILES string**, listed in
    `dataset.input_columns`. (Bundled demo uses `["Ligand","Additive","Base","Aryl halide"]`.)
  - **One numeric target column** = yield, named by `dataset.target_column` (demo: `"Output"`, range 0–100).
  - *(optional)* `existing_feature_columns` — columns holding pre-computed feature **lists** serialised as
    strings like `"[0, 1, 0, ...]"` (parsed with `ast.literal_eval`), e.g. DFT descriptors.
- **No positive/unlabeled/label column is supplied.** PAYN **derives** `true_bin` from the yield threshold
  and **controls the P/U split itself**. There is *no way in the input file to say "these rows are my
  positives and those are my unlabeled pool"* — that partition is simulated internally. (Key limitation for
  a real PU transfer; see §5.)
- Everything else is config. Internal bookkeeping columns (`true_bin`, `spy_inf_bin`, `augm_bin`, roles,
  `FP_combined`, …) are created by the framework; you don't provide them.

**Bundled example data:** `datasets/Dreher_and_Doyle_input_data.xlsx` (sheet `FullCV_01`, **3955 rows**,
columns `Ligand, Additive, Base, Aryl halide, Output`) — the classic Ahneman/Doyle BH C–N HTE screen.
Also `datasets/Richardson_Sach_OOS_Final.xlsx`.

---

## 3. Exact commands

**Upstream (Poetry, Python 3.12):**
```bash
poetry install
poetry run python run_spy.py                       # runs the whole config.yaml pipeline
poetry run python run_spy.py --spy_splitting_spy_tolerance 0.10   # (only the Experiments/* scripts read CLI)
poetry run pytest tests/                           # ~130 tests
poetry run mlflow ui                               # inspect ./mlruns
```
> Note: **`run_spy.py` hardcodes `ConfigLoader("config.yaml")` and takes NO CLI flags.** The dynamic
> `--section_key` override machinery (`ConfigArgParser`) is only wired into the legacy `Experiments/*.py`
> scripts. To change a run you edit `config.yaml` (or point the loader at another YAML/JSON).

**What actually worked in our isolated env (no Poetry, no GPU):**
```bash
cd external/PAYN
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -e .              # builds via poetry-core backend; ~239 pkgs
VIRTUAL_ENV=.venv uv pip install pytest            # dev dep, not pulled by -e .
.venv/bin/python -m pytest tests/ -q               # -> 130 passed
MLFLOW_TRACKING_URI="file:./mlruns" .venv/bin/python run_spy.py   # full pipeline, local file MLflow store
```

---

## 4. Did it run? **YES.**

- **Tests:** `130 passed, 2 warnings in 49.31s` (heavy deps CatBoost/RDKit/Optuna are mocked in tests).
- **Full pipeline:** `run_spy.py` completed end-to-end on the bundled Dreher–Doyle data with a reduced
  smoke config (2 folds, 1 Optuna trial, CatBoost `max_iterations=50`; MLflow → local `./mlruns`). It logged
  `PAYN Run Completed!`. Representative fold output (illustrates the PU thesis — R² on the held-out test):

  ```
  Evaluation test data: MAE 19.64   R2 0.287   RMSE 22.98   -> Augmented-Negative (PU) regressor
  Evaluation test data: MAE 16.50   R2 0.483   RMSE 19.55   -> True regressor (upper bound, real labels)
  Evaluation test data: MAE 24.46   R2 -0.151  RMSE 29.18   -> Positives-only regressor (naive)
  ```
  i.e. the PU-augmented model (0.287) clearly beats positives-only (−0.151) and moves toward the
  real-labels ceiling (0.483) — exactly PAYN's claim. (These numbers are a fast smoke config, **not**
  the paper's tuned results.)
- Both `config.yaml` and the working tree were **restored/cleaned** afterwards (smoke `mlruns/`,
  `catboost_info/` removed); `external/PAYN` git status is clean.
- **Environment note:** no GPU on this box; PAYN's compute is **CatBoost on CPU**, so GPU is not needed.
  `torch` and `umap-learn` are declared deps but are **never imported by `payn/` or `run_spy.py`** — they
  can be dropped from a slim install (saves the large Torch download).

---

## 5. How to run PAYN on OUR literature→JnJ task

Our data (`datasets/processed/literature_hte.parquet`, `.../features/literature_clean.parquet`,
`.../test_jnj_hte_2024.parquet`) is one row per reaction with SMILES + `Yield` (0–~114, %) + `labels`
(0/1). Column mapping to PAYN's featuriser:

| PAYN role | PAYN config field | OUR column |
|---|---|---|
| SMILES component 1 | `dataset.input_columns[0]` | `Aryl SMILES` |
| SMILES component 2 | `dataset.input_columns[1]` | `Amine SMILES` |
| SMILES component 3 | `dataset.input_columns[2]` | `Catalyst SMILES` (use SMILES, **not** `Catalyst Name`) |
| SMILES component 4 | `dataset.input_columns[3]` | `Base SMILES` |
| SMILES component 5 | `dataset.input_columns[4]` | `Solvent SMILES` |
| target (yield) | `dataset.target_column` | `Yield` |
| yield scale | `dataset.yield_limit` | `100` |
| positive cutoff | `dataset.yield_classification_threshold` | pick to reproduce our `labels` (see caveat) |

`Product SMILES` is available too and could be added as a 6th component. Our DRFP/QM columns are truncated
string reprs — **do not** feed them as `existing_feature_columns`; let PAYN compute ECFP, or precompute
clean feature lists ourselves.

### Preprocessing (minimal)
```python
# read-only conversion; write CSVs under external/PAYN or artifacts/, never into datasets/
import pandas as pd
cols = ["Aryl SMILES","Amine SMILES","Catalyst SMILES","Base SMILES","Solvent SMILES","Yield"]
lit = pd.read_parquet("datasets/processed/features/literature_clean.parquet")[cols]
jnj = pd.read_parquet("datasets/processed/test_jnj_hte_2024.parquet")[cols]
lit.to_csv("external/PAYN/datasets/our_literature.csv", index=False)
jnj.to_csv("external/PAYN/datasets/our_jnj.csv", index=False)
```
- **Condense-bits pitfall:** `condense_bits=True` drops zero-variance ECFP bits **relative to the dataframe
  it sees**. If literature and JnJ are featurised separately the two feature vectors can end up *different
  lengths*. For any train-lit/test-JnJ setup, **featurise the concatenation once** (or set
  `condense_bits: False`) so vectors align. The framework's own docstring flags this.
- Canonicalise SMILES first (our `chem/` canonicaliser) so ECFP is stable.

### Option A — quickest, but a *within-literature* sanity/feature-baseline only
Point stock `run_spy.py` at `our_literature.csv` (CSV path, `sheet_name` ignored). This reproduces PAYN's
own benchmark (PU vs positives-only vs true) **inside the literature set** via internal CV. Useful to sanity-
check the method and compare ECFP-PAYN against our DRFP+CatBoost, **but it does not touch JnJ** and so is not
the transfer experiment.

Config edits (`external/PAYN/config.yaml`):
```yaml
dataset:
  file_path: "./datasets/our_literature.csv"
  sheet_name: ""                     # ignored for CSV
  input_columns: ["Aryl SMILES","Amine SMILES","Catalyst SMILES","Base SMILES","Solvent SMILES"]
  target_column: "Yield"
  yield_limit: 100
  yield_classification_threshold: 0.2
```

### Option B — the real literature→JnJ transfer (needs a small custom driver — allowed, it's our code)
Stock `run_spy.py` **cannot** train on A and test on B. Two ways to get there:
1. **Scaffold-split hack:** concatenate lit+JnJ into one CSV, add a `domain` column (`"lit"`/`"jnj"`), and
   call `DataSplitting.kfold_train_val_scaffold_test_split(column_for_scaffolds="domain")` instead of the
   random KFold in `run_spy.py`. The fold whose test group is `jnj` gives train=lit / test=JnJ. (Edit the
   one call in `run_spy.py`; the method already exists.)
2. **Custom driver (recommended, cleanest):** a ~60-line script importing the payn modules and wiring our
   PU regime directly:
   - `Featurisation` on the **concatenated** lit+JnJ frame (single condense pass);
   - **P** = literature feasible rows; **U** = the combinatorial grid
     (`datasets/processed/features/unlabeled_grid.parquet`) *or* JnJ-without-labels;
   - `SpySplitting.spy_infiltration(P, U)` → `SpyModel` (train) → `AugmenNegativeIdentifier`
     (threshold + reliable negatives) → `DataRecombination.generate_augmented_negative_set` → `RegModel`;
   - **evaluate on JnJ** (threshold predicted yield → feasibility; ROC-AUC/PR vs `labels`).
   This reuses PAYN's spy/SCAR machinery honestly while imposing our train/test boundary.

### What makes a fair literature-only→JnJ comparison hard (be explicit in the write-up)
- **PAYN predicts yield, not feasibility.** Its headline model is a regressor; feasibility must be obtained
  by thresholding its output (or by taking the intermediate spy-classifier probability, which is not the
  paper's intended deliverable). Our metric is binary feasibility (ROC-AUC) — mind the mismatch.
- **SCAR is violated by design here.** Literature positives are a *biased* (publication-favoured) sample of
  all feasible reactions; propensity is **not** constant across the JnJ substrate space. PAYN's spy threshold
  assumes constant `c`, so we should expect the transfer to underperform — consistent with our current
  "cross-domain literature→JnJ ≈ random" finding. This is precisely the **beyond-SCAR** frontier
  (`literature/Beyond_SCAR/`); PAYN is the fair SCAR baseline to beat, not the solution.
- **Internal PU simulation vs. our real PU.** PAYN builds U by *hiding ground-truth negatives* inside one
  dataset. Our literature set actually *contains* measured negatives (labels 0 = 7393 of 11 857 rows in
  `literature_hte`). For a true PU framing you either ignore those negatives (positives-only literature) or
  set U = the combinatorial grid; feeding a literature set that already has negatives into stock `run_spy.py`
  measures "does PU recover negatives we deliberately hid", a different question.
- **`labels` vs PAYN's threshold.** Our `labels` may not equal a single 20 % yield cut. If we want PAYN's
  `true_bin` to match our `labels`, set `yield_classification_threshold` to reproduce it, or accept that
  PAYN re-binarises yield on its own terms.
- **Featurisation differs.** PAYN = per-component ECFP4(2048), condensed + concatenated (or its MFF); our
  pipeline = DRFP2048. Any score gap conflates *method* and *features* — hold one fixed to attribute cause.
- **No conditions we lack:** good news — PAYN only needs the SMILES columns we already have (aryl, amine,
  catalyst, base, solvent) + yield; no extra descriptors are mandatory.

---

## 6. Dependencies & environment that worked

- **Python:** 3.12 required (`>=3.12,<3.13`; we used CPython 3.12.2). Not 3.13 — so it **cannot** share our
  project's 3.13 `.venv`; kept fully isolated in `external/PAYN/.venv`.
- **Install that worked:** `uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -e .`
  (uv builds the Poetry project via the `poetry-core` backend). Then `uv pip install pytest` for the suite.
- **Resolved core versions:** numpy **1.26.0** (hard-pinned), catboost **1.2.10**, mlflow ^2.17 (got 2.x),
  optuna **4.x**, rdkit ^2023.9, scikit-learn 1.9.0, scipy 1.16.3, seaborn 0.13.2, plotly, xlsxwriter,
  openpyxl. **torch 2.12.1** and **umap-learn 0.5.12** install but are **unused** by the pipeline — safe to
  omit for a lighter/faster env.
- **Compute:** CPU-only (CatBoost). No GPU/CUDA needed; the box has no `nvidia-smi` and the run still
  completed. `SLURM_CPUS_PER_TASK` is auto-read for thread count if present.
- **MLflow:** `run_spy.py` doesn't set a tracking URI, so it defaults to a local `./mlruns` file store —
  runs offline, no server required. (The legacy `Run_natural_distribution.py` hardcodes a remote server; the
  main entrypoint does not.)
- **Isolation guarantees honoured:** only `external/PAYN/` was touched; our `.venv`, `pyproject.toml`,
  `uv.lock`, `src/`, `datasets/` untouched; `external/` added to `.gitignore`; nothing pushed.
```
