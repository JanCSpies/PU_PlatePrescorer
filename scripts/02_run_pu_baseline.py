"""End-to-end smoke run of the BH C-N feasibility pipeline (first real numbers).

Runs three honest experiments that together exercise every module (features →
leakage → folds → models → PU → metrics → plate):

* **A. In-domain OOD bar** — supervised CatBoost on DRFP, evaluated on one Neves
  leave-cluster-out fold of JnJ. Reproduces the "bar to beat".
* **B. PU-learner validation (within JnJ)** — hide labels, reveal half the true
  positives as P and treat the rest as unlabeled U; check SpyPU / BaggingPU recover
  the hidden positives. Validates the PU machinery in a clean same-domain setting.
* **C. Literature -> JnJ transfer + plate uplift** — train on the leakage-filtered
  literature HTE set and score the (fully held-out) JnJ wells; report OOD metrics and
  the plate hit-rate-uplift demo metric. This is the project's real cross-domain test.

Fast, modest settings (small DRFP, few boosting iterations) so it runs on CPU in a
few minutes. Results are written to ``artifacts/pu_baseline_metrics.json``.

Usage:
    uv run python scripts/02_run_pu_baseline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from bhpu.data import schema
from bhpu.eval import NevesFoldReader, feasibility_metrics, plate_hit_rate_uplift
from bhpu.features import DRFPFeaturizer
from bhpu.leakage import LeakageAuditor
from bhpu.models import CatBoostFeasibilityModel
from bhpu.pu import BaggingPULearner, SpyPULearner

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
PROCESSED: Path = REPO_ROOT / "datasets" / "processed"
ARTIFACTS: Path = REPO_ROOT / "artifacts"

# --- Fast, CPU-friendly settings for the smoke run. --------------------------
DRFP_BITS = 1024
CATBOOST_ITERS = 250
BAGGING_ESTIMATORS = 15
RANDOM_STATE = 0
console = Console()


def _catboost_factory() -> CatBoostFeasibilityModel:
    """Return a fresh, modestly-sized CatBoost feasibility model."""
    return CatBoostFeasibilityModel(iterations=CATBOOST_ITERS, depth=6, random_state=RANDOM_STATE)


def _featurize(frame: pd.DataFrame, label: str) -> np.ndarray:
    """DRFP-featurise ``frame`` with timing output."""
    start = time.perf_counter()
    features: np.ndarray = DRFPFeaturizer(n_bits=DRFP_BITS).fit_transform(frame)
    console.print(f"  DRFP[{label}] {features.shape} in {time.perf_counter() - start:.1f}s")
    return features


def experiment_a_in_domain_bar(jnj: pd.DataFrame, x_jnj: np.ndarray) -> dict[str, object]:
    """Supervised CatBoost on one Neves OOD fold of JnJ (the bar to beat)."""
    console.rule("[bold]A. In-domain OOD bar (supervised, Neves fold)")
    y = jnj[schema.LABEL].to_numpy().astype(int)
    fold = next(NevesFoldReader(jnj).iter_folds())  # iteration 0, fold 0
    model = _catboost_factory().fit(x_jnj[fold.train_index], y[fold.train_index])
    scores = model.predict_proba(x_jnj[fold.test_index])
    metrics = feasibility_metrics(y[fold.test_index], scores)
    console.print(
        f"  train={len(fold.train_index)} test={len(fold.test_index)} | "
        f"ROC-AUC={metrics.roc_auc:.3f} AUPRC={metrics.auprc:.3f} "
        f"BalAcc={metrics.balanced_accuracy:.3f} "
        f"prec@0.85={metrics.precision_high_confidence:.3f} "
        f"(frac={metrics.fraction_high_confidence:.2f})"
    )
    return {"fold": [fold.iteration, fold.fold], "metrics": metrics.model_dump()}


def experiment_b_pu_validation(jnj: pd.DataFrame, x_jnj: np.ndarray) -> dict[str, object]:
    """Within-JnJ PU: reveal half the positives, recover the rest from unlabeled."""
    console.rule("[bold]B. PU-learner validation (within JnJ)")
    rng = np.random.default_rng(RANDOM_STATE)
    y = jnj[schema.LABEL].to_numpy().astype(int)
    positive_index = np.flatnonzero(y == 1)
    revealed = rng.choice(positive_index, size=len(positive_index) // 2, replace=False)

    selection = np.zeros(len(y), dtype=int)
    selection[revealed] = 1  # labeled positives P; everything else is unlabeled U
    unlabeled_index = np.flatnonzero(selection == 0)
    y_unlabeled = y[unlabeled_index]
    console.print(
        f"  |P|={selection.sum()} labeled positives, |U|={len(unlabeled_index)} "
        f"(hidden positives in U={int(y_unlabeled.sum())})"
    )

    results: dict[str, object] = {}
    for name, learner in (
        ("spy", SpyPULearner(_catboost_factory, random_state=RANDOM_STATE)),
        (
            "bagging",
            BaggingPULearner(
                _catboost_factory, n_estimators=BAGGING_ESTIMATORS, random_state=RANDOM_STATE
            ),
        ),
    ):
        start = time.perf_counter()
        learner.fit(x_jnj, selection)
        scores = learner.predict_proba(x_jnj[unlabeled_index])
        metrics = feasibility_metrics(y_unlabeled, scores)
        console.print(
            f"  [{name:7}] recover hidden positives in U: "
            f"ROC-AUC={metrics.roc_auc:.3f} AUPRC={metrics.auprc:.3f} "
            f"(base={metrics.base_rate:.3f}) in {time.perf_counter() - start:.1f}s"
        )
        results[name] = metrics.model_dump()
    return results


def experiment_c_transfer_and_plate(
    literature: pd.DataFrame, jnj: pd.DataFrame, x_lit: np.ndarray, x_jnj: np.ndarray
) -> dict[str, object]:
    """Train on literature HTE, score held-out JnJ; report metrics + plate uplift."""
    console.rule("[bold]C. Literature -> JnJ transfer + plate uplift")
    y_lit = literature[schema.LABEL].to_numpy().astype(int)
    y_jnj = jnj[schema.LABEL].to_numpy().astype(int)

    model = _catboost_factory().fit(x_lit, y_lit)
    scores = model.predict_proba(x_jnj)
    metrics = feasibility_metrics(y_jnj, scores)
    console.print(
        f"  supervised transfer (lit->JnJ): ROC-AUC={metrics.roc_auc:.3f} "
        f"AUPRC={metrics.auprc:.3f} BalAcc={metrics.balanced_accuracy:.3f} "
        f"prec@0.85={metrics.precision_high_confidence:.3f}"
    )

    plate = plate_hit_rate_uplift(y_jnj, scores, budgets=(96, 384), random_state=RANDOM_STATE)
    for result in plate:
        console.print(
            f"  plate B={result.budget}: model={result.model_hit_rate:.3f} "
            f"random={result.random_hit_rate_mean:.3f} oracle={result.oracle_hit_rate:.3f} "
            f"-> uplift x{result.uplift_vs_random:.2f}, "
            f"+{result.extra_hits_vs_random:.1f} hits/plate"
        )
    return {"metrics": metrics.model_dump(), "plate": [p.model_dump() for p in plate]}


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    jnj = pd.read_parquet(PROCESSED / "test_jnj_hte_2024.parquet").reset_index(drop=True)
    literature = pd.read_parquet(PROCESSED / "literature_hte.parquet").reset_index(drop=True)

    console.rule("[bold]Leakage filter (literature vs JnJ, substrate-level)")
    auditor = LeakageAuditor(jnj)
    report = auditor.audit(literature)
    literature_clean = auditor.filter(literature, level="substrate").reset_index(drop=True)
    console.print(
        f"  shared aryls={report.n_shared_aryls} amines={report.n_shared_amines}; "
        f"rows sharing substrate={report.n_rows_sharing_any_substrate}; "
        f"literature {len(literature)} -> {len(literature_clean)} after filter"
    )

    console.rule("[bold]Featurise (DRFP)")
    x_jnj = _featurize(jnj, "JnJ")
    x_lit = _featurize(literature_clean, "lit")

    results = {
        "config": {
            "drfp_bits": DRFP_BITS,
            "catboost_iters": CATBOOST_ITERS,
            "bagging_estimators": BAGGING_ESTIMATORS,
            "n_jnj": len(jnj),
            "n_literature_clean": len(literature_clean),
            "leakage": report.model_dump(),
        },
        "A_in_domain_bar": experiment_a_in_domain_bar(jnj, x_jnj),
        "B_pu_validation": experiment_b_pu_validation(jnj, x_jnj),
        "C_transfer_plate": experiment_c_transfer_and_plate(literature_clean, jnj, x_lit, x_jnj),
    }
    out_path = ARTIFACTS / "pu_baseline_metrics.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    console.rule("[bold green]Done")
    console.print(f"  metrics -> {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
