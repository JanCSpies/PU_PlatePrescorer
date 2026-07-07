"""Benchmark sweep over features and PU strategies (SLURM array job).

Consumes the cached feature matrices from ``scripts/03_precompute_features.py`` and
runs one configuration per array task, so the whole grid parallelises across the
cluster. Two task types:

* ``supervised_ood`` — CatBoost across all Neves leave-cluster-out folds of JnJ
  (the in-domain "bar to beat"), reported as mean ± std over folds.
* ``pu_recovery`` — within-JnJ PU: reveal half the true positives as P, hide the
  rest in U, and measure how well a PU strategy (spy / bagging) recovers them,
  plus the plate hit-rate uplift.

Usage:
    uv run python scripts/04_benchmark.py --list                 # number of configs
    uv run python scripts/04_benchmark.py --config-id 0          # run one config
    uv run python scripts/04_benchmark.py --aggregate            # merge + summarise
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bhpu.data import schema
from bhpu.eval import NevesFoldReader, feasibility_metrics, plate_hit_rate_uplift
from bhpu.models import CatBoostFeasibilityModel, ModelFactory
from bhpu.pu import BaggingPULearner, PULearner, SpyPULearner

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
FEATURE_DIR: Path = REPO_ROOT / "datasets" / "processed" / "features"
RESULT_DIR: Path = REPO_ROOT / "artifacts" / "benchmark"

FEATURIZERS: tuple[str, ...] = ("drfp2048", "ecfp2048")
PU_STRATEGIES: tuple[str, ...] = ("spy", "bagging_k1", "bagging_k2")
SEEDS: tuple[int, ...] = (0, 1, 2)
CATBOOST_ITERS: int = 250
BAGGING_ESTIMATORS: int = 40


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """One benchmark configuration (a single array task).

    Attributes:
        task: ``"supervised_ood"`` or ``"pu_recovery"``.
        featurizer: Cached feature key (e.g. ``"drfp2048"``).
        strategy: PU strategy for ``pu_recovery`` (else ``""``).
        seed: Random seed for ``pu_recovery`` (else ``0``).
    """

    task: str
    featurizer: str
    strategy: str = ""
    seed: int = 0

    @property
    def label(self) -> str:
        """A filename-safe identifier for this config."""
        parts = [self.task, self.featurizer, self.strategy, f"s{self.seed}"]
        return "__".join(p for p in parts if p)


def build_config_grid() -> list[BenchmarkConfig]:
    """Return the full, deterministically-ordered benchmark grid."""
    configs = [BenchmarkConfig("supervised_ood", feat) for feat in FEATURIZERS]
    configs += [
        BenchmarkConfig("pu_recovery", feat, strategy, seed)
        for feat in FEATURIZERS
        for strategy in PU_STRATEGIES
        for seed in SEEDS
    ]
    return configs


def _thread_count() -> int:
    """CatBoost thread budget from the SLURM allocation (default 8)."""
    return int(os.environ.get("SLURM_CPUS_PER_TASK", 8))


def _catboost_factory(thread_count: int) -> ModelFactory:
    """Return a zero-arg factory producing CatBoost models with a fixed thread budget."""

    def factory() -> CatBoostFeasibilityModel:
        return CatBoostFeasibilityModel(
            iterations=CATBOOST_ITERS, depth=6, random_state=0, thread_count=thread_count
        )

    return factory


def _load_features(featurizer: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Load cached JnJ features and the aligned frame (labels + fold columns)."""
    features = np.load(FEATURE_DIR / f"test_jnj__{featurizer}.npz")["X"]
    frame = pd.read_parquet(FEATURE_DIR / "test_jnj.parquet")
    return features, frame


def run_supervised_ood(config: BenchmarkConfig) -> dict[str, object]:
    """CatBoost across all Neves OOD folds; report per-fold and aggregate metrics."""
    features, frame = _load_features(config.featurizer)
    labels = frame[schema.LABEL].to_numpy().astype(int)
    factory = _catboost_factory(_thread_count())

    roc_aucs: list[float] = []
    auprcs: list[float] = []
    for fold in NevesFoldReader(frame).iter_folds():
        model = factory().fit(features[fold.train_index], labels[fold.train_index])
        scores = model.predict_proba(features[fold.test_index])
        metrics = feasibility_metrics(labels[fold.test_index], scores)
        roc_aucs.append(metrics.roc_auc)
        auprcs.append(metrics.auprc)
    return {
        "config": asdict(config),
        "n_folds": len(roc_aucs),
        "roc_auc_mean": float(np.nanmean(roc_aucs)),
        "roc_auc_std": float(np.nanstd(roc_aucs)),
        "auprc_mean": float(np.nanmean(auprcs)),
        "auprc_std": float(np.nanstd(auprcs)),
    }


def _make_pu_learner(strategy: str, factory: ModelFactory) -> PULearner:
    """Instantiate the PU learner named by ``strategy``."""
    if strategy == "spy":
        return SpyPULearner(factory, random_state=0)
    if strategy == "bagging_k1":
        return BaggingPULearner(factory, n_estimators=BAGGING_ESTIMATORS, subsample_ratio=1.0)
    if strategy == "bagging_k2":
        return BaggingPULearner(factory, n_estimators=BAGGING_ESTIMATORS, subsample_ratio=2.0)
    raise ValueError(f"Unknown PU strategy: {strategy!r}")


def run_pu_recovery(config: BenchmarkConfig) -> dict[str, object]:
    """Within-JnJ PU: recover hidden positives from unlabeled; report metrics + uplift."""
    features, frame = _load_features(config.featurizer)
    labels = frame[schema.LABEL].to_numpy().astype(int)
    rng = np.random.default_rng(config.seed)

    positive_index = np.flatnonzero(labels == 1)
    revealed = rng.choice(positive_index, size=len(positive_index) // 2, replace=False)
    selection = np.zeros(len(labels), dtype=int)
    selection[revealed] = 1
    unlabeled_index = np.flatnonzero(selection == 0)

    learner = _make_pu_learner(config.strategy, _catboost_factory(_thread_count()))
    learner.fit(features, selection)
    scores = learner.predict_proba(features[unlabeled_index])
    metrics = feasibility_metrics(labels[unlabeled_index], scores)
    plate = plate_hit_rate_uplift(labels[unlabeled_index], scores, budgets=(96, 384))
    return {
        "config": asdict(config),
        "metrics": metrics.model_dump(),
        "plate": [p.model_dump() for p in plate],
    }


def run_config(config: BenchmarkConfig) -> dict[str, object]:
    """Dispatch a single config to its task runner."""
    if config.task == "supervised_ood":
        return run_supervised_ood(config)
    if config.task == "pu_recovery":
        return run_pu_recovery(config)
    raise ValueError(f"Unknown task: {config.task!r}")


def aggregate() -> None:
    """Merge all per-config result files into a summary table + JSON."""
    results = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(RESULT_DIR.glob("*.json"))
    ]
    rows: list[dict[str, object]] = []
    for result in results:
        config = result["config"]
        if config["task"] == "supervised_ood":
            rows.append(
                {
                    "task": config["task"],
                    "featurizer": config["featurizer"],
                    "strategy": "-",
                    "roc_auc": round(result["roc_auc_mean"], 3),
                    "auprc": round(result["auprc_mean"], 3),
                }
            )
        else:
            metrics = result["metrics"]
            plate96: dict[str, float] = next(
                (p for p in result["plate"] if p["budget"] == 96), {}
            )
            rows.append(
                {
                    "task": config["task"],
                    "featurizer": config["featurizer"],
                    "strategy": f"{config['strategy']}/s{config['seed']}",
                    "roc_auc": round(metrics["roc_auc"], 3),
                    "auprc": round(metrics["auprc"], 3),
                    "uplift@96": round(plate96.get("uplift_vs_random", float("nan")), 2),
                }
            )
    summary = pd.DataFrame(rows)
    out_path = REPO_ROOT / "artifacts" / "benchmark_summary.json"
    out_path.write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)} ({len(rows)} configs)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="Print the number of configs.")
    group.add_argument("--config-id", type=int, help="Run a single config by index.")
    group.add_argument("--aggregate", action="store_true", help="Merge per-config results.")
    args = parser.parse_args()

    grid = build_config_grid()
    if args.list:
        print(len(grid))
        return
    if args.aggregate:
        aggregate()
        return

    config = grid[args.config_id]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result = run_config(config)
    out_path = RESULT_DIR / f"{args.config_id:03d}__{config.label}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[{args.config_id}] {config.label} -> {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
