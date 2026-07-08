"""CatBoost-backed feasibility model (the PAYN-parity base learner)."""

from __future__ import annotations

import os
from typing import Self

import numpy as np
from catboost import CatBoostClassifier

from bhpu.models.base import FeasibilityModel


def _slurm_thread_count() -> int | None:
    """Thread cap from the SLURM allocation (``SLURM_CPUS_PER_TASK``), else ``None``.

    CatBoost otherwise spawns one thread per *node* core (e.g. 192), which oversubscribes a
    smaller cpus-per-task allocation many-fold and thrashes — the cause of ~10-100x slowdowns
    when several array tasks share a node. Capping to the allocation keeps runs fast and fair.
    """
    raw = os.environ.get("SLURM_CPUS_PER_TASK")
    return int(raw) if raw and raw.isdigit() else None


class CatBoostFeasibilityModel(FeasibilityModel):
    """Gradient-boosted-tree feasibility classifier (CatBoost).

    Defaults are modest so it is fast inside PU bagging / cross-validation sweeps;
    tune via Optuna later for the headline runs.

    Attributes:
        params: Keyword arguments forwarded to :class:`catboost.CatBoostClassifier`.
    """

    def __init__(
        self,
        *,
        iterations: int = 400,
        depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 0,
        auto_class_weights: str | None = "Balanced",
        **extra_params: object,
    ) -> None:
        """Initialise the CatBoost wrapper.

        Args:
            iterations: Number of boosting iterations.
            depth: Tree depth.
            learning_rate: Boosting learning rate.
            random_state: Random seed for reproducibility.
            auto_class_weights: CatBoost class-balancing strategy (``"Balanced"``
                counters the positive/unlabeled skew); pass ``None`` to disable.
            **extra_params: Additional CatBoostClassifier parameters.
        """
        self.params: dict[str, object] = {
            "iterations": iterations,
            "depth": depth,
            "learning_rate": learning_rate,
            "random_seed": random_state,
            "auto_class_weights": auto_class_weights,
            "verbose": False,
            "allow_writing_files": False,
            **extra_params,
        }
        # Respect the SLURM cpus-per-task allocation unless the caller overrode thread_count.
        if "thread_count" not in self.params:
            thread_count = _slurm_thread_count()
            if thread_count is not None:
                self.params["thread_count"] = thread_count
        self._model = CatBoostClassifier(**self.params)

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        """Fit the CatBoost classifier (see :meth:`FeasibilityModel.fit`)."""
        self._model.fit(features, labels, sample_weight=sample_weight)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return ``P(feasible)`` (the positive-class column of CatBoost)."""
        proba: np.ndarray = self._model.predict_proba(features)[:, 1]
        return proba
