"""Evaluator: turn model scores + ground truth into the shared metric row."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from pubench.eval import metrics

ThresholdStrategy = Literal["default", "val_youden", "val_f1"]


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Metrics for one (model, featurizer) on one split, plus the threshold provenance."""

    metrics: dict[str, float]
    threshold: float
    threshold_strategy: str

    def flat(self) -> dict[str, float | str]:
        """Return a flat mapping (metrics + threshold info) for a results-table row."""
        row: dict[str, float | str] = dict(self.metrics)
        row["threshold"] = self.threshold
        row["threshold_strategy"] = self.threshold_strategy
        return row


class Evaluator:
    """Compute ranking + threshold metrics, selecting the threshold from validation scores.

    Parameters
    ----------
    threshold_strategy:
        How to pick the decision threshold: a fixed ``"default"`` (0.5) or a validation-optimised
        ``"val_youden"`` / ``"val_f1"``. A spy-derived threshold is planned for Phase 2.
    """

    def __init__(self, *, threshold_strategy: ThresholdStrategy = "val_youden") -> None:
        self.threshold_strategy = threshold_strategy

    def evaluate(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        *,
        val_scores: np.ndarray | None = None,
        val_y: np.ndarray | None = None,
    ) -> EvalResult:
        """Evaluate ``scores`` against ``y_true``.

        Parameters
        ----------
        scores:
            Predicted ``P(y = 1 | x)`` for the split being scored (e.g. test).
        y_true:
            Ground-truth labels for that split.
        val_scores:
            Validation scores used to pick the threshold (required unless strategy is default).
        val_y:
            Validation labels used to pick the threshold.

        Returns:
        -------
        EvalResult
            The metric row plus the threshold and strategy used.
        """
        strategy = self.threshold_strategy
        if strategy == "default" or val_scores is None or val_y is None:
            threshold = 0.5
            strategy = "default"
        else:
            threshold = metrics.select_threshold(val_scores, val_y, strategy=strategy)
        row = {
            **metrics.ranking_metrics(scores, y_true),
            **metrics.threshold_metrics(scores, y_true, threshold),
        }
        return EvalResult(metrics=row, threshold=threshold, threshold_strategy=strategy)
