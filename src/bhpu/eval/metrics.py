"""Feasibility-classification metrics, matched to the Neves evaluation.

Reports the headline metrics used in the target paper plus calibration, so a
literature-trained PU model can be compared to the supervised OOD baseline on the same
footing: ROC-AUC, AUPRC (average precision), balanced accuracy, precision among
high-confidence positive predictions, and expected calibration error.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)


class FeasibilityMetrics(BaseModel):
    """Container of feasibility-classification metrics.

    Attributes:
        n: Number of scored samples.
        n_positive: Number of truly feasible samples.
        base_rate: Fraction of feasible samples.
        roc_auc: Area under the ROC curve.
        auprc: Area under the precision-recall curve (average precision).
        balanced_accuracy: Balanced accuracy at ``threshold``.
        precision_high_confidence: Precision among predictions with score ≥
            ``high_confidence`` (NaN if none qualify).
        fraction_high_confidence: Fraction of samples scored ≥ ``high_confidence``.
        expected_calibration_error: Binned |confidence − accuracy| gap.
    """

    n: int
    n_positive: int
    base_rate: float
    roc_auc: float
    auprc: float
    balanced_accuracy: float
    precision_high_confidence: float
    fraction_high_confidence: float
    expected_calibration_error: float


def feasibility_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
    high_confidence: float = 0.85,
    n_calibration_bins: int = 5,
) -> FeasibilityMetrics:
    """Compute feasibility metrics for scored predictions.

    Args:
        y_true: ``(n,)`` binary ground-truth feasibility labels.
        y_score: ``(n,)`` predicted ``P(feasible)`` in ``[0, 1]``.
        threshold: Decision threshold for balanced accuracy.
        high_confidence: Score cutoff defining a high-confidence positive prediction
            (Neves reported precision at > 0.85).
        n_calibration_bins: Number of equal-width bins for the calibration error.

    Returns:
        A populated :class:`FeasibilityMetrics`.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    high_conf_mask = y_score >= high_confidence
    precision_hc = float(y_true[high_conf_mask].mean()) if high_conf_mask.any() else float("nan")

    # ROC-AUC / AUPRC are undefined without both classes present.
    both_classes = 0 < y_true.sum() < len(y_true)
    return FeasibilityMetrics(
        n=len(y_true),
        n_positive=int(y_true.sum()),
        base_rate=float(y_true.mean()),
        roc_auc=float(roc_auc_score(y_true, y_score)) if both_classes else float("nan"),
        auprc=float(average_precision_score(y_true, y_score)) if both_classes else float("nan"),
        balanced_accuracy=(
            float(balanced_accuracy_score(y_true, (y_score >= threshold).astype(int)))
            if both_classes
            else float("nan")
        ),
        precision_high_confidence=precision_hc,
        fraction_high_confidence=float(high_conf_mask.mean()),
        expected_calibration_error=_expected_calibration_error(y_true, y_score, n_calibration_bins),
    )


def _expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int) -> float:
    """Return the equal-width expected calibration error (ECE)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(y_score, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bin_index == b
        if in_bin.any():
            confidence = float(y_score[in_bin].mean())
            accuracy = float(y_true[in_bin].mean())
            ece += float(in_bin.mean()) * abs(accuracy - confidence)
    return ece
