"""Classification + reliable-negative metric functions computed against ground truth."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)


def ranking_metrics(scores: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """Threshold-free ranking metrics.

    Parameters
    ----------
    scores:
        Predicted ``P(y = 1 | x)`` scores.
    y_true:
        Ground-truth binary labels.

    Returns:
    -------
    dict[str, float]
        ``roc_auc`` and ``pr_auc`` (average precision). NaN if only one class is present.
    """
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
    }


def threshold_metrics(scores: np.ndarray, y_true: np.ndarray, threshold: float) -> dict[str, float]:
    """Classification + reliable-negative metrics at a fixed decision threshold.

    Parameters
    ----------
    scores:
        Predicted ``P(y = 1 | x)`` scores.
    y_true:
        Ground-truth binary labels.
    threshold:
        Scores ``>= threshold`` are predicted positive.

    Returns:
    -------
    dict[str, float]
        ``balanced_accuracy``, ``mcc``, PAYN-style ``neg_precision`` / ``neg_recall``, and the
        confusion-matrix counts ``tn`` / ``fp`` / ``fn`` / ``tp``.
    """
    y_pred = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    neg_precision = tn / (tn + fn) if (tn + fn) else float("nan")
    neg_recall = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "neg_precision": float(neg_precision),
        "neg_recall": float(neg_recall),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def select_threshold(scores: np.ndarray, y_true: np.ndarray, *, strategy: str) -> float:
    """Choose a decision threshold from labelled (validation) scores.

    Parameters
    ----------
    scores:
        Validation ``P(y = 1 | x)`` scores.
    y_true:
        Validation ground-truth labels.
    strategy:
        ``"default"`` (0.5), ``"val_youden"`` (max TPR-FPR) or ``"val_f1"`` (max F1).

    Returns:
    -------
    float
        The chosen threshold.

    Raises:
    ------
    ValueError
        If ``strategy`` is unknown.
    """
    if strategy == "default":
        return 0.5
    candidates = np.unique(scores)
    if len(candidates) == 0:
        return 0.5
    if strategy == "val_youden":
        objective = lambda t: _youden(scores, y_true, t)  # noqa: E731
    elif strategy == "val_f1":
        objective = lambda t: _f1(scores, y_true, t)  # noqa: E731
    else:
        raise ValueError(f"Unknown threshold strategy {strategy!r}.")
    return float(max(candidates, key=objective))


def _youden(scores: np.ndarray, y_true: np.ndarray, threshold: float) -> float:
    """Youden's J = TPR - FPR at ``threshold``."""
    metrics = threshold_metrics(scores, y_true, threshold)
    tp, fn, fp, tn = metrics["tp"], metrics["fn"], metrics["fp"], metrics["tn"]
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return tpr - fpr


def _f1(scores: np.ndarray, y_true: np.ndarray, threshold: float) -> float:
    """F1 of the positive class at ``threshold``."""
    metrics = threshold_metrics(scores, y_true, threshold)
    tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom else 0.0
