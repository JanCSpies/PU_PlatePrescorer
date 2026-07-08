"""Tests for the metric functions and the evaluator."""

from __future__ import annotations

import numpy as np

from pubench.eval import Evaluator
from pubench.eval.metrics import ranking_metrics, select_threshold, threshold_metrics


def test_ranking_metrics_perfect_separation() -> None:
    """Perfectly ordered scores give AUC = PR-AUC = 1."""
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    out = ranking_metrics(scores, y)
    assert out["roc_auc"] == 1.0
    assert out["pr_auc"] == 1.0


def test_ranking_metrics_single_class_is_nan() -> None:
    """With only one class present, ranking metrics are NaN, not an error."""
    out = ranking_metrics(np.array([0.3, 0.7]), np.array([1, 1]))
    assert np.isnan(out["roc_auc"])


def test_threshold_metrics_confusion_and_negatives() -> None:
    """Confusion counts and PAYN-style negative precision/recall are correct at a threshold."""
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.4, 0.6, 0.1])  # at 0.5: pred = [1,0,1,0]
    m = threshold_metrics(scores, y, threshold=0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1.0, 1.0, 1.0, 1.0)
    assert m["neg_precision"] == 0.5  # tn / (tn + fn)
    assert m["neg_recall"] == 0.5  # tn / (tn + fp)


def test_select_threshold_youden() -> None:
    """Youden's-J threshold cleanly separates a separable validation set."""
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    thr = select_threshold(scores, y, strategy="val_youden")
    pred = (scores >= thr).astype(int)
    np.testing.assert_array_equal(pred, y)


def test_evaluator_uses_val_threshold() -> None:
    """The evaluator selects the threshold on val and reports which strategy it used."""
    val_scores = np.array([0.1, 0.2, 0.8, 0.9])
    val_y = np.array([0, 0, 1, 1])
    test_scores = np.array([0.15, 0.85])
    test_y = np.array([0, 1])
    result = Evaluator(threshold_strategy="val_youden").evaluate(
        test_scores, test_y, val_scores=val_scores, val_y=val_y
    )
    assert result.threshold_strategy == "val_youden"
    assert result.metrics["roc_auc"] == 1.0
