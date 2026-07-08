"""Tests for the Phase-2 density-ratio utility and shift/bias-aware PU models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from pubench.density_ratio import DENSITY_RATIOS, ULSIF, build_density_ratio
from pubench.models import (
    MODELS,
    BaggingPU,
    ElkanNotoClassifier,
    ImportanceWeightedPU,
    LabeledArrays,
    SpyTwoStep,
    build_model,
)


def _synthetic_pu(seed: int = 0) -> tuple[np.ndarray, np.ndarray, LabeledArrays, LabeledArrays]:
    """Positives near +1, negatives near -1; U is a 1:3 mixture (prior ~0.25)."""
    rng = np.random.default_rng(seed)
    dim = 20

    def pos(n: int) -> np.ndarray:
        return rng.normal(1.0, 1.0, (n, dim))

    def neg(n: int) -> np.ndarray:
        return rng.normal(-1.0, 1.0, (n, dim))

    p = pos(200)
    u = np.vstack([pos(100), neg(300)])
    rng.shuffle(u)
    val = LabeledArrays(np.vstack([pos(50), neg(50)]), np.r_[np.ones(50), np.zeros(50)].astype(int))
    test = LabeledArrays(
        np.vstack([pos(50), neg(50)]), np.r_[np.ones(50), np.zeros(50)].astype(int)
    )
    return p, u, val, test


def test_density_ratio_registered() -> None:
    """uLSIF/RuLSIF self-register and build by name."""
    assert {"ulsif", "rulsif", "kmm"}.issubset(DENSITY_RATIOS.names())
    assert isinstance(build_density_ratio("ulsif"), ULSIF)


def test_ulsif_recovers_shift_direction() -> None:
    """uLSIF gives higher p_num/p_den where the numerator density dominates."""
    rng = np.random.default_rng(0)
    numerator = rng.normal(2.0, 1.0, (400, 1))  # shifted right
    denominator = rng.normal(0.0, 1.0, (400, 1))
    est = ULSIF(n_centers=50, seed=0).fit(numerator, denominator)
    # ratio should increase from left (denominator region) to right (numerator region)
    left = est.ratio(np.array([[-1.0]]))[0]
    right = est.ratio(np.array([[3.0]]))[0]
    assert right > left
    assert (est.ratio(numerator) >= 0).all()


def test_iw_pu_registered_and_separates() -> None:
    """ImportanceWeightedPU fits (ratio + weighted nnPU) and ranks the test set."""
    assert "iw_pu" in MODELS.names()
    p, u, val, test = _synthetic_pu()
    model = build_model("iw_pu", {"prior": 0.25, "epochs": 200, "density_ratio": "rulsif"})
    assert isinstance(model, ImportanceWeightedPU)
    model.fit(p, u, val=val)
    assert model.ratio_ is not None
    assert model._p_weights is not None  # importance weights were set
    scores = model.predict_proba(test.X)
    assert scores.shape == (len(test.X),)
    assert roc_auc_score(test.y, scores) > 0.7


def test_spy_extracts_reliable_negatives_and_separates() -> None:
    """SpyTwoStep extracts reliable negatives and the final classifier ranks the test set."""
    p, u, val, test = _synthetic_pu()
    model = SpyTwoStep(iterations=60, depth=4, seed=0).fit(p, u, val=val)
    assert model.n_reliable_negatives_ > 0
    scores = model.predict_proba(test.X)
    assert roc_auc_score(test.y, scores) > 0.7


def test_bagging_pu_separates() -> None:
    """BaggingPU (ensemble over U subsamples) ranks the test set well above chance."""
    assert "bagging_pu" in MODELS.names()
    p, u, val, test = _synthetic_pu()
    model = BaggingPU(n_estimators=8, max_iter=60, seed=0).fit(p, u, val=val)
    assert isinstance(model, BaggingPU)
    scores = model.predict_proba(test.X)
    assert roc_auc_score(test.y, scores) > 0.8


def test_elkan_noto_classifier_separates() -> None:
    """The Elkan-Noto scaled classifier fits g/c and ranks the test set well above chance."""
    assert "elkan_noto" in MODELS.names()
    p, u, val, test = _synthetic_pu()
    model = ElkanNotoClassifier(max_iter=80, seed=0).fit(p, u, val=val)
    assert 0.0 < model.c_ <= 1.0
    scores = model.predict_proba(test.X)
    assert roc_auc_score(test.y, scores) > 0.8


def test_iw_pu_weights_vary_when_densities_differ() -> None:
    """When P spans two clusters but U covers only one, the ratio weights vary across P."""
    rng = np.random.default_rng(0)
    dim = 20
    near = rng.normal(np.r_[2.0, np.zeros(dim - 1)], 1.0, (150, dim))  # in U's support
    far = rng.normal(np.r_[-3.0, np.zeros(dim - 1)], 1.0, (150, dim))  # outside U's support
    p = np.vstack([near, far])
    u = rng.normal(np.r_[2.0, np.zeros(dim - 1)], 1.0, (400, dim))
    val = LabeledArrays(np.vstack([near[:20], far[:20]]), np.r_[np.ones(20), np.zeros(20)].astype(int))
    model = ImportanceWeightedPU(prior=0.5, epochs=20, project_dim=None, density_ratio="rulsif")
    model.fit(p, u, val=val)
    assert model._p_weights is not None
    # points in U's support should get higher weight than the out-of-support cluster
    assert model._p_weights[:150].mean() > model._p_weights[150:].mean()


def test_iw_pu_declares_shift_handling() -> None:
    """IW-PU advertises prior need and shift handling; spy advertises neither."""
    assert ImportanceWeightedPU.requires.needs_prior is True
    assert ImportanceWeightedPU.requires.handles_dd_shift is True
    assert SpyTwoStep.requires.needs_prior is False
