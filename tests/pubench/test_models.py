"""Tests for the PU-model fit/predict interface on tiny synthetic PU data."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from pubench.models import MODELS, LabeledArrays, NonNegativePU, SupervisedBaseline, build_model
from pubench.priors import ElkanNotoPrior


def _synthetic_pu(seed: int = 0) -> tuple[np.ndarray, np.ndarray, LabeledArrays, LabeledArrays]:
    """Two separated Gaussians: positives near +1, negatives near -1; U is a 1:3 mixture."""
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


def test_models_are_registered() -> None:
    """Both Phase-1 models self-register and build by name."""
    assert {"supervised", "nnpu"}.issubset(MODELS.names())
    assert isinstance(build_model("supervised"), SupervisedBaseline)
    assert isinstance(build_model("nnpu"), NonNegativePU)


def test_supervised_baseline_separates() -> None:
    """The supervised baseline fits, predicts a 1-D proba, and ranks the test set."""
    p, u, val, test = _synthetic_pu()
    model = SupervisedBaseline(iterations=80, depth=4, seed=0).fit(p, u, val=val)
    scores = model.predict_proba(test.X)
    assert scores.shape == (len(test.X),)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert roc_auc_score(test.y, scores) > 0.75


def test_nnpu_separates_with_prior() -> None:
    """nnPU (fixed prior) fits the risk minimiser and ranks the test set."""
    p, u, val, test = _synthetic_pu()
    model = NonNegativePU(prior=0.25, epochs=300, learning_rate=0.1, seed=0).fit(p, u, val=val)
    assert model.prior_ == 0.25
    scores = model.predict_proba(test.X)
    assert scores.shape == (len(test.X),)
    assert roc_auc_score(test.y, scores) > 0.75


def test_nnpu_declares_prior_requirement() -> None:
    """nnPU advertises that it needs a class prior."""
    assert NonNegativePU.requires.needs_prior is True
    assert SupervisedBaseline.requires.needs_prior is False


def test_elkan_noto_prior_in_range() -> None:
    """The Elkan-Noto prior estimate is a probability recovering the ~0.25 mixture."""
    p, u, _, _ = _synthetic_pu()
    pi = ElkanNotoPrior(seed=0).estimate(p, u)
    assert 0.0 < pi < 1.0
    assert abs(pi - 0.25) < 0.25  # loose: right ballpark on separable data
