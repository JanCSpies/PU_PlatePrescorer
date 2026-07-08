"""Concrete class-prior estimators.

Phase 1 ships the Elkan–Noto estimator (simple, SCAR-based). The shift-robust KM2 and TIcE
estimators are declared with their interface and left as Phase-2 stubs so downstream code can be
written against them now.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from pubench.priors.base import PRIORS, ClassPriorEstimator


@PRIORS.register("elkan_noto")
class ElkanNotoPrior(ClassPriorEstimator):
    """Elkan & Noto (2008) scaled-classifier class-prior estimator.

    Trains a *non-traditional* classifier ``g(x) ≈ P(labelled = 1 | x)`` on ``P`` (labelled) vs
    ``U`` (unlabelled), estimates the label-frequency ``c = E[g(x) | x ∈ P]`` on a held-out slice
    of ``P``, and returns ``pi = clip(mean_U g(x) / c)`` — the estimated fraction of positives in
    the unlabeled pool.

    Assumptions
    -----------
    Relies on **SCAR** (labelled positives are a random subset of all positives). Under the
    reporting bias / distribution shift this study targets, this estimate is biased; it is
    included as a cheap, well-understood baseline against the shift-robust estimators (KM2, TIcE).

    Parameters
    ----------
    holdout_fraction:
        Fraction of ``P`` reserved to estimate ``c``.
    clip:
        Lower/upper bound applied to the returned prior for numerical safety.
    seed:
        RNG seed for the P split and the classifier.

    References:
    ----------
    C. Elkan and K. Noto, "Learning classifiers from only positive and unlabeled data", KDD 2008.
    """

    name: ClassVar[str] = "elkan_noto"

    def __init__(
        self,
        *,
        holdout_fraction: float = 0.25,
        clip: tuple[float, float] = (0.01, 0.99),
        max_iter: int = 300,
        max_unlabeled: int = 20000,
        seed: int = 0,
    ) -> None:
        self.holdout_fraction = holdout_fraction
        self.clip = clip
        self.max_iter = max_iter
        self.max_unlabeled = max_unlabeled
        self.seed = seed

    def estimate(self, positives: np.ndarray, unlabeled: np.ndarray) -> float:  # noqa: D102
        unlabeled = _subsample(unlabeled, self.max_unlabeled, self.seed)
        p_train, p_holdout = train_test_split(
            positives, test_size=self.holdout_fraction, random_state=self.seed
        )
        x = np.vstack([p_train, unlabeled])
        s = np.concatenate([np.ones(len(p_train)), np.zeros(len(unlabeled))])
        classifier = LogisticRegression(max_iter=self.max_iter, random_state=self.seed)
        classifier.fit(x, s)
        g = classifier.predict_proba
        c = float(np.mean(g(p_holdout)[:, 1]))
        if c <= 0.0:
            return self.clip[0]
        pi = float(np.mean(g(unlabeled)[:, 1]) / c)
        return float(np.clip(pi, *self.clip))


def _subsample(matrix: np.ndarray, limit: int, seed: int) -> np.ndarray:
    """Return at most ``limit`` rows of ``matrix``, sampled without replacement."""
    if len(matrix) <= limit:
        return matrix
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(matrix), size=limit, replace=False)
    return matrix[idx]


@PRIORS.register("km2")
class KM2Prior(ClassPriorEstimator):
    """KM2 mixture-proportion estimator, Ramaswamy et al. ICML 2016 (Phase-2 stub).

    Kernel-mean-embedding mixture-proportion estimation that does not assume SCAR, making it a
    better fit for the shifted setting. Not yet implemented.

    References:
    ----------
    H. Ramaswamy, C. Scott, A. Tewari, "Mixture proportion estimation via kernel embeddings of
    distributions", ICML 2016.
    """

    name: ClassVar[str] = "km2"

    def estimate(self, positives: np.ndarray, unlabeled: np.ndarray) -> float:  # noqa: D102
        raise NotImplementedError("KM2Prior is a Phase-2 stub; use 'elkan_noto' for now.")


@PRIORS.register("tice")
class TIcEPrior(ClassPriorEstimator):
    """TIcE tree-induced class-prior estimator, Bekker & Davis ECML 2018 (Phase-2 stub).

    Estimates a lower bound on the label frequency by finding a tree-induced subdomain where the
    unlabeled set is nearly pure positive. Not yet implemented.

    References:
    ----------
    J. Bekker and J. Davis, "Estimating the class prior in positive and unlabeled data through
    decision tree induction", ECML-PKDD 2018.
    """

    name: ClassVar[str] = "tice"

    def estimate(self, positives: np.ndarray, unlabeled: np.ndarray) -> float:  # noqa: D102
        raise NotImplementedError("TIcEPrior is a Phase-2 stub; use 'elkan_noto' for now.")
