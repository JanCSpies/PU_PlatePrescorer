"""Bagging PU: ensemble of classifiers, each trained on P vs a bootstrap subsample of U."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from pubench.models.base import MODELS, LabeledArrays, PUModel, Requires


@MODELS.register("bagging_pu")
class BaggingPU(PUModel):
    """Bagging PU learning (Mordelet & Vert, 2014).

    Trains ``n_estimators`` classifiers, each on all of P (positive) versus a bootstrap subsample of
    U (treated as negative), and averages their scores. Averaging over many random U-subsamples
    dilutes the label noise from the latent positives hidden in U, so it is a strong nonlinear PU
    method — and, unlike the risk-based estimators, it inherits the capacity of the base learner
    (here histogram gradient boosting, fast at this scale). Needs no prior or propensity model.

    Parameters
    ----------
    n_estimators:
        Number of bagged classifiers.
    max_samples:
        U-subsample size per bag, as a fraction of ``len(P)`` (the balanced-bag heuristic).
    max_iter:
        Boosting iterations of each base classifier.
    learning_rate:
        Base-classifier learning rate.
    seed:
        Global seed (each bag uses ``seed + i``).
    """

    name: ClassVar[str] = "bagging_pu"
    requires: ClassVar[Requires] = Requires()

    def __init__(
        self,
        *,
        n_estimators: int = 15,
        max_samples: float = 1.0,
        max_iter: int = 200,
        learning_rate: float = 0.1,
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self._models: list[HistGradientBoostingClassifier] = []

    def fit(  # noqa: D102 (inherited)
        self, P: np.ndarray, U: np.ndarray, *, val: LabeledArrays | None = None
    ) -> BaggingPU:
        rng = np.random.default_rng(self.seed)
        n_neg = max(1, min(len(U), round(self.max_samples * len(P))))
        self._models = []
        for i in range(self.n_estimators):
            neg_idx = rng.choice(len(U), size=n_neg, replace=True)
            x = np.vstack([P, U[neg_idx]])
            y = np.concatenate([np.ones(len(P)), np.zeros(n_neg)]).astype(np.int64)
            clf = HistGradientBoostingClassifier(
                max_iter=self.max_iter,
                learning_rate=self.learning_rate,
                random_state=self.seed + i,
            )
            clf.fit(x, y)
            self._models.append(clf)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: D102 (inherited)
        if not self._models:
            raise RuntimeError("BaggingPU.predict_proba called before fit.")
        scores = np.mean([clf.predict_proba(X)[:, 1] for clf in self._models], axis=0)
        return scores
