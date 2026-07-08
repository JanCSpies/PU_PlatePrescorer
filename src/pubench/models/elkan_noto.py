"""Elkan–Noto scaled-classifier PU learning (a nonlinear, prior-free posterior estimator)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from pubench.models.base import MODELS, LabeledArrays, PUModel, Requires


@MODELS.register("elkan_noto")
class ElkanNotoClassifier(PUModel):
    """Elkan & Noto (2008) scaled-classifier PU learner.

    Trains a *non-traditional* classifier ``g(x) ≈ P(labelled = 1 | x)`` on P (labelled) vs U
    (unlabelled), then recovers the true posterior ``P(y = 1 | x) = g(x) / c`` where the label
    frequency ``c = E[g(x) | x ∈ P]`` is estimated on a held-out slice of P. Because the base
    learner is a strong nonlinear model (histogram gradient boosting), this is the nonlinear
    counterpart to the risk-based methods and typically a much stronger PU baseline than a linear
    scorer. Needs no prior (it estimates ``c`` internally).

    Assumptions
    -----------
    Relies on SCAR (labelled positives are a random subset of all positives); the ``c`` correction
    only rescales scores, so *ranking* (ROC-AUC) is unaffected by ``c`` — it matters for calibrated
    thresholds, not for the AUC.

    Parameters
    ----------
    holdout_fraction:
        Fraction of P held out to estimate ``c``.
    max_iter, learning_rate:
        Base HistGradientBoosting hyperparameters.
    seed:
        Global seed.

    References:
    ----------
    C. Elkan and K. Noto, "Learning classifiers from only positive and unlabeled data", KDD 2008.
    """

    name: ClassVar[str] = "elkan_noto"
    requires: ClassVar[Requires] = Requires()

    def __init__(
        self,
        *,
        holdout_fraction: float = 0.25,
        max_iter: int = 300,
        learning_rate: float = 0.1,
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed)
        self.holdout_fraction = holdout_fraction
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.c_: float = 1.0
        self._model: HistGradientBoostingClassifier | None = None

    def fit(  # noqa: D102 (inherited)
        self, P: np.ndarray, U: np.ndarray, *, val: LabeledArrays | None = None
    ) -> ElkanNotoClassifier:
        p_train, p_holdout = train_test_split(
            P, test_size=self.holdout_fraction, random_state=self.seed
        )
        x = np.vstack([p_train, U])
        y = np.concatenate([np.ones(len(p_train)), np.zeros(len(U))]).astype(np.int64)
        self._model = HistGradientBoostingClassifier(
            max_iter=self.max_iter, learning_rate=self.learning_rate, random_state=self.seed
        )
        self._model.fit(x, y)
        self.c_ = max(float(np.mean(self._model.predict_proba(p_holdout)[:, 1])), 1e-6)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: D102 (inherited)
        if self._model is None:
            raise RuntimeError("ElkanNotoClassifier.predict_proba called before fit.")
        g = self._model.predict_proba(X)[:, 1]
        posterior: np.ndarray = np.clip(g / self.c_, 0.0, 1.0)
        return posterior
