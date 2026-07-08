"""Naive supervised baseline: treat the unlabeled pool as negative (swappable base learner)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from pubench.models.base import MODELS, LabeledArrays, PUModel, Requires


@MODELS.register("supervised")
class SupervisedBaseline(PUModel):
    """Treat P as positive and all of U as negative, then train a standard classifier.

    The naive PU lower bound: it ignores that U contains latent positives, so it mislabels them and
    under-estimates feasibility. Needs no prior/propensity/shift assumption. The base learner is
    swappable (``catboost`` or ``histgb``) — E4 found HistGradientBoosting transfers markedly better
    than CatBoost on this shifted P-vs-U task, so the base learner is itself a lever.

    Parameters
    ----------
    base:
        Base classifier: ``"catboost"`` or ``"histgb"``.
    iterations:
        Boosting iterations (CatBoost ``iterations`` / HistGB ``max_iter``).
    depth:
        CatBoost tree depth (ignored by HistGB).
    learning_rate:
        Boosting learning rate.
    l2_regularization:
        HistGB L2 regularisation (ignored by CatBoost).
    max_leaf_nodes:
        HistGB tree size (ignored by CatBoost).
    seed:
        Global seed.
    """

    name: ClassVar[str] = "supervised"
    requires: ClassVar[Requires] = Requires()

    def __init__(
        self,
        *,
        base: str = "catboost",
        iterations: int = 300,
        depth: int = 6,
        learning_rate: float = 0.05,
        l2_regularization: float = 0.0,
        max_leaf_nodes: int = 31,
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed)
        self.base = base
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.max_leaf_nodes = max_leaf_nodes
        self._model = self._build()

    def _build(self) -> CatBoostClassifier | HistGradientBoostingClassifier:
        """Construct the configured base classifier."""
        if self.base == "catboost":
            return CatBoostClassifier(
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                random_seed=self.seed,
                verbose=False,
                allow_writing_files=False,
            )
        if self.base == "histgb":
            return HistGradientBoostingClassifier(
                max_iter=self.iterations,
                learning_rate=self.learning_rate,
                l2_regularization=self.l2_regularization,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=self.seed,
            )
        raise ValueError(f"Unknown base learner {self.base!r}; choose 'catboost' or 'histgb'.")

    def fit(  # noqa: D102 (inherited)
        self, P: np.ndarray, U: np.ndarray, *, val: LabeledArrays | None = None
    ) -> SupervisedBaseline:
        x = np.vstack([P, U])
        y = np.concatenate([np.ones(len(P), dtype=np.int64), np.zeros(len(U), dtype=np.int64)])
        self._model.fit(x, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: D102 (inherited)
        proba: np.ndarray = self._model.predict_proba(X)[:, 1]
        return proba
