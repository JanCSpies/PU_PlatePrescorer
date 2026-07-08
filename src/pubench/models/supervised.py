"""Naive supervised baseline: treat the unlabeled pool as negative."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from catboost import CatBoostClassifier

from pubench.models.base import MODELS, LabeledArrays, PUModel, Requires


@MODELS.register("supervised")
class SupervisedBaseline(PUModel):
    """Treat P as positive and all of U as negative, then train a standard classifier.

    This is the naive PU lower bound: it ignores the fact that U contains latent positives, so it
    systematically mislabels them and under-estimates feasibility. It is the reference every PU
    method must beat, and it needs no prior, propensity, or shift assumption.

    Parameters
    ----------
    iterations:
        CatBoost boosting iterations.
    depth:
        CatBoost tree depth.
    learning_rate:
        CatBoost learning rate.
    seed:
        Global seed.
    """

    name: ClassVar[str] = "supervised"
    requires: ClassVar[Requires] = Requires()

    def __init__(
        self,
        *,
        iterations: int = 300,
        depth: int = 6,
        learning_rate: float = 0.05,
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed)
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self._model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )

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
