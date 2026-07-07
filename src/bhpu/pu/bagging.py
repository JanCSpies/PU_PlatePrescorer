"""Bagging PU learning (Mordelet & Vert 2014; the "many weak predictors" ensemble).

Each of ``T`` weak learners is trained on all positives versus a fresh balanced
subsample (size ``K ≈ k·|P|``) of the unlabeled set, drawn negative. A point's score
is aggregated over the learners for which it was *out-of-bag*. Because any latent
positive that lands in a negative subsample is out-voted across bags, this is more
robust to unreliable negatives than the single hard commitment of the spy method, and
each weak learner sees a balanced problem — so the large-U / small-P imbalance never
bites. See docs/literature/notes/pu-phenol-homocoupling-ensemble.md.
"""

from __future__ import annotations

from typing import Self

import numpy as np

from bhpu.models.base import FeasibilityModel, ModelFactory
from bhpu.pu.base import PULearner


class BaggingPULearner(PULearner):
    """Out-of-bag bagging ensemble of weak PU classifiers.

    Attributes:
        oob_scores_: Out-of-bag feasibility scores for the unlabeled points seen at
            :meth:`fit`, aligned to their order in the input (NaN where a point was
            never out-of-bag). Use these for transductive scoring of the unlabeled set.
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        *,
        n_estimators: int = 100,
        subsample_ratio: float = 1.0,
        replacement: bool = False,
        random_state: int = 0,
    ) -> None:
        """Initialise the bagging PU learner.

        Args:
            model_factory: Zero-arg callable returning a fresh feasibility model.
            n_estimators: Number of weak learners (bags).
            subsample_ratio: Unlabeled subsample size as a multiple of the positive
                count (``K = round(subsample_ratio * |P|)``, capped at ``|U|``).
            replacement: Whether unlabeled subsampling is with replacement.
            random_state: Random seed.
        """
        self._model_factory = model_factory
        self.n_estimators = n_estimators
        self.subsample_ratio = subsample_ratio
        self.replacement = replacement
        self.random_state = random_state
        self._models: list[FeasibilityModel] = []
        self.oob_scores_: np.ndarray = np.empty(0)

    def fit(self, features: np.ndarray, selection: np.ndarray) -> Self:
        """Train the ensemble and compute out-of-bag scores for the unlabeled set."""
        rng = np.random.default_rng(self.random_state)
        positives = features[selection == 1]
        unlabeled = features[selection == 0]
        if len(positives) == 0 or len(unlabeled) == 0:
            raise ValueError("BaggingPULearner needs both positive and unlabeled samples.")

        subsample_size = min(len(unlabeled), max(1, round(self.subsample_ratio * len(positives))))
        positive_labels = np.ones(len(positives))
        oob_sum = np.zeros(len(unlabeled), dtype=np.float64)
        oob_count = np.zeros(len(unlabeled), dtype=np.int64)
        self._models = []

        for _ in range(self.n_estimators):
            bag = rng.choice(len(unlabeled), size=subsample_size, replace=self.replacement)
            in_bag = np.zeros(len(unlabeled), dtype=bool)
            in_bag[bag] = True

            model = self._model_factory()
            train_features = np.vstack([positives, unlabeled[bag]])
            train_labels = np.concatenate([positive_labels, np.zeros(subsample_size)])
            model.fit(train_features, train_labels)
            self._models.append(model)

            out_of_bag = ~in_bag
            if out_of_bag.any():
                oob_sum[out_of_bag] += model.predict_proba(unlabeled[out_of_bag])
                oob_count[out_of_bag] += 1

        self.oob_scores_ = np.where(oob_count > 0, oob_sum / np.maximum(oob_count, 1), np.nan)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return the mean feasibility score across all weak learners.

        Args:
            features: Feature matrix to score. For *new* points (inductive use); for
                the training unlabeled set prefer :attr:`oob_scores_` to avoid the
                small optimistic bias of scoring a point with bags that saw it.

        Returns:
            ``(n_samples,)`` averaged feasibility scores.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self._models:
            raise RuntimeError("BaggingPULearner must be fit before predict_proba.")
        scores = np.mean(
            [model.predict_proba(features) for model in self._models],
            axis=0,
        )
        return np.asarray(scores)
