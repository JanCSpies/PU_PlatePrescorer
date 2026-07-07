"""Spy-based PU learning (the PAYN-style reliable-negative recovery baseline).

Algorithm (Liu et al. "spy" technique, as used by PAYN):

1. Hide a random fraction of the known positives among the unlabeled set as *spies*.
2. Train a first-stage classifier: remaining positives (1) vs unlabeled ∪ spies (0).
3. Score the spies; the low-percentile spy score is a threshold below which unlabeled
   points are confidently negative — the **reliable negatives** (RN).
4. Retrain a final classifier on all positives (1) vs RN (0) and use it to score.

The single hard RN commitment is this method's known weakness (a latent positive
mined as RN poisons the retrain); :mod:`bhpu.pu.bagging` dilutes that risk instead.
"""

from __future__ import annotations

from typing import Self

import numpy as np

from bhpu.models.base import ModelFactory
from bhpu.pu.base import PULearner


class SpyPULearner(PULearner):
    """PU learner via the spy technique for reliable-negative recovery.

    Attributes:
        reliable_negative_fraction_: Fraction of unlabeled points selected as RN
            (populated after :meth:`fit`).
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        *,
        spy_fraction: float = 0.15,
        spy_quantile: float = 0.05,
        n_repeats: int = 1,
        rn_vote_threshold: float = 0.5,
        random_state: int = 0,
    ) -> None:
        """Initialise the spy PU learner.

        Args:
            model_factory: Zero-arg callable returning a fresh feasibility model.
            spy_fraction: Fraction of positives held out as spies each repeat.
            spy_quantile: Quantile of spy scores used as the RN threshold (lower →
                stricter/fewer, purer reliable negatives).
            n_repeats: Number of independent spy draws to average over.
            rn_vote_threshold: An unlabeled point is a reliable negative if flagged in
                at least this fraction of repeats.
            random_state: Random seed.
        """
        self._model_factory = model_factory
        self.spy_fraction = spy_fraction
        self.spy_quantile = spy_quantile
        self.n_repeats = n_repeats
        self.rn_vote_threshold = rn_vote_threshold
        self.random_state = random_state
        self.reliable_negative_fraction_: float = 0.0
        self._final_model = model_factory()

    def fit(self, features: np.ndarray, selection: np.ndarray) -> Self:
        """Recover reliable negatives via spies and fit the final classifier."""
        rng = np.random.default_rng(self.random_state)
        positives = features[selection == 1]
        unlabeled = features[selection == 0]
        if len(positives) == 0 or len(unlabeled) == 0:
            raise ValueError("SpyPULearner needs both positive and unlabeled samples.")

        rn_votes = np.zeros(len(unlabeled), dtype=np.float64)
        for _ in range(self.n_repeats):
            rn_votes += self._reliable_negative_mask(positives, unlabeled, rng)
        reliable_mask = (rn_votes / self.n_repeats) >= self.rn_vote_threshold

        if not reliable_mask.any():
            # Degenerate threshold: fall back to treating all unlabeled as negative.
            reliable_mask = np.ones(len(unlabeled), dtype=bool)
        self.reliable_negative_fraction_ = float(reliable_mask.mean())

        reliable_negatives = unlabeled[reliable_mask]
        train_features = np.vstack([positives, reliable_negatives])
        train_labels = np.concatenate([np.ones(len(positives)), np.zeros(len(reliable_negatives))])
        self._final_model = self._model_factory()
        self._final_model.fit(train_features, train_labels)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Score ``features`` with the final positive-vs-RN classifier."""
        return self._final_model.predict_proba(features)

    def _reliable_negative_mask(
        self, positives: np.ndarray, unlabeled: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Return a boolean RN mask over ``unlabeled`` for one spy draw."""
        n_spies = max(1, int(self.spy_fraction * len(positives)))
        perm = rng.permutation(len(positives))
        spies = positives[perm[:n_spies]]
        kept_positives = positives[perm[n_spies:]]

        train_features = np.vstack([kept_positives, unlabeled, spies])
        train_labels = np.concatenate(
            [
                np.ones(len(kept_positives)),
                np.zeros(len(unlabeled) + len(spies)),
            ]
        )
        classifier = self._model_factory()
        classifier.fit(train_features, train_labels)

        spy_scores = classifier.predict_proba(spies)
        unlabeled_scores = classifier.predict_proba(unlabeled)
        threshold = float(np.quantile(spy_scores, self.spy_quantile))
        return unlabeled_scores <= threshold
