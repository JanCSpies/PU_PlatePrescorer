"""Abstract base class for Positive-Unlabeled learners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np


class PULearner(ABC):
    """Learn a feasibility scorer from positive and unlabeled examples.

    The label vector passed to :meth:`fit` is the PU *selection* indicator ``s``:
    ``s = 1`` marks a labeled positive, ``s = 0`` marks an unlabeled example (which
    may be latently positive or negative). :meth:`predict_proba` returns a feasibility
    score usable for ranking (well/plate design) and thresholded classification.
    """

    @abstractmethod
    def fit(self, features: np.ndarray, selection: np.ndarray) -> Self:
        """Fit the PU learner.

        Args:
            features: ``(n_samples, n_features)`` feature matrix.
            selection: ``(n_samples,)`` PU indicator; ``1`` = labeled positive,
                ``0`` = unlabeled.

        Returns:
            ``self``.
        """

    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return a ``(n_samples,)`` feasibility score in ``[0, 1]``."""
