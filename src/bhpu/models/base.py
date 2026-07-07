"""Abstract base class and factory typing for supervised feasibility models.

A :class:`FeasibilityModel` is a thin, uniform wrapper around a binary probabilistic
classifier that consumes a dense feature matrix. PU learners (:mod:`bhpu.pu`) are
built on top of these and create fresh instances via a :data:`ModelFactory`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Self

import numpy as np


class FeasibilityModel(ABC):
    """A binary probabilistic classifier of reaction feasibility.

    Implementations wrap a concrete learner (CatBoost, TabPFN, …) behind a minimal
    fit / predict_proba interface so they are interchangeable inside PU learners and
    the evaluation harness.
    """

    @abstractmethod
    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        """Fit the model.

        Args:
            features: ``(n_samples, n_features)`` feature matrix.
            labels: ``(n_samples,)`` binary labels in ``{0, 1}``.
            sample_weight: Optional per-sample weights.

        Returns:
            ``self``.
        """

    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return ``P(feasible)`` as a ``(n_samples,)`` array in ``[0, 1]``."""


#: A zero-argument callable that returns a fresh, unfitted :class:`FeasibilityModel`.
type ModelFactory = Callable[[], FeasibilityModel]
