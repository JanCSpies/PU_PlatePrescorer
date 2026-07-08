"""Class-prior (label-frequency) estimation extension point.

Several PU models need the positive class prior ``pi = P(y = 1)`` in the unlabeled pool. This
module defines the common interface and a registry; concrete estimators live in
:mod:`pubench.priors.estimators`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from pubench.registry import Registry

#: Registry of class-prior estimators, keyed by ``name``.
PRIORS: Registry[ClassPriorEstimator] = Registry("class_prior")


class ClassPriorEstimator(ABC):
    """Estimate the positive class prior ``pi`` from positive and unlabeled features.

    Attributes:
    ----------
    name:
        Unique registry key.
    """

    name: ClassVar[str]

    @abstractmethod
    def estimate(self, positives: np.ndarray, unlabeled: np.ndarray) -> float:
        """Return an estimate of ``pi = P(y = 1)`` in the unlabeled pool.

        Parameters
        ----------
        positives:
            Feature matrix of the positive (labelled) set P.
        unlabeled:
            Feature matrix of the unlabeled set U.

        Returns:
        -------
        float
            The estimated prior in ``(0, 1)``.
        """


def build_prior_estimator(name: str, params: dict[str, Any] | None = None) -> ClassPriorEstimator:
    """Instantiate a registered class-prior estimator by name."""
    return PRIORS.get(name)(**(params or {}))
