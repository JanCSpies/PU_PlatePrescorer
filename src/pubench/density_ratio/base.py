"""Density-ratio estimation extension point.

The importance-weighting family of PU models needs an estimate of the density ratio
``r(x) = p_num(x) / p_den(x)`` between two sample sets (e.g. the unlabeled/target marginal over
the positive marginal). This module defines the common interface and a registry; concrete
estimators live alongside it (:mod:`pubench.density_ratio.ulsif`, ``kmm``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from pubench.registry import Registry

#: Registry of density-ratio estimators, keyed by ``name``.
DENSITY_RATIOS: Registry[DensityRatioEstimator] = Registry("density_ratio")


class DensityRatioEstimator(ABC):
    """Estimate ``r(x) = p_num(x) / p_den(x)`` from samples of the two densities.

    The convention is fixed: ``fit`` receives *numerator* samples first, then *denominator*
    samples, and :meth:`ratio` returns the (non-negative) ratio at arbitrary points.

    Attributes:
    ----------
    name:
        Unique registry key.
    """

    name: ClassVar[str]

    @abstractmethod
    def fit(self, numerator: np.ndarray, denominator: np.ndarray) -> DensityRatioEstimator:
        """Fit the estimator from numerator and denominator samples.

        Parameters
        ----------
        numerator:
            Samples drawn from ``p_num`` (the density in the ratio's numerator).
        denominator:
            Samples drawn from ``p_den`` (the density in the ratio's denominator).

        Returns:
        -------
        DensityRatioEstimator
            ``self``, fitted.
        """

    @abstractmethod
    def ratio(self, x: np.ndarray) -> np.ndarray:
        """Return the estimated, non-negative density ratio ``r(x)`` for each row of ``x``."""


def build_density_ratio(name: str, params: dict[str, Any] | None = None) -> DensityRatioEstimator:
    """Instantiate a registered density-ratio estimator by name."""
    return DENSITY_RATIOS.get(name)(**(params or {}))
