"""CORAL: closed-form second-order feature alignment (Sun, Feng & Saenko, AAAI 2016).

CORrelation ALignment removes the *domain shortcut* a P-vs-U classifier otherwise exploits by
matching the source (literature) feature distribution's mean and covariance to the target (JnJ)
distribution. It is the cheapest domain-adaptation lever — no labels, no iteration, just a
whiten-then-recolour linear map — and is our first escalation beyond naive transfer
(see ``docs/decisions/003-crossdomain-transfer-diagnosis.md``).

Given source stats ``(μ_s, Σ_s)`` and target stats ``(μ_t, Σ_t)``, a source point ``x`` maps to::

    x' = (x − μ_s) · Σ_s^{-1/2} · Σ_t^{1/2} + μ_t

which sends a distribution with mean/covariance ``(μ_s, Σ_s)`` to ``(μ_t, Σ_t)``. Covariances are
shrinkage-regularised (``Σ + λ·tr(Σ)/d · I``) because high-dimensional sparse fingerprint
covariances are near-singular.
"""

from __future__ import annotations

from typing import Self

import numpy as np


def _psd_power(matrix: np.ndarray, power: float, floor: float = 1e-10) -> np.ndarray:
    """Raise a symmetric PSD matrix to a real power via its eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    clipped = np.clip(eigenvalues, floor, None)
    scaled: np.ndarray = eigenvectors @ np.diag(clipped**power) @ eigenvectors.T
    return scaled


class CoralAligner:
    """Align a source feature distribution to a target distribution (mean + covariance).

    Attributes:
        transform_: The ``(d, d)`` recolouring matrix ``Σ_s^{-1/2} · Σ_t^{1/2}`` (available after
            :meth:`fit`).
    """

    def __init__(self, *, shrinkage: float = 1e-2, align_mean: bool = True) -> None:
        """Initialise the aligner.

        Args:
            shrinkage: Relative diagonal loading added to each covariance for stability
                (``Σ ← Σ + shrinkage · mean(diag(Σ)) · I``).
            align_mean: If ``True``, also shift the mean from ``μ_s`` to ``μ_t`` (recommended);
                if ``False``, only the covariance is aligned (classic "frustratingly easy" CORAL).
        """
        self.shrinkage = shrinkage
        self.align_mean = align_mean
        self._source_mean: np.ndarray = np.empty(0)
        self._target_mean: np.ndarray = np.empty(0)
        self.transform_: np.ndarray = np.empty(0)

    def _regularised_cov(self, features: np.ndarray) -> np.ndarray:
        """Covariance with relative diagonal shrinkage for numerical stability."""
        cov = np.cov(features, rowvar=False)
        cov = np.atleast_2d(cov)
        loading = self.shrinkage * float(np.mean(np.diag(cov)))
        return cov + loading * np.eye(cov.shape[0])

    def fit(self, source: np.ndarray, target: np.ndarray) -> Self:
        """Compute the whiten-recolour map from source stats to target stats.

        Args:
            source: ``(n_source, d)`` source-domain features (literature).
            target: ``(n_target, d)`` target-domain features (JnJ / grid).

        Returns:
            ``self``.
        """
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        self._source_mean = source.mean(axis=0)
        self._target_mean = target.mean(axis=0)
        whiten = _psd_power(self._regularised_cov(source), -0.5)
        recolour = _psd_power(self._regularised_cov(target), 0.5)
        self.transform_ = whiten @ recolour
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Map source-domain features into the target distribution.

        Args:
            features: ``(n, d)`` features drawn from the *source* domain.

        Returns:
            ``(n, d)`` aligned features with target mean/covariance.
        """
        if self.transform_.size == 0:
            raise RuntimeError("CoralAligner must be fit before transform.")
        aligned: np.ndarray = (
            np.asarray(features, dtype=np.float64) - self._source_mean
        ) @ self.transform_
        aligned = aligned + (self._target_mean if self.align_mean else self._source_mean)
        return aligned

    def fit_transform(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Fit on ``(source, target)`` and return the aligned ``source``."""
        return self.fit(source, target).transform(source)
