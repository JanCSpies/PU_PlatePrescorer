"""Dimensionality-reduced ECFP featurizer.

The raw Morgan fingerprint is high-dimensional and very sparse (~1.7 % of bits set), which has two
costs for cross-domain PU transfer: (1) in thousands of sparse dimensions almost any two finite
molecule sets are linearly separable, so a P-vs-U discriminator trivially hits AUC ~1.0 (the domain
shortcut) partly as an artifact of width; (2) Gaussian-kernel density-ratio estimators (RuLSIF, used
by :class:`~pubench.models.iw_pu.ImportanceWeightedPU`) degenerate in high dimension, collapsing the
importance weights to uniform. Condensing the fingerprint to a compact, dense representation attacks
both. This featurizer wraps :class:`~pubench.features.ecfp.ECFPFeaturizer` and applies a fitted
reduction on top; the reducer is fit on whatever pool the runner passes in (P plus the unlabeled U),
using **no labels**, so it is leakage-free and consistent with the transductive protocol.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from pubench.features.base import FEATURIZERS, Featurizer
from pubench.features.ecfp import ECFPFeaturizer


@FEATURIZERS.register("reduced_ecfp")
class ReducedECFPFeaturizer(Featurizer):
    """ECFP fingerprints condensed to a compact representation for cross-domain PU.

    The pipeline is: raw Morgan bits (via :class:`ECFPFeaturizer`) -> a frequency filter that drops
    constant and (optionally) ultra-rare bits -> the reduction selected by ``method``. The frequency
    filter alone removes the near-unique bits that let a discriminator memorise which molecules came
    from literature; ``"svd"`` additionally projects to a dense low-dimensional space (the form the
    density-ratio / IW machinery needs).

    Parameters
    ----------
    radius:
        Morgan radius forwarded to the inner :class:`ECFPFeaturizer`.
    n_bits:
        Bits per per-molecule fingerprint block, forwarded to the inner featurizer.
    use_conditions:
        If true, the inner featurizer appends ligand/base/solvent fingerprints and the metal
        one-hot *before* reduction. Defaults to ``False``: substrate condensation is the intended
        use, and structural condition fingerprints are handled by a dedicated descriptor featurizer.
    method:
        Reduction strategy applied after the frequency filter:

        * ``"svd"`` (default) -- :class:`~sklearn.decomposition.TruncatedSVD` to ``n_components``
          dense components (LSA; the correct reducer for a sparse binary matrix).
        * ``"correlation"`` -- greedily drop any bit whose absolute Pearson correlation with an
          already-kept bit exceeds ``corr_threshold`` (redundancy pruning; keeps a sparse subset).
        * ``"variance"`` -- keep only the frequency filter (drop constant / ultra-rare bits).
        * ``"none"`` -- passthrough of the raw fingerprint (for A/B against the reduction).
    n_components:
        Target dimensionality for ``method="svd"`` (capped at ``n_kept_bits - 1``).
    min_freq:
        Bits set in a fraction of fit rows outside ``[min_freq, 1 - min_freq]`` are dropped by the
        frequency filter. ``0.0`` drops only all-zero / all-one columns.
    corr_threshold:
        Absolute-correlation cutoff for ``method="correlation"``.
    corr_sample:
        Rows subsampled (without replacement) to estimate the correlation matrix; keeps the
        ``O(n * n_bits^2)`` covariance tractable on the full P+U pool.
    standardize:
        If true, standardise the reduced features (zero mean / unit variance). Helpful for
        kernel-based density-ratio estimators; off by default.
    seed:
        Random state for the randomized SVD and the correlation subsample.
    """

    name: ClassVar[str] = "reduced_ecfp"

    def __init__(
        self,
        *,
        radius: int = 2,
        n_bits: int = 2048,
        use_conditions: bool = False,
        method: str = "svd",
        n_components: int = 256,
        min_freq: float = 0.0,
        corr_threshold: float = 0.8,
        corr_sample: int = 20000,
        standardize: bool = False,
        seed: int = 0,
    ) -> None:
        valid = {"svd", "correlation", "variance", "none"}
        if method not in valid:
            raise ValueError(f"method={method!r} not in {sorted(valid)}.")
        if not 0.0 <= min_freq < 0.5:
            raise ValueError(f"min_freq must be in [0, 0.5), got {min_freq}.")
        self.radius = radius
        self.n_bits = n_bits
        self.use_conditions = use_conditions
        self.method = method
        self.n_components = n_components
        self.min_freq = min_freq
        self.corr_threshold = corr_threshold
        self.corr_sample = corr_sample
        self.standardize = standardize
        self.seed = seed
        self._ecfp = ECFPFeaturizer(radius=radius, n_bits=n_bits, use_conditions=use_conditions)
        self._keep_mask: np.ndarray | None = None
        self._svd: TruncatedSVD | None = None
        self._scaler: StandardScaler | None = None
        self._n_out: int | None = None

    def fit(self, records: pd.DataFrame) -> ReducedECFPFeaturizer:  # noqa: D102 (inherited)
        self._ecfp.fit(records)
        raw = self._ecfp.transform(records)
        mask = self._frequency_mask(raw)
        if self.method == "correlation":
            mask = self._correlation_mask(raw, mask)
        self._keep_mask = mask
        reduced = raw[:, mask]
        if self.method == "svd":
            k = int(min(self.n_components, reduced.shape[1] - 1))
            if k < 1:
                raise ValueError("Too few surviving bits for SVD; relax min_freq or n_components.")
            self._svd = TruncatedSVD(n_components=k, random_state=self.seed)
            reduced = self._svd.fit_transform(reduced)
        if self.standardize:
            self._scaler = StandardScaler().fit(reduced)
            reduced = self._scaler.transform(reduced)
        self._n_out = int(reduced.shape[1])
        return self

    def transform(self, records: pd.DataFrame) -> np.ndarray:  # noqa: D102 (inherited)
        if self._keep_mask is None:
            raise RuntimeError("ReducedECFPFeaturizer.transform called before fit().")
        reduced = self._ecfp.transform(records)[:, self._keep_mask]
        if self._svd is not None:
            reduced = self._svd.transform(reduced)
        if self._scaler is not None:
            reduced = self._scaler.transform(reduced)
        return np.asarray(reduced, dtype=np.float32)

    @property
    def n_features(self) -> int:
        """Reduced feature-vector width (available only after :meth:`fit`)."""
        if self._n_out is None:
            raise RuntimeError("n_features is only defined after fit().")
        return self._n_out

    def _frequency_mask(self, raw: np.ndarray) -> np.ndarray:
        """Boolean column mask keeping bits with document frequency in ``[min_freq, 1 - min_freq]``.

        Fingerprint bits are 0/1, so the column mean is the document frequency. Constant columns
        (all-zero or all-one) are always dropped; ``min_freq`` additionally removes ultra-rare (and
        ultra-common) bits that mostly memorise individual molecules.
        """
        freq = raw.mean(axis=0)
        low = max(self.min_freq, 0.0)
        keep = (freq > low) & (freq < 1.0 - low)
        if not keep.any():
            raise ValueError("Frequency filter removed all bits; lower min_freq.")
        return keep

    def _correlation_mask(self, raw: np.ndarray, freq_mask: np.ndarray) -> np.ndarray:
        """Extend ``freq_mask`` by greedily dropping bits correlated above ``corr_threshold``.

        The correlation matrix is estimated on a random row subsample (``corr_sample``) of the
        frequency-kept bits; among any group of mutually correlated bits the earliest-indexed one is
        retained. Returns a mask over the *original* fingerprint width.
        """
        kept_idx = np.flatnonzero(freq_mask)
        sub = raw[:, kept_idx]
        if sub.shape[0] > self.corr_sample:
            rng = np.random.default_rng(self.seed)
            rows = rng.choice(sub.shape[0], size=self.corr_sample, replace=False)
            sub = sub[rows]
        corr = np.abs(np.corrcoef(sub, rowvar=False))
        corr = np.nan_to_num(corr)  # constant columns after subsampling -> nan -> treat as 0 corr
        np.fill_diagonal(corr, 0.0)
        dropped: set[int] = set()
        for i in range(corr.shape[0]):
            if i in dropped:
                continue
            correlated = np.flatnonzero(corr[i] > self.corr_threshold)
            dropped.update(int(j) for j in correlated if j > i)
        survivors = [j for j in range(corr.shape[0]) if j not in dropped]
        mask = np.zeros(raw.shape[1], dtype=bool)
        mask[kept_idx[survivors]] = True
        return mask
