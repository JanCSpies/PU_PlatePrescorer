"""uLSIF / RuLSIF kernel density-ratio estimators."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from pubench.density_ratio.base import DENSITY_RATIOS, DensityRatioEstimator


@DENSITY_RATIOS.register("ulsif")
class ULSIF(DensityRatioEstimator):
    """Unconstrained Least-Squares Importance Fitting (Kanamori et al., JMLR 2009).

    Models the ratio ``r(x) = sum_l theta_l * K(x, c_l)`` with Gaussian kernels centred on a
    random subset of the numerator samples, and solves the regularised least-squares problem
    ``min_theta 1/2 theta' H theta - h' theta + (lambda/2)||theta||^2`` in closed form
    (``theta = (H + lambda I)^-1 h``, clipped to be non-negative). Used by the importance-weighting
    PU family to reweight the (biased) positives toward the target/unlabeled marginal.

    Parameters
    ----------
    n_centers:
        Number of Gaussian kernel centres (subsampled from the numerator).
    sigma:
        Kernel bandwidth, or ``"median"`` to use the median-distance heuristic.
    reg_lambda:
        L2 regularisation strength.
    alpha:
        Relative-ratio mixing (``0`` = plain uLSIF; ``>0`` = RuLSIF, bounding the ratio).
    seed:
        RNG seed for centre subsampling and the median-heuristic sample.

    References:
    ----------
    T. Kanamori, S. Hido, M. Sugiyama, "A least-squares approach to direct importance estimation",
    JMLR 2009.
    """

    name: ClassVar[str] = "ulsif"

    def __init__(
        self,
        *,
        n_centers: int = 100,
        sigma: float | str = "median",
        reg_lambda: float = 1e-3,
        alpha: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.n_centers = n_centers
        self.sigma = sigma
        self.reg_lambda = reg_lambda
        self.alpha = alpha
        self.seed = seed
        self._centers: np.ndarray | None = None
        self._theta: np.ndarray | None = None
        self._sigma_value: float = 1.0

    def fit(self, numerator: np.ndarray, denominator: np.ndarray) -> ULSIF:  # noqa: D102
        rng = np.random.default_rng(self.seed)
        num = numerator.astype(np.float64)
        den = denominator.astype(np.float64)
        n_centers = min(self.n_centers, len(num))
        centre_idx = rng.choice(len(num), size=n_centers, replace=False)
        self._centers = num[centre_idx]
        self._sigma_value = self._resolve_sigma(num, rng)

        phi_num = self._kernel(num)
        phi_den = self._kernel(den)
        h = phi_num.mean(axis=0)
        hh_den = phi_den.T @ phi_den / len(den)
        hh_num = phi_num.T @ phi_num / len(num)
        big_h = (1.0 - self.alpha) * hh_den + self.alpha * hh_num
        theta = np.linalg.solve(big_h + self.reg_lambda * np.eye(n_centers), h)
        self._theta = np.clip(theta, 0.0, None)
        return self

    def ratio(self, x: np.ndarray) -> np.ndarray:  # noqa: D102
        if self._theta is None:
            raise RuntimeError("ULSIF.ratio called before fit.")
        r: np.ndarray = self._kernel(x.astype(np.float64)) @ self._theta
        clipped: np.ndarray = np.clip(r, 0.0, None)
        return clipped

    def _kernel(self, x: np.ndarray) -> np.ndarray:
        """Gaussian kernel matrix between ``x`` and the stored centres."""
        assert self._centers is not None
        sq = _squared_distances(x, self._centers)
        return np.exp(-sq / (2.0 * self._sigma_value**2))

    def _resolve_sigma(self, num: np.ndarray, rng: np.random.Generator) -> float:
        """Return the bandwidth: the configured value, or the median-distance heuristic."""
        if isinstance(self.sigma, (int, float)):
            return float(self.sigma)
        sample = num[rng.choice(len(num), size=min(len(num), 500), replace=False)]
        sq = _squared_distances(sample, sample)
        median = float(np.median(np.sqrt(sq[sq > 0]))) if np.any(sq > 0) else 1.0
        return median or 1.0


@DENSITY_RATIOS.register("rulsif")
class RuLSIF(ULSIF):
    """Relative uLSIF (Yamada et al., NeurIPS 2011): bounds the ratio for stability.

    Identical to :class:`ULSIF` but with ``alpha > 0``, estimating the *relative* density ratio
    ``p_num / (alpha p_num + (1 - alpha) p_den)`` which is bounded by ``1/alpha`` and therefore
    more robust when the densities have little overlap (the shifted PU setting).

    References:
    ----------
    M. Yamada, T. Suzuki, T. Kanamori, H. Hachiya, M. Sugiyama, "Relative density-ratio
    estimation for robust distribution comparison", NeurIPS 2011.
    """

    name: ClassVar[str] = "rulsif"

    def __init__(self, *, alpha: float = 0.1, **kwargs: object) -> None:
        super().__init__(alpha=alpha, **kwargs)  # type: ignore[arg-type]


@DENSITY_RATIOS.register("kmm")
class KernelMeanMatching(DensityRatioEstimator):
    """Kernel Mean Matching, Huang et al. NeurIPS 2006 (Phase-2 stub).

    Reweights denominator samples so their kernel mean embedding matches the numerator's, by
    solving a bounded quadratic program. Not yet implemented (needs a QP solver); use ``ulsif`` or
    ``rulsif`` meanwhile.

    References:
    ----------
    J. Huang, A. Gretton, K. Borgwardt, B. Schoelkopf, A. Smola, "Correcting sample selection bias
    by unlabeled data", NeurIPS 2006.
    """

    name: ClassVar[str] = "kmm"

    def fit(self, numerator: np.ndarray, denominator: np.ndarray) -> KernelMeanMatching:  # noqa: D102
        raise NotImplementedError("KMM is a Phase-2 stub; use 'ulsif' or 'rulsif' for now.")

    def ratio(self, x: np.ndarray) -> np.ndarray:  # noqa: D102
        raise NotImplementedError("KMM is a Phase-2 stub; use 'ulsif' or 'rulsif' for now.")


def _squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the pairwise squared Euclidean distances between rows of ``a`` and ``b``."""
    a_sq = np.sum(a**2, axis=1, keepdims=True)
    b_sq = np.sum(b**2, axis=1, keepdims=True).T
    out: np.ndarray = np.maximum(a_sq + b_sq - 2.0 * (a @ b.T), 0.0)
    return out
