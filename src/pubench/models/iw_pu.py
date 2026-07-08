"""Importance-weighted PU learning: reweight the biased positives toward the target marginal."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from pubench.density_ratio import DensityRatioEstimator, build_density_ratio
from pubench.models.base import MODELS, LabeledArrays, Requires
from pubench.models.nnpu import NonNegativePU


@MODELS.register("iw_pu")
class ImportanceWeightedPU(NonNegativePU):
    """Density-ratio importance-weighted PU learning (covariate-shift correction).

    The literature positives P are sampled from a shifted marginal ``p_P`` that differs from the
    target/unlabeled marginal ``p_U``. This model estimates the density ratio
    ``r(x) = p_U(x) / p_P(x)`` (via a :mod:`pubench.density_ratio` estimator) and uses it to
    reweight the positive-risk terms of the nnPU objective, so training targets the deployment
    marginal rather than the reporting-biased one. Weights are clipped and self-normalised to tame
    the heavy tail. This reproduces the IW-PU arm already in use.

    Assumptions
    -----------
    Assumes the shift is **covariate shift** on the positives (``p(y | x)`` invariant, only the
    marginal changes) and needs the class prior. It does *not* model a shifting negative
    population, so ``handles_dd_shift`` is marked True only in the weak sense of correcting the
    positive marginal.

    Parameters
    ----------
    density_ratio:
        Registered density-ratio estimator name (``"rulsif"`` recommended for stability).
    density_ratio_params:
        Extra constructor params for the density-ratio estimator.
    weight_clip:
        ``(low, high)`` clip applied to the importance weights before self-normalisation.
    prior, prior_method, epochs, learning_rate, beta, gamma, l2, seed:
        Forwarded to :class:`~pubench.models.nnpu.NonNegativePU`.
    """

    name: ClassVar[str] = "iw_pu"
    requires: ClassVar[Requires] = Requires(needs_prior=True, handles_dd_shift=True)

    def __init__(
        self,
        *,
        density_ratio: str = "rulsif",
        density_ratio_params: dict[str, Any] | None = None,
        weight_clip: tuple[float, float] = (0.0, 10.0),
        prior: float | None = None,
        prior_method: str = "elkan_noto",
        epochs: int = 200,
        learning_rate: float = 0.05,
        beta: float = 0.0,
        gamma: float = 1.0,
        l2: float = 1e-4,
        seed: int = 0,
    ) -> None:
        super().__init__(
            prior=prior,
            prior_method=prior_method,
            epochs=epochs,
            learning_rate=learning_rate,
            beta=beta,
            gamma=gamma,
            l2=l2,
            seed=seed,
        )
        self.density_ratio = density_ratio
        self.density_ratio_params = density_ratio_params or {}
        self.weight_clip = weight_clip
        self.ratio_: DensityRatioEstimator | None = None

    def fit(  # noqa: D102 (inherited)
        self, P: np.ndarray, U: np.ndarray, *, val: LabeledArrays | None = None
    ) -> ImportanceWeightedPU:
        estimator = build_density_ratio(
            self.density_ratio, {"seed": self.seed, **self.density_ratio_params}
        )
        estimator.fit(U, P)  # numerator = U (target), denominator = P (biased positives)
        self.ratio_ = estimator
        weights = np.clip(estimator.ratio(P), *self.weight_clip)
        self._p_weights = weights if weights.sum() > 0 else None
        super().fit(P, U, val=val)
        return self
