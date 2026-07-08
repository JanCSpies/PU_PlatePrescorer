"""Non-negative PU learning (nnPU) with a self-contained linear risk minimiser.

Implemented in NumPy so the core benchmark needs no deep-learning dependency; a torch-MLP variant
can be added later behind the ``models`` extra. The scorer is linear, which is adequate for the
high-dimensional sparse ECFP representation used in Phase 1.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from pubench.models.base import MODELS, LabeledArrays, PUModel, Requires
from pubench.priors import build_prior_estimator


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    out: np.ndarray = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    return out


def _softplus(z: np.ndarray) -> np.ndarray:
    """Numerically stable ``log(1 + exp(z))``."""
    out: np.ndarray = np.logaddexp(0.0, z)
    return out


@MODELS.register("nnpu")
class NonNegativePU(PUModel):
    """Non-negative PU risk estimator (Kiryo et al., NeurIPS 2017).

    Minimises the PU risk ``pi * R_p^+ + max(0, R_u^- - pi * R_p^-)`` with the logistic surrogate
    over a linear scorer. The non-negativity guard prevents the empirical negative risk from going
    negative (the overfitting failure mode of the unbiased uPU estimator); when it does, the
    update follows Kiryo's gradient-ascent correction.

    Assumptions
    -----------
    Needs the positive class prior ``pi`` (supplied via ``prior`` or estimated by
    ``prior_method``). The classic estimator assumes SCAR; under this study's shift the prior is
    itself only approximate, which is part of what the benchmark probes.

    Parameters
    ----------
    prior:
        Fixed class prior ``pi``. If ``None``, it is estimated from (P, U) via ``prior_method``.
    prior_method:
        Registered class-prior estimator name used when ``prior`` is ``None``.
    epochs:
        Full-batch gradient-descent epochs.
    learning_rate:
        Step size.
    beta:
        Non-negativity threshold (Kiryo's ``beta``; 0 = standard nnPU).
    gamma:
        Scaling of the correction step when the negative risk dips below ``-beta``.
    l2:
        L2 regularisation strength on the weights.
    seed:
        Global seed (used for the prior estimator and weight init).

    References:
    ----------
    R. Kiryo, G. Niu, M. C. du Plessis, M. Sugiyama, "Positive-unlabeled learning with
    non-negative risk estimator", NeurIPS 2017.
    """

    name: ClassVar[str] = "nnpu"
    requires: ClassVar[Requires] = Requires(needs_prior=True)

    def __init__(
        self,
        *,
        prior: float | None = None,
        prior_method: str = "elkan_noto",
        epochs: int = 200,
        learning_rate: float = 0.05,
        beta: float = 0.0,
        gamma: float = 1.0,
        l2: float = 1e-4,
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed)
        self.prior = prior
        self.prior_method = prior_method
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.beta = beta
        self.gamma = gamma
        self.l2 = l2
        self.prior_: float | None = None
        self._w: np.ndarray | None = None
        self._b: float = 0.0

    def fit(  # noqa: D102 (inherited)
        self, P: np.ndarray, U: np.ndarray, *, val: LabeledArrays | None = None
    ) -> NonNegativePU:
        pi: float = (
            self.prior
            if self.prior is not None
            else build_prior_estimator(self.prior_method, {"seed": self.seed}).estimate(P, U)
        )
        self.prior_ = pi
        rng = np.random.default_rng(self.seed)
        n_features = P.shape[1]
        # Match the feature dtype (float32): a float64 w would upcast-copy the whole U matrix
        # on every matmul, making each epoch ~20x slower.
        w: np.ndarray = np.asarray(rng.normal(scale=0.01, size=n_features), dtype=P.dtype)
        b: float = 0.0
        for _ in range(self.epochs):
            w, b = self._step(P, U, w, b, pi)
        self._w, self._b = w, b
        return self

    def _step(
        self, P: np.ndarray, U: np.ndarray, w: np.ndarray, b: float, pi: float
    ) -> tuple[np.ndarray, float]:
        """Perform one full-batch nnPU gradient update and return the new ``(w, b)``."""
        s_p = P @ w + b
        s_u = U @ w + b
        # d(positive loss)/ds = -sigmoid(-s); d(negative loss)/ds = sigmoid(s)
        grad_p_plus_w = P.T @ (-_sigmoid(-s_p)) / len(P)
        grad_p_plus_b = float(np.mean(-_sigmoid(-s_p)))
        grad_p_minus_w = P.T @ _sigmoid(s_p) / len(P)
        grad_p_minus_b = float(np.mean(_sigmoid(s_p)))
        grad_u_minus_w = U.T @ _sigmoid(s_u) / len(U)
        grad_u_minus_b = float(np.mean(_sigmoid(s_u)))

        neg_risk = float(np.mean(_softplus(s_u)) - pi * np.mean(_softplus(s_p)))
        grad_neg_w = grad_u_minus_w - pi * grad_p_minus_w
        grad_neg_b = grad_u_minus_b - pi * grad_p_minus_b

        if neg_risk >= -self.beta:
            grad_w = pi * grad_p_plus_w + grad_neg_w + self.l2 * w
            grad_b = pi * grad_p_plus_b + grad_neg_b
            lr = self.learning_rate
        else:  # Kiryo negative-risk correction: descend on (pi R_p^- - R_u^-)
            grad_w = -grad_neg_w + self.l2 * w
            grad_b = -grad_neg_b
            lr = self.learning_rate * self.gamma
        return w - lr * grad_w, b - lr * grad_b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: D102 (inherited)
        if self._w is None:
            raise RuntimeError("NonNegativePU.predict_proba called before fit.")
        return _sigmoid(X @ self._w + self._b)
