"""Beyond-SCAR PU learners: propensity/SAR methods that use reactivity-orthogonal shadow features.

The SCAR PU methods (``supervised``, ``elkan_noto``, ``bagging_pu`` …) assume the probability a
positive is *reported* (∈ P) is a constant. Literature positives violate this — reporting favours
popular / available building blocks — so the labeling propensity ``e(x)`` is feature-dependent
(SAR: Selected At Random). Correcting it needs a **shadow variable**: features that drive reporting
but not reaction feasibility. If the propensity is modelled on the same substrate fingerprints that
carry feasibility it is not separately identifiable and the estimator collapses to the naive P-vs-U
classifier — which is exactly why the ECFP-space importance-weighter :class:`ImportanceWeightedPU`
degenerated to nnPU on this data.

Both models here take a separate **shadow-feature matrix** (popularity / availability; see
:mod:`pubench.features.propensity`) injected via :meth:`set_propensity` before :meth:`fit`, keeping
the base :class:`~pubench.models.base.PUModel` ``fit(P, U)`` contract intact.

* :class:`PropensityIPS` — inverse-propensity reweighting of the positives in *shadow* space
  (density ratio ``p_U/p_P`` on shadow features), then a naive P-vs-U HistGB on outcome features.
  The shadow-space analogue of IW-PU; because shadow features separate P from U (unlike ECFP) the
  weights are non-trivial, so this genuinely tests whether debiasing the reporting selection helps.
* :class:`SAREM` — Bekker, Robberechts & Davis (2019) SAR-EM: jointly fit the feasibility classifier
  ``f(x)`` on outcome features and the propensity ``e(x_e)`` on shadow features by EM.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pubench.models.base import MODELS, LabeledArrays, PUModel, Requires

_EPS = 1e-6


class _ShadowMixin(PUModel):
    """Shared plumbing for SAR models: injected shadow features + a standardised propensity fit."""

    def __init__(self, *, seed: int = 0) -> None:
        """Initialise with no propensity attached yet (set via :meth:`set_propensity`)."""
        super().__init__(seed=seed)
        self._e_p: np.ndarray | None = None
        self._e_u: np.ndarray | None = None

    def set_propensity(self, e_P: np.ndarray, e_U: np.ndarray) -> None:
        """Attach the shadow-feature matrices for the positive and unlabeled rows.

        Must be called before :meth:`fit`; ``e_P`` / ``e_U`` are row-aligned with the ``P`` / ``U``
        outcome-feature matrices passed to ``fit``.

        Args:
            e_P: ``(n_P, d_e)`` shadow features for the positives.
            e_U: ``(n_U, d_e)`` shadow features for the unlabeled pool.
        """
        self._e_p = np.asarray(e_P, dtype=np.float64)
        self._e_u = np.asarray(e_U, dtype=np.float64)

    def _require_propensity(self, n_p: int, n_u: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the attached shadow matrices, validating they were set and are row-aligned."""
        if self._e_p is None or self._e_u is None:
            raise RuntimeError(
                f"{type(self).__name__} needs shadow features; call set_propensity(e_P, e_U) "
                "before fit()."
            )
        if len(self._e_p) != n_p or len(self._e_u) != n_u:
            raise ValueError(
                "Shadow features are not row-aligned with P/U "
                f"(e_P {len(self._e_p)} vs P {n_p}; e_U {len(self._e_u)} vs U {n_u})."
            )
        return self._e_p, self._e_u


@MODELS.register("prop_ips")
class PropensityIPS(_ShadowMixin):
    """Inverse-propensity-weighted P-vs-U in shadow space (SAR reporting-bias correction).

    A logistic propensity ``g(x_e) = P(reported | x_e)`` is fit on the shadow features that
    discriminate P from U; each positive is reweighted by the clipped, self-normalised density ratio
    ``w = (1 - g) / g`` (so positives that *look like* the under-reported candidate grid are
    up-weighted, popular over-reported ones down-weighted). A HistGB feasibility classifier is then
    fit on the **outcome** features with those positive weights (U at weight 1). This is the
    shadow-space, non-degenerate counterpart of :class:`ImportanceWeightedPU`.
    """

    name: ClassVar[str] = "prop_ips"
    requires: ClassVar[Requires] = Requires(needs_propensity=True, handles_dd_shift=True)

    def __init__(
        self,
        *,
        iterations: int = 400,
        learning_rate: float = 0.05,
        l2_regularization: float = 1.0,
        max_leaf_nodes: int = 31,
        clip_quantile: float = 0.05,
        prop_C: float = 1.0,
        seed: int = 0,
    ) -> None:
        """Configure the HistGB feasibility learner and the propensity/weight-clipping controls.

        Args:
            iterations: HistGB ``max_iter`` for the feasibility classifier.
            learning_rate: HistGB learning rate.
            l2_regularization: HistGB L2 penalty.
            max_leaf_nodes: HistGB leaf cap.
            clip_quantile: Two-sided quantile at which importance weights are winsorised
                (``0.05`` clips to the 5th/95th percentiles) to tame the heavy tail.
            prop_C: Inverse-regularisation strength of the logistic propensity model.
            seed: Global RNG seed.
        """
        super().__init__(seed=seed)
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.max_leaf_nodes = max_leaf_nodes
        self.clip_quantile = clip_quantile
        self.prop_C = prop_C
        self._model: HistGradientBoostingClassifier | None = None

    def _weights(self, e_p: np.ndarray, e_u: np.ndarray) -> np.ndarray:
        """Fit the shadow propensity and return clipped, self-normalised positive IPS weights."""
        scaler = StandardScaler().fit(np.vstack([e_p, e_u]))
        x = scaler.transform(np.vstack([e_p, e_u]))
        s = np.r_[np.ones(len(e_p)), np.zeros(len(e_u))]
        prop = LogisticRegression(
            C=self.prop_C, max_iter=1000, class_weight="balanced", random_state=self.seed
        ).fit(x, s)
        g = np.clip(prop.predict_proba(scaler.transform(e_p))[:, 1], _EPS, 1.0 - _EPS)
        w = (1.0 - g) / g
        lo, hi = np.quantile(w, [self.clip_quantile, 1.0 - self.clip_quantile])
        w = np.clip(w, lo, hi)
        normalised: np.ndarray = w * (len(w) / w.sum())  # self-normalise to mean 1
        return normalised

    def fit(  # noqa: D102 (inherited)
        self, P: np.ndarray, U: np.ndarray, *, val: LabeledArrays | None = None
    ) -> PropensityIPS:
        e_p, e_u = self._require_propensity(len(P), len(U))
        w = self._weights(e_p, e_u)
        x = np.vstack([P, U])
        y = np.concatenate([np.ones(len(P)), np.zeros(len(U))]).astype(np.int64)
        sw = np.concatenate([w, np.ones(len(U))])
        self._model = HistGradientBoostingClassifier(
            max_iter=self.iterations,
            learning_rate=self.learning_rate,
            l2_regularization=self.l2_regularization,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.seed,
        )
        self._model.fit(x, y, sample_weight=sw)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: D102 (inherited)
        if self._model is None:
            raise RuntimeError("PropensityIPS.predict_proba called before fit.")
        proba: np.ndarray = self._model.predict_proba(X)[:, 1]
        return proba


@MODELS.register("sar_em")
class SAREM(_ShadowMixin):
    """SAR-EM (Bekker, Robberechts & Davis 2019): joint EM of classifier ``f`` and propensity ``e``.

    ``f(x) = P(feasible | x)`` is a HistGB on outcome features; ``e(x_e) = P(reported | feasible,
    x_e)`` is a logistic on shadow features. EM alternates (E) computing each unlabeled point's
    posterior probability of being a hidden positive,
    ``P(y=1 | s=0, x) = f (1 - e) / (1 - f e)``, and (M) refitting ``f`` on the soft-labelled data
    and ``e`` on the labeling indicator weighted by the expected positive mass. Because ``e`` is
    tied to *shadow* features it is identifiable and does not collapse into ``f``. Predictions are
    the feasibility posterior ``f(x)`` (propensity-free), the quantity we rank plates by.
    """

    name: ClassVar[str] = "sar_em"
    requires: ClassVar[Requires] = Requires(needs_propensity=True, handles_dd_shift=True)

    def __init__(
        self,
        *,
        iterations: int = 200,
        learning_rate: float = 0.05,
        l2_regularization: float = 1.0,
        max_leaf_nodes: int = 31,
        em_iters: int = 5,
        prop_C: float = 1.0,
        seed: int = 0,
    ) -> None:
        """Configure the HistGB classifier ``f``, the logistic propensity ``e``, and the EM budget.

        Args:
            iterations: HistGB ``max_iter`` for ``f`` (kept modest — refit every EM step).
            learning_rate: HistGB learning rate.
            l2_regularization: HistGB L2 penalty.
            max_leaf_nodes: HistGB leaf cap.
            em_iters: Number of EM iterations.
            prop_C: Inverse-regularisation strength of the logistic propensity ``e``.
            seed: Global RNG seed.
        """
        super().__init__(seed=seed)
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.max_leaf_nodes = max_leaf_nodes
        self.em_iters = em_iters
        self.prop_C = prop_C
        self._model: HistGradientBoostingClassifier | None = None

    def _new_f(self) -> HistGradientBoostingClassifier:
        """Construct a fresh HistGB feasibility classifier with the configured hyperparameters."""
        return HistGradientBoostingClassifier(
            max_iter=self.iterations,
            learning_rate=self.learning_rate,
            l2_regularization=self.l2_regularization,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.seed,
        )

    def _fit_f(self, x: np.ndarray, x_u: np.ndarray, yhat: np.ndarray, yhat_u: np.ndarray) -> None:
        """M-step for ``f``: soft-label fit (each row a pos w=ŷ; each U row a neg w=1-ŷ)."""
        xf = np.vstack([x, x_u])
        yf = np.concatenate([np.ones(len(x)), np.zeros(len(x_u))]).astype(np.int64)
        wf = np.concatenate([yhat, 1.0 - yhat_u])
        keep = wf > _EPS  # HistGB rejects zero/negative weights (labeled rows have yhat≈1)
        self._model = self._new_f()
        self._model.fit(xf[keep], yf[keep], sample_weight=wf[keep])

    def fit(  # noqa: D102 (inherited)
        self, P: np.ndarray, U: np.ndarray, *, val: LabeledArrays | None = None
    ) -> SAREM:
        e_p, e_u = self._require_propensity(len(P), len(U))
        x = np.vstack([P, U])
        s = np.r_[np.ones(len(P)), np.zeros(len(U))]
        unlab = s == 0
        scaler = StandardScaler().fit(np.vstack([e_p, e_u]))
        xe = scaler.transform(np.vstack([e_p, e_u]))

        # Init: naive P-vs-U classifier for f; constant propensity (Elkan–Noto c on labeled).
        self._model = self._new_f()
        self._model.fit(x, s.astype(np.int64))
        f = self._model.predict_proba(x)[:, 1]
        e = np.full(len(x), float(np.clip(f[~unlab].mean(), 0.05, 0.95)))

        for _ in range(self.em_iters):
            # E-step: posterior P(y=1 | observed).
            yhat = np.ones(len(x))
            denom = np.clip(1.0 - f[unlab] * e[unlab], _EPS, None)
            yhat[unlab] = np.clip(f[unlab] * (1.0 - e[unlab]) / denom, 0.0, 1.0)

            # M-step (f): soft-label refit on outcome features.
            self._fit_f(x, x[unlab], yhat, yhat[unlab])
            f = self._model.predict_proba(x)[:, 1]

            # M-step (e): P(reported | feasible, x_e) — logistic on shadow features, weighted by ŷ.
            w = np.clip(yhat, _EPS, None)
            prop = LogisticRegression(C=self.prop_C, max_iter=1000, random_state=self.seed).fit(
                xe, s.astype(np.int64), sample_weight=w
            )
            e = np.clip(prop.predict_proba(xe)[:, 1], _EPS, 1.0 - _EPS)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: D102 (inherited)
        if self._model is None:
            raise RuntimeError("SAREM.predict_proba called before fit.")
        proba: np.ndarray = self._model.predict_proba(X)[:, 1]
        return proba
