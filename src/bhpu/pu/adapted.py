"""The domain-adapted PU learner — our "better PAYN" for cross-domain feasibility transfer.

PAYN and plain PU-bagging assume the positives and the unlabeled pool share one feature
distribution (SCAR, in-domain). Transferring literature positives onto a JnJ HTE plate breaks
that: a naive P-vs-U classifier learns a *domain shortcut* and ranks ≈ randomly
(see ``docs/decisions/003-crossdomain-transfer-diagnosis.md``). :class:`DomainAdaptedPULearner`
composes the fixes behind the standard :class:`~bhpu.pu.base.PULearner` interface:

1. **Transductive U** — the caller supplies the *target-domain* unlabeled grid as ``selection == 0``
   (``U`` must live where the test lives).
2. **CORAL alignment** — the positives are recoloured into the target feature distribution, removing
   the second-order component of the shortcut.
3. **Covariate-shift importance weights** — positives are reweighted by ``d(x)/(1−d(x))`` toward the
   target marginal (Sakai & Shimizu, AAAI 2019).
4. **PU-bagging base** — a robust out-of-bag ensemble over balanced positive/unlabeled draws.

Because the positives are aligned *into* the target space at fit time, :meth:`predict_proba` expects
**target-domain** features (e.g. the JnJ reactions) and scores them without further transformation.
"""

from __future__ import annotations

from typing import Self

import numpy as np

from bhpu.adapt.coral import CoralAligner
from bhpu.adapt.discriminator import DomainDiscriminator
from bhpu.models.base import ModelFactory
from bhpu.pu.bagging import BaggingPULearner
from bhpu.pu.base import PULearner


class DomainAdaptedPULearner(PULearner):
    """Transductive, domain-debiased PU-bagging learner.

    Attributes:
        domain_auc_: Positive-vs-unlabeled discriminator AUC measured at fit (the shortcut
            diagnostic; only populated when importance weighting is enabled).
        effective_sample_size_: Kish effective sample size of the positive weights (``nan`` if
            importance weighting is disabled) — a low value flags weight collapse under disjoint
            support.
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        *,
        use_coral: bool = False,
        use_importance_weights: bool = True,
        n_estimators: int = 100,
        subsample_ratio: float = 1.0,
        clip_quantile: float = 0.99,
        random_state: int = 0,
    ) -> None:
        """Initialise the domain-adapted PU learner.

        Args:
            model_factory: Zero-arg factory of the base feasibility model (e.g. CatBoost).
            use_coral: Align positives to the unlabeled/target distribution via CORAL. **Off by
                default:** empirically CORAL *hurts* on high-dimensional sparse fingerprints (the
                covariance whiten/recolor is unstable and smears the class signal — see RESULTS.md);
                enable only for dense, low-dimensional feature spaces.
            use_importance_weights: Reweight positives by covariate-shift importance weights — the
                lever that lifts literature→JnJ transfer from ≈random to ROC-AUC ~0.72.
            n_estimators: Number of PU-bagging weak learners.
            subsample_ratio: Unlabeled subsample size per bag as a multiple of ``|P|``.
            clip_quantile: Upper-quantile clip for importance weights (tames the heavy tail).
            random_state: Global seed.
        """
        self._model_factory = model_factory
        self.use_coral = use_coral
        self.use_importance_weights = use_importance_weights
        self.n_estimators = n_estimators
        self.subsample_ratio = subsample_ratio
        self.clip_quantile = clip_quantile
        self.random_state = random_state

        self._aligner: CoralAligner | None = None
        self._bagging: BaggingPULearner | None = None
        self.domain_auc_: float = float("nan")
        self.effective_sample_size_: float = float("nan")

    def fit(
        self,
        features: np.ndarray,
        selection: np.ndarray,
        *,
        source_marginal: np.ndarray | None = None,
    ) -> Self:
        """Fit the adapted PU ensemble.

        Args:
            features: ``(n_samples, n_features)`` feature matrix. Rows with ``selection == 1`` are
                literature positives; rows with ``selection == 0`` are the target-domain unlabeled
                grid.
            selection: ``(n_samples,)`` PU indicator (``1`` = positive, ``0`` = unlabeled/target).
            source_marginal: Optional ``(m, n_features)`` sample of the **source (literature)
                feature marginal** — ideally all literature reactions (positives *and* measured
                negatives) — used to fit the CORAL map and the covariate-shift discriminator. This
                matters: fitting them on the positives alone inflates the positive covariance to the
                grid's bimodal spread and smears the feasibility signal. If ``None``, the positives
                are used as a (weaker) proxy.

        Returns:
            ``self``.
        """
        selection = np.asarray(selection).astype(int)
        positives = features[selection == 1]
        unlabeled = features[selection == 0]
        if len(positives) == 0 or len(unlabeled) == 0:
            raise ValueError("DomainAdaptedPULearner needs both positive and unlabeled samples.")
        source = positives if source_marginal is None else np.asarray(source_marginal)

        weights: np.ndarray | None = None
        if self.use_importance_weights:
            discriminator = DomainDiscriminator(
                clip_quantile=self.clip_quantile, random_state=self.random_state
            ).fit(source, unlabeled)
            self.domain_auc_ = discriminator.domain_auc_
            weights = discriminator.importance_weights(positives)
            self.effective_sample_size_ = DomainDiscriminator.effective_sample_size(weights)

        train_positives = positives
        if self.use_coral:
            self._aligner = CoralAligner().fit(source, unlabeled)
            train_positives = self._aligner.transform(positives)

        train_features = np.vstack([train_positives, unlabeled])
        train_selection = np.concatenate([np.ones(len(positives)), np.zeros(len(unlabeled))])
        sample_weight = None
        if weights is not None:
            sample_weight = np.concatenate([weights, np.ones(len(unlabeled))])

        self._bagging = BaggingPULearner(
            self._model_factory,
            n_estimators=self.n_estimators,
            subsample_ratio=self.subsample_ratio,
            random_state=self.random_state,
        )
        self._bagging.fit(train_features, train_selection, sample_weight=sample_weight)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Score **target-domain** points (positives were aligned into this space at fit).

        Args:
            features: ``(n, n_features)`` target-domain feature matrix (e.g. JnJ reactions).

        Returns:
            ``(n,)`` feasibility scores in ``[0, 1]``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._bagging is None:
            raise RuntimeError("DomainAdaptedPULearner must be fit before predict_proba.")
        return self._bagging.predict_proba(features)
