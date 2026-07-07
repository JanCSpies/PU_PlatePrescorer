"""Domain discriminator: the cross-domain smoking-gun diagnostic and importance weights.

A literature→JnJ PU model that scores ≈ 0.52 ROC-AUC has learned a **domain shortcut**
("was this reaction *reported* in the literature?") instead of feasibility. The cleanest way
to *measure* that shortcut is to train a classifier to tell the two domains apart:

* If literature and JnJ are trivially separable, its held-out AUC ≈ 1.0 — that number **is** the
  explanation of the 0.52 baseline (see ``docs/decisions/003-crossdomain-transfer-diagnosis.md``).
* The same classifier ``d(x) = P(x ∈ target)`` yields covariate-shift **importance weights**
  ``w(x) = d(x) / (1 − d(x))`` that reweight literature training points toward the JnJ marginal
  (Sakai & Shimizu, AAAI 2019 — PU under covariate shift).

Weights for the training (source) points are computed **out-of-fold** so they are not inflated by
the discriminator overfitting its own training data.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from bhpu.models.base import FeasibilityModel, ModelFactory
from bhpu.models.boosting import CatBoostFeasibilityModel


def _default_factory() -> CatBoostFeasibilityModel:
    """A modest CatBoost discriminator (fast; class-balanced for the domain split)."""
    return CatBoostFeasibilityModel(iterations=300, depth=6, learning_rate=0.05, random_state=0)


class DomainDiscriminator:
    """Distinguish a *source* feature distribution from a *target* one.

    Fitted on ``source`` (label 0) versus ``target`` (label 1). Exposes the cross-fitted
    held-out domain AUC (the diagnostic) and covariate-shift importance weights.

    Attributes:
        domain_auc_: Out-of-fold ROC-AUC of the source-vs-target classifier. ``~0.5`` means the
            domains are indistinguishable (no shortcut); ``~1.0`` means a strong domain shortcut.
        source_oob_proba_: Out-of-fold ``d(x) = P(target)`` for each source point (fit order).
    """

    def __init__(
        self,
        model_factory: ModelFactory | None = None,
        *,
        n_splits: int = 5,
        clip_quantile: float = 0.99,
        random_state: int = 0,
    ) -> None:
        """Initialise the domain discriminator.

        Args:
            model_factory: Zero-arg factory of the base classifier (defaults to CatBoost).
            n_splits: Cross-fitting folds for the out-of-fold domain AUC and source weights.
            clip_quantile: Importance weights are clipped at this upper quantile to tame the
                heavy tail that a near-disjoint support produces (set ``1.0`` to disable).
            random_state: Seed for the fold split.
        """
        self._model_factory = model_factory or _default_factory
        self.n_splits = n_splits
        self.clip_quantile = clip_quantile
        self.random_state = random_state
        self.domain_auc_: float = float("nan")
        self.source_oob_proba_: np.ndarray = np.empty(0)
        self._full_model: FeasibilityModel | None = None
        self._n_source = 0

    def fit(self, source: np.ndarray, target: np.ndarray) -> Self:
        """Fit the discriminator on source (0) vs target (1).

        Args:
            source: ``(n_source, n_features)`` feature matrix of the source domain (literature).
            target: ``(n_target, n_features)`` feature matrix of the target domain (JnJ / grid).

        Returns:
            ``self``.
        """
        self._n_source = len(source)
        features = np.vstack([source, target])
        domain = np.concatenate([np.zeros(len(source)), np.ones(len(target))]).astype(int)

        oof = np.full(len(features), np.nan, dtype=np.float64)
        splitter = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        for train_idx, test_idx in splitter.split(features, domain):
            model = self._model_factory()
            model.fit(features[train_idx], domain[train_idx])
            oof[test_idx] = model.predict_proba(features[test_idx])

        self.domain_auc_ = float(roc_auc_score(domain, oof))
        self.source_oob_proba_ = oof[: len(source)]

        self._full_model = self._model_factory()
        self._full_model.fit(features, domain)
        return self

    def target_proba(self, features: np.ndarray) -> np.ndarray:
        """Return ``d(x) = P(x ∈ target)`` from the full-data discriminator."""
        if self._full_model is None:
            raise RuntimeError("DomainDiscriminator must be fit before target_proba.")
        return self._full_model.predict_proba(features)

    def importance_weights(self, features: np.ndarray | None = None) -> np.ndarray:
        """Covariate-shift weights ``w(x) = d(x) / (1 − d(x))``, clipped and mean-normalised.

        Args:
            features: Points to weight. If ``None``, uses the stored **out-of-fold** ``d(x)`` for
                the source points seen at :meth:`fit` (the correct choice for training weights).

        Returns:
            ``(n,)`` positive weights with mean 1 (so total mass ≈ ``n`` — comparable to
            unweighted training).
        """
        if features is None:
            proba = self.source_oob_proba_
            if proba.size == 0:
                raise RuntimeError("DomainDiscriminator must be fit before importance_weights.")
        else:
            proba = self.target_proba(features)
        proba = np.clip(proba, 1e-6, 1 - 1e-6)
        weights = proba / (1.0 - proba)
        if self.clip_quantile < 1.0:
            cap = float(np.quantile(weights, self.clip_quantile))
            weights = np.minimum(weights, cap)
        mean = float(weights.mean())
        normalised: np.ndarray = weights / mean if mean > 0 else weights
        return normalised

    @staticmethod
    def effective_sample_size(weights: np.ndarray) -> float:
        """Kish effective sample size ``(Σw)² / Σw²`` — how many points the weights *really* use.

        A value far below ``len(weights)`` warns that a few literature points dominate training,
        the classic failure mode of importance weighting under near-disjoint support.
        """
        total = float(np.sum(weights))
        sq = float(np.sum(weights**2))
        return (total * total) / sq if sq > 0 else 0.0
