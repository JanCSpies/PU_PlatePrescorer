"""Honest literature→JnJ transfer experiment under the frozen protocol.

This is the harness for the "better PAYN" question: *can a model trained only on literature/
external data rank JnJ Buchwald–Hartwig feasibility?* It runs a matrix of arms — naive vs
domain-adapted (CORAL / importance-weighted), and PU (positives + unlabeled grid) vs
literature-supervised (published positives *and* HTE negatives) — and scores every arm on the
frozen substrate-disjoint VAL and TEST splits.

Guarantees (see ``docs/decisions/003-crossdomain-transfer-diagnosis.md``):

* **No JnJ labels ever enter training.** JnJ is used only to score VAL/TEST. The unlabeled grid
  ``U`` spans JnJ *substrate structures* (no labels) — the transductive premise.
* Model *selection* is on VAL only; TEST is reported once for all arms in a single final pass.
* :meth:`~bhpu.eval.protocol.FrozenEvalProtocol.assert_training_clean` verifies the literature
  corpus reuses no JnJ substrate/pair before any arm trains.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from bhpu.adapt import CoralAligner, DomainDiscriminator
from bhpu.data import schema
from bhpu.eval.metrics import FeasibilityMetrics, feasibility_metrics
from bhpu.eval.protocol import FrozenEvalProtocol
from bhpu.models.boosting import CatBoostFeasibilityModel
from bhpu.pu.bagging import BaggingPULearner

#: A fitted scorer: features -> P(feasible).
Scorer = Callable[[np.ndarray], np.ndarray]


@dataclass
class ArmResult:
    """Metrics for one transfer arm on VAL and TEST.

    Attributes:
        name: Arm identifier (e.g. ``"pu_coral_iw"``).
        family: ``"pu"`` or ``"supervised"``.
        val: Feasibility metrics on the JnJ VAL split (used for model selection).
        test: Feasibility metrics on the JnJ TEST split (reported once; ``None`` until computed).
        notes: Free-form provenance (effective sample size, domain AUC, …).
    """

    name: str
    family: str
    val: FeasibilityMetrics
    test: FeasibilityMetrics | None = None
    notes: dict[str, float] = field(default_factory=dict)


class TransferExperiment:
    """Run the literature→JnJ transfer arm matrix on cached DRFP features.

    Attributes:
        protocol: The frozen substrate-disjoint evaluation protocol.
        domain_auc: Cross-fitted literature-vs-JnJ-grid discriminator AUC (the shortcut diagnostic).
    """

    def __init__(
        self,
        features_dir: Path,
        *,
        featurizer: str = "drfp2048",
        n_estimators: int = 60,
        random_state: int = 0,
    ) -> None:
        """Load cached features and build the frozen split.

        Args:
            features_dir: Directory holding ``literature_clean.parquet``, ``test_jnj.parquet`` and
                the ``*__{featurizer}.npz`` matrices (see ``scripts/03_precompute_features.py``).
            featurizer: Feature key to load (``"drfp2048"`` or ``"ecfp2048"``).
            n_estimators: Bags per PU ensemble (kept modest for the sweep).
            random_state: Global seed.
        """
        self._dir = features_dir
        self._featurizer = featurizer
        self.n_estimators = n_estimators
        self.random_state = random_state

        self._lit_frame = pd.read_parquet(features_dir / "literature_clean.parquet")
        self._jnj_frame = pd.read_parquet(features_dir / "test_jnj.parquet")
        self._x_lit = self._load(f"literature_clean__{featurizer}.npz")
        self._x_grid = self._load(f"unlabeled_grid__{featurizer}.npz")
        self._x_jnj = self._load(f"test_jnj__{featurizer}.npz")

        self.protocol = FrozenEvalProtocol(self._jnj_frame)
        self.protocol.assert_training_clean(self._lit_frame)

        self._pos_mask = self._lit_frame[schema.LABEL].to_numpy() == 1
        self._x_pos = self._x_lit[self._pos_mask]
        self._y_lit = self._lit_frame[schema.LABEL].to_numpy().astype(int)

        self._val_idx = self.protocol.val_index
        self._test_idx = self.protocol.test_index
        self._x_val, self._y_val = self._x_jnj[self._val_idx], self.protocol.labels("val")
        self._x_test, self._y_test = self._x_jnj[self._test_idx], self.protocol.labels("test")

        # One covariate-shift discriminator (literature marginal vs JnJ-grid marginal) reused by
        # every importance-weighted arm; its AUC is the headline domain-shortcut diagnostic.
        self._discriminator = DomainDiscriminator(random_state=random_state).fit(
            self._x_lit, self._x_grid
        )
        self.domain_auc: float = self._discriminator.domain_auc_

    def _load(self, name: str) -> np.ndarray:
        """Load the ``X`` matrix from a cached ``.npz`` feature file."""
        with np.load(self._dir / name) as data:
            matrix: np.ndarray = data["X"].astype(np.float32)
        return matrix

    # -- Base learners ------------------------------------------------------

    def _catboost(self) -> CatBoostFeasibilityModel:
        """Fresh modest CatBoost base learner (shared config across arms)."""
        return CatBoostFeasibilityModel(
            iterations=400, depth=6, learning_rate=0.05, random_state=self.random_state
        )

    def _fit_supervised(self, x_train: np.ndarray, weights: np.ndarray | None) -> Scorer:
        """Train a literature-supervised classifier (positives + measured negatives)."""
        model = self._catboost().fit(x_train, self._y_lit, sample_weight=weights)
        return model.predict_proba

    def _fit_pu(self, x_pos: np.ndarray, weights: np.ndarray | None) -> Scorer:
        """Train a PU-bagging learner: positives vs the unlabeled JnJ grid."""
        features = np.vstack([x_pos, self._x_grid])
        selection = np.concatenate([np.ones(len(x_pos)), np.zeros(len(self._x_grid))])
        sample_weight = None
        if weights is not None:
            sample_weight = np.concatenate([weights, np.ones(len(self._x_grid))])
        learner = BaggingPULearner(
            self._catboost, n_estimators=self.n_estimators, random_state=self.random_state
        )
        learner.fit(features, selection, sample_weight=sample_weight)
        return learner.predict_proba

    # -- Arm assembly -------------------------------------------------------

    def _coral_source(self, source: np.ndarray) -> np.ndarray:
        """Align a literature source matrix to the JnJ-grid distribution via CORAL."""
        return CoralAligner().fit(source, self._x_grid).transform(source)

    def run(self, *, compute_test: bool = True) -> list[ArmResult]:
        """Fit and evaluate every arm; return results (VAL always, TEST if requested).

        Args:
            compute_test: If ``True``, also score every arm on TEST (the single final pass).

        Returns:
            One :class:`ArmResult` per arm, in a stable order.
        """
        pos_weights = self._discriminator.importance_weights(self._x_pos)
        lit_weights = self._discriminator.importance_weights(self._x_lit)
        ess_pos = DomainDiscriminator.effective_sample_size(pos_weights)
        ess_lit = DomainDiscriminator.effective_sample_size(lit_weights)

        x_pos_coral = self._coral_source(self._x_pos)
        x_lit_coral = self._coral_source(self._x_lit)

        # (name, family, fitted scorer, notes)
        arms: list[tuple[str, str, Scorer, dict[str, float]]] = [
            ("pu_naive", "pu", self._fit_pu(self._x_pos, None), {}),
            ("pu_coral", "pu", self._fit_pu(x_pos_coral, None), {}),
            ("pu_iw", "pu", self._fit_pu(self._x_pos, pos_weights), {"ess": ess_pos}),
            ("pu_coral_iw", "pu", self._fit_pu(x_pos_coral, pos_weights), {"ess": ess_pos}),
            ("sup_naive", "supervised", self._fit_supervised(self._x_lit, None), {}),
            ("sup_coral", "supervised", self._fit_supervised(x_lit_coral, None), {}),
            (
                "sup_iw",
                "supervised",
                self._fit_supervised(self._x_lit, lit_weights),
                {"ess": ess_lit},
            ),
        ]

        results: list[ArmResult] = []
        for name, family, scorer, notes in arms:
            notes = {**notes, "domain_auc": self.domain_auc}
            val = feasibility_metrics(self._y_val, scorer(self._x_val))
            result = ArmResult(name=name, family=family, val=val, notes=notes)
            if compute_test:
                result.test = feasibility_metrics(self._y_test, scorer(self._x_test))
            results.append(result)
        return results

    def summary(self) -> dict[str, object]:
        """Return split + diagnostic provenance for logging."""
        return {
            "featurizer": self._featurizer,
            "n_estimators": self.n_estimators,
            "domain_auc_lit_vs_grid": self.domain_auc,
            "n_positives": int(self._pos_mask.sum()),
            "n_literature": len(self._x_lit),
            "n_grid_unlabeled": len(self._x_grid),
            **self.protocol.summary(),
        }
