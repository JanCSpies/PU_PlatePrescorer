"""Spy-based two-step PU learning (reliable-negative extraction; the PAYN baseline)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from catboost import CatBoostClassifier

from pubench.models.base import MODELS, LabeledArrays, PUModel, Requires


@MODELS.register("spy")
class SpyTwoStep(PUModel):
    """Spy technique (Liu et al., 2002) for reliable-negative extraction, then supervised fit.

    Step 1 plants a random subset of the positives (the *spies*) into the unlabeled pool and trains
    a classifier of P (minus spies) vs U (plus spies). The spy scores calibrate a threshold: an
    unlabeled point scoring below the ``spy_percentile``-th percentile of the spies is deemed a
    **reliable negative**, since it looks less positive than known-hidden positives. Step 2 trains
    the final classifier on P (positive) vs the extracted reliable negatives.

    Assumptions
    -----------
    Assumes the spies are representative of the positives hidden in U (a SCAR-flavoured
    assumption). This is the PAYN-style baseline, for comparison against the shift-aware methods.

    Parameters
    ----------
    spy_fraction:
        Fraction of P used as spies.
    spy_percentile:
        Percentile of the spy score distribution used as the reliable-negative threshold.
    iterations, depth, learning_rate:
        CatBoost hyperparameters for both stages.
    seed:
        Global seed.
    """

    name: ClassVar[str] = "spy"
    requires: ClassVar[Requires] = Requires()

    def __init__(
        self,
        *,
        spy_fraction: float = 0.15,
        spy_percentile: float = 10.0,
        iterations: int = 200,
        depth: int = 6,
        learning_rate: float = 0.05,
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed)
        self.spy_fraction = spy_fraction
        self.spy_percentile = spy_percentile
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.n_reliable_negatives_: int = 0
        self._final: CatBoostClassifier | None = None

    def _classifier(self) -> CatBoostClassifier:
        """Return a fresh, quietly-configured CatBoost classifier."""
        return CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            random_seed=self.seed,
            verbose=False,
            allow_writing_files=False,
        )

    def fit(  # noqa: D102 (inherited)
        self, P: np.ndarray, U: np.ndarray, *, val: LabeledArrays | None = None
    ) -> SpyTwoStep:
        rng = np.random.default_rng(self.seed)
        n_spies = max(1, round(self.spy_fraction * len(P)))
        spy_idx = rng.choice(len(P), size=n_spies, replace=False)
        spy_mask = np.zeros(len(P), dtype=bool)
        spy_mask[spy_idx] = True
        p_train, spies = P[~spy_mask], P[spy_mask]

        # Step 1: P_train (positive) vs U + spies (unlabeled -> negative).
        stage1 = self._classifier()
        x1 = np.vstack([p_train, U, spies])
        y1 = np.concatenate([np.ones(len(p_train)), np.zeros(len(U) + len(spies))]).astype(np.int64)
        stage1.fit(x1, y1)
        spy_scores = stage1.predict_proba(spies)[:, 1]
        threshold = float(np.percentile(spy_scores, self.spy_percentile))
        u_scores = stage1.predict_proba(U)[:, 1]
        reliable = U[u_scores < threshold]
        self.n_reliable_negatives_ = len(reliable)
        if len(reliable) == 0:  # fall back to the lowest-scoring decile of U
            reliable = U[np.argsort(u_scores)[: max(1, len(U) // 10)]]
            self.n_reliable_negatives_ = len(reliable)

        # Step 2: all P (positive) vs reliable negatives.
        self._final = self._classifier()
        x2 = np.vstack([P, reliable])
        y2 = np.concatenate([np.ones(len(P)), np.zeros(len(reliable))]).astype(np.int64)
        self._final.fit(x2, y2)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: D102 (inherited)
        if self._final is None:
            raise RuntimeError("SpyTwoStep.predict_proba called before fit.")
        proba: np.ndarray = self._final.predict_proba(X)[:, 1]
        return proba
