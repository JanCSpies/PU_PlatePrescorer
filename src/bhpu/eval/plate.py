"""The demo metric: hit-rate uplift of a model-designed plate vs a random plate.

Given feasibility scores and true outcomes for a pool of candidate wells, we pick the
top-B by score (the "model plate"), and compare its realised hit rate to a random
plate (bootstrapped) and to the oracle (best possible top-B). This quantifies, with
**zero new experiments**, how many more usable/testable products a model-designed
plate yields — the project's headline impact number.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel


class PlateUpliftResult(BaseModel):
    """Hit-rate uplift for a single plate budget.

    Attributes:
        budget: Number of wells selected (plate size B).
        base_rate: Pool-wide feasible fraction.
        model_hit_rate: Feasible fraction among the top-B by predicted score.
        random_hit_rate_mean: Mean feasible fraction of a random B-well plate.
        random_hit_rate_ci: 95 % bootstrap CI for the random plate hit rate.
        oracle_hit_rate: Best achievable feasible fraction for B wells.
        uplift_vs_random: ``model_hit_rate / random_hit_rate_mean``.
        enrichment_vs_base: ``model_hit_rate / base_rate``.
        extra_hits_vs_random: Extra feasible wells vs a random plate (``Δrate × B``).
    """

    budget: int
    base_rate: float
    model_hit_rate: float
    random_hit_rate_mean: float
    random_hit_rate_ci: tuple[float, float]
    oracle_hit_rate: float
    uplift_vs_random: float
    enrichment_vs_base: float
    extra_hits_vs_random: float


def plate_hit_rate_uplift(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    budgets: tuple[int, ...] = (96, 384),
    n_bootstrap: int = 1000,
    random_state: int = 0,
) -> list[PlateUpliftResult]:
    """Compute hit-rate uplift for each plate budget.

    Args:
        y_true: ``(n,)`` binary true feasibility of the candidate wells.
        y_score: ``(n,)`` predicted feasibility scores for the same wells.
        budgets: Plate sizes to evaluate (skipped if larger than the pool).
        n_bootstrap: Random-plate bootstrap resamples.
        random_state: Seed for the bootstrap.

    Returns:
        One :class:`PlateUpliftResult` per feasible budget.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    base_rate = float(y_true.mean())
    ranked = np.argsort(y_score)[::-1]  # descending score
    n_positive = int(y_true.sum())

    results: list[PlateUpliftResult] = []
    for budget in budgets:
        if budget > n:
            continue
        model_hits = float(y_true[ranked[:budget]].mean())
        oracle_hits = min(n_positive, budget) / budget
        random_means = np.array(
            [y_true[rng.choice(n, size=budget, replace=False)].mean() for _ in range(n_bootstrap)]
        )
        random_mean = float(random_means.mean())
        ci = (float(np.quantile(random_means, 0.025)), float(np.quantile(random_means, 0.975)))
        results.append(
            PlateUpliftResult(
                budget=budget,
                base_rate=base_rate,
                model_hit_rate=model_hits,
                random_hit_rate_mean=random_mean,
                random_hit_rate_ci=ci,
                oracle_hit_rate=oracle_hits,
                uplift_vs_random=model_hits / random_mean if random_mean > 0 else float("nan"),
                enrichment_vs_base=model_hits / base_rate if base_rate > 0 else float("nan"),
                extra_hits_vs_random=(model_hits - random_mean) * budget,
            )
        )
    return results
