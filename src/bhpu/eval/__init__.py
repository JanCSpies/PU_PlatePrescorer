"""Evaluation: Neves OOD folds, feasibility metrics, and plate hit-rate uplift."""

from __future__ import annotations

from bhpu.eval.folds import NevesFoldReader, OODFold
from bhpu.eval.metrics import FeasibilityMetrics, feasibility_metrics
from bhpu.eval.plate import PlateUpliftResult, plate_hit_rate_uplift
from bhpu.eval.protocol import FrozenEvalProtocol

__all__ = [
    "FeasibilityMetrics",
    "FrozenEvalProtocol",
    "NevesFoldReader",
    "OODFold",
    "PlateUpliftResult",
    "feasibility_metrics",
    "plate_hit_rate_uplift",
]
