"""PU-model extension point (ABC + registry) and built-in models."""

from __future__ import annotations

from pubench.models import (  # noqa: F401  (imports register the models)
    iw_pu,
    nnpu,
    spy,
    supervised,
)
from pubench.models.base import MODELS, LabeledArrays, PUModel, Requires, build_model
from pubench.models.iw_pu import ImportanceWeightedPU
from pubench.models.nnpu import NonNegativePU
from pubench.models.spy import SpyTwoStep
from pubench.models.supervised import SupervisedBaseline

__all__ = [
    "MODELS",
    "ImportanceWeightedPU",
    "LabeledArrays",
    "NonNegativePU",
    "PUModel",
    "Requires",
    "SpyTwoStep",
    "SupervisedBaseline",
    "build_model",
]
