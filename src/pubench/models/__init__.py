"""PU-model extension point (ABC + registry) and built-in models."""

from __future__ import annotations

from pubench.models import (  # noqa: F401  (imports register the models)
    bagging_pu,
    elkan_noto,
    iw_pu,
    nnpu,
    sar,
    spy,
    supervised,
)
from pubench.models.bagging_pu import BaggingPU
from pubench.models.base import MODELS, LabeledArrays, PUModel, Requires, build_model
from pubench.models.elkan_noto import ElkanNotoClassifier
from pubench.models.iw_pu import ImportanceWeightedPU
from pubench.models.nnpu import NonNegativePU
from pubench.models.sar import PropensityIPS, SAREM
from pubench.models.spy import SpyTwoStep
from pubench.models.supervised import SupervisedBaseline

__all__ = [
    "MODELS",
    "BaggingPU",
    "ElkanNotoClassifier",
    "ImportanceWeightedPU",
    "LabeledArrays",
    "NonNegativePU",
    "PropensityIPS",
    "PUModel",
    "Requires",
    "SAREM",
    "SpyTwoStep",
    "SupervisedBaseline",
    "build_model",
]
