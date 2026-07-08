"""PU-model extension point (ABC + registry) and built-in models."""

from __future__ import annotations

from pubench.models import nnpu, supervised  # noqa: F401  (import registers the models)
from pubench.models.base import MODELS, LabeledArrays, PUModel, Requires, build_model
from pubench.models.nnpu import NonNegativePU
from pubench.models.supervised import SupervisedBaseline

__all__ = [
    "MODELS",
    "LabeledArrays",
    "NonNegativePU",
    "PUModel",
    "Requires",
    "SupervisedBaseline",
    "build_model",
]
