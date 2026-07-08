"""Featurizer extension point (ABC + registry) and built-in featurizers."""

from __future__ import annotations

from pubench.features import ecfp  # noqa: F401  (import registers ECFPFeaturizer)
from pubench.features.base import FEATURIZERS, Featurizer, build_featurizer
from pubench.features.ecfp import ECFPFeaturizer

__all__ = ["FEATURIZERS", "ECFPFeaturizer", "Featurizer", "build_featurizer"]
