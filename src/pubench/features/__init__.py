"""Featurizer extension point (ABC + registry) and built-in featurizers."""

from __future__ import annotations

from pubench.features import drfp, ecfp  # noqa: F401  (imports register the featurizers)
from pubench.features.base import FEATURIZERS, Featurizer, build_featurizer
from pubench.features.drfp import DRFPFeaturizer
from pubench.features.ecfp import ECFPFeaturizer

__all__ = ["FEATURIZERS", "DRFPFeaturizer", "ECFPFeaturizer", "Featurizer", "build_featurizer"]
