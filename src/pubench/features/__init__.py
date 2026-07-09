"""Featurizer extension point (ABC + registry) and built-in featurizers."""

from __future__ import annotations

from pubench.features import (  # noqa: F401  (imports register the featurizers)
    composite,
    condition_descriptors,
    drfp,
    ecfp,
    reduced_ecfp,
)
from pubench.features.base import FEATURIZERS, Featurizer, build_featurizer
from pubench.features.composite import CompositeFeaturizer
from pubench.features.condition_descriptors import ConditionDescriptorFeaturizer
from pubench.features.drfp import DRFPFeaturizer
from pubench.features.ecfp import ECFPFeaturizer
from pubench.features.reduced_ecfp import ReducedECFPFeaturizer

__all__ = [
    "FEATURIZERS",
    "CompositeFeaturizer",
    "ConditionDescriptorFeaturizer",
    "DRFPFeaturizer",
    "ECFPFeaturizer",
    "Featurizer",
    "ReducedECFPFeaturizer",
    "build_featurizer",
]
