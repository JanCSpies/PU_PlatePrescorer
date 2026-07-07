"""Feasibility models: CatBoost (core) plus optional TabPFN and ChemProp wrappers.

Only :class:`CatBoostFeasibilityModel` and the base types are imported eagerly. The
TabPFN and ChemProp wrappers live in their own modules and import their heavy
dependencies lazily; import them directly when the ``models`` extra is installed::

    from bhpu.models.tabpfn_model import TabPFNFeasibilityModel
    from bhpu.models.chemprop_model import ChempropFeasibilityModel
"""

from __future__ import annotations

from bhpu.models.base import FeasibilityModel, ModelFactory
from bhpu.models.boosting import CatBoostFeasibilityModel

__all__ = [
    "CatBoostFeasibilityModel",
    "FeasibilityModel",
    "ModelFactory",
]
