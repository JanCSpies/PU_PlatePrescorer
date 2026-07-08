"""Featurizer extension point: turn reaction records into a numeric matrix."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from pubench.registry import Registry

#: Registry of featurizers, keyed by ``name``; populated by the ``@FEATURIZERS.register`` decorator.
FEATURIZERS: Registry[Featurizer] = Registry("featurizer")


class Featurizer(ABC):
    """Abstract base class mapping reaction records to a dense feature matrix.

    Subclasses declare a class-level ``name`` and register themselves with
    :data:`FEATURIZERS`. A featurizer is *stateful* only if it needs to fit vocabulary from
    data; fingerprint featurizers are effectively stateless but still honour ``fit``.

    Attributes:
    ----------
    name:
        Unique registry key for the featurizer.
    """

    name: ClassVar[str]

    @abstractmethod
    def fit(self, records: pd.DataFrame) -> Featurizer:
        """Fit any data-dependent state (vocabularies, scalers) on ``records``.

        Parameters
        ----------
        records:
            Reaction rows with the canonical schema columns.

        Returns:
        -------
        Featurizer
            ``self``, fitted.
        """

    @abstractmethod
    def transform(self, records: pd.DataFrame) -> np.ndarray:
        """Transform ``records`` into a ``(n_rows, n_features)`` float array."""

    def fit_transform(self, records: pd.DataFrame) -> np.ndarray:
        """Fit on ``records`` then transform them (convenience)."""
        return self.fit(records).transform(records)


def build_featurizer(name: str, params: dict[str, Any] | None = None) -> Featurizer:
    """Instantiate a registered featurizer by name.

    Parameters
    ----------
    name:
        Registry key (e.g. ``"ecfp"``).
    params:
        Keyword arguments forwarded to the featurizer constructor.

    Returns:
    -------
    Featurizer
        The constructed featurizer.
    """
    return FEATURIZERS.get(name)(**(params or {}))
