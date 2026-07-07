"""Abstract base class for reaction featurisers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd


class ReactionFeaturizer(ABC):
    """Turn a DataFrame of reactions into a dense numeric feature matrix.

    Subclasses implement :meth:`transform`; stateless featurisers may rely on the
    default no-op :meth:`fit`. The contract is deliberately close to scikit-learn's
    transformer API so featurisers compose and slot into pipelines.
    """

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Names of the produced feature columns, in output order."""

    def fit(self, frame: pd.DataFrame) -> Self:
        """Fit any internal state on ``frame`` (default: no-op).

        Args:
            frame: Reactions to fit on.

        Returns:
            ``self``, to allow chaining.
        """
        return self

    @abstractmethod
    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Transform ``frame`` into a ``(n_rows, n_features)`` float array."""

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Fit on ``frame`` then transform it.

        Args:
            frame: Reactions to fit on and transform.

        Returns:
            The feature matrix for ``frame``.
        """
        return self.fit(frame).transform(frame)
