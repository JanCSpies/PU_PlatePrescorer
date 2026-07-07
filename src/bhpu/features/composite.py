"""Compose several featurisers into one horizontally-concatenated feature matrix."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd

from bhpu.features.base import ReactionFeaturizer


class CompositeFeaturizer(ReactionFeaturizer):
    """Concatenate the outputs of several featurisers side by side.

    Example:
        ``CompositeFeaturizer([ECFPFeaturizer(), ConditionEncoder()])`` yields
        substrate fingerprints followed by one-hot condition features.

    Attributes:
        featurizers: The component featurisers, applied in order.
    """

    def __init__(self, featurizers: list[ReactionFeaturizer]) -> None:
        """Initialise the composite.

        Args:
            featurizers: Component featurisers whose outputs are concatenated.

        Raises:
            ValueError: If no featurisers are supplied.
        """
        if not featurizers:
            raise ValueError("CompositeFeaturizer requires at least one featuriser.")
        self.featurizers = featurizers

    def fit(self, frame: pd.DataFrame) -> Self:
        """Fit every component featuriser on ``frame``.

        Args:
            frame: Reactions to fit on.

        Returns:
            ``self``.
        """
        for featurizer in self.featurizers:
            featurizer.fit(frame)
        return self

    @property
    def feature_names(self) -> list[str]:
        """Concatenated feature names across all component featurisers."""
        names: list[str] = []
        for featurizer in self.featurizers:
            names.extend(featurizer.feature_names)
        return names

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Transform ``frame`` with each featuriser and horizontally stack.

        Args:
            frame: Reactions to transform.

        Returns:
            A ``(n_rows, sum_of_component_widths)`` float array.
        """
        return np.hstack([featurizer.transform(frame) for featurizer in self.featurizers])
