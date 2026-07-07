"""One-hot encoder for reaction conditions (catalyst / base / solvent names)."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from bhpu.data import schema
from bhpu.features.base import ReactionFeaturizer


class ConditionEncoder(ReactionFeaturizer):
    """One-hot encode the categorical reaction-condition columns.

    Unseen categories at transform time map to an all-zero block for that field
    (``handle_unknown="ignore"``), which is the desired behaviour when scoring a
    source whose exact reagents were not present at fit time.

    Attributes:
        condition_columns: The categorical condition-name columns to encode.
    """

    def __init__(
        self, *, condition_columns: tuple[str, ...] = schema.CONDITION_NAME_COLUMNS
    ) -> None:
        """Initialise the encoder.

        Args:
            condition_columns: Ordered categorical columns to one-hot encode.
        """
        self.condition_columns = condition_columns
        self._encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32
        )
        self._fitted = False

    def fit(self, frame: pd.DataFrame) -> Self:
        """Learn the category vocabulary from ``frame``.

        Args:
            frame: Reactions containing the condition columns.

        Returns:
            ``self``.
        """
        self._encoder.fit(self._select(frame))
        self._fitted = True
        return self

    @property
    def feature_names(self) -> list[str]:
        """Return the one-hot feature names (requires a prior :meth:`fit`)."""
        if not self._fitted:
            raise RuntimeError("ConditionEncoder.feature_names requires fit() first.")
        return [
            f"cond_{name}" for name in self._encoder.get_feature_names_out(self.condition_columns)
        ]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """One-hot encode the condition columns of ``frame``.

        Args:
            frame: Reactions containing the condition columns.

        Returns:
            A ``(n_rows, n_categories)`` float32 array.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self._fitted:
            raise RuntimeError("ConditionEncoder must be fit before transform.")
        result: np.ndarray = self._encoder.transform(self._select(frame))
        return result

    def _select(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the condition columns as strings (NaNs → empty category)."""
        return frame[list(self.condition_columns)].astype("string").fillna("")
