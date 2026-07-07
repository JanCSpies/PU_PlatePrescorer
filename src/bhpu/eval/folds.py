"""Reader for the Neves DRFP + K-means leave-cluster-out OOD folds.

The curated file bakes in the exact out-of-distribution splits from Neves et al.: for
each clustering ``iteration_{i}``, the column ``iteration_{i} train_test`` holds, per
row, a 4-element list of ``"train"``/``"test"`` assignments (four held-out folds per
iteration). Reusing these reproduces the paper's evaluation instead of inventing our
own splits.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class OODFold:
    """A single out-of-distribution train/test split.

    Attributes:
        iteration: Clustering-iteration index (0-based).
        fold: Fold index within the iteration (0-based).
        train_index: Positional row indices of the training rows.
        test_index: Positional row indices of the held-out test rows.
    """

    iteration: int
    fold: int
    train_index: np.ndarray
    test_index: np.ndarray


class NevesFoldReader:
    """Yield the pre-computed Neves OOD folds from the ``iteration_*`` columns.

    Attributes:
        n_iterations: Number of clustering iterations available in the frame.
        n_folds_per_iteration: Folds encoded per iteration (list length per cell).
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        """Index the available iteration columns in ``frame``.

        Args:
            frame: A partition exposing ``iteration_{i} train_test`` columns.

        Raises:
            ValueError: If no ``iteration_* train_test`` columns are present.
        """
        self._columns = sorted(
            (c for c in frame.columns if c.startswith("iteration_") and c.endswith("train_test")),
            key=lambda c: int(c.split("_", 2)[1].split(" ", 1)[0]),
        )
        if not self._columns:
            raise ValueError("No 'iteration_* train_test' columns found in frame.")
        self._assignments = {col: self._parse_column(frame[col]) for col in self._columns}
        self.n_iterations = len(self._columns)
        self.n_folds_per_iteration = self._assignments[self._columns[0]].shape[1]

    def iter_folds(self) -> Iterator[OODFold]:
        """Iterate over every (iteration, fold) OOD split.

        Yields:
            One :class:`OODFold` per fold; a row is *test* in fold ``j`` iff its
            assignment list has ``"test"`` at position ``j``.
        """
        for iteration, column in enumerate(self._columns):
            matrix = self._assignments[column]
            for fold in range(matrix.shape[1]):
                is_test = matrix[:, fold] == "test"
                yield OODFold(
                    iteration=iteration,
                    fold=fold,
                    train_index=np.flatnonzero(~is_test),
                    test_index=np.flatnonzero(is_test),
                )

    @staticmethod
    def _parse_column(series: pd.Series) -> np.ndarray:
        """Parse a ``train_test`` column into an ``(n_rows, n_folds)`` string array."""

        def parse_cell(cell: object) -> list[str]:
            if isinstance(cell, (list, tuple, np.ndarray)):
                return [str(x) for x in cell]
            return [str(x) for x in ast.literal_eval(str(cell))]

        return np.array([parse_cell(cell) for cell in series], dtype=object).astype(str)
