"""Typed container for the four PU splits and their (optional) featurised matrices."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd


class Split(StrEnum):
    """The four roles a set of reaction records can play in the PU benchmark."""

    P = "P"
    """Positives: literature-mined reactions treated as ``y = 1`` (biased sample)."""
    U = "U"
    """Unlabeled pool: a substrate x condition matrix; a mixture of latent +/- reactions."""
    VAL = "val"
    """Validation split: labelled, for model selection / threshold tuning."""
    TEST = "test"
    """Test split: fully labelled HTE data, for final evaluation only (never trained on)."""


@dataclass(frozen=True, slots=True)
class SplitFrame:
    """Raw reaction records for one split, plus ground truth where it exists.

    Parameters
    ----------
    role:
        Which :class:`Split` this frame represents.
    records:
        Reaction rows with the canonical substrate + condition columns (see
        :mod:`pubench.data.schema`).
    y:
        Ground-truth binary labels aligned to ``records`` (``1`` works, ``0`` does not), or
        ``None`` for P and U where no ground truth exists.
    """

    role: Split
    records: pd.DataFrame
    y: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate that labels, if present, align with the records."""
        if self.y is not None and len(self.y) != len(self.records):
            raise ValueError(
                f"{self.role}: y has {len(self.y)} labels but {len(self.records)} records."
            )

    @property
    def n(self) -> int:
        """Number of reaction records in the split."""
        return len(self.records)


@dataclass
class PUDataset:
    """The four splits together with their featurised matrices.

    Featurisation is filled in later (the runner featurises every split with a shared
    featurizer), so ``features`` starts empty and is populated via :meth:`with_features`.

    Parameters
    ----------
    frames:
        Mapping of each :class:`Split` to its :class:`SplitFrame`. ``P`` and ``U`` are required.
    features:
        Optional mapping of split -> feature matrix, aligned row-wise to each frame's records.
    """

    frames: dict[Split, SplitFrame]
    features: dict[Split, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check that the mandatory PU training splits are present."""
        for required in (Split.P, Split.U):
            if required not in self.frames:
                raise ValueError(f"PUDataset is missing the required {required} split.")

    def frame(self, split: Split) -> SplitFrame:
        """Return the :class:`SplitFrame` for ``split``."""
        return self.frames[split]

    def records(self, split: Split) -> pd.DataFrame:
        """Return the raw records for ``split``."""
        return self.frames[split].records

    def labels(self, split: Split) -> np.ndarray | None:
        """Return the ground-truth labels for ``split`` (``None`` if unlabelled)."""
        return self.frames[split].y

    def has_features(self, split: Split) -> bool:
        """Whether ``split`` has been featurised."""
        return split in self.features

    def matrix(self, split: Split) -> np.ndarray:
        """Return the feature matrix for ``split``.

        Raises:
        ------
        KeyError
            If the split has not been featurised yet.
        """
        if split not in self.features:
            raise KeyError(f"Split {split} has not been featurised.")
        return self.features[split]

    def with_features(self, features: Mapping[Split, np.ndarray]) -> PUDataset:
        """Return a copy of this dataset carrying the given feature matrices."""
        return PUDataset(frames=dict(self.frames), features=dict(features))

    @property
    def available_splits(self) -> list[Split]:
        """The splits present in this dataset, in canonical P/U/val/test order."""
        order = [Split.P, Split.U, Split.VAL, Split.TEST]
        return [s for s in order if s in self.frames]
