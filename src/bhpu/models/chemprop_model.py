"""ChemProp (D-MPNN) graph-model pathway (optional; ``models`` extra).

Unlike the fingerprint-based :class:`~bhpu.models.base.FeasibilityModel` learners,
ChemProp is a *graph* model: it consumes SMILES directly and learns its own
representation with a directed message-passing neural network. It therefore takes a
list of SMILES rather than a precomputed feature matrix, so it is exposed with its
own fit/predict signature instead of subclassing ``FeasibilityModel``.

This is a thin, lazily-imported wrapper delimiting the integration point; the heavy
Torch/ChemProp dependency is only imported when the class is actually used. It is a
planned benchmark arm (not exercised in the core CPU pipeline).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np


class ChempropFeasibilityModel:
    """Directed-MPNN feasibility classifier over reaction/molecule SMILES.

    Attributes:
        epochs: Training epochs.
        depth: Message-passing depth.
        hidden_size: Hidden dimension of the MPNN.
    """

    def __init__(self, *, epochs: int = 30, depth: int = 3, hidden_size: int = 300) -> None:
        """Initialise the ChemProp wrapper.

        Args:
            epochs: Number of training epochs.
            depth: Number of message-passing steps.
            hidden_size: MPNN hidden dimension.
        """
        self.epochs = epochs
        self.depth = depth
        self.hidden_size = hidden_size
        self._model: object | None = None

    @staticmethod
    def _require_chemprop() -> None:
        """Raise a helpful error if ChemProp is not installed."""
        try:
            import chemprop  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "ChempropFeasibilityModel requires the 'models' extra: `uv sync --extra models`."
            ) from exc

    def fit(self, smiles: Sequence[str], labels: np.ndarray) -> Self:
        """Train the D-MPNN on ``(smiles, labels)``.

        Args:
            smiles: Reaction or molecule SMILES, one per sample.
            labels: Binary feasibility labels.

        Returns:
            ``self``.

        Raises:
            ImportError: If the optional ``chemprop`` dependency is missing.
            NotImplementedError: Training loop is intentionally deferred until this
                arm is activated (kept as a documented integration point).
        """
        self._require_chemprop()
        raise NotImplementedError(
            "ChemProp training arm is a planned benchmark; wire up chemprop v2 "
            "MoleculeDataset + trainer here when activating this pathway."
        )

    def predict_proba(self, smiles: Sequence[str]) -> np.ndarray:
        """Return ``P(feasible)`` for ``smiles`` (requires a fitted model)."""
        self._require_chemprop()
        raise NotImplementedError("Activate the ChemProp arm before predicting.")
