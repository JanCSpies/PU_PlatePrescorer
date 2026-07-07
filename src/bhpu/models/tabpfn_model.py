"""TabPFN-backed feasibility model (optional; install with the ``models`` extra).

TabPFN is a pretrained tabular foundation model that is strong in the limited-label
regime we care about (few positives). It is bounded in the number of features and
samples it ingests, so high-dimensional fingerprints are optionally reduced with PCA
before being handed to TabPFN. The ``tabpfn`` package (and its Torch dependency) is
imported lazily so the core pipeline does not require it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
from sklearn.decomposition import PCA

from bhpu.models.base import FeasibilityModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tabpfn import TabPFNClassifier


class TabPFNFeasibilityModel(FeasibilityModel):
    """Pretrained tabular-transformer feasibility classifier (TabPFN v2).

    Attributes:
        max_features: If the input width exceeds this, a PCA reduces it to this many
            components (TabPFN performs best within its trained feature budget).
        device: Torch device string passed to TabPFN (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        *,
        max_features: int = 100,
        device: str = "cpu",
        random_state: int = 0,
    ) -> None:
        """Initialise the TabPFN wrapper.

        Args:
            max_features: Feature-count cap; PCA is applied above it.
            device: Torch device for inference.
            random_state: Seed for PCA and TabPFN.
        """
        self.max_features = max_features
        self.device = device
        self.random_state = random_state
        self._pca: PCA | None = None
        self._model: TabPFNClassifier | None = None

    def _build_model(self) -> TabPFNClassifier:
        """Import and construct a TabPFN classifier (lazy; requires ``models`` extra)."""
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "TabPFNFeasibilityModel requires the 'models' extra: `uv sync --extra models`."
            ) from exc
        return TabPFNClassifier(device=self.device, random_state=self.random_state)

    def _maybe_reduce(self, features: np.ndarray, *, fit: bool) -> np.ndarray:
        """Optionally PCA-reduce ``features`` to ``max_features`` columns."""
        if features.shape[1] <= self.max_features:
            return features
        if fit:
            self._pca = PCA(n_components=self.max_features, random_state=self.random_state)
            reduced: np.ndarray = self._pca.fit_transform(features)
            return reduced
        if self._pca is None:
            raise RuntimeError("PCA not fitted; call fit() before predict_proba().")
        transformed: np.ndarray = self._pca.transform(features)
        return transformed

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        """Fit TabPFN (context-set the training data); see :meth:`FeasibilityModel.fit`."""
        self._model = self._build_model()
        reduced = self._maybe_reduce(features, fit=True)
        self._model.fit(reduced, labels)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return ``P(feasible)`` from TabPFN."""
        if self._model is None:
            raise RuntimeError("TabPFNFeasibilityModel must be fit before predict_proba.")
        reduced = self._maybe_reduce(features, fit=False)
        proba: np.ndarray = self._model.predict_proba(reduced)[:, 1]
        return proba
