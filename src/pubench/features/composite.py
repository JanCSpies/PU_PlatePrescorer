"""Composite featurizer: concatenate several featurizers into one block.

Used to combine the condensed substrate representation with the physical condition descriptors
(e.g. ``reduced_ecfp`` ⊕ ``condition_descriptors``) into a single substrate+condition feature
matrix, which is the representation the condition-aware transfer study targets. Each component is
fit and transformed on the same records; the outputs are horizontally stacked in declared order.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from pubench.features.base import FEATURIZERS, Featurizer, build_featurizer


@FEATURIZERS.register("composite")
class CompositeFeaturizer(Featurizer):
    """Concatenate the outputs of several sub-featurizers.

    Parameters
    ----------
    components:
        Ordered list of ``{"name": str, "params": dict}`` specs, each naming a registered
        featurizer. Every component is fit on the pool the runner supplies (P+U) and transformed
        per split; the resulting matrices are concatenated column-wise.
    """

    name: ClassVar[str] = "composite"

    def __init__(self, *, components: list[dict[str, Any]]) -> None:
        if not components:
            raise ValueError("CompositeFeaturizer needs at least one component.")
        self.components = components
        self._featurizers: list[Featurizer] = [
            build_featurizer(spec["name"], spec.get("params")) for spec in components
        ]

    def fit(self, records: pd.DataFrame) -> CompositeFeaturizer:  # noqa: D102 (inherited)
        for featurizer in self._featurizers:
            featurizer.fit(records)
        return self

    def transform(self, records: pd.DataFrame) -> np.ndarray:  # noqa: D102 (inherited)
        blocks = [featurizer.transform(records) for featurizer in self._featurizers]
        return np.hstack(blocks).astype(np.float32)
