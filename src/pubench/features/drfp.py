"""DRFP (differential reaction fingerprint) featurizer over the substrate->product reaction."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from drfp import DrfpEncoder

from pubench.data import schema
from pubench.features.base import FEATURIZERS, Featurizer


@FEATURIZERS.register("drfp")
class DRFPFeaturizer(Featurizer):
    """Differential Reaction FingerPrint of the ``aryl.amine >> product`` reaction.

    DRFP (Probst et al., 2022) hashes the symmetric difference of circular substructures between
    the reactant and product sides of a reaction SMILES. Here the reaction is built from the
    substrates and the (stored) coupling product, giving a condition-free, reaction-aware
    representation — an alternative to substrate ECFP that avoids the condition-fingerprint
    domain-tag failure. Fingerprints are cached per unique reaction SMILES.

    Parameters
    ----------
    n_bits:
        Folded fingerprint length.
    radius:
        Maximum circular-substructure radius.

    References:
    ----------
    D. Probst, P. Schwaller, J.-L. Reymond, "Reaction classification and yield prediction using the
    differential reaction fingerprint DRFP", Digital Discovery 2022.
    """

    name: ClassVar[str] = "drfp"

    def __init__(self, *, n_bits: int = 2048, radius: int = 3) -> None:
        self.n_bits = n_bits
        self.radius = radius
        self._cache: dict[str, np.ndarray] = {}

    def fit(self, records: pd.DataFrame) -> DRFPFeaturizer:  # noqa: D102 (inherited)
        required = {schema.ARYL_SMILES, schema.AMINE_SMILES, schema.PRODUCT_SMILES}
        missing = required - set(records.columns)
        if missing:
            raise ValueError(f"DRFPFeaturizer: records missing columns {sorted(missing)}.")
        return self

    def transform(self, records: pd.DataFrame) -> np.ndarray:  # noqa: D102 (inherited)
        rxns = [
            self._reaction_smiles(a, n, p)
            for a, n, p in zip(
                records[schema.ARYL_SMILES],
                records[schema.AMINE_SMILES],
                records[schema.PRODUCT_SMILES],
                strict=True,
            )
        ]
        self._encode_uncached(sorted({r for r in rxns if r is not None}))
        rows = [
            self._cache[r] if r is not None else np.zeros(self.n_bits, dtype=np.uint8) for r in rxns
        ]
        return np.vstack(rows).astype(np.float32)

    @property
    def n_features(self) -> int:
        """Feature-vector width (one folded DRFP block)."""
        return self.n_bits

    def _reaction_smiles(self, aryl: object, amine: object, product: object) -> str | None:
        """Build the ``aryl.amine>>product`` reaction SMILES, or ``None`` if any part is missing."""
        if not all(isinstance(s, str) and s for s in (aryl, amine, product)):
            return None
        return f"{aryl}.{amine}>>{product}"

    def _encode_uncached(self, rxns: list[str]) -> None:
        """Encode any reaction SMILES not yet in the cache (batched)."""
        pending = [r for r in rxns if r not in self._cache]
        if not pending:
            return
        fps = DrfpEncoder.encode(pending, n_folded_length=self.n_bits, radius=self.radius)
        for rxn, fp in zip(pending, fps, strict=True):
            self._cache[rxn] = np.asarray(fp, dtype=np.uint8)
