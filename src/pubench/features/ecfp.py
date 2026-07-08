"""ECFP (Morgan fingerprint) featurizer over substrates and reaction conditions."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from pubench.data import schema
from pubench.features.base import FEATURIZERS, Featurizer


@FEATURIZERS.register("ecfp")
class ECFPFeaturizer(Featurizer):
    """Concatenated Morgan fingerprints of the substrates and (optionally) the conditions.

    Each SMILES column is hashed to a ``n_bits`` Morgan (ECFP) bit vector and the per-column
    blocks are concatenated (``aggregation="concat"``); an unparseable or missing SMILES yields
    a zero block. When ``use_conditions`` is true, the ligand/base/solvent fingerprints and a
    small one-hot over the catalytic metal are appended — this is the substrate ⊕ condition
    representation the condition-aware study targets.

    Parameters
    ----------
    radius:
        Morgan radius (ECFP diameter = ``2 * radius``).
    n_bits:
        Length of each per-molecule fingerprint block.
    use_conditions:
        If true, append ligand/base/solvent fingerprints and the metal one-hot.
    metal_vocab:
        Catalytic metals encoded as a one-hot block (order fixed for reproducibility).
    aggregation:
        Reaction -> vector strategy. ``"concat"`` concatenates per-column blocks (only strategy
        implemented; ``"rxn_diff"`` is reserved for a future reaction-difference fingerprint).

    Notes:
    -----
    This is a deliberately simple, transfer-blind representation. ADR-005 in the repo shows that
    *structural* condition fingerprints can act as a domain tag and hurt naive transfer; the point
    of the benchmark is to see which PU algorithms cope with that. Descriptor featurizers are a
    planned pluggable alternative.
    """

    name: ClassVar[str] = "ecfp"

    def __init__(
        self,
        *,
        radius: int = 2,
        n_bits: int = 2048,
        use_conditions: bool = True,
        metal_vocab: tuple[str, ...] = ("Pd", "Cu", "Ni"),
        aggregation: str = "concat",
    ) -> None:
        if aggregation != "concat":
            raise NotImplementedError(f"aggregation={aggregation!r} is not implemented yet.")
        self.radius = radius
        self.n_bits = n_bits
        self.use_conditions = use_conditions
        self.metal_vocab = metal_vocab
        self.aggregation = aggregation
        self._generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        self._cache: dict[str, np.ndarray] = {}
        self._substrate_columns: tuple[str, ...] = (schema.ARYL_SMILES, schema.AMINE_SMILES)
        self._condition_columns: tuple[str, ...] = schema.CONDITION_SMILES_COLUMNS

    def fit(self, records: pd.DataFrame) -> ECFPFeaturizer:  # noqa: D102 (inherited)
        required = set(self._substrate_columns)
        if self.use_conditions:
            required |= set(self._condition_columns) | {schema.METAL}
        missing = required - set(records.columns)
        if missing:
            raise ValueError(f"ECFPFeaturizer: records missing columns {sorted(missing)}.")
        return self

    def transform(self, records: pd.DataFrame) -> np.ndarray:  # noqa: D102 (inherited)
        blocks: list[np.ndarray] = [self._fp_column(records[c]) for c in self._substrate_columns]
        if self.use_conditions:
            blocks.extend(self._fp_column(records[c]) for c in self._condition_columns)
            blocks.append(self._metal_onehot(records[schema.METAL]))
        return np.hstack(blocks).astype(np.float32)

    @property
    def n_features(self) -> int:
        """Total feature-vector width for the current configuration."""
        n_smiles_blocks = len(self._substrate_columns) + (
            len(self._condition_columns) if self.use_conditions else 0
        )
        metal = len(self.metal_vocab) if self.use_conditions else 0
        return n_smiles_blocks * self.n_bits + metal

    def _fp_column(self, values: pd.Series) -> np.ndarray:
        """Fingerprint one SMILES column into a ``(n_rows, n_bits)`` array (0 block if invalid)."""
        return np.vstack([self._fingerprint(v) for v in values])

    def _fingerprint(self, smiles: object) -> np.ndarray:
        """Return the cached Morgan bit vector for a SMILES (zeros if missing/unparseable)."""
        if not isinstance(smiles, str) or not smiles:
            return np.zeros(self.n_bits, dtype=np.uint8)
        cached = self._cache.get(smiles)
        if cached is not None:
            return cached
        mol = Chem.MolFromSmiles(smiles)
        fp = (
            np.zeros(self.n_bits, dtype=np.uint8)
            if mol is None
            else self._generator.GetFingerprintAsNumPy(mol)
        )
        self._cache[smiles] = fp
        return fp

    def _metal_onehot(self, values: pd.Series) -> np.ndarray:
        """One-hot encode the catalytic metal against :attr:`metal_vocab` (unknown -> all zero)."""
        index = {metal: i for i, metal in enumerate(self.metal_vocab)}
        out = np.zeros((len(values), len(self.metal_vocab)), dtype=np.uint8)
        for row, value in enumerate(values):
            for token in str(value).split("|") if isinstance(value, str) else ():
                if token in index:
                    out[row, index[token]] = 1
        return out
