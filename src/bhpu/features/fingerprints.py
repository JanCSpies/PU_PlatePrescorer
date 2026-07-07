"""Fingerprint featurisers: substrate ECFP (Morgan) and reaction DRFP.

Two complementary reaction representations, both recomputed from SMILES (the raw
``DRFP``/``QM`` columns in the source file are truncated string reprs and unusable):

* :class:`ECFPFeaturizer` — concatenated Morgan/ECFP bit vectors of the individual
  substrates (aryl halide, amine). Product-free, so it works directly on the
  enumerated unlabeled space, and condition-agnostic (pair it with a
  :class:`~bhpu.features.conditions.ConditionEncoder`).
* :class:`DRFPFeaturizer` — the Differential Reaction FingerPrint over the assembled
  reaction (optionally including reagents), so a single vector captures both
  substrates and conditions and transfers across data sources.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from drfp import DrfpEncoder
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from bhpu.chem.reaction import ReactionAssembler
from bhpu.data import schema
from bhpu.features.base import ReactionFeaturizer


class ECFPFeaturizer(ReactionFeaturizer):
    """Concatenated Morgan (ECFP) fingerprints of the substrate molecules.

    Attributes:
        radius: Morgan radius (ECFP diameter = ``2 * radius``; ``radius=2`` → ECFP4).
        n_bits: Folded bit-vector length per substrate.
        substrate_columns: SMILES columns to fingerprint and concatenate.
    """

    def __init__(
        self,
        *,
        radius: int = 2,
        n_bits: int = 2048,
        substrate_columns: tuple[str, ...] = schema.SUBSTRATE_COLUMNS,
    ) -> None:
        """Initialise the ECFP featuriser.

        Args:
            radius: Morgan radius.
            n_bits: Bits per substrate fingerprint.
            substrate_columns: Ordered SMILES columns to fingerprint.
        """
        self.radius = radius
        self.n_bits = n_bits
        self.substrate_columns = substrate_columns
        self._generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    @property
    def feature_names(self) -> list[str]:
        """Return ``ecfp_<column>_<bit>`` names for every substrate and bit."""
        return [
            f"ecfp_{column}_{bit}"
            for column in self.substrate_columns
            for bit in range(self.n_bits)
        ]

    def _fingerprint(self, smiles: object) -> np.ndarray:
        """Return the ECFP bit vector of one SMILES (zeros if missing/invalid)."""
        if not isinstance(smiles, str) or not smiles:
            return np.zeros(self.n_bits, dtype=np.float32)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.n_bits, dtype=np.float32)
        fingerprint: np.ndarray = np.asarray(
            self._generator.GetFingerprintAsNumPy(mol), dtype=np.float32
        )
        return fingerprint

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Fingerprint each substrate column and concatenate per row.

        Args:
            frame: Reactions with the configured substrate SMILES columns.

        Returns:
            A ``(n_rows, n_substrates * n_bits)`` float32 array.
        """
        blocks = [
            np.vstack([self._fingerprint(value) for value in frame[column].to_numpy()])
            for column in self.substrate_columns
        ]
        return np.hstack(blocks)


class DRFPFeaturizer(ReactionFeaturizer):
    """Differential Reaction FingerPrint over the assembled reaction SMILES.

    Attributes:
        n_bits: Folded fingerprint length.
        radius: Maximum substructure radius.
        include_reagents: Whether the assembled reaction includes catalyst/base/solvent.
    """

    def __init__(
        self,
        *,
        n_bits: int = 2048,
        radius: int = 3,
        include_reagents: bool = True,
    ) -> None:
        """Initialise the DRFP featuriser.

        Args:
            n_bits: Folded bit-vector length.
            radius: Maximum circular-substructure radius.
            include_reagents: Include reagents on the reactant side of the reaction.
        """
        self.n_bits = n_bits
        self.radius = radius
        self.include_reagents = include_reagents
        self._assembler = ReactionAssembler(include_reagents=include_reagents)

    @property
    def feature_names(self) -> list[str]:
        """Return ``drfp_<bit>`` names."""
        return [f"drfp_{bit}" for bit in range(self.n_bits)]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Assemble reactions and encode them as DRFP vectors.

        Rows whose reaction SMILES cannot be assembled (missing product) yield an
        all-zero vector so the output stays aligned to ``frame``.

        Args:
            frame: Reactions with component + product SMILES columns.

        Returns:
            A ``(n_rows, n_bits)`` float32 array.
        """
        reactions = self._assembler.assemble(frame)
        matrix = np.zeros((len(reactions), self.n_bits), dtype=np.float32)

        valid_indices = [i for i, rxn in enumerate(reactions) if rxn is not None]
        if not valid_indices:
            return matrix

        valid_smiles = [reactions[i] for i in valid_indices]
        encoded = DrfpEncoder.encode(valid_smiles, n_folded_length=self.n_bits, radius=self.radius)
        for position, fingerprint in zip(valid_indices, encoded, strict=True):
            matrix[position] = np.asarray(fingerprint, dtype=np.float32)
        return matrix
