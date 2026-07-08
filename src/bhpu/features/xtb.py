"""Semiempirical GFN2-xTB electronic descriptors per substrate (the WS4 QM feature arm).

Fingerprints encode *connectivity*; they cannot see the electronics that actually drive a
Buchwald–Hartwig coupling — the aryl-halide's electrophilicity and the amine's nucleophilicity.
This module computes fast GFN2-xTB descriptors (via ``tblite``) for each substrate and exposes them
as a reaction featuriser, so we can test whether quantum descriptors add lift over ECFP.

Per substrate we embed a single 3D conformer (RDKit ETKDG + MMFF), run one GFN2-xTB single point,
and extract a compact, physically meaningful vector: heavy-atom count, energy per atom, HOMO, LUMO,
HOMO–LUMO gap, dipole magnitude, and partial-charge summaries. Results are cached by canonical
SMILES (computing once per unique molecule), so featurising the grid/JnJ/literature sets — which
share the same substrates — is cheap after a one-off precompute.
"""

from __future__ import annotations

import contextlib
from typing import Self

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from bhpu.chem.canonical import canonical_smiles
from bhpu.data import schema
from bhpu.features.base import ReactionFeaturizer

#: Ordered descriptor names (one block per substrate).
XTB_DESCRIPTORS: tuple[str, ...] = (
    "n_heavy",
    "energy_per_atom",
    "homo",
    "lumo",
    "gap",
    "dipole",
    "q_min",
    "q_max",
    "q_absmean",
)
_BOHR_PER_ANGSTROM = 1.8897259886


def compute_xtb_descriptors(smiles: str, *, seed: int = 42) -> np.ndarray | None:
    """Compute GFN2-xTB electronic descriptors for one molecule.

    Args:
        smiles: Substrate SMILES.
        seed: RNG seed for conformer embedding (fixed for reproducibility).

    Returns:
        A ``(len(XTB_DESCRIPTORS),)`` float array, or ``None`` if parsing / embedding /
        the xTB single point fails.
    """
    from tblite.interface import Calculator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        return None
    with contextlib.suppress(RuntimeError, ValueError):
        AllChem.MMFFOptimizeMolecule(mol)  # keep the raw embedding if MMFF fails

    numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    coords = np.asarray(mol.GetConformer().GetPositions()) * _BOHR_PER_ANGSTROM
    try:
        calc = Calculator("GFN2-xTB", numbers, coords)
        calc.set("verbosity", 0)
        result = calc.singlepoint()
    except Exception:
        return None

    charges = np.asarray(result.get("charges"))
    orbital_energies = np.asarray(result.get("orbital-energies"))
    occupations = np.asarray(result.get("orbital-occupations"))
    occupied = occupations > 0.5
    if not occupied.any() or occupied.all():
        return None
    homo = float(orbital_energies[occupied].max())
    lumo = float(orbital_energies[~occupied].min())
    n_heavy = int((numbers > 1).sum())
    return np.array(
        [
            n_heavy,
            float(result.get("energy")) / max(len(numbers), 1),
            homo,
            lumo,
            lumo - homo,
            float(np.linalg.norm(result.get("dipole"))),
            float(charges.min()),
            float(charges.max()),
            float(np.abs(charges).mean()),
        ],
        dtype=np.float64,
    )


class XTBDescriptorFeaturizer(ReactionFeaturizer):
    """Per-substrate GFN2-xTB descriptors (aryl ⊕ amine), backed by a SMILES cache.

    Attributes:
        substrate_columns: SMILES columns to describe and concatenate.
    """

    def __init__(
        self,
        cache: dict[str, np.ndarray] | None = None,
        *,
        substrate_columns: tuple[str, ...] = schema.SUBSTRATE_COLUMNS,
    ) -> None:
        """Initialise the featuriser.

        Args:
            cache: Optional prebuilt ``canonical SMILES -> descriptor vector`` cache (from
                :meth:`load_cache`). Missing substrates are computed lazily (slow) at transform.
            substrate_columns: Ordered SMILES columns to describe.
        """
        self._cache: dict[str, np.ndarray] = cache or {}
        self.substrate_columns = substrate_columns

    @classmethod
    def load_cache(cls, path: str) -> dict[str, np.ndarray]:
        """Load a precomputed descriptor cache from a parquet (``smiles`` + descriptor columns)."""
        frame = pd.read_parquet(path)
        values = frame[list(XTB_DESCRIPTORS)].to_numpy(dtype=np.float64)
        return {smiles: values[i] for i, smiles in enumerate(frame["smiles"])}

    def fit(self, frame: pd.DataFrame) -> Self:
        """No-op (descriptors are content-defined); present for interface parity."""
        return self

    @property
    def feature_names(self) -> list[str]:
        """Descriptor names, prefixed per substrate column."""
        return [
            f"{column}__{name}" for column in self.substrate_columns for name in XTB_DESCRIPTORS
        ]

    def _descriptor(self, smiles: object) -> np.ndarray:
        """Return the cached/computed descriptor for a SMILES (zeros on failure)."""
        canonical = canonical_smiles(smiles) if isinstance(smiles, str) else None
        if canonical is None:
            return np.zeros(len(XTB_DESCRIPTORS))
        cached = self._cache.get(canonical)
        if cached is not None:
            return cached
        computed = compute_xtb_descriptors(canonical)
        vector = computed if computed is not None else np.zeros(len(XTB_DESCRIPTORS))
        self._cache[canonical] = vector
        return vector

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Return ``(n_rows, len(substrate_columns) * len(XTB_DESCRIPTORS))`` descriptors."""
        blocks = [
            np.vstack([self._descriptor(s) for s in frame[column]])
            for column in self.substrate_columns
        ]
        return np.hstack(blocks).astype(np.float32)
