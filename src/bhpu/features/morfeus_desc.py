"""Site-specific *steric* descriptors at the reacting atoms, via morfeus.

Whole-molecule descriptors (``features/xtb.py``) average away the local environment that
actually governs a Buchwald–Hartwig coupling. This featuriser instead measures sterics **at
the reacting atoms**: Sterimol L/B1/B5 and %V_bur/SASA at the aryl C–X carbon (along the
axis the nitrogen approaches), and %V_bur/pyramidalisation/SASA at the amine nitrogen.

Electronics are provided separately by the g-xTB featuriser (``features/gxtb.py``); this
module is deliberately QM-free (pure geometry) so it is fast and dependency-light. Descriptors
cache per unique SMILES, since each substrate's reactive site is partner-independent
(see :mod:`bhpu.chem.sites`).
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from morfeus import SASA, BuriedVolume, Pyramidalization, Sterimol
from rdkit import Chem

from bhpu.chem.canonical import canonical_smiles
from bhpu.chem.sites import (
    aggregate_conformer_descriptors,
    embed_conformers,
    locate_amine_site,
    locate_aryl_site,
)
from bhpu.data import schema
from bhpu.features.base import ReactionFeaturizer

#: Default conformers per substrate for aggregated sterics.
DEFAULT_N_CONFORMERS: int = 20


def _leaving_halide(aryl_smiles: str, carbon_idx: int) -> int | None:
    """Index of the halide leaving atom bonded to a given aryl carbon (``None`` if none)."""
    mol = Chem.MolFromSmiles(aryl_smiles)
    if mol is None or carbon_idx >= mol.GetNumAtoms():
        return None
    return next(
        (
            neighbour.GetIdx()
            for neighbour in mol.GetAtomWithIdx(carbon_idx).GetNeighbors()
            if neighbour.GetSymbol() in ("F", "Cl", "Br", "I")
        ),
        None,
    )


#: Ordered aryl-site steric descriptor names.
ARYL_STERIC_DESCRIPTORS: tuple[str, ...] = (
    "sterimol_L",
    "sterimol_B1",
    "sterimol_B5",
    "vbur_carbon",
    "sasa_carbon",
)
#: Ordered amine-site steric descriptor names.
AMINE_STERIC_DESCRIPTORS: tuple[str, ...] = (
    "vbur_nitrogen",
    "pyramidalization",
    "sasa_nitrogen",
)

_VBUR_RADIUS: float = 3.5


def _aryl_geometry_descriptors(
    elements: list[str], coords: np.ndarray, carbon_idx: int, leaving_idx: int | None
) -> np.ndarray:
    """Aryl-site steric descriptors for a single conformer geometry."""
    carbon_1 = carbon_idx + 1  # morfeus is 1-indexed
    # Sterimol along the leaving-group → carbon axis: the trajectory the nitrogen approaches.
    if leaving_idx is not None:
        try:
            sterimol = Sterimol(elements, coords, leaving_idx + 1, carbon_1)
            length, b1, b5 = sterimol.L_value, sterimol.B_1_value, sterimol.B_5_value
        except (ValueError, np.linalg.LinAlgError):
            length = b1 = b5 = np.nan
    else:
        length = b1 = b5 = np.nan  # pseudohalide: no single leaving atom to define the axis
    vbur = _buried_volume(elements, coords, carbon_1)
    sasa = _atom_sasa(elements, coords, carbon_1)
    return np.array([length, b1, b5, vbur, sasa], dtype=np.float64)


def _amine_geometry_descriptors(
    elements: list[str], coords: np.ndarray, nitrogen_idx: int
) -> np.ndarray:
    """Amine-site steric descriptors for a single conformer geometry."""
    nitrogen_1 = nitrogen_idx + 1
    vbur = _buried_volume(elements, coords, nitrogen_1)
    try:
        pyramid = Pyramidalization(coords, nitrogen_1).P
    except (ValueError, IndexError, np.linalg.LinAlgError):
        pyramid = np.nan
    sasa = _atom_sasa(elements, coords, nitrogen_1)
    return np.array([vbur, pyramid, sasa], dtype=np.float64)


def compute_aryl_conformer_matrix(
    aryl_smiles: str,
    *,
    n_conformers: int = DEFAULT_N_CONFORMERS,
    site_override: int | None = None,
) -> tuple[np.ndarray, list[float]] | None:
    """Per-conformer aryl steric descriptors and MMFF energies (for any aggregation).

    Args:
        aryl_smiles: SMILES of the aryl (pseudo)halide.
        n_conformers: Conformers to embed.
        site_override: Reacting-carbon index to use instead of the template's canonical site
            (e.g. a product-resolved site).

    Returns:
        ``(matrix, energies)`` — ``matrix`` shaped ``(n_conformers, n_aryl_descriptors)`` and
        ``energies`` the per-conformer MMFF energies, or ``None`` on failure.
    """
    if site_override is not None:
        carbon_idx = site_override
        leaving_idx = _leaving_halide(aryl_smiles, site_override)
    else:
        site = locate_aryl_site(aryl_smiles)
        if site is None:
            return None
        carbon_idx, leaving_idx = site
    ensemble = embed_conformers(aryl_smiles, carbon_idx, n_conformers=n_conformers)
    if ensemble is None:
        return None
    matrix = np.vstack(
        [
            _aryl_geometry_descriptors(ensemble.elements, coords, carbon_idx, leaving_idx)
            for coords in ensemble.conformers
        ]
    )
    return matrix, ensemble.mmff_energies


def compute_amine_conformer_matrix(
    amine_smiles: str,
    *,
    n_conformers: int = DEFAULT_N_CONFORMERS,
    site_override: int | None = None,
) -> tuple[np.ndarray, list[float]] | None:
    """Per-conformer amine steric descriptors and MMFF energies (for any aggregation).

    Args:
        amine_smiles: SMILES of the amine nucleophile.
        n_conformers: Conformers to embed.
        site_override: Reacting-nitrogen index to use instead of the template's canonical site.

    Returns:
        ``(matrix, energies)`` with ``matrix`` shaped
        ``(n_conformers, len(AMINE_STERIC_DESCRIPTORS))``, or ``None`` on failure.
    """
    nitrogen_idx = site_override if site_override is not None else locate_amine_site(amine_smiles)
    if nitrogen_idx is None:
        return None
    ensemble = embed_conformers(amine_smiles, nitrogen_idx, n_conformers=n_conformers)
    if ensemble is None:
        return None
    matrix = np.vstack(
        [
            _amine_geometry_descriptors(ensemble.elements, coords, nitrogen_idx)
            for coords in ensemble.conformers
        ]
    )
    return matrix, ensemble.mmff_energies


def compute_aryl_steric_descriptors(
    aryl_smiles: str, *, n_conformers: int = DEFAULT_N_CONFORMERS, aggregate: str = "boltzmann"
) -> np.ndarray | None:
    """Aggregated steric descriptors at an aryl halide's reacting carbon.

    Args:
        aryl_smiles: SMILES of the aryl (pseudo)halide.
        n_conformers: Conformers to embed.
        aggregate: Conformer-aggregation mode (see :data:`bhpu.chem.sites.AGGREGATION_MODES`).

    Returns:
        A ``(len(ARYL_STERIC_DESCRIPTORS),)`` array, or ``None`` if no conformer can be embedded.
    """
    result = compute_aryl_conformer_matrix(aryl_smiles, n_conformers=n_conformers)
    if result is None:
        return None
    matrix, energies = result
    return aggregate_conformer_descriptors(matrix, energies, mode=aggregate)


def compute_amine_steric_descriptors(
    amine_smiles: str, *, n_conformers: int = DEFAULT_N_CONFORMERS, aggregate: str = "boltzmann"
) -> np.ndarray | None:
    """Aggregated steric descriptors at an amine's reacting nitrogen.

    Args:
        amine_smiles: SMILES of the amine nucleophile.
        n_conformers: Conformers to embed.
        aggregate: Conformer-aggregation mode (see :data:`bhpu.chem.sites.AGGREGATION_MODES`).

    Returns:
        A ``(len(AMINE_STERIC_DESCRIPTORS),)`` array, or ``None`` if no conformer can be embedded.
    """
    result = compute_amine_conformer_matrix(amine_smiles, n_conformers=n_conformers)
    if result is None:
        return None
    matrix, energies = result
    return aggregate_conformer_descriptors(matrix, energies, mode=aggregate)


def _buried_volume(elements: list[str], coords: np.ndarray, center_1indexed: int) -> float:
    """Percent buried volume in a sphere around a 1-indexed atom (``nan`` on failure)."""
    try:
        buried = BuriedVolume(elements, coords, center_1indexed, radius=_VBUR_RADIUS)
        return float(buried.fraction_buried_volume * 100.0)
    except (ValueError, IndexError, np.linalg.LinAlgError):
        return float("nan")


def _atom_sasa(elements: list[str], coords: np.ndarray, atom_1indexed: int) -> float:
    """Solvent-accessible surface area of a single 1-indexed atom (``nan`` on failure)."""
    try:
        return float(SASA(elements, coords).atom_areas[atom_1indexed])
    except (ValueError, KeyError, IndexError, np.linalg.LinAlgError):
        return float("nan")


class MorfeusSiteFeaturizer(ReactionFeaturizer):
    """Reactive-site steric descriptors (aryl carbon ⊕ amine nitrogen), SMILES-cached.

    Attributes:
        aryl_column: SMILES column of the aryl halide.
        amine_column: SMILES column of the amine.
    """

    def __init__(
        self,
        aryl_cache: dict[str, np.ndarray] | None = None,
        amine_cache: dict[str, np.ndarray] | None = None,
        *,
        aryl_column: str = schema.ARYL_SMILES,
        amine_column: str = schema.AMINE_SMILES,
    ) -> None:
        """Initialise the featuriser.

        Args:
            aryl_cache: Optional ``canonical SMILES -> aryl descriptor`` cache.
            amine_cache: Optional ``canonical SMILES -> amine descriptor`` cache.
            aryl_column: Aryl SMILES column name.
            amine_column: Amine SMILES column name.
        """
        self._aryl_cache: dict[str, np.ndarray] = aryl_cache or {}
        self._amine_cache: dict[str, np.ndarray] = amine_cache or {}
        self.aryl_column = aryl_column
        self.amine_column = amine_column

    @property
    def feature_names(self) -> list[str]:
        """Descriptor names, prefixed by substrate role."""
        return [f"aryl__{name}" for name in ARYL_STERIC_DESCRIPTORS] + [
            f"amine__{name}" for name in AMINE_STERIC_DESCRIPTORS
        ]

    def fit(self, frame: pd.DataFrame) -> Self:
        """No-op (descriptors are content-defined); present for interface parity."""
        return self

    def _aryl_block(self, smiles: object) -> np.ndarray:
        """Cached/computed aryl descriptors (zeros on failure)."""
        return self._lookup(
            smiles, self._aryl_cache, compute_aryl_steric_descriptors, len(ARYL_STERIC_DESCRIPTORS)
        )

    def _amine_block(self, smiles: object) -> np.ndarray:
        """Cached/computed amine descriptors (zeros on failure)."""
        return self._lookup(
            smiles,
            self._amine_cache,
            compute_amine_steric_descriptors,
            len(AMINE_STERIC_DESCRIPTORS),
        )

    @staticmethod
    def _lookup(
        smiles: object,
        cache: dict[str, np.ndarray],
        compute: object,
        width: int,
    ) -> np.ndarray:
        """Resolve a substrate descriptor from cache or by computing it (zeros on failure)."""
        canonical = canonical_smiles(smiles) if isinstance(smiles, str) else None
        if canonical is None:
            return np.zeros(width)
        cached = cache.get(canonical)
        if cached is not None:
            return cached
        computed = compute(canonical)  # type: ignore[operator]
        vector = computed if computed is not None else np.zeros(width)
        vector = np.nan_to_num(vector, nan=0.0)
        cache[canonical] = vector
        return vector

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Return ``(n_rows, len(feature_names))`` reactive-site steric descriptors."""
        aryl = np.vstack([self._aryl_block(s) for s in frame[self.aryl_column]])
        amine = np.vstack([self._amine_block(s) for s in frame[self.amine_column]])
        return np.hstack([aryl, amine]).astype(np.float32)
