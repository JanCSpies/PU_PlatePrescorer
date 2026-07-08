"""Locate the reactive atoms of each substrate and embed a 3D conformer.

Site-specific descriptors (Sterimol / %V_bur at the reacting carbon, local charge / Fukui
at the amine nitrogen) need three things per substrate: the reacting atom index, a 3D
geometry, and the element/coordinate arrays. This module centralises all three so the
morfeus and g-xTB featurisers share one implementation.

The reacting site of an aryl halide depends only on the aryl (the template picks a halide
independently of the amine), and the site of an amine depends only on the amine. We exploit
that by locating each substrate's site against a fixed *reference partner* — ammonia for an
aryl, bromobenzene for an amine — so a substrate has one well-defined site regardless of the
pair it appears in, and descriptors cache per unique SMILES (as ECFP / xTB already do).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from bhpu.chem.reaction import coupling_site_indices

#: Reference partners that make each substrate's coupling site well-defined on its own.
#: The amine must be primary/secondary (the template needs ``[N;H1,H2]``), so ammonia (H3)
#: would fail — methylamine is the minimal valid nucleophile.
_REFERENCE_AMINE: str = "CN"  # methylamine — a minimal H2 nucleophile the template accepts
_REFERENCE_ARYL: str = "Brc1ccccc1"  # bromobenzene — a minimal aryl halide

_BOHR_PER_ANGSTROM: float = 1.8897259886


#: Boltzmann constant in kcal/(mol·K); MMFF energies are in kcal/mol.
_KCAL_PER_MOL_K: float = 0.0019872041
_ROOM_TEMPERATURE_K: float = 298.15


@dataclass(frozen=True, slots=True)
class EmbeddedMolecule:
    """A substrate with a 3D conformer and its reactive-atom index.

    Attributes:
        mol: The RDKit molecule with explicit hydrogens and one embedded conformer.
        site_index: Index (in ``mol``) of the reacting atom — the aryl C–X carbon or the
            amine nitrogen.
        elements: ``(n_atoms,)`` element symbols, aligned to ``mol`` atom order.
        coordinates: ``(n_atoms, 3)`` Cartesian coordinates in Ångström.
    """

    mol: Chem.Mol
    site_index: int
    elements: list[str]
    coordinates: np.ndarray

    @property
    def coordinates_bohr(self) -> np.ndarray:
        """Coordinates converted to atomic units (Bohr) for QM engines."""
        return self.coordinates * _BOHR_PER_ANGSTROM


@dataclass(frozen=True, slots=True)
class ConformerEnsemble:
    """A substrate's multi-conformer 3D ensemble sharing one reactive-atom index.

    Descriptors that depend on geometry (%V_bur, Sterimol, dipole, frontier orbitals) vary
    across conformers; computing them per conformer and Boltzmann-averaging (or picking the
    lowest-energy geometry) is more robust than a single embedding.

    Attributes:
        site_index: Reactive-atom index shared by every conformer (atom order is identical).
        elements: ``(n_atoms,)`` element symbols.
        conformers: List of ``(n_atoms, 3)`` Ångström coordinate arrays, low energy first.
        mmff_energies: Per-conformer MMFF energy (kcal/mol); ``nan`` where MMFF was unavailable.
    """

    site_index: int
    elements: list[str]
    conformers: list[np.ndarray]
    mmff_energies: list[float]

    def boltzmann_weights(self, temperature_k: float = _ROOM_TEMPERATURE_K) -> np.ndarray:
        """Boltzmann weights over conformers from MMFF energies (uniform if energies absent).

        Args:
            temperature_k: Temperature for the Boltzmann factor.

        Returns:
            A ``(n_conformers,)`` array of non-negative weights summing to 1.
        """
        return boltzmann_weights(self.mmff_energies, temperature_k=temperature_k)


def boltzmann_weights(
    energies: list[float] | np.ndarray, *, temperature_k: float = _ROOM_TEMPERATURE_K
) -> np.ndarray:
    """Boltzmann weights from relative energies (kcal/mol); uniform if all are non-finite.

    Args:
        energies: Per-item energies in kcal/mol (``nan``/``inf`` treated as missing).
        temperature_k: Temperature for the Boltzmann factor.

    Returns:
        A ``(n,)`` array of non-negative weights summing to 1.
    """
    values = np.asarray(energies, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(len(values), 1.0 / max(len(values), 1))
    shifted = np.where(finite, values - np.min(values[finite]), np.inf)
    factors = np.exp(-shifted / (_KCAL_PER_MOL_K * temperature_k))
    total = float(factors.sum())
    return factors / total if total > 0 else np.full(len(values), 1.0 / len(values))


#: Supported conformer-aggregation strategies for site descriptors.
AGGREGATION_MODES: tuple[str, ...] = ("boltzmann", "lowest", "mean")


def aggregate_conformer_descriptors(
    matrix: np.ndarray,
    energies: list[float] | np.ndarray,
    *,
    mode: str = "boltzmann",
    temperature_k: float = _ROOM_TEMPERATURE_K,
) -> np.ndarray:
    """Collapse a per-conformer descriptor matrix to one vector by the chosen strategy.

    Args:
        matrix: ``(n_conformers, n_descriptors)`` per-conformer descriptor values.
        energies: ``(n_conformers,)`` energies (kcal/mol) for weighting / lowest selection.
        mode: ``"boltzmann"`` (energy-weighted average), ``"lowest"`` (lowest-energy conformer),
            or ``"mean"`` (uniform average).
        temperature_k: Temperature for the Boltzmann factor (``mode="boltzmann"`` only).

    Returns:
        A ``(n_descriptors,)`` aggregated descriptor vector.

    Raises:
        ValueError: If ``mode`` is not one of :data:`AGGREGATION_MODES`.
    """
    if mode not in AGGREGATION_MODES:
        raise ValueError(f"Unknown aggregation mode {mode!r}; expected one of {AGGREGATION_MODES}.")
    values = np.asarray(energies, dtype=np.float64)
    if mode == "lowest":
        index = int(np.nanargmin(values)) if np.isfinite(values).any() else 0
        return np.asarray(matrix[index], dtype=np.float64)
    if mode == "mean":
        weights = np.ones(len(matrix), dtype=np.float64)
    else:
        weights = boltzmann_weights(values, temperature_k=temperature_k)
    return weighted_nanmean(matrix, weights)


def weighted_nanmean(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Column-wise weighted mean of ``(n_conformers, n_descriptors)``, ignoring ``nan``.

    Each descriptor is averaged only over the conformers where it is finite (weights
    renormalised over that subset); a descriptor that failed for every conformer stays ``nan``.

    Args:
        matrix: ``(n_conformers, n_descriptors)`` per-conformer descriptor values.
        weights: ``(n_conformers,)`` conformer weights (e.g. Boltzmann).

    Returns:
        A ``(n_descriptors,)`` aggregated descriptor vector.
    """
    out = np.full(matrix.shape[1], np.nan)
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        finite = np.isfinite(values)
        if finite.any():
            subset = weights[finite]
            total = float(subset.sum())
            normalised = subset / total if total > 0 else np.full(int(finite.sum()), 1.0)
            out[column] = float(np.dot(values[finite], normalised))
    return out


def embed_conformers(
    smiles: str,
    site_index: int,
    *,
    n_conformers: int = 5,
    seed: int = 42,
    prune_rms: float = 0.5,
) -> ConformerEnsemble | None:
    """Embed and MMFF-optimise a small conformer ensemble, keeping the site index aligned.

    Args:
        smiles: Substrate SMILES.
        site_index: Reactive-atom index on the heavy-atom molecule (from the locators).
        n_conformers: Number of conformers to request (RMS-pruned, so fewer may survive).
        seed: Embedding seed (fixed for reproducibility).
        prune_rms: RMS pruning threshold (Å) to drop near-duplicate conformers.

    Returns:
        A :class:`ConformerEnsemble` sorted by MMFF energy (low first), or ``None`` if
        parsing/embedding fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or site_index >= mol.GetNumAtoms():
        return None
    mol = Chem.AddHs(mol)  # heavy-atom indices (hence site_index) are preserved
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = prune_rms
    conformer_ids = list(
        AllChem.EmbedMultipleConfs(mol, numConfs=max(n_conformers, 1), params=params)
    )
    if not conformer_ids:
        return None
    energies: dict[int, float] = {}
    try:
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
        for cid, (_, energy) in zip(conformer_ids, results, strict=False):
            energies[cid] = float(energy)
    except (RuntimeError, ValueError):
        pass  # keep raw ETKDG geometries; energies stay unknown → uniform Boltzmann weights
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    ordered = sorted(conformer_ids, key=lambda cid: energies.get(cid, float("inf")))
    conformers = [
        np.asarray(mol.GetConformer(cid).GetPositions(), dtype=np.float64) for cid in ordered
    ]
    mmff_energies = [energies.get(cid, float("nan")) for cid in ordered]
    return ConformerEnsemble(
        site_index=site_index,
        elements=elements,
        conformers=conformers,
        mmff_energies=mmff_energies,
    )


@lru_cache(maxsize=200_000)
def locate_aryl_site(aryl_smiles: str) -> tuple[int, int | None] | None:
    """Return ``(carbon_idx, leaving_halide_idx)`` for an aryl halide, or ``None``.

    Args:
        aryl_smiles: SMILES of the aryl (pseudo)halide.

    Returns:
        The reacting carbon index and its leaving-halide index (``None`` for a pseudohalide),
        or ``None`` if no aryl C–X coupling site is found.
    """
    sites = coupling_site_indices(aryl_smiles, _REFERENCE_AMINE)
    if sites is None:
        return None
    carbon_idx, leaving_idx, _ = sites
    return carbon_idx, leaving_idx


@lru_cache(maxsize=200_000)
def locate_amine_site(amine_smiles: str) -> int | None:
    """Return the reacting nitrogen index of an amine, or ``None`` if none applies.

    Args:
        amine_smiles: SMILES of the amine nucleophile.

    Returns:
        The reacting nitrogen index, or ``None`` if the amine has no BH-eligible N–H site.
    """
    sites = coupling_site_indices(_REFERENCE_ARYL, amine_smiles)
    if sites is None:
        return None
    return sites[2]


def embed_with_site(smiles: str, site_index: int, *, seed: int = 42) -> EmbeddedMolecule | None:
    """Embed a 3D conformer and keep the reactive-atom index aligned.

    Adds explicit hydrogens (appended after heavy atoms, so ``site_index`` stays valid),
    embeds one ETKDG conformer, and MMFF-optimises it (keeping the raw embedding if MMFF
    fails).

    Args:
        smiles: Substrate SMILES.
        site_index: Reactive-atom index on the heavy-atom molecule (from the locators).
        seed: Conformer-embedding seed (fixed for reproducibility).

    Returns:
        An :class:`EmbeddedMolecule`, or ``None`` if parsing/embedding fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or site_index >= mol.GetNumAtoms():
        return None
    mol = Chem.AddHs(mol)  # heavy-atom indices (hence site_index) are preserved
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        return None
    with contextlib.suppress(RuntimeError, ValueError):
        AllChem.MMFFOptimizeMolecule(mol)  # keep the raw ETKDG embedding if MMFF fails
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coordinates = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    return EmbeddedMolecule(
        mol=mol, site_index=site_index, elements=elements, coordinates=coordinates
    )
