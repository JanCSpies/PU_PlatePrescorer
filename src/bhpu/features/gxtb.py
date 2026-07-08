"""Site-specific *electronic* descriptors from the g-xTB semiempirical method.

g-xTB (Grimme lab, 2025) is a general semiempirical method that approximates
ωB97M-V/def2-TZVPPD properties — a substantial electronic-structure upgrade over GFN2-xTB.
This featuriser runs the ``gxtb`` binary once per substrate on an RDKit conformer and reads
back both **global** electronics (HOMO–LUMO gap, Janak IP/EA, dipole, energy per atom) and,
crucially, the **per-atom EEQ_BC charge and Wiberg bond order at the reacting atom** — the
site-specific electrophilicity (aryl C–X carbon) / nucleophilicity (amine N) that whole-
molecule descriptors average away.

The binary and its parameter files ship outside the project; paths default to the shared
install and are overridable via the ``GXTB_BINARY`` / ``GXTBHOME`` environment variables.
Descriptors cache per unique SMILES (each substrate's site is partner-independent).
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd

from bhpu.chem.canonical import canonical_smiles
from bhpu.chem.sites import (
    aggregate_conformer_descriptors,
    embed_conformers,
    locate_amine_site,
    locate_aryl_site,
)
from bhpu.data import schema
from bhpu.features.base import ReactionFeaturizer

#: Default conformers per substrate for aggregated electronics (g-xTB is the costly engine).
DEFAULT_N_CONFORMERS: int = 10
#: Hartree → kcal/mol (g-xTB energies are in Hartree; Boltzmann weighting works in kcal/mol).
_HARTREE_TO_KCAL: float = 627.5094740631

#: Ordered g-xTB descriptor names (one block per substrate; ``site`` = reacting atom).
GXTB_DESCRIPTORS: tuple[str, ...] = (
    "gap_ev",
    "ip_ev",
    "ea_ev",
    "dipole",
    "energy_per_atom",
    "site_charge",
    "site_wbo",
)

_DEFAULT_BINARY: str = "/mnt/share/public/software/g-xtb/binary/gxtb"
_DEFAULT_GXTBHOME: str = "/mnt/share/public/software/g-xtb/parameters"


def _binary() -> str:
    """Resolve the ``gxtb`` binary path (``GXTB_BINARY`` env override)."""
    return os.environ.get("GXTB_BINARY", _DEFAULT_BINARY)


def _gxtbhome() -> str:
    """Resolve the parameter directory (``GXTBHOME`` env override)."""
    return os.environ.get("GXTBHOME", _DEFAULT_GXTBHOME)


def _write_xyz(path: Path, elements: list[str], coords: np.ndarray) -> None:
    """Write an XYZ file (Ångström) in molecule atom order (so index i → gxtb atom i+1)."""
    lines = [str(len(elements)), ""]
    lines += [
        f"{el} {x:.6f} {y:.6f} {z:.6f}" for el, (x, y, z) in zip(elements, coords, strict=True)
    ]
    path.write_text("\n".join(lines) + "\n")


def _parse_site_charge(text: str, atom_1indexed: int) -> float:
    """Read the EEQ_BC partial charge (Hirshfeld-like) of a 1-indexed atom from stdout."""
    block = re.search(r"CN\(basis\)\s+q\n(.*?)\n\s*\n", text, re.DOTALL)
    if block is None:
        return float("nan")
    for line in block.group(1).splitlines():
        tokens = line.split()
        if len(tokens) >= 6 and tokens[0].isdigit() and int(tokens[0]) == atom_1indexed:
            return float(tokens[-1])
    return float("nan")


def _parse_site_wbo(text: str, atom_1indexed: int) -> float:
    """Read the total Wiberg bond order of a 1-indexed atom from the WBO block."""
    block = re.search(r"largest Wiberg.*?\n(.*)", text, re.DOTALL)
    if block is None:
        return float("nan")
    for line in block.group(1).splitlines():
        tokens = line.split()
        if len(tokens) >= 3 and tokens[0].isdigit() and int(tokens[0]) == atom_1indexed:
            try:
                return float(tokens[2])
            except ValueError:
                return float("nan")
    return float("nan")


def _parse_scalar(text: str, label: str) -> float:
    """Read a ``label : value`` scalar line from stdout (``nan`` if absent)."""
    match = re.search(rf"{re.escape(label)}\s*:\s*(-?\d+\.\d+)", text)
    return float(match.group(1)) if match else float("nan")


def _run_gxtb_geometry(
    elements: list[str], coords: np.ndarray, site_index: int, *, timeout_s: float
) -> tuple[np.ndarray, float] | None:
    """Run g-xTB on one geometry; return ``(descriptor_vector, total_energy_hartree)``.

    Args:
        elements: Element symbols in molecule atom order.
        coords: ``(n_atoms, 3)`` Ångström coordinates (same atom order as ``elements``).
        site_index: Reactive-atom index (0-based) whose charge/WBO to read.
        timeout_s: Per-call wall-clock budget for the ``gxtb`` binary.

    Returns:
        The ``(len(GXTB_DESCRIPTORS),)`` descriptor vector and the total energy in Hartree
        (for Boltzmann weighting across conformers), or ``None`` on failure.
    """
    env = {**os.environ, "GXTBHOME": _gxtbhome(), "OMP_NUM_THREADS": "1"}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        _write_xyz(work / "mol.xyz", elements, coords)
        try:
            result = subprocess.run(
                [_binary(), "-c", "mol.xyz"],
                cwd=work,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        text = result.stdout
        energy = _read_total_energy(work / "energy")

    atom_1 = site_index + 1
    vector = np.array(
        [
            _parse_scalar(text, "gap (eV)"),
            _parse_scalar(text, "Janaks theorem for IP"),
            _parse_scalar(text, "Janaks theorem for EA"),
            _dipole_debye(text),
            energy / max(len(elements), 1),
            _parse_site_charge(text, atom_1),
            _parse_site_wbo(text, atom_1),
        ],
        dtype=np.float64,
    )
    return vector, energy


def _read_total_energy(energy_file: Path) -> float:
    """Read the g-xTB total energy (Hartree) from the TURBOMOLE ``energy`` file."""
    if not energy_file.exists():
        return float("nan")
    for line in energy_file.read_text().splitlines():
        tokens = line.split()
        if len(tokens) >= 2 and tokens[0].isdigit():
            return float(tokens[1])
    return float("nan")


def compute_gxtb_conformer_matrix(
    smiles: str,
    site_index: int,
    *,
    n_conformers: int = DEFAULT_N_CONFORMERS,
    timeout_s: float = 120.0,
) -> tuple[np.ndarray, list[float]] | None:
    """Run g-xTB per conformer; return ``(matrix, energies_kcal)`` for any aggregation.

    Args:
        smiles: Substrate SMILES.
        site_index: Reactive-atom index (from the site locators) on the heavy-atom molecule.
        n_conformers: Conformers to embed and run through g-xTB.
        timeout_s: Per-conformer wall-clock budget for the ``gxtb`` call.

    Returns:
        ``(matrix, energies)`` with ``matrix`` shaped ``(n_ok_conformers, len(GXTB_DESCRIPTORS))``
        and ``energies`` the g-xTB total energies in kcal/mol, or ``None`` on failure.
    """
    ensemble = embed_conformers(smiles, site_index, n_conformers=n_conformers)
    if ensemble is None:
        return None
    vectors: list[np.ndarray] = []
    energies: list[float] = []
    for coords in ensemble.conformers:
        outcome = _run_gxtb_geometry(ensemble.elements, coords, site_index, timeout_s=timeout_s)
        if outcome is None:
            continue
        vector, energy = outcome
        vectors.append(vector)
        energies.append(energy * _HARTREE_TO_KCAL)
    if not vectors:
        return None
    return np.vstack(vectors), energies


def compute_gxtb_descriptors(
    smiles: str,
    site_index: int,
    *,
    n_conformers: int = DEFAULT_N_CONFORMERS,
    aggregate: str = "boltzmann",
    timeout_s: float = 120.0,
) -> np.ndarray | None:
    """Run g-xTB over a conformer ensemble and aggregate global + site electronics.

    Args:
        smiles: Substrate SMILES.
        site_index: Reactive-atom index (from the site locators) on the heavy-atom molecule.
        n_conformers: Conformers to embed and run through g-xTB.
        aggregate: Conformer-aggregation mode (see :data:`bhpu.chem.sites.AGGREGATION_MODES`);
            weighting uses the g-xTB total energy.
        timeout_s: Per-conformer wall-clock budget for the ``gxtb`` call.

    Returns:
        A ``(len(GXTB_DESCRIPTORS),)`` array, or ``None`` if embedding/execution/parsing fails.
    """
    result = compute_gxtb_conformer_matrix(
        smiles, site_index, n_conformers=n_conformers, timeout_s=timeout_s
    )
    if result is None:
        return None
    matrix, energies = result
    return aggregate_conformer_descriptors(matrix, energies, mode=aggregate)


def _dipole_debye(text: str) -> float:
    """Read the total dipole moment (Debye) from the final dipole line."""
    matches = re.findall(r"total\s+\(au/Debye\)\s*:\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)", text)
    return float(matches[-1][1]) if matches else float("nan")


def compute_aryl_gxtb_conformer_matrix(
    aryl_smiles: str, *, n_conformers: int = DEFAULT_N_CONFORMERS, site_override: int | None = None
) -> tuple[np.ndarray, list[float]] | None:
    """Per-conformer g-xTB descriptors + energies at an aryl carbon (``None`` on failure)."""
    carbon_idx = site_override if site_override is not None else _aryl_carbon(aryl_smiles)
    if carbon_idx is None:
        return None
    return compute_gxtb_conformer_matrix(aryl_smiles, carbon_idx, n_conformers=n_conformers)


def compute_amine_gxtb_conformer_matrix(
    amine_smiles: str, *, n_conformers: int = DEFAULT_N_CONFORMERS, site_override: int | None = None
) -> tuple[np.ndarray, list[float]] | None:
    """Per-conformer g-xTB descriptors + energies at an amine nitrogen (``None`` on failure)."""
    nitrogen_idx = site_override if site_override is not None else locate_amine_site(amine_smiles)
    if nitrogen_idx is None:
        return None
    return compute_gxtb_conformer_matrix(amine_smiles, nitrogen_idx, n_conformers=n_conformers)


def _aryl_carbon(aryl_smiles: str) -> int | None:
    """Reacting-carbon index of an aryl halide (``None`` if no site)."""
    site = locate_aryl_site(aryl_smiles)
    return site[0] if site is not None else None


def compute_aryl_gxtb_descriptors(
    aryl_smiles: str, *, n_conformers: int = DEFAULT_N_CONFORMERS, aggregate: str = "boltzmann"
) -> np.ndarray | None:
    """Aggregated g-xTB descriptors at an aryl halide's reacting carbon (``None`` on failure)."""
    result = compute_aryl_gxtb_conformer_matrix(aryl_smiles, n_conformers=n_conformers)
    if result is None:
        return None
    matrix, energies = result
    return aggregate_conformer_descriptors(matrix, energies, mode=aggregate)


def compute_amine_gxtb_descriptors(
    amine_smiles: str, *, n_conformers: int = DEFAULT_N_CONFORMERS, aggregate: str = "boltzmann"
) -> np.ndarray | None:
    """Aggregated g-xTB descriptors at an amine's reacting nitrogen (``None`` on failure)."""
    result = compute_amine_gxtb_conformer_matrix(amine_smiles, n_conformers=n_conformers)
    if result is None:
        return None
    matrix, energies = result
    return aggregate_conformer_descriptors(matrix, energies, mode=aggregate)


class GXTBSiteFeaturizer(ReactionFeaturizer):
    """Reactive-site g-xTB electronic descriptors (aryl ⊕ amine), SMILES-cached.

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
        return [f"aryl__{name}" for name in GXTB_DESCRIPTORS] + [
            f"amine__{name}" for name in GXTB_DESCRIPTORS
        ]

    def fit(self, frame: pd.DataFrame) -> Self:
        """No-op (descriptors are content-defined); present for interface parity."""
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Return ``(n_rows, 2 * len(GXTB_DESCRIPTORS))`` reactive-site electronics."""
        width = len(GXTB_DESCRIPTORS)
        aryl = np.vstack(
            [
                self._lookup(s, self._aryl_cache, compute_aryl_gxtb_descriptors, width)
                for s in frame[self.aryl_column]
            ]
        )
        amine = np.vstack(
            [
                self._lookup(s, self._amine_cache, compute_amine_gxtb_descriptors, width)
                for s in frame[self.amine_column]
            ]
        )
        return np.hstack([aryl, amine]).astype(np.float32)

    @staticmethod
    def _lookup(
        smiles: object, cache: dict[str, np.ndarray], compute: object, width: int
    ) -> np.ndarray:
        """Resolve a substrate descriptor from cache or by computing it (zeros on failure)."""
        canonical = canonical_smiles(smiles) if isinstance(smiles, str) else None
        if canonical is None:
            return np.zeros(width)
        cached = cache.get(canonical)
        if cached is not None:
            return cached
        computed = compute(canonical)  # type: ignore[operator]
        vector = np.nan_to_num(computed if computed is not None else np.zeros(width), nan=0.0)
        cache[canonical] = vector
        return vector
