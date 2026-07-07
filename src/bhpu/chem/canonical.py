"""Canonical SMILES / InChIKey utilities for de-duplication and leakage auditing.

All comparisons of substrates across data sources (leakage audit, unlabeled-space
de-duplication) must be done on a *canonical* representation so that different
but equivalent SMILES strings (kekulisation, atom order, salts, charges) map to the
same key. We standardise on RDKit canonical SMILES and the first block of the
InChIKey (the connectivity hash) as the substrate identity.
"""

from __future__ import annotations

from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from rdkit.Chem.SaltRemover import SaltRemover

# RDKit is noisy on unparsable inputs; silence its C++ logger (we handle failures).
RDLogger.DisableLog("rdApp.*")

_SALT_REMOVER = SaltRemover()


@lru_cache(maxsize=100_000)
def _mol_from_smiles(smiles: str) -> Chem.Mol | None:
    """Parse a SMILES string to an RDKit ``Mol`` (cached), or ``None`` if invalid."""
    if not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


@lru_cache(maxsize=100_000)
def canonical_smiles(smiles: str) -> str | None:
    """Return the RDKit canonical SMILES of ``smiles``.

    Args:
        smiles: An input SMILES string.

    Returns:
        The canonical SMILES, or ``None`` if the input cannot be parsed.
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    canonical: str = Chem.MolToSmiles(mol)
    return canonical


@lru_cache(maxsize=100_000)
def strip_and_canonicalise(smiles: str) -> str | None:
    """Neutralise-free salt-strip then canonicalise a SMILES.

    Removes counter-ions/salts (e.g. bases stored as ``base.[Cs+]``) so that the
    organic fragment identity is compared, not the packaging. Isotopes and atom-map
    numbers are cleared. Charges are *not* forcibly neutralised (that can change
    identity); only salt fragments are removed.

    Args:
        smiles: An input SMILES string.

    Returns:
        The salt-stripped canonical SMILES, or ``None`` if unparsable.
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    mol = _SALT_REMOVER.StripMol(mol, dontRemoveEverything=True)
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
        atom.SetAtomMapNum(0)
    if mol.GetNumAtoms() == 0:
        return None
    cleaned: str = Chem.MolToSmiles(mol)
    return cleaned


@lru_cache(maxsize=100_000)
def inchikey(smiles: str, *, connectivity_only: bool = True) -> str | None:
    """Return the InChIKey of ``smiles`` for use as a substrate identity key.

    Args:
        smiles: An input SMILES string.
        connectivity_only: If ``True``, return only the first block (the 14-character
            connectivity hash), which ignores stereochemistry/protonation — the right
            granularity for "is this the same substrate" overlap checks.

    Returns:
        The (possibly truncated) InChIKey, or ``None`` if it cannot be computed.
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    key: str = inchi.MolToInchiKey(mol)
    if not key:
        return None
    return key.split("-", 1)[0] if connectivity_only else key
