"""Canonicalise reaction *conditions* so the same chemistry gets the same representation.

The single biggest obstacle to learning condition effects across data sources is that the *same*
catalytic system is written differently: a precatalyst as one molecule (``RuPhos-Pd-G4``) vs. the
metal source and free ligand written separately (``Pd(OAc)2`` + ``RuPhos``). Fingerprinting those
naively yields unrelated vectors, so a model cannot learn "RuPhos works here" across sources.

This module unifies them by decomposing every condition into functional slots:

* **metal(s)** — which transition metal is present (Pd/Cu/Ni/…), the coarse catalyst identity;
* **ligand** — the free phosphine/donor, recovered from a precatalyst by deleting the metal and
  keeping the largest phosphorus-containing fragment, then neutralising it. A precatalyst and the
  corresponding free ligand collapse to the **same** canonical SMILES / InChIKey (validated on
  RuPhos: ``RuPhos-Pd-G3`` and free ``RuPhos`` → identical key);
* **base** and **solvent** — canonical connectivity InChIKeys, which already unify write-order
  differences (e.g. ``[Cs+].[Cs+].O=C([O-])[O-]`` vs ``O=C([O-])[O-].[Cs+].[Cs+]``).

Both the HTE tables (explicit catalyst/base/solvent columns) and the mined corpora (a bag of
reagent SMILES) are normalised to the same :class:`CanonicalCondition`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from rdkit import Chem

from bhpu.chem.canonical import canonical_smiles, inchikey

#: Transition metals whose removal exposes the free ligand (Pd dominates Buchwald–Hartwig).
_METAL_ATOMIC_NUMBERS: frozenset[int] = frozenset({46, 29, 28, 78, 79, 45, 44, 27, 26})
_ATOMIC_SYMBOL: dict[int, str] = {
    46: "Pd",
    29: "Cu",
    28: "Ni",
    78: "Pt",
    79: "Au",
    45: "Rh",
    44: "Ru",
    27: "Co",
    26: "Fe",
}

#: Group-1/2 (and common) cations stripped from a base to expose its unifying anion.
_CATION_ATOMIC_NUMBERS: frozenset[int] = frozenset({3, 11, 19, 37, 55, 4, 12, 20})

#: Connectivity InChIKeys of common BH solvents (for classifying mined reagent bags).
_SOLVENT_KEYS: frozenset[str] = frozenset(
    key
    for k in (
        "C1CCOC1",
        "C1COCCO1",
        "Cc1ccccc1",
        "CC(C)O",
        "CCO",
        "CO",
        "CS(C)=O",
        "CN(C)C=O",
        "CC#N",
        "ClCCl",
        "CC(C)(C)O",
        "CCCCCC",
        "O",
        "C1CCCCC1",
        "Cc1ccccc1C",
    )
    if (key := inchikey(k)) is not None
)
#: Connectivity InChIKeys of common BH base *anions* / neutral amine bases (post cation-strip).
_BASE_KEYS: frozenset[str] = frozenset(
    key
    for k in (
        "O=C([O-])[O-]",
        "O=P([O-])([O-])[O-]",
        "CC(C)(C)[O-]",
        "[OH-]",
        "[F-]",
        "CCN(CC)CC",
        "CN(C)C",
        "C1CCC2=NCCCN2CC1",  # carbonate/phosphate/tBuO/OH/F/Et3N/DABCO/DBU
    )
    if (key := inchikey(k)) is not None
)


@dataclass(frozen=True, slots=True)
class CanonicalCondition:
    """A reaction condition decomposed into unified functional slots.

    Attributes:
        metals: Transition-metal element symbols present (e.g. ``frozenset({"Pd"})``).
        ligand_smiles: Canonical free-ligand SMILES, or ``None`` if ligand-free/undetected.
        base_smiles: Canonical base SMILES, or ``None``.
        solvent_smiles: Canonical solvent SMILES, or ``None``.
    """

    metals: frozenset[str]
    ligand_smiles: str | None
    base_smiles: str | None
    solvent_smiles: str | None

    @property
    def ligand_key(self) -> str | None:
        """Connectivity InChIKey of the ligand (unifies precatalyst vs free-ligand forms)."""
        return inchikey(self.ligand_smiles) if self.ligand_smiles else None

    @property
    def base_key(self) -> str | None:
        """Connectivity InChIKey of the base."""
        return inchikey(self.base_smiles) if self.base_smiles else None

    @property
    def solvent_key(self) -> str | None:
        """Connectivity InChIKey of the solvent."""
        return inchikey(self.solvent_smiles) if self.solvent_smiles else None


@lru_cache(maxsize=100_000)
def extract_ligand(smiles: str) -> str | None:
    """Recover the free-ligand SMILES from a precatalyst or a free ligand reagent.

    Deletes any transition-metal atoms, neutralises the exposed phosphorus, and returns the
    canonical SMILES of the largest phosphorus-containing fragment — so ``RuPhos-Pd-G4`` and free
    ``RuPhos`` yield the same string.

    Args:
        smiles: A reagent SMILES (a metal complex, a precatalyst, or a free phosphine).

    Returns:
        Canonical free-ligand SMILES, or ``None`` if no phosphine ligand is present.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    editable = Chem.RWMol(mol)
    metal_indices = [
        a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in _METAL_ATOMIC_NUMBERS
    ]
    for index in sorted(metal_indices, reverse=True):
        editable.RemoveAtom(index)
    candidate = editable.GetMol()
    for atom in candidate.GetAtoms():
        if atom.GetAtomicNum() == 15 and atom.GetFormalCharge() != 0:
            atom.SetFormalCharge(0)  # coordinated P+ becomes a neutral free phosphine
    try:
        fragments = Chem.GetMolFrags(candidate, asMols=True, sanitizeFrags=False)
    except (Chem.AtomValenceException, ValueError):
        return None
    phosphine_fragments = [
        f for f in fragments if any(atom.GetAtomicNum() == 15 for atom in f.GetAtoms())
    ]
    if not phosphine_fragments:
        return None
    ligand = max(phosphine_fragments, key=lambda f: f.GetNumAtoms())
    try:
        Chem.SanitizeMol(ligand)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError):
        return None
    canonical: str = Chem.MolToSmiles(ligand)
    return canonical


def _metals_in(smiles: str) -> frozenset[str]:
    """Transition-metal element symbols present in a SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return frozenset()
    return frozenset(
        _ATOMIC_SYMBOL[a.GetAtomicNum()]
        for a in mol.GetAtoms()
        if a.GetAtomicNum() in _METAL_ATOMIC_NUMBERS
    )


@lru_cache(maxsize=100_000)
def base_anion(smiles: str) -> str | None:
    """Canonical SMILES of a base's anion (cations stripped), so Cs₂CO₃ and K₂CO₃ unify.

    Group-1/2 cations are removed and the largest remaining fragment is returned; neutral amine
    bases (no cation) pass through unchanged. This gives one representation per base chemistry,
    independent of counter-ion and write order.

    Args:
        smiles: Base SMILES (salt or neutral amine).

    Returns:
        Canonical anion/neutral-base SMILES, or ``None`` if unparsable.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    cation_indices = [
        a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in _CATION_ATOMIC_NUMBERS
    ]
    if not cation_indices:
        return canonical_smiles(smiles)
    editable = Chem.RWMol(mol)
    for index in sorted(cation_indices, reverse=True):
        editable.RemoveAtom(index)
    try:
        fragments = Chem.GetMolFrags(editable.GetMol(), asMols=True, sanitizeFrags=False)
    except (Chem.AtomValenceException, ValueError):
        return None
    if not fragments:
        return None
    anion = max(fragments, key=lambda f: f.GetNumAtoms())
    try:
        Chem.SanitizeMol(anion)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError):
        return None
    canonical: str = Chem.MolToSmiles(anion)
    return canonical


def condition_from_columns(
    catalyst_smiles: str | None, base_smiles: str | None, solvent_smiles: str | None
) -> CanonicalCondition:
    """Canonicalise an HTE condition from explicit catalyst/base/solvent columns.

    Args:
        catalyst_smiles: Precatalyst (or metal+ligand) SMILES.
        base_smiles: Base SMILES.
        solvent_smiles: Solvent SMILES.

    Returns:
        The unified :class:`CanonicalCondition`.
    """
    metals: frozenset[str] = frozenset()
    ligand: str | None = None
    if isinstance(catalyst_smiles, str) and catalyst_smiles:
        metals = _metals_in(catalyst_smiles)
        ligand = extract_ligand(catalyst_smiles)
    return CanonicalCondition(
        metals=metals,
        ligand_smiles=ligand,
        base_smiles=base_anion(base_smiles)
        if isinstance(base_smiles, str) and base_smiles
        else None,
        solvent_smiles=canonical_smiles(solvent_smiles)
        if isinstance(solvent_smiles, str)
        else None,
    )


def condition_from_reagents(reagents: tuple[str, ...]) -> CanonicalCondition:
    """Canonicalise a condition from a mined reaction's bag of reagent/agent SMILES.

    Classifies each reagent into metal source / ligand / base / solvent by structure, so mined
    ORD/Pistachio reactions (metal and ligand written separately) map to the same slots as an HTE
    precatalyst. The ligand is recovered from whichever reagent carries phosphorus (free or
    metal-bound); base/solvent are matched against small canonical libraries.

    Args:
        reagents: SMILES of all non-substrate reactant/reagent molecules.

    Returns:
        The unified :class:`CanonicalCondition`.
    """
    metals: set[str] = set()
    ligand: str | None = None
    base: str | None = None
    solvent: str | None = None
    for smiles in reagents:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        metals |= _metals_in(smiles)
        has_phosphorus = any(a.GetAtomicNum() == 15 for a in mol.GetAtoms())
        if has_phosphorus and ligand is None:
            ligand = extract_ligand(smiles)
        anion = base_anion(smiles)
        if base is None and anion is not None and inchikey(anion) in _BASE_KEYS:
            base = anion
        elif solvent is None and inchikey(canonical_smiles(smiles) or smiles) in _SOLVENT_KEYS:
            solvent = canonical_smiles(smiles)
    return CanonicalCondition(
        metals=frozenset(metals), ligand_smiles=ligand, base_smiles=base, solvent_smiles=solvent
    )
