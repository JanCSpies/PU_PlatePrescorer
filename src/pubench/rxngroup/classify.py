"""Reaction-class assignment: a named-reaction SMARTS library + a reaction-center signature.

Two complementary layers (the user chose "both"):

* :class:`NamedReactionLibrary` -- an ordered set of interpretable, plate-relevant transformations.
  A reaction is labelled by the **first** matching named reaction. Three detection modes are
  supported so the library covers couplings *and* functional-group interconversions:

  1. **Bimolecular coupling** -- two coupling-partner roles filled by two *distinct* reactant
     molecules (Suzuki, Buchwald-Hartwig, amide coupling, SNAr, ...), with a catalyst-metal gate
     that disambiguates same-substrate/different-mechanism pairs (aryl-halide + amine is Buchwald
     under Pd, Ullmann under Cu, SNAr under no metal).
  2. **Reagent-gated coupling** -- as (1) but also requires a signature reagent in the agent bag
     (Mitsunobu's DIAD/PPh3, Wittig's ylide, borylation's diboron, CuAAC's Cu + azide).
  3. **Unimolecular FGI** -- a functional-group interconversion detected by a motif that is
     *lost* (present in reactants, absent from products) and/or *gained* (Boc/Cbz/Fmoc deprotection,
     ester/amide hydrolysis, nitro reduction, carbonyl reduction, oxidation, aromatic halogenation).

  Ordering is specific -> general so the metal-gated couplings win, then reagent-gated bimolecular
  reactions, then FGIs on whatever is left. This is fast (pure substructure matching, no product
  enumeration); it labels by substrate + condition + net motif change, which is what a chemist reads
  when choosing a plate, and does not *confirm* a mechanism.
* :func:`reaction_center_signature` -- an unsupervised formed/broken-bond fingerprint from the
  atom-mapped reaction, used to sub-split whatever remains in the ``"other"`` bucket.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

OTHER: Final[str] = "other"


def _smarts(*patterns: str) -> tuple[Chem.Mol, ...]:
    """Compile one or more SMARTS strings, dropping any that fail to parse."""
    return tuple(m for p in patterns if (m := Chem.MolFromSmarts(p)) is not None)


# --- Coupling-partner role patterns (a molecule *fills a role* if it matches any) -----------------
_HALIDE_AR = _smarts("[Cl,Br,I][c]", "[Cl,Br,I][CX3]=[CX3]")
_HALIDE_SNAR = _smarts("[F,Cl,Br,I][c]")
_HALIDE_ALKYL = _smarts("[Cl,Br,I][CX4]")
_BORON = _smarts("[#6][BX3]([OX2])[OX2]", "[#6][B-](F)(F)F", "[#6][BX3]")
_DIBORON = _smarts("[BX3][BX3]", "[BX3;H1]")  # B2pin2 / pinacolborane (borylation reagent)
_NH_AMINE = _smarts("[NX3;H1,H2;!$([NX3]=*);!$([NX3]#*);!$([NX3]-[!#6;!#1]);!$([NX3][CX3]=[OX1])]")
_ALKYNE_TERM = _smarts("[CX2;H1]#[CX2]")
_AZIDE = _smarts("[#6][NX2]=[NX2+]=[NX1-]", "[#6][NX2-][NX2+]#[NX1]", "[#6]N=[N+]=[N-]")
_ALKENE = _smarts("[CX3;!$([CX3]=[OX1])]=[CX3;!$([CX3]=[OX1])]")
_ORGANOZINC = _smarts("[Zn]")
_ORGANOMAGNESIUM = _smarts("[Mg]")
_ORGANOLITHIUM = _smarts("[Li]")
_ORGANOTIN = _smarts("[Sn]")
_CARBOX_ACID = _smarts("[CX3](=[OX1])[OX2H1]", "[CX3](=[OX1])[Cl,Br]", "[CX3](=O)[OX2][CX3]=[OX1]")
_SULFONYL_HALIDE = _smarts("[SX4](=[OX1])(=[OX1])[Cl,F,Br]")
_CARBONYL = _smarts("[CX3H1]=[OX1]", "[#6][CX3](=[OX1])[#6]")
_ALCOHOL = _smarts("[OX2H][CX4]", "[OX2H][c]")
_HYDROXYL_THIOL = _smarts("[OX2H][CX4]", "[OX2H][c]", "[SX2H]")
_ISOCYANATE = _smarts("[NX2]=[CX2]=[OX1]")
_EPOXIDE = _smarts("[CR1]1[OR1][CR1]1")
_EPOXIDE_NUC = _smarts("[NX3;H1,H2]", "[OX2H]", "[SX2H]")
_YLIDE = _smarts("[#6]=[PX4]", "[CX4][PX4](=O)([OX2])[OX2]", "[P+]([#6])([#6])([#6])[#6]")
_ACIDIC_NUC = _smarts(  # Mitsunobu pronucleophiles: imide N-H, phenol, acid, sulfonamide N-H
    "[NX3;H1]([CX3]=[OX1])[CX3]=[OX1]",
    "[OX2H][c]",
    "[CX3](=[OX1])[OX2H1]",
    "[SX4](=[OX1])(=[OX1])[NX3;H1,H2]",
)
_HETEROARENE_CH = _smarts(  # electron-rich 5-membered heteroarene C-H (C-H arylation partner)
    "[cH]1cco1",
    "[cH]1ccs1",
    "[cH]1cc[nH]1",
    "[cH]1nc[nH]1",
    "[cH]1ncs1",
    "[cH]1nco1",
)

# --- Reagent-gate patterns (matched against the agent bag, not the reactants) ---------------------
_DIAD = _smarts("[#6]OC(=O)N=NC(=O)O[#6]", "[#7]C(=O)N=NC(=O)[#7]")
_PPH3 = _smarts("[PX3](c)(c)c")

# --- FGI motifs (compared between reactant side and product side) ---------------------------------
_N_BOC = _smarts("[NX3]C(=O)OC(C)(C)C")
_N_CBZ = _smarts("[NX3]C(=O)OCc1ccccc1")
_N_FMOC = _smarts("[NX3]C(=O)OCC1c2ccccc2-c2ccccc21")
_ESTER = _smarts("[CX3](=[OX1])[OX2][#6;!$([CX3]=[OX1])]")
_AMIDE = _smarts("[CX3](=[OX1])[NX3]")
_ACID = _smarts("[CX3](=[OX1])[OX2H1]")
_NITRO = _smarts("[$([NX3](=O)=O),$([NX3+](=O)[O-])]")
_PRIMARY_AMINE = _smarts("[NX3;H2][#6]")
_CARBONYL_OX = _smarts("[CX3H1]=[OX1]", "[#6][CX3](=[OX1])[#6]")
_ALCOHOL_C = _smarts("[CX4][OX2H]")
_SULFOXIDE_SULFONE = _smarts("[$([SX3]=O),$([SX4](=O)=O)]")


def _fills(mol: Chem.Mol, role: Sequence[Chem.Mol]) -> bool:
    """True if ``mol`` matches any SMARTS alternative of ``role``."""
    return any(mol.HasSubstructMatch(pat) for pat in role)


def _present(mols: Sequence[Chem.Mol], role: Sequence[Chem.Mol]) -> bool:
    """True if any molecule in ``mols`` matches any SMARTS alternative of ``role``."""
    return any(_fills(m, role) for m in mols)


@dataclass(frozen=True, slots=True)
class NamedReaction:
    """One named transformation: a bimolecular coupling, a reagent-gated coupling, or an FGI.

    Attributes:
        name: Human-readable class label.
        role_a: SMARTS alternatives one reactant must match (partner A), or ``None`` for an FGI.
        role_b: SMARTS alternatives a *distinct* reactant must match (partner B), or ``None``.
        metal_in: If set, at least one of these catalyst metals must be present.
        metal_forbidden: If True, no transition metal may be present (mechanistic disambiguation).
        reagent_any: If set, at least one *agent* must match one of these SMARTS (reagent gate).
        lost: FGI motif that must be present in a reactant and absent from all products.
        gained: FGI motif that must be present in a product and absent from all reactants.
    """

    name: str
    role_a: tuple[Chem.Mol, ...] | None = None
    role_b: tuple[Chem.Mol, ...] | None = None
    metal_in: frozenset[str] | None = None
    metal_forbidden: bool = False
    reagent_any: tuple[Chem.Mol, ...] | None = None
    lost: tuple[Chem.Mol, ...] | None = None
    gained: tuple[Chem.Mol, ...] | None = None

    def match(
        self,
        reactants: Sequence[tuple[str, Chem.Mol]],
        products: Sequence[Chem.Mol],
        agents: Sequence[Chem.Mol],
        metals: frozenset[str],
    ) -> tuple[str | None, str | None] | None:
        """Return ``(partner_a, partner_b)`` if this reaction matches, else ``None``.

        Args:
            reactants: ``(clean_smiles, mol)`` for each reactant molecule.
            products: Product molecules (for FGI motif comparison).
            agents: Condition-bag molecules (for the reagent gate).
            metals: Catalyst-metal symbols present in the condition bag.

        Returns:
            The two partner SMILES (partner B is ``None`` for an FGI), or ``None`` if no match.
        """
        if self.metal_forbidden and metals:
            return None
        if self.metal_in is not None and not (metals & self.metal_in):
            return None
        if self.reagent_any is not None and not _present(agents, self.reagent_any):
            return None

        if self.role_a is not None and self.role_b is not None:  # bimolecular
            return self._match_pair(reactants)
        return self._match_fgi(reactants, products)  # unimolecular FGI

    def _match_pair(
        self, reactants: Sequence[tuple[str, Chem.Mol]]
    ) -> tuple[str | None, str | None] | None:
        """Find two distinct reactants filling role A and role B."""
        assert self.role_a is not None and self.role_b is not None
        for i, (smi_a, mol_a) in enumerate(reactants):
            if not _fills(mol_a, self.role_a):
                continue
            for j, (smi_b, mol_b) in enumerate(reactants):
                if i != j and _fills(mol_b, self.role_b):
                    return smi_a, smi_b
        return None

    def _match_fgi(
        self, reactants: Sequence[tuple[str, Chem.Mol]], products: Sequence[Chem.Mol]
    ) -> tuple[str | None, str | None] | None:
        """Match a net motif change (lost in reactants->products, and/or gained)."""
        reactant_mols = [m for _, m in reactants]
        if self.lost is not None and not (
            _present(reactant_mols, self.lost) and not _present(products, self.lost)
        ):
            return None
        if self.gained is not None and not (
            _present(products, self.gained) and not _present(reactant_mols, self.gained)
        ):
            return None
        return None, None  # matched, but an FGI has no coupling-partner matrix


def _default_reactions() -> tuple[NamedReaction, ...]:
    """Build the ordered named-reaction library (couplings -> reagent-gated -> FGIs)."""
    pd_ni = frozenset({"Pd", "Ni"})
    pd = frozenset({"Pd"})
    cu = frozenset({"Cu"})
    return (
        # --- metal-gated bimolecular couplings ---
        NamedReaction("Suzuki-Miyaura", _HALIDE_AR, _BORON, metal_in=pd_ni),
        NamedReaction("Miyaura borylation", _HALIDE_AR, _DIBORON, metal_in=pd),
        NamedReaction("Sonogashira", _HALIDE_AR, _ALKYNE_TERM, metal_in=pd),
        NamedReaction("CuAAC click", _AZIDE, _ALKYNE_TERM, metal_in=cu),
        NamedReaction("Negishi", _HALIDE_AR, _ORGANOZINC, metal_in=pd_ni),
        NamedReaction(
            "Kumada", _HALIDE_AR, _ORGANOMAGNESIUM, metal_in=frozenset({"Pd", "Ni", "Fe"})
        ),
        NamedReaction("Stille", _HALIDE_AR, _ORGANOTIN, metal_in=pd),
        NamedReaction("Heck", _HALIDE_AR, _ALKENE, metal_in=pd),
        NamedReaction("Buchwald-Hartwig", _HALIDE_AR, _NH_AMINE, metal_in=pd_ni),
        NamedReaction("Ullmann C-N", _HALIDE_AR, _NH_AMINE, metal_in=cu),
        NamedReaction("Ullmann C-O", _HALIDE_AR, _ALCOHOL, metal_in=cu),
        NamedReaction("Chan-Lam", _BORON, _NH_AMINE, metal_in=cu),
        NamedReaction("C-H arylation", _HALIDE_AR, _HETEROARENE_CH, metal_in=pd),
        # --- reagent-gated / metal-free bimolecular ---
        NamedReaction("sulfonamide", _SULFONYL_HALIDE, _NH_AMINE),
        NamedReaction("Mitsunobu", _ALCOHOL, _ACIDIC_NUC, reagent_any=_DIAD + _PPH3),
        NamedReaction("amide coupling", _CARBOX_ACID, _NH_AMINE, metal_forbidden=True),
        NamedReaction("Wittig/HWE", _CARBONYL, _YLIDE, metal_forbidden=True),
        NamedReaction(
            "Grignard/organolithium addition",
            _CARBONYL,
            _ORGANOMAGNESIUM + _ORGANOLITHIUM,
            metal_forbidden=True,
        ),
        NamedReaction("reductive amination", _CARBONYL, _NH_AMINE, metal_forbidden=True),
        NamedReaction("epoxide opening", _EPOXIDE, _EPOXIDE_NUC, metal_forbidden=True),
        NamedReaction("SNAr amination", _HALIDE_SNAR, _NH_AMINE, metal_forbidden=True),
        NamedReaction("SNAr etherification", _HALIDE_SNAR, _HYDROXYL_THIOL, metal_forbidden=True),
        NamedReaction("N-alkylation", _HALIDE_ALKYL, _NH_AMINE, metal_forbidden=True),
        NamedReaction("Williamson ether", _HALIDE_ALKYL, _ALCOHOL, metal_forbidden=True),
        NamedReaction("esterification", _CARBOX_ACID, _ALCOHOL, metal_forbidden=True),
        NamedReaction("urea/carbamate", _ISOCYANATE, _NH_AMINE),
        # --- unimolecular functional-group interconversions (product-compared) ---
        NamedReaction("Boc deprotection", lost=_N_BOC),
        NamedReaction("Cbz deprotection", lost=_N_CBZ),
        NamedReaction("Fmoc deprotection", lost=_N_FMOC),
        NamedReaction("ester hydrolysis", lost=_ESTER, gained=_ACID),
        NamedReaction("amide hydrolysis", lost=_AMIDE, gained=_ACID),
        NamedReaction("nitro reduction", lost=_NITRO, gained=_PRIMARY_AMINE),
        NamedReaction("carbonyl reduction", lost=_CARBONYL_OX, gained=_ALCOHOL_C),
        NamedReaction("oxidation", gained=_SULFOXIDE_SULFONE),
        NamedReaction("alcohol oxidation", lost=_ALCOHOL_C, gained=_CARBONYL_OX),
        NamedReaction("aromatic halogenation", gained=_HALIDE_SNAR),
    )


@dataclass(slots=True)
class ClassResult:
    """The class assignment for one reaction."""

    reaction_class: str
    partner_a: str | None
    partner_b: str | None


class NamedReactionLibrary:
    """Ordered named-reaction classifier over a reaction's reactant/product/agent molecules."""

    def __init__(self, reactions: Sequence[NamedReaction] | None = None) -> None:
        """Store the (ordered) reactions; defaults to the built-in plate-relevant library."""
        self._reactions: tuple[NamedReaction, ...] = tuple(reactions or _default_reactions())

    @property
    def class_names(self) -> list[str]:
        """The ordered class labels this library can assign (plus ``other``)."""
        return [r.name for r in self._reactions] + [OTHER]

    def classify(
        self,
        reactants: Sequence[str],
        products: Sequence[str],
        agents: Sequence[str],
        metals: frozenset[str],
    ) -> ClassResult:
        """Assign the first-matching named reaction, or ``other``.

        Args:
            reactants: Clean (map-stripped) reactant SMILES.
            products: Clean (map-stripped) product SMILES (for FGI motif comparison).
            agents: Raw agent SMILES (for the reagent gate).
            metals: Catalyst-metal symbols present in the condition bag.

        Returns:
            A :class:`ClassResult` with the class label and the two coupling partners (or ``None``).
        """
        r_mols: list[tuple[str, Chem.Mol]] = [
            (s, m) for s in reactants if (m := Chem.MolFromSmiles(s)) is not None
        ]
        p_mols: list[Chem.Mol] = [m for s in products if (m := Chem.MolFromSmiles(s)) is not None]
        a_mols: list[Chem.Mol] = [m for s in agents if (m := Chem.MolFromSmiles(s)) is not None]
        for reaction in self._reactions:
            hit = reaction.match(r_mols, p_mols, a_mols, metals)
            if hit is not None:
                return ClassResult(reaction.name, hit[0], hit[1])
        return ClassResult(OTHER, None, None)


def _atom_desc(atom: Chem.Atom) -> str:
    """Coarse atom descriptor for a reaction-center environment: element + aromaticity."""
    symbol: str = atom.GetSymbol()
    return symbol + ("a" if atom.GetIsAromatic() else "")


def _bond_map(block: str) -> tuple[dict[frozenset[int], str], dict[int, Chem.Atom]]:
    """Map (mapnum-pair -> bond type) and (mapnum -> atom) for one reaction-SMILES block."""
    mol = Chem.MolFromSmiles(block, sanitize=False)
    bonds: dict[frozenset[int], str] = {}
    atoms: dict[int, Chem.Atom] = {}
    if mol is None:
        return bonds, atoms
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum():
            atoms[atom.GetAtomMapNum()] = atom
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetAtomMapNum() and b.GetAtomMapNum():
            bonds[frozenset((a.GetAtomMapNum(), b.GetAtomMapNum()))] = str(bond.GetBondType())
    return bonds, atoms


def reaction_center_signature(mapped_rxn: str) -> str | None:
    """Return a formed/broken-bond signature for an atom-mapped reaction, or ``None``.

    Bonds are compared by the map-number pair of their endpoints between the reactant and product
    sides; a bond present on only one side (or with a changed order) is a reaction-center bond. Each
    center bond is described by the sorted pair of its endpoint atom descriptors (element +
    aromaticity), giving a compact, template-free transformation key that groups mechanistically
    identical reactions in the ``other`` bucket.

    Args:
        mapped_rxn: Full atom-mapped ``reactants>agents>products`` reaction SMILES.

    Returns:
        A signature string ``"F=<formed>|B=<broken>"``, or ``None`` if the reaction cannot be parsed
        or carries no usable atom map.
    """
    parts = mapped_rxn.split(">")
    if len(parts) < 3:
        return None
    r_bonds, r_atoms = _bond_map(parts[0])
    p_bonds, p_atoms = _bond_map(parts[-1])
    if not r_bonds or not p_bonds:
        return None

    def _desc(pair: frozenset[int], atoms: dict[int, Chem.Atom]) -> str:
        ds = sorted(_atom_desc(atoms[m]) for m in pair if m in atoms)
        return "-".join(ds)

    formed = sorted(_desc(pair, p_atoms) for pair in p_bonds if r_bonds.get(pair) != p_bonds[pair])
    broken = sorted(_desc(pair, r_atoms) for pair in r_bonds if p_bonds.get(pair) != r_bonds[pair])
    if not formed and not broken:
        return None
    return f"F={';'.join(formed)}|B={';'.join(broken)}"
