"""General condition extraction: reagent bag -> (metal, ligand, base family, solvent class, temp).

Generalises the BH-tuned :mod:`bhpu.chem.conditions` engine to the whole reaction corpus. It reuses
that module's validated primitives -- :func:`~bhpu.chem.conditions.extract_ligand` (precatalyst ==
free ligand) and :func:`~bhpu.chem.conditions.base_anion` (counter-ion-invariant base) -- and adds
broad **family maps** (base -> carbonate/phosphate/amine/..., solvent -> ethereal/polar_aprotic/
protic/...) so that "small deviations" (a different carbonate salt, a different ethereal solvent)
collapse into the same condition cluster. The family tables are seeded from the curated
:data:`~pubench.features.condition_descriptors.BASE_TABLE` / ``SOLVENT_TABLE`` and extended with the
most common remaining reagents; unrecognised bases/solvents return ``None`` (their own bucket).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final

from rdkit import Chem, RDLogger

from bhpu.chem.canonical import inchikey
from bhpu.chem.conditions import base_anion, extract_ligand

RDLogger.DisableLog("rdApp.*")

#: Transition metals that mark a catalytic system (same set as the BH condition engine).
_METALS: Final[dict[int, str]] = {
    46: "Pd",
    29: "Cu",
    28: "Ni",
    78: "Pt",
    79: "Au",
    45: "Rh",
    44: "Ru",
    27: "Co",
    26: "Fe",
    23: "V",
    25: "Mn",
    47: "Ag",
}


def _family_map(smiles_to_family: dict[str, str]) -> dict[str, str]:
    """Convert a {SMILES -> family} map into a {connectivity InChIKey -> family} map."""
    out: dict[str, str] = {}
    for smi, fam in smiles_to_family.items():
        key = inchikey(smi)
        if key is not None:
            out[key] = fam
    return out


#: Base connectivity-InChIKey -> family (built from representative base anions/neutral bases).
_BASE_FAMILY: Final[dict[str, str]] = _family_map(
    {
        "O=C([O-])[O-]": "carbonate",
        "OC(=O)[O-]": "carbonate",
        "O=P([O-])([O-])[O-]": "phosphate",
        "O=P(O)([O-])[O-]": "phosphate",
        "[OH-]": "hydroxide",
        "CC(C)(C)[O-]": "alkoxide",
        "[O-]C": "alkoxide",
        "CC[O-]": "alkoxide",
        "[F-]": "fluoride",
        "CC(=O)[O-]": "acetate",
        "[H-]": "hydride",
        "CCN(CC)CC": "amine",
        "CN(C)C": "amine",
        "CCN(C(C)C)C(C)C": "amine",
        "c1ccncc1": "pyridine",
        "Cc1ccncc1": "pyridine",
        "CN(C)c1ccncc1": "pyridine",
        "C1CCC2=NCCCN2CC1": "amidine",
        "CN1CCCN2CCCN=C12": "amidine",
        "[N-]([Si](C)(C)C)[Si](C)(C)C": "silazide",
        "NC(N)=N": "guanidine",
    }
)

#: Solvent connectivity-InChIKey -> class.
_SOLVENT_FAMILY: Final[dict[str, str]] = _family_map(
    {
        "C1CCOC1": "ethereal",
        "C1COCCO1": "ethereal",
        "CC1CCCO1": "ethereal",
        "CCOCC": "ethereal",
        "COCCOC": "ethereal",
        "COC(C)(C)C": "ethereal",
        "Cc1ccccc1": "aromatic",
        "Cc1ccccc1C": "aromatic",
        "c1ccccc1": "aromatic",
        "Clc1ccccc1": "aromatic",
        "CS(C)=O": "polar_aprotic",
        "CN(C)C=O": "polar_aprotic",
        "CC(=O)N(C)C": "polar_aprotic",
        "CN1CCCC1=O": "polar_aprotic",
        "CC#N": "polar_aprotic",
        "CC(C)=O": "polar_aprotic",
        "O=CN(C)C": "polar_aprotic",
        "CO": "protic",
        "CCO": "protic",
        "CC(C)O": "protic",
        "CC(C)(C)O": "protic",
        "CCC(C)(C)O": "protic",
        "O": "protic",
        "ClCCl": "chlorinated",
        "ClC(Cl)Cl": "chlorinated",
        "ClCCCl": "chlorinated",
        "CCCCCC": "alkane",
        "C1CCCCC1": "alkane",
        "CCCCCCC": "alkane",
        "CCCCC": "alkane",
        "CCOC(C)=O": "ester",
        "COC(C)=O": "ester",
    }
)


@lru_cache(maxsize=200_000)
def metals_in(smiles: str) -> frozenset[str]:
    """Transition-metal element symbols present in a reagent SMILES (cached; reagents repeat)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return frozenset()
    return frozenset(
        _METALS[a.GetAtomicNum()] for a in mol.GetAtoms() if a.GetAtomicNum() in _METALS
    )


@lru_cache(maxsize=200_000)
def is_reagent(smiles: str) -> bool:
    """Heuristically decide if a molecule is a reagent/condition rather than a substrate.

    Used to separate substrates from catalysts/bases/solvents/gases in sources that pool everything
    into one reactant list (CHORISO, CJHIF). A molecule is treated as a reagent if it carries a
    transition metal, is a recognised base or solvent, or is a single-heavy-atom / diatomic species
    (H2, HCl, O2, monatomic ions). Everything else is a substrate.

    Args:
        smiles: A molecule SMILES from a pooled reactant bag.

    Returns:
        ``True`` if the molecule should be treated as a condition reagent, else ``False``.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    if any(a.GetAtomicNum() in _METALS for a in mol.GetAtoms()):
        return True
    if mol.GetNumHeavyAtoms() <= 1:  # H2 / HCl / O2 / monatomic ions
        return True
    key = inchikey(smiles) or ""
    return key in _SOLVENT_FAMILY or key in _BASE_FAMILY


def metal_family(metals: frozenset[str]) -> str:
    """Collapse the metal set to a cluster label (``"none"`` if metal-free)."""
    return "+".join(sorted(metals)) if metals else "none"


def temp_bin(temperature: float | None) -> str:
    """Bin a temperature (deg C; ORDerly is Celsius) into rt / 40-80 / 80-120 / >120 / unknown."""
    if temperature is None:
        return "unknown"
    if temperature < 35.0:
        return "rt"
    if temperature < 80.0:
        return "40-80"
    if temperature < 120.0:
        return "80-120"
    return ">120"


class ConditionExtractor:
    """Extract the clustering condition fields from a reaction's reagent (agent) bag."""

    def extract(
        self, agents: tuple[str, ...], temperature: float | None
    ) -> tuple[str, str | None, str | None, str | None, str]:
        """Return ``(metal, ligand_key, base_family, solvent_family, temp_bin)`` for the agents.

        Args:
            agents: SMILES of the condition bag (catalyst/ligand/base/solvent/reagents).
            temperature: Reported temperature in degrees Celsius, if any.

        Returns:
            A 5-tuple: metal-family label, ligand connectivity InChIKey (or ``None``), base family
            (or ``None``), solvent class (or ``None``), and the temperature bin label.
        """
        metals: frozenset[str] = frozenset()
        ligand_key: str | None = None
        base_family: str | None = None
        solvent_family: str | None = None
        for agent in agents:
            metals |= metals_in(agent)
            if ligand_key is None and (lig := extract_ligand(agent)) is not None:
                ligand_key = inchikey(lig)
            if base_family is None and (anion := base_anion(agent)) is not None:
                base_family = _BASE_FAMILY.get(inchikey(anion) or "")
            if solvent_family is None:
                solvent_family = _SOLVENT_FAMILY.get(inchikey(agent) or "")
        return metal_family(metals), ligand_key, base_family, solvent_family, temp_bin(temperature)
