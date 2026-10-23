"""A class's reactive groups on one molecule: how many, where, and which subclass.

Positives and the lab library ask how many groups of each role a substrate carries; features and the
per-class evaluation ask where the reacting atom is and which subclass it belongs to. This module is
the one place that answers them.
"""

from rdkit import Chem

from platepu.classify.engine import ClassRule, Subclass


def group_counts(smiles: str, rule: ClassRule) -> tuple[int, int]:
    """(groups of role A, groups of role B) in one molecule.

    A group is a distinct atom that can be the role's reacting atom, so a diacid has two acid groups.

    Raises:
        ValueError: if ``smiles`` cannot be read (standardised substrates always can).
    """
    mol = _mol(smiles)
    return (
        _reacting_atoms(mol, rule.pattern_a, rule.reacting_a),
        _reacting_atoms(mol, rule.pattern_b, rule.reacting_b),
    )


def reacting_group(
    mol: Chem.Mol, role: str, rule: ClassRule
) -> tuple[frozenset[int], int]:
    """The atoms of the role's group and its reacting atom R, for a molecule with exactly one group.

    The group is every atom of every match at R (for the acid: C, =O and OH).

    Raises:
        ValueError: the role is not in the class, or the molecule has zero or several groups.
    """
    pattern, reacting, _ = _role(role, rule)
    matches = mol.GetSubstructMatches(pattern)
    centres = {match[reacting] for match in matches}
    if len(centres) != 1:
        raise ValueError(
            f"{Chem.MolToSmiles(mol)}: {len(centres)} {role} groups, expected exactly one"
        )
    return frozenset(atom for match in matches for atom in match), centres.pop()


def subclass_of(smiles: str, role: str, rule: ClassRule) -> str | None:
    """The first subclass of ``role`` whose ``:1`` atom is the reacting atom; ``"other"`` if none.

    None when the role has no subclasses. Anchoring at R matters: an amine with an anilide N-H
    elsewhere is still aliphatic primary (821 of 9,973 positive amines change without it).

    Raises:
        ValueError: as ``reacting_group``, or when ``smiles`` cannot be read.
    """
    _, _, subclasses = _role(role, rule)
    mol = _mol(smiles)
    _, centre = reacting_group(mol, role, rule)
    if not subclasses:
        return None
    for subclass in subclasses:
        matches = mol.GetSubstructMatches(subclass.pattern)
        if any(match[subclass.reacting] == centre for match in matches):
            return subclass.name
    return "other"


def _role(role: str, rule: ClassRule) -> tuple[Chem.Mol, int, tuple[Subclass, ...]]:
    """(pattern, index of its ``:1`` atom, subclasses) of one role; a typo stops the run."""
    if role == rule.role_a:
        return rule.pattern_a, rule.reacting_a, rule.subclasses_a
    if role == rule.role_b:
        return rule.pattern_b, rule.reacting_b, rule.subclasses_b
    raise ValueError(f"class {rule.name} has no role {role!r}")


def _mol(smiles: str) -> Chem.Mol:
    """The molecule of a standardised SMILES; unreadable input stops the run."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparsable SMILES: {smiles}")
    return mol


def _reacting_atoms(mol: Chem.Mol, pattern: Chem.Mol, reacting: int) -> int:
    """Number of distinct atoms in ``mol`` that can be the pattern's reacting atom."""
    return len({match[reacting] for match in mol.GetSubstructMatches(pattern)})
