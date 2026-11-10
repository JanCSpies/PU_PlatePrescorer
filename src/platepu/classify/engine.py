"""Generic two-substrate class engine (docs/design/rebuild.md, Stage 5; ADR 002).

A class (``configs/classes/<name>.yml``) names two roles, each a SMARTS with the reacting atom marked
``:1``, and the bond formed between them. In a mapped reaction, a *formed bond* is a product bond
between atoms that came from two different reactant molecules. The class matches when one atom of a
formed bond is role A's reacting atom and the other is role B's. Nothing here knows about acids or
amines; the YAML does. Dynamic bonds are recorded, never filtered on (positives do that).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import yaml
from rdkit import Chem, rdBase

from platepu.io import REJECTED_SCHEMA

CLASSIFIED_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("licence", pa.string()),
        ("class", pa.string()),
        ("substrate_a", pa.string()),
        ("substrate_b", pa.string()),
        ("n_reactants", pa.int64()),
        ("n_products", pa.int64()),
        ("n_dynamic_bonds", pa.int64()),
        ("clean_rsmi", pa.string()),
    ]
)

# (reactant index, atom map number, reactant index, atom map number) of one formed bond
FormedBond = tuple[int, int, int, int]
# (class name, substrate A, substrate B), substrates as unmapped SMILES
Match = tuple[str, str, str]


@dataclass(frozen=True)
class Subclass:
    """One named subclass of a role (e.g. aniline among amines).

    Attributes:
        name: Subclass name from the YAML.
        pattern: Its SMARTS; it counts only when the atom at ``reacting`` is the role's reacting
            atom, so a group elsewhere in the molecule cannot name the subclass.
        reacting: Index of the ``:1`` atom in ``pattern``.
    """

    name: str
    pattern: Chem.Mol
    reacting: int


@dataclass(frozen=True)
class ClassRule:
    """One reaction class from its YAML file.

    Attributes:
        name: Class name, e.g. ``"amide_coupling"``.
        role_a: Name of the first role in ``formed_bond`` (e.g. ``"acid"``).
        role_b: Name of the second role (e.g. ``"amine"``).
        pattern_a: SMARTS of role A; ``reacting_a`` is the index of its ``:1`` atom.
        pattern_b: SMARTS of role B; ``reacting_b`` is the index of its ``:1`` atom.
        allowed_dynamic_bonds: CGR dynamic-bond counts a positive may have.
        subclasses_a: Subclasses of role A in YAML order (first match wins); empty if none.
        subclasses_b: Subclasses of role B.
    """

    name: str
    role_a: str
    role_b: str
    pattern_a: Chem.Mol
    pattern_b: Chem.Mol
    reacting_a: int
    reacting_b: int
    allowed_dynamic_bonds: tuple[int, ...]
    subclasses_a: tuple[Subclass, ...] = ()
    subclasses_b: tuple[Subclass, ...] = ()


def load_class(path: Path) -> ClassRule:
    """Read a class YAML; a SMARTS that does not parse or has no ``:1`` atom raises ValueError."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    role_a, role_b = config["formed_bond"]
    pattern_a = parse_smarts(config["roles"][role_a]["smarts"])
    pattern_b = parse_smarts(config["roles"][role_b]["smarts"])
    return ClassRule(
        name=config["name"],
        role_a=role_a,
        role_b=role_b,
        pattern_a=pattern_a,
        pattern_b=pattern_b,
        reacting_a=_reacting_index(pattern_a),
        reacting_b=_reacting_index(pattern_b),
        allowed_dynamic_bonds=tuple(config["allowed_dynamic_bonds"]),
        subclasses_a=_subclasses(config["roles"][role_a]),
        subclasses_b=_subclasses(config["roles"][role_b]),
    )


def _subclasses(role: Mapping[str, Any]) -> tuple[Subclass, ...]:
    """A role's optional ``subclasses`` list, each SMARTS checked like a role SMARTS."""
    subclasses = []
    for entry in role.get("subclasses", []):
        pattern = parse_smarts(entry["smarts"])
        subclasses.append(Subclass(entry["name"], pattern, _reacting_index(pattern)))
    return tuple(subclasses)


def match_reaction(rsmi: str, rules: Sequence[ClassRule]) -> list[Match]:
    """All ``(class, substrate_a, substrate_b)`` matches of one mapped reaction, substrates unmapped."""
    return _classify(rsmi, rules)[1]


def fits_roles(substrate_a: str, substrate_b: str, rule: ClassRule) -> bool:
    """Whether two given molecules can play the class's roles (benchmarks: roles from their columns).

    Used where the roles are known without a mapping, so a mapping error never changes the answer.
    """
    with (
        rdBase.BlockLogs()
    ):  # an unreadable benchmark molecule is "not in class", not a log line
        mol_a, mol_b = Chem.MolFromSmiles(substrate_a), Chem.MolFromSmiles(substrate_b)
    if mol_a is None or mol_b is None:
        return False
    return mol_a.HasSubstructMatch(rule.pattern_a) and mol_b.HasSubstructMatch(
        rule.pattern_b
    )


def classify_table(
    cleaned: pa.Table, rules: Sequence[ClassRule]
) -> tuple[pa.Table, pa.Table]:
    """Stage 5: one row per (reaction, class match).

    Rejects, with the first reason that applies: ``unparsable`` (a molecule RDKit cannot read),
    ``no_formed_bond`` (no bond between atoms of two reactant molecules; includes one-substrate and
    intramolecular reactions), ``no_class`` (a formed bond no class fits).
    """
    rows, rejected = [], []
    for row in cleaned.to_pylist():
        reason, matches = _classify(row["clean_rsmi"], rules)
        if reason:
            rejected.append({"record_id": row["record_id"], "reason": reason})
        for name, substrate_a, substrate_b in matches:
            rows.append(
                {
                    **{key: row[key] for key in CLASSIFIED_SCHEMA.names if key in row},
                    "class": name,
                    "substrate_a": substrate_a,
                    "substrate_b": substrate_b,
                }
            )
    return (
        pa.Table.from_pylist(rows, schema=CLASSIFIED_SCHEMA),
        pa.Table.from_pylist(rejected, schema=REJECTED_SCHEMA),
    )


def parse_smarts(smarts: str) -> Chem.Mol:
    """Parse a SMARTS from a config file, stopping on a typo instead of matching nothing.

    The one place a config SMARTS is read, for roles, subclasses and the feature patterns alike.
    """
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        raise ValueError(f"unparsable SMARTS: {smarts}")
    return pattern


def _reacting_index(pattern: Chem.Mol) -> int:
    """Index of the atom marked ``:1`` in a role SMARTS."""
    marked = [atom.GetIdx() for atom in pattern.GetAtoms() if atom.GetAtomMapNum() == 1]
    if len(marked) != 1:
        raise ValueError("a role SMARTS needs exactly one atom marked :1")
    return marked[0]


def _classify(rsmi: str, rules: Sequence[ClassRule]) -> tuple[str, list[Match]]:
    """The matches of one mapped reaction and "", or no matches and the reason why.

    The reaction is parsed once; stage 5 rejects most reactions, so a second parse to find the
    reason would double its run time.
    """
    parsed = _parse(rsmi)
    if parsed is None:
        return "unparsable", []
    reactants, product = parsed
    formed = _formed_bonds(reactants, product)
    if not formed:
        return "no_formed_bond", []
    matches = [
        (rule.name, _unmapped(reactants[index_a]), _unmapped(reactants[index_b]))
        for rule in rules
        for index_a, index_b in _role_pairs(reactants, formed, rule)
    ]
    return ("" if matches else "no_class"), matches


def _parse(rsmi: str) -> tuple[list[Chem.Mol], Chem.Mol] | None:
    """Reactant molecules and the product side of ``reactants>reagents>products``; None if RDKit
    cannot read one of them."""
    left, _, right = rsmi.split(">")
    with (
        rdBase.BlockLogs()
    ):  # unreadable reactions are counted as rejects, not logged per row
        reactants = [Chem.MolFromSmiles(smiles) for smiles in left.split(".")]
        product = Chem.MolFromSmiles(right)
    if product is None or any(mol is None for mol in reactants):
        return None
    return reactants, product


def _formed_bonds(reactants: Sequence[Chem.Mol], product: Chem.Mol) -> list[FormedBond]:
    """Product bonds between atoms that came from two different reactant molecules."""
    owner = {
        atom.GetAtomMapNum(): index
        for index, mol in enumerate(reactants)
        for atom in mol.GetAtoms()
        if atom.GetAtomMapNum()
    }
    formed = []
    for bond in product.GetBonds():
        begin_map = bond.GetBeginAtom().GetAtomMapNum()
        end_map = bond.GetEndAtom().GetAtomMapNum()
        if (
            begin_map in owner
            and end_map in owner
            and owner[begin_map] != owner[end_map]
        ):
            formed.append((owner[begin_map], begin_map, owner[end_map], end_map))
    return formed


def _role_pairs(
    reactants: Sequence[Chem.Mol], formed: Sequence[FormedBond], rule: ClassRule
) -> set[tuple[int, int]]:
    """``(index of role A molecule, index of role B molecule)`` for every formed bond the rule fits.

    Each formed bond is tried both ways round, since the mapping does not say which end is role A.
    """
    pairs = set()
    for index_1, map_1, index_2, map_2 in formed:
        for index_a, map_a, index_b, map_b in (
            (index_1, map_1, index_2, map_2),
            (index_2, map_2, index_1, map_1),
        ):
            fits_a = map_a in _reacting_maps(
                reactants[index_a], rule.pattern_a, rule.reacting_a
            )
            if fits_a and map_b in _reacting_maps(
                reactants[index_b], rule.pattern_b, rule.reacting_b
            ):
                pairs.add((index_a, index_b))
    return pairs


def _reacting_maps(mol: Chem.Mol, pattern: Chem.Mol, index: int) -> set[int]:
    """Atom-map numbers of the atoms in ``mol`` that can be the pattern's reacting atom."""
    return {
        mol.GetAtomWithIdx(match[index]).GetAtomMapNum()
        for match in mol.GetSubstructMatches(pattern)
    }


def _unmapped(mol: Chem.Mol) -> str:
    """Canonical SMILES of a molecule without its atom-map numbers."""
    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    return str(Chem.MolToSmiles(copy))
