"""One row of features per substrate (docs/design/modeling.md, "Stage F — features"; ADR 006, item 2).

Morgan counts and ten descriptors around the reacting atom R of the role SMARTS: nine from the
graph, plus one electron-withdrawing count from SMARTS (``n_ewg_on_ring``). R and its
group G come from the class YAML, so nothing here knows about acids or amines. The alpha atoms are
the heavy neighbours of R outside G; shell counts are heavy atoms outside G at 2 and 3 bonds from R.
A substrate is identified by ``key_stereo``; a key seen with several SMILES (tautomers merged by the
InChIKey, counted 29 on 2026-10-21) takes the smallest string, so a run is repeatable.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from platepu.classify.engine import ClassRule, parse_smarts
from platepu.classify.groups import reacting_group, subclass_of
from platepu.io import strictest_licence

DESCRIPTORS = (
    "n_heavy_atoms",
    "centre_in_ring",
    "n_alpha",
    "alpha_n_aromatic",
    "alpha_n_in_ring",
    "alpha_max_heavy_degree",
    "alpha_sum_heavy_degree",
    "n_heavy_at_2_bonds",
    "n_heavy_at_3_bonds",
    "n_ewg_on_ring",
)
SOURCES = ("positives", "library", "validation")
SUBSTRATE = ["key_stereo", "key_flat", "smiles", "role", "licence", "std_version"]
MAX_COUNT = int(np.iinfo(np.uint8).max)


@dataclass(frozen=True)
class FeatureConfig:
    """Feature settings from ``configs/features.yml``.

    Attributes:
        morgan_radius: Morgan radius (2, as PAYN).
        morgan_bits: Fingerprint length (2,048).
        centre_radius: Radius of the fingerprint rooted at the reacting atom.
        centre_bits: Its length.
        electron_withdrawing: Patterns counted on the aromatic ring of the reacting atom, in file
            order. Which groups withdraw electrons does not depend on the reaction class, so they
            live here and not in the class YAML.
    """

    morgan_radius: int
    morgan_bits: int
    centre_radius: int
    centre_bits: int
    electron_withdrawing: tuple[Chem.Mol, ...]


def load_feature_config(path: Path) -> FeatureConfig:
    """Read ``configs/features.yml``; a SMARTS that does not parse raises ValueError."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    morgan, centre = config["morgan"], config["morgan_centre"]
    return FeatureConfig(
        morgan_radius=int(morgan["radius"]),
        morgan_bits=int(morgan["n_bits"]),
        centre_radius=int(centre["radius"]),
        centre_bits=int(centre["n_bits"]),
        electron_withdrawing=tuple(
            parse_smarts(entry["smarts"]) for entry in config["electron_withdrawing"]
        ),
    )


def substrate_features(
    smiles: str, role: str, rule: ClassRule, config: FeatureConfig
) -> dict[str, object]:
    """The descriptors, subclass and Morgan counts (uint8 array) of one substrate.

    Raises:
        ValueError: unreadable SMILES, not exactly one group of ``role``, or a Morgan count
            above 255 (counted max 51), which uint8 cannot hold.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparsable SMILES: {smiles}")
    group, centre = reacting_group(mol, role, rule)
    return {
        **_descriptors(mol, group, centre, config.electron_withdrawing),
        "subclass": subclass_of(smiles, role, rule),
        "morgan": _morgan(mol, smiles, config.morgan_radius, config.morgan_bits),
        "morgan_centre": _morgan(
            mol, smiles, config.centre_radius, config.centre_bits, root=centre
        ),
    }


def substrate_table(
    positives: pd.DataFrame,
    library: pd.DataFrame,
    validation: pd.DataFrame,
    rule: ClassRule,
    config: FeatureConfig,
) -> pd.DataFrame:
    """One row per ``key_stereo`` over the three inputs, sorted by (role, key_stereo).

    Args:
        positives: Positive pairs (any family) with side columns ``key_{a,b}_stereo``,
            ``key_{a,b}_flat``, ``substrate_{a,b}``, plus ``licence`` and ``std_version``.
        library: Library compounds with ``key_stereo``, ``key_flat``, ``smiles``, ``role``,
            ``licence``, ``std_version``.
        validation: Benchmark pairs with the side columns, ``std_version`` and ``licence``. The
            licence is read, never stamped: until 2026-10-22 this assigned ``"open"``, which
            mislabelled every Chemlex substrate (ADR 009). A table without the column stops the
            run rather than defaulting.
        rule: The class (roles, SMARTS, subclasses).
        config: Fingerprint settings.

    Raises:
        ValueError: mixed ``std_version``, a library role outside the class, a key under two
            roles, or any failure of ``substrate_features``.
    """
    seen = pd.concat(
        [
            _sides(positives, rule).assign(source="positives"),
            library[SUBSTRATE].assign(source="library"),
            _sides(validation, rule).assign(source="validation"),
        ],
        ignore_index=True,
    )
    _check(seen, rule)
    keys = (
        seen.groupby("key_stereo", sort=True)
        .agg(
            key_flat=("key_flat", "first"),
            role=("role", "first"),
            smiles=("smiles", "min"),
            n_smiles=("smiles", "nunique"),
            std_version=("std_version", "first"),
            licence=("licence", strictest_licence),
        )
        .reset_index()
    )
    for source in SOURCES:
        keys[f"in_{source}"] = keys["key_stereo"].isin(
            seen.loc[seen["source"] == source, "key_stereo"]
        )
    features = pd.DataFrame(
        [
            substrate_features(smiles, role, rule, config)
            for smiles, role in zip(keys["smiles"], keys["role"], strict=True)
        ],
        columns=[*DESCRIPTORS, "subclass", "morgan", "morgan_centre"],
    )
    table = pd.concat([keys, features], axis=1)
    table = table.astype(dict.fromkeys([*DESCRIPTORS, "n_smiles"], "int16"))
    return table.sort_values(["role", "key_stereo"]).reset_index(drop=True)


def _descriptors(
    mol: Chem.Mol, group: frozenset[int], centre: int, patterns: tuple[Chem.Mol, ...]
) -> dict[str, int]:
    """The ten descriptors around R; atoms of the group G are never counted."""
    alpha = [
        atom
        for atom in mol.GetAtomWithIdx(centre).GetNeighbors()
        if atom.GetIdx() not in group and atom.GetAtomicNum() > 1
    ]
    degrees = [_heavy_degree(atom) for atom in alpha]
    distance = Chem.GetDistanceMatrix(mol)[centre]
    outside = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetIdx() not in group and atom.GetAtomicNum() > 1
    ]
    return {
        "n_heavy_atoms": mol.GetNumHeavyAtoms(),
        "centre_in_ring": int(mol.GetAtomWithIdx(centre).IsInRing()),
        "n_alpha": len(alpha),
        "alpha_n_aromatic": sum(int(atom.GetIsAromatic()) for atom in alpha),
        "alpha_n_in_ring": sum(int(atom.IsInRing()) for atom in alpha),
        "alpha_max_heavy_degree": max(degrees, default=0),
        "alpha_sum_heavy_degree": sum(degrees),
        "n_heavy_at_2_bonds": _at_distance(distance, outside, 2),
        "n_heavy_at_3_bonds": _at_distance(distance, outside, 3),
        "n_ewg_on_ring": _ewg_on_ring(mol, group, centre, patterns),
    }


def _ewg_on_ring(
    mol: Chem.Mol, group: frozenset[int], centre: int, patterns: tuple[Chem.Mol, ...]
) -> int:
    """Electron-withdrawing substituents on the aromatic ring system that carries R.

    Zero when no neighbour of R is aromatic. Substituents are counted by their attachment atom,
    so a ring-bound amide counts once although ``C(=O)X`` matches it through both X choices.
    Anilines whose ring carries one succeeded in 2.6 % of the Chemlex HATU validation reactions
    against 21.7 % for the rest (3 of 16 anilines, 2026-10-21): thin, but the Hammett slope for
    aniline acylation is steep, so the group belongs in the table.
    """
    system = _aromatic_system(mol, centre)
    if not system:
        return 0
    attached = {
        match[0]
        for pattern in patterns
        for match in mol.GetSubstructMatches(pattern)
        if match[0] not in system
        and match[0] not in group
        and any(
            neighbour.GetIdx() in system
            for neighbour in mol.GetAtomWithIdx(match[0]).GetNeighbors()
        )
    }
    return len(attached)


def _aromatic_system(mol: Chem.Mol, centre: int) -> frozenset[int]:
    """The aromatic rings fused to any aromatic neighbour of R; a biaryl partner is not fused.

    A diarylamine has two aromatic neighbours and both rings count, so the value does not depend on
    the order the atoms happen to have in the molecule.
    """
    system = {
        atom.GetIdx()
        for atom in mol.GetAtomWithIdx(centre).GetNeighbors()
        if atom.GetIsAromatic()
    }
    if not system:
        return frozenset()
    rings = [
        set(ring)
        for ring in mol.GetRingInfo().AtomRings()
        if all(mol.GetAtomWithIdx(index).GetIsAromatic() for index in ring)
    ]
    grown = True
    while grown:
        grown = False
        for ring in rings:
            if ring & system and not ring <= system:
                system |= ring
                grown = True
    return frozenset(system)


def _heavy_degree(atom: Chem.Atom) -> int:
    """Heavy neighbours of an atom, the bond to R included (methyl 1, tert-butyl C 4)."""
    return sum(1 for neighbour in atom.GetNeighbors() if neighbour.GetAtomicNum() > 1)


def _at_distance(distance: np.ndarray, atoms: Iterable[int], bonds: int) -> int:
    """How many of ``atoms`` lie exactly ``bonds`` bonds from R."""
    return sum(1 for atom in atoms if distance[atom] == bonds)


def _morgan(
    mol: Chem.Mol, smiles: str, radius: int, bits: int, root: int | None = None
) -> np.ndarray:
    """Morgan count fingerprint as uint8, without chirality (stereo stays in the key).

    With ``root`` only the environments centred on that atom are counted, so the result carries at
    most ``radius + 1`` bits and describes the reaction centre alone.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    counts = (
        generator.GetCountFingerprintAsNumPy(mol, fromAtoms=[root])
        if root is not None
        else generator.GetCountFingerprintAsNumPy(mol)
    )
    if int(counts.max()) > MAX_COUNT:
        raise ValueError(
            f"{smiles}: a Morgan count of {counts.max()} does not fit uint8"
        )
    return counts.astype(np.uint8)


def _sides(pairs: pd.DataFrame, rule: ClassRule) -> pd.DataFrame:
    """The substrates of a pair table, one row per pair side, with the role of that side."""
    sides = []
    for side, role in (("a", rule.role_a), ("b", rule.role_b)):
        names = {
            f"key_{side}_stereo": "key_stereo",
            f"key_{side}_flat": "key_flat",
            f"substrate_{side}": "smiles",
        }
        sides.append(pairs.rename(columns=names).assign(role=role)[SUBSTRATE])
    return pd.concat(sides, ignore_index=True)


def _check(seen: pd.DataFrame, rule: ClassRule) -> None:
    """Stop on mixed standardisation, a role outside the class, or a key under two roles."""
    versions = sorted(set(seen["std_version"]))
    if len(versions) > 1:
        raise ValueError(f"inputs mix std_version values {versions}")
    unknown = sorted(set(seen["role"]) - {rule.role_a, rule.role_b})
    if unknown:
        raise ValueError(f"roles {unknown} are not in class {rule.name}")
    n_roles = seen.groupby("key_stereo")["role"].nunique()
    both = sorted(n_roles[n_roles > 1].index)
    if both:
        raise ValueError(f"keys under both roles: {both[:5]} ({len(both)} in all)")
