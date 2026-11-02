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
        near_radius: Radius of the fingerprint rooted on every atom within ``near_distance`` bonds
            of the reacting atom.
        near_bits: Its length.
        near_distance: How far from the reacting atom a root may sit. 0 is ``morgan_centre``, and a
            distance past the molecule's diameter is ``morgan``. It is chosen **per reaction class**
            by the label-free rule of ADR 030 -- amide coupling gets 6 -- which is why the column is
            called ``morgan_near`` and carries no number: ``morgan_d6`` would lie for a class that
            selects 3.
        electron_withdrawing: Patterns counted on the aromatic ring of the reacting atom, in file
            order. Which groups withdraw electrons does not depend on the reaction class, so they
            live here and not in the class YAML.
    """

    morgan_radius: int
    morgan_bits: int
    centre_radius: int
    centre_bits: int
    near_radius: int
    near_bits: int
    near_distance: int
    electron_withdrawing: tuple[Chem.Mol, ...]


def load_feature_config(path: Path) -> FeatureConfig:
    """Read ``configs/features.yml``; a missing key or a SMARTS that does not parse raises.

    Raises:
        ValueError: a block is missing, ``morgan_near``'s distance is absent, fractional, a word or
            negative, or a SMARTS does not parse. The message names the file, because the caller
            sees only a path.
    """
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        morgan, centre, near = (
            config["morgan"],
            config["morgan_centre"],
            config["morgan_near"],
        )
        distance = near["distance"]
        if isinstance(distance, bool) or not isinstance(distance, int) or distance < 0:
            raise ValueError(
                f"morgan_near.distance must be a whole number of bonds, 0 or more, "
                f"not {distance!r}"
            )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} is not a features config: {error}") from error
    return FeatureConfig(
        morgan_radius=int(morgan["radius"]),
        morgan_bits=int(morgan["n_bits"]),
        centre_radius=int(centre["radius"]),
        centre_bits=int(centre["n_bits"]),
        near_radius=int(near["radius"]),
        near_bits=int(near["n_bits"]),
        near_distance=distance,
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
            mol, smiles, config.centre_radius, config.centre_bits, roots=[centre]
        ),
        "morgan_near": _morgan(
            mol,
            smiles,
            config.near_radius,
            config.near_bits,
            roots=_within(mol, centre, config.near_distance),
        ),
    }


def _within(mol: Chem.Mol, centre: int, distance: int) -> list[int]:
    """Every atom at most ``distance`` bonds from the reacting atom, the centre included.

    The same distance matrix ``_descriptors`` already walks, so the mechanism is not new to this
    module. At distance 0 this is ``[centre]``, which is exactly ``morgan_centre``.
    """
    bonds = Chem.GetDistanceMatrix(mol)[centre]
    return [atom for atom in range(mol.GetNumAtoms()) if bonds[atom] <= distance]


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
        columns=[*DESCRIPTORS, "subclass", "morgan", "morgan_centre", "morgan_near"],
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
    mol: Chem.Mol, smiles: str, radius: int, bits: int, roots: list[int] | None = None
) -> np.ndarray:
    """Morgan count fingerprint as uint8, without chirality (stereo stays in the key).

    With ``roots`` only the environments centred on those atoms are counted: one root is the
    reaction centre alone (``morgan_centre``, at most ``radius + 1`` bits), every atom within d
    bonds is ``morgan_near``, and ``None`` is the whole molecule.

    **The series is not nested at its first step**, which is worth knowing before anyone reasons
    from it. RDKit drops an environment whose bond set lies inside another it has recorded, so the
    radius-2 environment of a *terminal* reacting atom survives when it is the only root and is
    swallowed once its neighbour becomes one: on ethylamine, bit 317 is set at distance 0 and gone
    at distance 1 (measured on 7 of 14 fixtures, 2026-11-02;
    ``tests/test_features.py::test_near_loses_the_radius_two_bit_of_a_terminal_centre_at_the_first_step``).
    From distance 1 upward containment holds. ``configs/features.yml`` already records the same
    quirk for ``morgan_centre``. Default RDKit is kept rather than
    ``includeRedundantEnvironments=True``, which would restore nesting but break the bit-for-bit
    match between a large distance and ``morgan`` -- and that match is what makes the whole distance
    table comparable with every number measured under the two older columns (ADR 030).
    """
    if roots is not None and not roots:
        # RDKit reads an empty `fromAtoms` as the WHOLE molecule, verified bit for bit on ethylamine
        # (2026-11-02), so an empty root set would silently return `morgan` and contradict every
        # sentence above. Unreachable today -- `_within` always holds the centre, because its
        # distance to itself is 0 and `load_feature_config` refuses a negative distance -- and
        # refused rather than branched on, so it stays unreachable (§8).
        raise ValueError(
            "roots must not be empty: RDKit would read it as the whole molecule and return "
            "`morgan` under another block's name"
        )
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    counts = (
        generator.GetCountFingerprintAsNumPy(mol, fromAtoms=roots)
        if roots is not None
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
