"""Molecule standardisation and the condition key (docs/design/rebuild.md, Stage 6).

Runs after mapping and classification (RXNMapper rewrites molecules; Gimadiev et al. 2021). Substrates
are reduced to their largest organic fragment (lab inventory SMILES carry counter-ions and ``[H+]``;
reactions never do) and neutralised with RDKit's Uncharger, exactly as rxnutils'
``neutralize_molecules`` does (re-parse, uncharge, write). It is called directly because importing
rxnutils.chem pulls in rdchiral, which switches RDKit's log off for the whole process:
carboxylate → acid, ammonium → amine; nitro, azide and quaternary N stay. Agents and solvents keep their ions and get the
synonyms from ``configs/standardise.yml``. Built only for what the data has (counted 2026-10-09).
"""

from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

Role = Literal["substrate", "agent", "solvent"]

# How a condition that could not be read is written in reports and benchmark labels.
UNREADABLE = "(unreadable)"

# Keep the largest organic fragment, and neutralise; both stateless, so shared instances are safe.
_LARGEST = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
_UNCHARGER = rdMolStandardize.Uncharger()


@dataclass(frozen=True)
class StdConfig:
    """Standardisation choices, canonicalised when loaded.

    Attributes:
        std_version: Version written into every result; change it with the config.
        synonyms: Canonical agent/solvent SMILES → the canonical SMILES used instead.
        counter_ions: Canonical SMILES dropped from condition keys.
    """

    std_version: str
    synonyms: Mapping[str, str]
    counter_ions: frozenset[str]


@dataclass(frozen=True)
class StdResult:
    """One standardised molecule.

    Attributes:
        original: The SMILES as given.
        smiles: Canonical standardised SMILES, or None when it failed (see ``reason``).
        reason: Why it failed (``"unparsable"``, ``"neutralise_failed"``, ``"no_inchikey"``),
            else None.
        key_stereo: Full InChIKey of the standardised molecule.
        key_flat: First InChIKey block: stereo-agnostic, used to match P against benchmarks.
        std_version: The config version that produced it.
    """

    original: str
    smiles: str | None
    reason: str | None
    key_stereo: str | None
    key_flat: str | None
    std_version: str


def load_std_config(path: Path) -> StdConfig:
    """Read ``standardise.yml``; every SMILES in it must parse, else ValueError."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    synonyms = {_canonical(k): _canonical(v) for k, v in config["synonyms"].items()}
    return StdConfig(
        std_version=str(config["std_version"]),
        synonyms=synonyms,
        counter_ions=frozenset(_canonical(ion) for ion in config["counter_ions"]),
    )


def standardize(smiles: str, role: Role, config: StdConfig) -> StdResult:
    """Standardise one molecule for its role; an unreadable SMILES gives None with a reason.

    RDKit's log is blocked while it runs: failures are counted by reason, not logged per row.
    """
    with rdBase.BlockLogs():
        return _standardize(smiles, role, config)


def _standardize(smiles: str, role: Role, config: StdConfig) -> StdResult:
    """``standardize`` without the log block."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return StdResult(smiles, None, "unparsable", None, None, config.std_version)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    if role == "substrate":
        neutral = _neutral_smiles(_LARGEST.choose(mol))
        parsed = Chem.MolFromSmiles(neutral) if neutral is not None else None
        if neutral is None or parsed is None:
            return StdResult(
                smiles, None, "neutralise_failed", None, None, config.std_version
            )
        mol, canonical = parsed, neutral
    else:
        canonical = str(Chem.MolToSmiles(mol))
        canonical = config.synonyms.get(canonical, canonical)
        mol = Chem.MolFromSmiles(canonical)
    key = str(Chem.MolToInchiKey(mol))
    # RDKit writes no key for a dummy atom (*); without one nothing ever matches
    if not key:
        return StdResult(smiles, None, "no_inchikey", None, None, config.std_version)
    return StdResult(
        smiles, canonical, None, key, key.split("-")[0], config.std_version
    )


def required(result: StdResult, record_id: str) -> StdResult:
    """``result``, or ValueError naming the record when standardisation failed.

    For substrates and products of classified reactions and benchmarks: none failed in full ORDerly
    or Chemlex (counted 2026-10-20), so a failure is input we have not seen. It stops the stage
    instead of leaving a null molecule that later joins and groupings would drop silently.
    """
    if result.smiles is None:
        raise ValueError(
            f"{record_id}: {result.original} failed standardisation ({result.reason})"
        )
    return result


def condition_key(
    agents: Iterable[str], solvents: Iterable[str], config: StdConfig
) -> tuple[str, ...] | None:
    """Role-agnostic condition: sorted set of standardised agents and solvents, counter-ions dropped.

    Returns None when any of them cannot be read, so an unknown condition is never matched.
    """
    parts = set()
    for smiles in (*agents, *solvents):
        result = standardize(smiles, "agent", config)
        if result.smiles is None:
            return None
        if result.smiles not in config.counter_ions:
            parts.add(result.smiles)
    return tuple(sorted(parts))


def in_family(condition: Iterable[str] | None, members: AbstractSet[str]) -> bool:
    """A condition is in a family when it contains every member; extra reagents are allowed.

    The one rule for positives and benchmark splits (decided 2026-10-09). An unreadable condition
    (None) is in no family.
    """
    return condition is not None and members <= set(condition)


def condition_text(key: Iterable[str] | None) -> str:
    """A condition key as one string, ``"A + B"``; ``"(unreadable)"`` for None."""
    return " + ".join(key) if key is not None else UNREADABLE


def condition_from_text(text: str) -> tuple[str, ...] | None:
    """The inverse of ``condition_text``: ``"A + B"`` → ``("A", "B")``, ``""`` → ``()``,
    ``"(unreadable)"`` → None. SMILES hold no spaces, so the separator is unambiguous."""
    if text == UNREADABLE:
        return None
    return tuple(text.split(" + ")) if text else ()


def _neutral_smiles(mol: Chem.Mol) -> str | None:
    """SMILES of the uncharged molecule; None when its SMILES cannot be read back first.

    The same steps as rxnutils' ``neutralize_molecules``: SMILES → molecule → Uncharger → SMILES.
    The caller reads the result once more, which catches charges the Uncharger cannot fix
    (cyclopentadienides: ``neutralise_failed``).
    """
    parsed = Chem.MolFromSmiles(str(Chem.MolToSmiles(mol)))
    if parsed is None:
        return None
    return str(Chem.MolToSmiles(_UNCHARGER.uncharge(parsed)))


def _canonical(smiles: str) -> str:
    """Canonical SMILES of a config entry; a typo stops loading instead of never matching."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparsable SMILES in the standardisation config: {smiles}")
    return str(Chem.MolToSmiles(mol))
