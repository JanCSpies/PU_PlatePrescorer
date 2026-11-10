"""Stage 2: one canonical, unmapped reaction SMILES per record, and its hash.

Every molecule of a record (reactants, agents, solvents) goes left of ``>>``; RXNMapper and rxnutils
decide later which of them are reactants (ADR 002). Each molecule is written as RDKit canonical SMILES
without atom-map numbers and each side is sorted, so the same reaction from two sources gives the same
string and is mapped once. This is not standardisation: charges, salts and stereo stay as given.
"""

import hashlib
from collections.abc import MutableMapping, Sequence

import pyarrow as pa
from rdkit import Chem, rdBase

from platepu.ingest.record import ReactionRecord, table_to_records
from platepu.io import REJECTED_SCHEMA

RSMI_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("licence", pa.string()),
        ("rsmi", pa.string()),
        ("rsmi_hash", pa.string()),
    ]
)

Cache = MutableMapping[str, str | None]


def reaction_smiles(record: ReactionRecord, cache: Cache) -> str | None:
    """Return ``"sorted left molecules>>sorted products"``, or None if a molecule is unparsable.

    Args:
        record: The reaction as the adapter gave it.
        cache: SMILES → canonical SMILES (or None) seen so far; pass one dict per table so a
            molecule that occurs in many reactions is parsed once.
    """
    left = _canonical_all((*record.reactants, *record.agents, *record.solvents), cache)
    right = _canonical_all(record.products, cache)
    if left is None or right is None:
        return None
    return ".".join(sorted(left)) + ">>" + ".".join(sorted(right))


def rsmi_table(records: pa.Table) -> tuple[pa.Table, pa.Table]:
    """Build the stage-2 table from the stage-1 table: one row per readable record.

    Returns:
        The table (``record_id``, ``licence``, ``rsmi``, ``rsmi_hash``) and the rejected rows
        (``record_id``, ``reason``; the only reason is ``unparsable``).
    """
    cache: Cache = {}
    rows, rejected = [], []
    for record in table_to_records(records):
        rsmi = reaction_smiles(record, cache)
        if rsmi is None:
            rejected.append({"record_id": record.record_id, "reason": "unparsable"})
            continue
        rows.append(
            {
                "record_id": record.record_id,
                "licence": record.licence,
                "rsmi": rsmi,
                "rsmi_hash": hashlib.sha1(rsmi.encode()).hexdigest(),
            }
        )
    return (
        pa.Table.from_pylist(rows, schema=RSMI_SCHEMA),
        pa.Table.from_pylist(rejected, schema=REJECTED_SCHEMA),
    )


def _canonical_all(smiles: Sequence[str], cache: Cache) -> list[str] | None:
    """Canonical SMILES of every molecule, or None as soon as one cannot be read."""
    out = []
    for one in smiles:
        if one not in cache:
            cache[one] = _canonical(one)
        canonical = cache[one]
        if canonical is None:
            return None
        out.append(canonical)
    return out


def _canonical(smiles: str) -> str | None:
    """RDKit canonical SMILES without atom-map numbers; None if RDKit cannot read it.

    Unparsable molecules are counted as rejects, so RDKit's log is blocked while parsing, not logged
    one line per row.
    """
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return str(Chem.MolToSmiles(mol))
