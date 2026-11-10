"""The common record every source adapter produces (see docs/design/rebuild.md, Stage 1)."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import pyarrow as pa

_SMILES = pa.list_(pa.string())
_SMILES_FIELDS = ("reactants", "agents", "solvents", "products")

RECORD_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("source", pa.string()),
        ("source_id", pa.string()),
        ("licence", pa.string()),
        ("reactants", _SMILES),
        ("agents", _SMILES),
        ("solvents", _SMILES),
        ("products", _SMILES),
        ("source_rxn", pa.string()),
        ("temperature_c", pa.float64()),
        ("reaction_time_h", pa.float64()),
        ("yield_pct", pa.float64()),
        ("yield_kind", pa.string()),
        ("year", pa.int64()),
        ("source_document", pa.string()),
    ]
)

REJECTED_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("source", pa.string()),
        ("source_id", pa.string()),
        ("reason", pa.string()),
    ]
)


@dataclass(frozen=True)
class ReactionRecord:
    """One literature reaction, reshaped from its source but chemically untouched.

    Adapters only read and reshape: every SMILES is kept exactly as the source gives it,
    so any later result can be traced back to the original row. Missing values are
    ``None``, never ``""``, ``"empty"`` or ``"nan"``.

    Attributes:
        source: Short key of the source file, e.g. ``"orderly"``.
        source_id: ID of the row in that source (for ORDerly: the row ``index``).
        licence: ``"open"``, ``"licensed"`` (Pistachio) or ``"noncommercial"`` (Chemlex,
            CC BY-NC-ND, ADR 009). Anything but ``"open"`` may only be written under a
            ``private/`` folder. Not ``"internal"``: the group inventory is not a reaction.
        reactants: Reactant SMILES as given by the source.
        agents: Catalyst and reagent SMILES as given; empty if the source pools them
            with the reactants.
        solvents: Solvent SMILES as given; empty if the source does not separate them.
        products: Product SMILES as given.
        source_rxn: The source's own (often atom-mapped) reaction string, if it has one. Kept
            for checks only: every reaction is mapped again with RXNMapper (ADR 002).
        temperature_c: Reaction temperature in °C.
        reaction_time_h: Reaction time in hours.
        yield_pct: Yield in percent as given; may be above 100 (filtered later, with a
            reason, not here).
        yield_kind: What ``yield_pct`` measures: ``"isolated"`` (literature; assumed, as
            patents sometimes report HPLC or crude yields), ``"lcms_uv_ratio"`` (HTE
            conversion from LC-MS UV peak areas), or ``None`` when there is no number. A label
            only: the two kinds are never put on one scale.
        year: Year of the record's date; for patents this is the grant year, not the
            year of the experiment.
        source_document: The source's own grouping of records (ORDerly: the ORD dataset file,
            a time block of the patent corpus, not a patent); None if the source has none.
            Kept so splits can follow time or document (ADR 006).
    """

    source: str
    source_id: str
    licence: Literal["open", "licensed", "noncommercial"]
    reactants: tuple[str, ...]
    agents: tuple[str, ...]
    solvents: tuple[str, ...]
    products: tuple[str, ...]
    source_rxn: str | None
    temperature_c: float | None
    reaction_time_h: float | None
    yield_pct: float | None
    yield_kind: Literal["isolated", "lcms_uv_ratio"] | None
    year: int | None
    source_document: str | None

    @property
    def record_id(self) -> str:
        """``source:source_id``, the key of this reaction in every later stage table."""
        return f"{self.source}:{self.source_id}"


@dataclass(frozen=True)
class RejectedRow:
    """A source row an adapter could not turn into a ``ReactionRecord``.

    Rejected rows are saved and counted in the stage report, never dropped silently.

    Attributes:
        source: Short key of the source file, e.g. ``"orderly"``.
        source_id: ID of the row in that source.
        reason: Short, fixed reason such as ``"no reactants"``, so rows can be counted
            by reason.
    """

    source: str
    source_id: str
    reason: str


def records_to_table(records: Sequence[ReactionRecord]) -> pa.Table:
    """Turn records into the stage-1 table: ``record_id`` first, SMILES tuples as string lists."""
    rows = [{"record_id": record.record_id, **asdict(record)} for record in records]
    return pa.Table.from_pylist(rows, schema=RECORD_SCHEMA)


def table_to_records(table: pa.Table) -> list[ReactionRecord]:
    """Read a stage-1 table back into records, the exact inverse of ``records_to_table``."""
    records = []
    for row in table.to_pylist():
        del row["record_id"]
        for name in _SMILES_FIELDS:
            row[name] = tuple(row[name])
        records.append(ReactionRecord(**row))
    return records


def rejected_to_table(rejected: Sequence[RejectedRow]) -> pa.Table:
    """Turn rejected rows into a table with the same ``record_id`` key as the records."""
    rows = [
        {"record_id": f"{row.source}:{row.source_id}", **asdict(row)}
        for row in rejected
    ]
    return pa.Table.from_pylist(rows, schema=REJECTED_SCHEMA)
