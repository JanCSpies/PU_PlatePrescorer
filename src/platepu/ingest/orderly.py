"""ORDerly adapter: read an ORDerly condition parquet file into ``ReactionRecord``s.

The adapter only reads and reshapes; it does no chemistry. Roles, split salts, yields above
100 % and ORDerly's own conventions ("room temperature" → 25 °C, "overnight" → 8 h) are kept as
given. The reasons are in docs/design/rebuild.md, "ORDerly specifics".
"""

import logging
import math
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from platepu.ingest.record import ReactionRecord, RejectedRow

logger = logging.getLogger(__name__)

SOURCE = "orderly"


def read_orderly(path: Path) -> tuple[list[ReactionRecord], list[RejectedRow]]:
    """Read an ORDerly parquet file.

    Args:
        path: ORDerly condition parquet file. Its row index is used as ``source_id``.

    Returns:
        The records, in file order, and the rows that could not be read, with a reason.
    """
    table = pd.read_parquet(path).reset_index()
    records: list[ReactionRecord] = []
    rejected: list[RejectedRow] = []
    for raw_row in table.to_dict(orient="records"):
        # pandas types column names as Hashable; ORDerly's are all strings.
        row = {str(name): value for name, value in raw_row.items()}
        result = record_from_row(row)
        if isinstance(result, RejectedRow):
            rejected.append(result)
        else:
            records.append(result)
    logger.info(
        "%s: %d rows in, %d records out, %d rejected",
        path.name,
        len(table),
        len(records),
        len(rejected),
    )
    return records, rejected


def record_from_row(row: Mapping[str, object]) -> ReactionRecord | RejectedRow:
    """Reshape one ORDerly row into a record, or reject it with a reason.

    Args:
        row: One row as a column-name → value mapping; ``index`` is ORDerly's row index.

    Returns:
        The record, or a ``RejectedRow`` when the row has no reactant or no product,
        because such a row is not a reaction.
    """
    source_id = str(row["index"])
    reactants = _smiles_columns(row, "reactant_")
    products = _smiles_columns(row, "product_")
    if not reactants:
        return RejectedRow(source=SOURCE, source_id=source_id, reason="no reactants")
    if not products:
        return RejectedRow(source=SOURCE, source_id=source_id, reason="no products")
    yield_pct = _number(row.get("yield_000"))
    return ReactionRecord(
        source=SOURCE,
        source_id=source_id,
        licence="open",
        reactants=reactants,
        agents=_smiles_columns(row, "agent_"),
        solvents=_smiles_columns(row, "solvent_"),
        products=products,
        source_rxn=_text(row.get("rxn_str")),
        temperature_c=_number(row.get("temperature")),
        reaction_time_h=_number(row.get("rxn_time")),
        yield_pct=yield_pct,
        yield_kind="isolated" if yield_pct is not None else None,
        year=_year(row.get("grant_date")),
        source_document=_text(row.get("extracted_from_file")),
    )


def _smiles_columns(row: Mapping[str, object], prefix: str) -> tuple[str, ...]:
    """Collect the SMILES of numbered columns (``agent_000``, ``agent_001``, …), in order.

    Empty cells are skipped, so a row with one agent gives a one-element tuple.
    """
    columns = sorted(name for name in row if name.startswith(prefix))
    return tuple(smiles for name in columns if (smiles := _text(row[name])) is not None)


def _text(value: object) -> str | None:
    """Return a non-empty string as is, and anything missing (NaN, None, "") as None."""
    if isinstance(value, str) and value:
        return value
    return None


def _number(value: object) -> float | None:
    """Return a number as float, and a missing one (NaN, None) as None, never NaN."""
    if isinstance(value, int | float) and not math.isnan(value):
        return float(value)
    return None


def _year(value: object) -> int | None:
    """Return the year of a date; ORDerly's month is garbled, so only the year is used."""
    if isinstance(value, pd.Timestamp):
        return int(value.year)
    return None
