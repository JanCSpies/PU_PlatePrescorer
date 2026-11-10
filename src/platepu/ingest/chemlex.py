"""Chemlex adapter: read the Chemlex amide HTE sheet into ``ReactionRecord``s and a split table.

Chemlex (Zhong et al., Nat. Commun. 2025, 16:4522) is the held-out TEST set. The adapter only reads
and reshapes; it does no chemistry. Both HATU drawings and PyBrOP's mis-charged PF6 stay as written.
The reasons are in docs/design/rebuild.md, "Chemlex".
"""

from pathlib import Path

import pandas as pd

from platepu.ingest.record import ReactionRecord, RejectedRow

SOURCE = "chemlex"

# Every row has all of these (checked 2026-10-09); only the both-unseen split may be missing.
REQUIRED = ["Acid", "Amine", "Reagents", "Solvent", "Products", "Conversion"]

SPLITS = {
    "Random_Split": "random",
    "Stratified_Split_One_Unseen": "one_unseen",
    "Stratified_Split_Both_Unseen": "both_unseen",
}


def load_chemlex(path: Path) -> pd.DataFrame:
    """Read the Chemlex sheet; the 0-based row index becomes each record's ``source_id``."""
    return pd.read_excel(path)


def read_chemlex(path: Path) -> tuple[list[ReactionRecord], list[RejectedRow]]:
    """Read the Chemlex sheet into records, with the same signature as every other adapter.

    Chemlex has no row an adapter must reject (checked 2026-10-09), so the second list is always
    empty; a broken file raises ``ValueError`` instead (see ``records_from_table``).
    """
    return records_from_table(load_chemlex(path)), []


def records_from_table(table: pd.DataFrame) -> list[ReactionRecord]:
    """Reshape every Chemlex row into a record, in table order.

    Args:
        table: Chemlex rows as read by ``load_chemlex`` (or a slice of them, index kept).

    Returns:
        One record per row. ``agents`` is ``Reagents`` split on "." so it is ion by ion.

    Raises:
        ValueError: if a required value is missing or a conversion is outside 0 to 100. The real
            file has neither, so either means the input is not the file this adapter was built for.
    """
    _check(table)
    return [
        ReactionRecord(
            source=SOURCE,
            source_id=str(index),
            licence="noncommercial",  # CC BY-NC-ND 4.0 (ADR 009)
            reactants=(str(row["Acid"]), str(row["Amine"])),
            agents=tuple(str(row["Reagents"]).split(".")),
            solvents=(str(row["Solvent"]),),
            products=(str(row["Products"]),),
            source_rxn=None,
            temperature_c=None,
            reaction_time_h=None,
            yield_pct=float(row["Conversion"]),
            yield_kind="lcms_uv_ratio",
            year=None,
            source_document=None,
        )
        for index, row in table.to_dict(orient="index").items()
    ]


def splits_from_table(table: pd.DataFrame) -> pd.DataFrame:
    """Return the three split labels per row: ``source_id``, ``random``, ``one_unseen``,
    ``both_unseen``; each ``"train"``, ``"test"`` or ``None`` (missing)."""
    columns: dict[str, list[str | None]] = {
        "source_id": [str(index) for index in table.index]
    }
    for column, name in SPLITS.items():
        columns[name] = [
            value if isinstance(value, str) else None for value in table[column]
        ]
    return pd.DataFrame(columns)


def _check(table: pd.DataFrame) -> None:
    """Stop on a missing required value or a conversion outside 0 to 100."""
    missing = int(table[REQUIRED].isna().sum().sum())
    if missing:
        raise ValueError(f"Chemlex table has {missing} missing required values")
    conversion = table["Conversion"]
    if not conversion.between(0, 100).all():
        raise ValueError("Chemlex conversion must be 0 to 100 (percent)")
