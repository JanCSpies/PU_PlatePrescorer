"""Tests for writing ReactionRecords as a table and reading them back (stage 1 output)."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from platepu.ingest.chemlex import records_from_table
from platepu.ingest.orderly import read_orderly
from platepu.ingest.record import (
    ReactionRecord,
    RejectedRow,
    records_to_table,
    rejected_to_table,
    table_to_records,
)

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def records() -> list[ReactionRecord]:
    """The ORDerly and Chemlex fixture records: tuples, None values, both yield kinds."""
    orderly, _ = read_orderly(DATA / "orderly_5rows.parquet")
    chemlex = records_from_table(pd.read_parquet(DATA / "chemlex_like_6rows.parquet"))
    return orderly + chemlex


def test_round_trip_gives_the_same_records(records: list[ReactionRecord]) -> None:
    """Writing and reading back loses nothing: tuples stay tuples, None stays None."""
    assert table_to_records(records_to_table(records)) == records


def test_table_has_record_id_first(records: list[ReactionRecord]) -> None:
    """record_id = source:source_id is the key of every later stage."""
    table = records_to_table(records)
    assert table.column_names[0] == "record_id"
    assert table["record_id"][0].as_py() == "orderly:638744"


def test_smiles_tuples_are_string_lists(records: list[ReactionRecord]) -> None:
    """The schema is explicit, so an empty tuple is still a list of strings, never null."""
    table = records_to_table(records)
    assert table.schema.field("agents").type == pa.list_(pa.string())
    assert table.schema.field("year").type == pa.int64()
    assert table.schema.field("source_document").type == pa.string()


def test_empty_input_gives_an_empty_table() -> None:
    """A source with nothing left still writes a table with the full schema."""
    assert records_to_table([]).num_rows == 0


def test_rejected_rows_keep_source_id_and_reason() -> None:
    """Rejected rows are a table too, counted by reason in the report."""
    table = rejected_to_table([RejectedRow("orderly", "7", "no products")])
    assert table.to_pylist() == [
        {
            "record_id": "orderly:7",
            "source": "orderly",
            "source_id": "7",
            "reason": "no products",
        }
    ]
