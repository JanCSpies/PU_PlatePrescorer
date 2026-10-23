"""Tests for the Chemlex adapter, on six hand-written rows in Chemlex's layout.

Fixture ``tests/data/chemlex_like_6rows.parquet``: hand-written rows (all 9 Chemlex columns, row index
0-5 as the sheet row index) with simple building blocks, so no Chemlex data is in the repository
(CC BY-NC-ND, CLAUDE.md §9). Built by ``scratch/build_chemlex_like_fixtures.py`` on 2026-10-20. They
keep the cases of the real file: HATU drawn as the O-form (0), HATU drawn as the N-oxide with no
both-unseen label (1), EDCI (2), PyBrOP with PF6 written as a cation (3), and one HATU replicate pair
whose labels disagree at 20 % (4: 31 %, 5: 0 %). The expected records follow
``docs/design/rebuild.md`` ("Chemlex"): every SMILES is kept exactly as written.
"""

from pathlib import Path

import pandas as pd
import pytest

from platepu.ingest.chemlex import load_chemlex, records_from_table, splits_from_table
from platepu.ingest.record import ReactionRecord

FIXTURE = Path(__file__).parent / "data" / "chemlex_like_6rows.parquet"

DIEA = "CCN(C(C)C)C(C)C"
PF6 = "F[P-](F)(F)(F)(F)F"
HATU_O_FORM = "CN(C)C(On1nnc2cccnc21)=[N+](C)C"
DMF = "CN(C)C=O"


def _chemlex(
    source_id: str,
    acid: str,
    amine: str,
    agents: tuple[str, ...],
    product: str,
    conversion: float,
) -> ReactionRecord:
    """Build an expected Chemlex record; only these fields differ between rows."""
    return ReactionRecord(
        source="chemlex",
        source_id=source_id,
        licence="noncommercial",
        reactants=(acid, amine),
        agents=agents,
        solvents=(DMF,),
        products=(product,),
        source_rxn=None,
        temperature_c=None,
        reaction_time_h=None,
        yield_pct=conversion,
        yield_kind="lcms_uv_ratio",
        year=None,
        source_document=None,
    )


EXPECTED: dict[str, ReactionRecord] = {
    # HATU, O-form drawing; a stereocentre in the amine
    "0": _chemlex(
        "0",
        "O=C(O)CCCc1ccccc1",
        "C[C@H](N)c1ccccc1",
        (DIEA, HATU_O_FORM, PF6),
        "C[C@H](NC(=O)CCCc1ccccc1)c1ccccc1",
        80.0,
    ),
    # HATU, N-oxide drawing (the real structure); kept as written
    "1": _chemlex(
        "1",
        "O=C(O)c1ccco1",
        "Nc1ccc([N+](=O)[O-])cc1",
        (DIEA, "CN(C)C(n1n[n+]([O-])c2ncccc21)=[N+](C)C", PF6),
        "O=C(Nc1ccc([N+](=O)[O-])cc1)c1ccco1",
        0.0,
    ),
    # EDCI as its hydrochloride; no HOBt in the string
    "2": _chemlex(
        "2",
        "O=C(O)c1ccco1",
        "C1CCNCC1",
        ("CCN=C=NCCCN(C)C", "Cl", DIEA),
        "O=C(c1ccco1)N1CCCCC1",
        12.5,
    ),
    # PyBrOP; PF6 written with the wrong charge, kept as written
    "3": _chemlex(
        "3",
        "O=C(O)c1ccccc1",
        "Nc1ccc(N2CCOCC2)cc1",
        (DIEA, "Br[P+](N1CCCC1)(N1CCCC1)N1CCCC1", "F[P+](F)(F)(F)(F)F"),
        "O=C(Nc1ccc(N2CCOCC2)cc1)c1ccccc1",
        0.0,
    ),
    # HATU replicate pair, labels disagree at 20 %
    "4": _chemlex(
        "4",
        "O=C(O)c1ccc(Cl)cc1",
        "Nc1nccs1",
        (DIEA, HATU_O_FORM, PF6),
        "O=C(Nc1nccs1)c1ccc(Cl)cc1",
        31.0,
    ),
    "5": _chemlex(
        "5",
        "O=C(O)c1ccc(Cl)cc1",
        "Nc1nccs1",
        (DIEA, HATU_O_FORM, PF6),
        "O=C(Nc1nccs1)c1ccc(Cl)cc1",
        0.0,
    ),
}


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    """The fixture rows, with the original sheet row index."""
    return pd.read_parquet(FIXTURE)


@pytest.fixture(scope="module")
def records(table: pd.DataFrame) -> dict[str, ReactionRecord]:
    """Reshape the fixture once and index the records by ``source_id``."""
    return {record.source_id: record for record in records_from_table(table)}


def test_all_six_rows_become_records(records: dict[str, ReactionRecord]) -> None:
    """Every fixture row becomes one record; none is lost."""
    assert sorted(records) == sorted(EXPECTED)


@pytest.mark.parametrize("source_id", sorted(EXPECTED))
def test_record_matches_hand_checked_expectation(
    records: dict[str, ReactionRecord], source_id: str
) -> None:
    """Each record equals the hand-checked expectation, field by field."""
    assert records[source_id] == EXPECTED[source_id]


def test_source_id_is_the_sheet_row_index(records: dict[str, ReactionRecord]) -> None:
    """The ID is the row's position in the sheet, so every record traces back to one row."""
    assert "1" in records


def test_both_hatu_drawings_stay_as_written(records: dict[str, ReactionRecord]) -> None:
    """The adapter does no chemistry: unifying the two HATU drawings is Stage 2's job."""
    assert records["0"].agents != records["1"].agents


def test_conversion_is_labelled_lcms(records: dict[str, ReactionRecord]) -> None:
    """Chemlex numbers are LC-MS UV conversions, never to be read as isolated yields."""
    assert all(record.yield_kind == "lcms_uv_ratio" for record in records.values())


def test_missing_value_is_an_error(table: pd.DataFrame) -> None:
    """Chemlex has no missing value (checked 2026-10-09), so one means a different file:
    stop instead of guessing."""
    broken = table.assign(Conversion=[80.0, float("nan"), 12.5, 0.0, 31.0, 0.0])
    with pytest.raises(ValueError, match="missing"):
        records_from_table(broken)


def test_conversion_outside_0_to_100_is_an_error(table: pd.DataFrame) -> None:
    """A conversion is a percentage; anything else means the column is not what we think."""
    broken = table.assign(Conversion=[80.0, 0.0, 12.5, 0.0, 131.0, 0.0])
    with pytest.raises(ValueError, match="0 to 100"):
        records_from_table(broken)


def test_splits_are_kept_per_row(table: pd.DataFrame) -> None:
    """The three split labels are kept, keyed by ``source_id``; a missing label is None."""
    splits = splits_from_table(table).set_index("source_id")
    assert splits.loc["0"].to_dict() == {
        "random": "test",
        "one_unseen": "test",
        "both_unseen": "test",
    }
    assert splits.loc["3"].to_dict() == {
        "random": "train",
        "one_unseen": "train",
        "both_unseen": "test",
    }
    assert splits.loc["1", "both_unseen"] is None


def test_load_chemlex_reads_the_xlsx_sheet(table: pd.DataFrame, tmp_path: Path) -> None:
    """From an .xlsx like the real file: one sheet, row index from 0 in sheet order."""
    path = tmp_path / "chemlex.xlsx"
    table.to_excel(path, index=False)
    read = records_from_table(load_chemlex(path))
    assert [record.source_id for record in read] == ["0", "1", "2", "3", "4", "5"]
    assert read[0].reactants == EXPECTED["0"].reactants
