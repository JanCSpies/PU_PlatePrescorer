"""Tests for scripts/ingest.py (stage 1), on the five-row ORDerly fixture."""

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
from ingest import main

import platepu.io
from platepu.io import source_dir

FIXTURE = Path(__file__).parent / "data" / "orderly_5rows.parquet"
CHEMLEX_FIXTURE = Path(__file__).parent / "data" / "chemlex_like_6rows.parquet"


def test_ingest_writes_records_rejected_and_report(tmp_path: Path) -> None:
    """One run writes the three stage files under <root>/<source>/, with matching counts."""
    main(["orderly", str(FIXTURE), str(tmp_path)])
    out = tmp_path / "orderly"
    assert pq.read_table(out / "1_records.parquet").num_rows == 5
    assert pq.read_table(out / "1_records.rejected.parquet").num_rows == 0
    report = json.loads((out / "1_records.report.json").read_text(encoding="utf-8"))
    assert report["stage"] == "ingest"
    assert report["rows_out"] == 5
    assert report["rejected"] == {}


def test_licensed_source_goes_under_private(tmp_path: Path) -> None:
    """Pistachio's stage tables live under <root>/private/, so the licence guard lets them pass."""
    assert source_dir(tmp_path, "pistachio") == tmp_path / "private" / "pistachio"
    assert source_dir(tmp_path, "orderly") == tmp_path / "orderly"


def test_chemlex_goes_under_private(tmp_path: Path) -> None:
    """ADR 009 item 2: Chemlex (CC BY-NC-ND) is withheld from the open outputs like Pistachio; an
    open source such as ORDerly still goes to <root>/orderly."""
    assert source_dir(tmp_path, "chemlex") == tmp_path / "private" / "chemlex"
    assert source_dir(tmp_path, "orderly") == tmp_path / "orderly"


def test_private_sources_name_holds_chemlex_and_pistachio() -> None:
    """ADR 009 item 2: LICENSED_SOURCES is renamed PRIVATE_SOURCES, because membership means
    "withhold from the open outputs", and gains chemlex."""
    assert platepu.io.PRIVATE_SOURCES == {"pistachio", "chemlex"}
    assert not hasattr(platepu.io, "LICENSED_SOURCES")


def test_chemlex_ingest_is_noncommercial_and_lands_under_private(
    tmp_path: Path,
) -> None:
    """Six hand-written Chemlex-layout rows: every record is noncommercial, and the stage tables
    are written under <root>/private/chemlex/ (so the guard accepted them)."""
    raw = tmp_path / "chemlex.xlsx"
    pd.read_parquet(CHEMLEX_FIXTURE).to_excel(raw)  # the real file's format
    main(["chemlex", str(raw), str(tmp_path)])
    records = pq.read_table(
        tmp_path / "private" / "chemlex" / "1_records.parquet"
    ).to_pandas()
    assert len(records) == 6
    assert set(records["licence"]) == {"noncommercial"}
    assert not (tmp_path / "chemlex").exists()


def test_unknown_source_is_an_error(tmp_path: Path) -> None:
    """A typo in the source name stops the run instead of writing an empty folder."""
    with pytest.raises(SystemExit):
        main(["ordrely", str(FIXTURE), str(tmp_path)])
