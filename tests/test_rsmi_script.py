"""Tests for scripts/rsmi.py (stage 2), run after stage 1 on the five-row ORDerly fixture."""

import json
from pathlib import Path

import pyarrow.parquet as pq
from ingest import main as ingest
from rsmi import main as rsmi

FIXTURE = Path(__file__).parent / "data" / "orderly_5rows.parquet"


def test_rsmi_follows_ingest(tmp_path: Path) -> None:
    """Stage 2 reads stage 1's table and writes one reaction string per record."""
    ingest(["orderly", str(FIXTURE), str(tmp_path)])
    rsmi(["orderly", str(tmp_path)])
    out = tmp_path / "orderly"
    assert pq.read_table(out / "2_rsmi.parquet").num_rows == 5
    report = json.loads((out / "2_rsmi.report.json").read_text(encoding="utf-8"))
    assert (report["rows_in"], report["rows_out"], report["rejected"]) == (5, 5, {})
