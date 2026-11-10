"""Test for scripts/positives.py on the ORDerly fixture rows, run through stages 1 and 4-6 in a tmp root."""

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from ingest import main as ingest
from positives import main as positives

from platepu.classify.engine import classify_table, load_class
from platepu.io import write_table
from platepu.reactions.clean import clean_table, load_pipeline
from platepu.standardise.molecule import load_std_config
from platepu.standardise.table import standardise_table

ROOT = Path(__file__).parents[1]
DATA = ROOT / "tests" / "data"


def prepare(tmp_path: Path) -> pa.Table:
    """Stages 1 and 4-6 for the ORDerly fixture in ``tmp_path``; returns the stage-6 table."""
    ingest(["orderly", str(DATA / "orderly_5rows.parquet"), str(tmp_path)])
    mapped = pq.read_table(DATA / "mapped_12rows.parquet")
    mapped = mapped.filter(pc.starts_with(mapped["record_id"], "orderly:"))
    cleaned, _ = clean_table(mapped, load_pipeline(ROOT / "configs" / "clean.yml"))
    classified, _ = classify_table(
        cleaned, [load_class(ROOT / "configs" / "classes" / "amide_coupling.yml")]
    )
    table = standardise_table(
        classified, load_std_config(ROOT / "configs" / "standardise.yml")
    )
    write_table(table, tmp_path / "orderly" / "6_standardised.parquet", root=tmp_path)
    return table


def test_positives_from_the_orderly_fixture(tmp_path: Path) -> None:
    """Open source only: one open folder, no private folder; the report has the yield cut-offs.
    The benchmark labels given share nothing with it: a run needs some, see the test below."""
    table = prepare(tmp_path)
    benchmark = tmp_path / "no_overlap.parquet"
    pd.DataFrame(
        {"key_a_flat": ["made_up_acid"], "key_b_flat": ["made_up_amine"]}
    ).to_parquet(benchmark)
    positives(
        ["amide_coupling", str(tmp_path), "--sources", "orderly"]
        + ["--benchmark-labels", str(benchmark)]
    )
    out = tmp_path / "positives" / "amide_coupling"
    report = json.loads((out / "positives.report.json").read_text(encoding="utf-8"))
    assert report["reactions_in"] == table.num_rows
    assert set(report["sizes_by_yield_cutoff"]) == {
        "min_yield_0",
        "min_yield_10",
        "min_yield_20",
    }
    assert pq.read_table(out / "positives.parquet").num_rows >= 1
    assert not (tmp_path / "private").exists()


def test_positives_table_has_no_leak_columns_and_report_has_the_counts(
    tmp_path: Path,
) -> None:
    """ADR 009: the open table carries no benchmark-derived column; the report keeps the counts."""
    test_positives_from_the_orderly_fixture(tmp_path)
    out = tmp_path / "positives" / "amide_coupling"
    first = pd.read_parquet(out / "positives.parquet").iloc[0]
    benchmark = tmp_path / "benchmark_labels.parquet"
    pd.DataFrame(
        {"key_a_flat": [first["key_a_flat"]], "key_b_flat": [first["key_b_flat"]]}
    ).to_parquet(benchmark)

    positives(
        ["amide_coupling", str(tmp_path), "--sources", "orderly"]
        + ["--benchmark-labels", str(benchmark)]
    )
    table = pd.read_parquet(out / "positives.parquet")
    assert not {"leak_pairs_one_side", "leak_pairs_both_sides"} & set(table.columns)
    report = json.loads((out / "positives.report.json").read_text(encoding="utf-8"))
    assert report["leak_pairs_both_sides"] >= 1
    assert report["leak_pairs_one_side"] >= report["leak_pairs_both_sides"]
    assert pq.read_table(out / "positives.parquet").schema.names == list(table.columns)


def test_run_without_benchmark_labels_writes_nothing(tmp_path: Path) -> None:
    """Without --benchmark-labels the report said leak_pairs_one_side 0 and leak_pairs_both_sides 0, i.e. 'no overlap
    with Chemlex', when no overlap was ever checked. The run must stop, before any file is written."""
    prepare(tmp_path)
    with pytest.raises((ValueError, SystemExit)):
        positives(["amide_coupling", str(tmp_path), "--sources", "orderly"])
    assert not (tmp_path / "positives").exists()
