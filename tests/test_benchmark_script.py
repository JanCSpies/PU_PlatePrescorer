"""Tests for scripts/benchmark.py (stage 7b labels) and the licence of its table, ADR 009 item 3.

Run on the six hand-written Chemlex-layout rows (tests/data/chemlex_like_6rows.parquet, no Chemlex
data), ingested by scripts/ingest.py into a tmp root. The script reads the real
configs/classes/amide_coupling.yml and configs/standardise.yml, so run from the repository root.
Written 2026-10-22 before the code.
"""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml
from benchmark import main as benchmark
from benchmark import summary
from ingest import main as ingest

from platepu.io import write_table

FIXTURE = Path(__file__).parent / "data" / "chemlex_like_6rows.parquet"
CONFIG = Path("configs/benchmarks/chemlex.yml")


def prepare(root: Path) -> Path:
    """Stage 1 for the fixture, and a stage-5 table that puts every record in amide_coupling.
    Returns the raw .xlsx (the real file's format, written from the fixture rows) for the splits."""
    raw = root / "chemlex.xlsx"
    pd.read_parquet(FIXTURE).to_excel(raw)
    ingest(["chemlex", str(raw), str(root)])
    records = pq.read_table(root / "private" / "chemlex" / "1_records.parquet")
    classified = pd.DataFrame(
        {"record_id": records["record_id"].to_pylist(), "class": "amide_coupling"}
    )
    classified.to_parquet(root / "private" / "chemlex" / "5_classified.parquet")
    return raw


def test_chemlex_config_declares_the_benchmark_licence() -> None:
    """The licence comes from the benchmark config (ADR 009 item 3), not from the code."""
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["licence"] == "noncommercial"


def test_labels_are_written_under_private_with_the_config_licence(
    tmp_path: Path,
) -> None:
    """labels.parquet moves to <root>/private/benchmarks/chemlex/ and carries licence in every row."""
    raw = prepare(tmp_path)
    benchmark(["chemlex", str(tmp_path), str(raw)])
    labels = pd.read_parquet(
        tmp_path / "private" / "benchmarks" / "chemlex" / "labels.parquet"
    )
    assert set(labels["licence"]) == {"noncommercial"}
    assert not (tmp_path / "benchmarks").exists()


def test_labels_licence_is_read_from_the_config_not_hard_coded(tmp_path: Path) -> None:
    """A copy of the config that says licence: internal gives internal labels."""
    raw = prepare(tmp_path)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["licence"] = "internal"
    other = tmp_path / "chemlex_internal.yml"
    other.write_text(yaml.safe_dump(config), encoding="utf-8")
    benchmark(["chemlex", str(tmp_path), str(raw), "--config", str(other)])
    labels = pd.read_parquet(
        tmp_path / "private" / "benchmarks" / "chemlex" / "labels.parquet"
    )
    assert set(labels["licence"]) == {"internal"}


def test_labels_are_refused_outside_private_on_content_alone(tmp_path: Path) -> None:
    """The point of ADR 009 item 3: take the table the script wrote and try to put it in the OLD
    open location. write_table must refuse it because of the licence column, not because of a
    path convention. Before the fix the table has no such column and this write succeeds."""
    raw = prepare(tmp_path)
    benchmark(["chemlex", str(tmp_path), str(raw)])
    labels = pq.read_table(
        tmp_path / "private" / "benchmarks" / "chemlex" / "labels.parquet"
    )
    outside = tmp_path / "benchmarks" / "chemlex" / "labels.parquet"
    with pytest.raises(ValueError, match="private"):
        write_table(labels, outside, root=tmp_path)
    assert not outside.exists()


def test_labels_report_averages_only_over_rows_outside_the_test_side() -> None:
    """positive_share and the per-condition shares of labels.report.json averaged over every labelled
    row, test included, which published a test-side outcome (CLAUDE.md §6: never tune on TEST).
    Here the non-test rows are 3 of 4 positive (condition X: 2 of 3) and the five test rows are all
    negative, so the old figures were 3/9 overall and 2/7 for X, and condition T (test rows only)
    appeared in top_conditions.

    The keys carry ``_outside_test`` on purpose (agreed 2026-10-22): the first fix of this defect
    removed one test-side mean but left ``positive_share`` and ``top_conditions`` averaging over the
    test rows, and a comment claimed the opposite, because nothing in the key said which rows it
    covered. A key that names its rows cannot be misread the same way."""
    rows = [
        ("X", True, "train"),
        ("X", True, None),
        ("X", False, "train"),
        ("Y", True, None),
        ("X", False, "test"),
        ("X", False, "test"),
        ("X", False, "test"),
        ("X", False, "test"),
        ("T", False, "test"),
    ]
    labels = pd.DataFrame(rows, columns=["condition", "label", "both_unseen"]).assign(
        label=lambda t: t["label"].astype("boolean"),
        status="ok",
        split_conflict=False,
        n_mapped_in_class=1,
        n_measurements=1,
    )
    report = summary(labels, pd.DataFrame({"in_class": [True] * len(labels)}))
    assert report["positive_share_outside_test"] == 0.75
    assert report["labelled_outside_test"] == 4
    top = report["top_conditions_outside_test"]
    assert isinstance(top, dict)
    assert top["X"]["positive_share"] == round(2 / 3, 4)
    assert "T" not in top
