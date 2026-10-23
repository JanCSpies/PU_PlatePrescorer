"""Test for scripts/benchmark_split.py on tiny hand-written labels in a tmp root (made-up keys, no
Chemlex row). The script reads the real configs/classes/amide_coupling.yml, so run from the repo root.
"""

import json
from pathlib import Path

import benchmark_split as split_module
import pandas as pd
import pyarrow.parquet as pq
import pytest
from benchmark_split import main as benchmark_split
from benchmark_split import shared_substrates

from platepu.io import write_table
from platepu.positives.build import load_positives_config
from platepu.standardise.molecule import load_std_config

FAMILY = "HATU_DIPEA_DMF"


def BENCHMARK_DIR(root: Path) -> Path:
    """ADR 009 item 3: the benchmark tables live under <root>/private/benchmarks/<name>/."""
    return root / "private" / "benchmarks" / "chemlex"


def family_condition() -> str:
    """The condition text of the family: its standardised members joined as the labels table does."""
    families = load_positives_config(
        Path("configs/classes/amide_coupling.yml"),
        load_std_config(Path("configs/standardise.yml")),
    ).families
    return " + ".join(sorted(families[FAMILY]))


def write_labels(root: Path) -> None:
    """Three family rows: one test, two others on different acids and amines (disjoint sides);
    one row outside the family."""
    rows = [
        (family_condition(), "tacid-S", "tamine-S", "test", True),
        (family_condition(), "vacid-S", "vamine-S", "train", True),
        (family_condition(), "wacid-S", "wamine-S", None, False),
        ("CCO", "xacid-S", "xamine-S", "train", True),
    ]
    labels = pd.DataFrame(
        rows,
        columns=["condition", "key_a_stereo", "key_b_stereo", "both_unseen", "label"],
    ).astype({"label": "boolean"})
    labels["key_a_flat"] = labels["key_a_stereo"].str.split("-").str[0]
    labels["key_b_flat"] = labels["key_b_stereo"].str.split("-").str[0]
    labels["licence"] = (
        "noncommercial"  # ADR 009: stage 7b writes the benchmark's licence
    )
    folder = BENCHMARK_DIR(root)
    folder.mkdir(parents=True)
    labels.to_parquet(folder / "labels.parquet")


def test_split_report_records_that_the_seal_held(tmp_path: Path) -> None:
    """Validation holds acids v, w and amines v, w; test holds acid t and amine t: no flat key is
    shared, so the report must carry shared_acids and shared_amines, both 0 (the only audit trail
    that no test substrate reached validation)."""
    write_labels(tmp_path)
    benchmark_split(["chemlex", "amide_coupling", FAMILY, str(tmp_path)])
    out = BENCHMARK_DIR(tmp_path) / FAMILY
    report = json.loads((out / "split.report.json").read_text(encoding="utf-8"))
    assert (report["validation"], report["test"]) == (2, 1)
    assert report["shared_acids"] == 0
    assert report["shared_amines"] == 0


def test_split_tables_carry_the_benchmark_licence(tmp_path: Path) -> None:
    """validation.parquet and test.parquet both hold licence noncommercial in every row (ADR 009
    item 3); the old open folder is not created."""
    write_labels(tmp_path)
    benchmark_split(["chemlex", "amide_coupling", FAMILY, str(tmp_path)])
    out = BENCHMARK_DIR(tmp_path) / FAMILY
    for name in ("validation", "test"):
        part = pd.read_parquet(out / f"{name}.parquet")
        assert set(part["licence"]) == {"noncommercial"}, name
    assert not (tmp_path / "benchmarks").exists()


@pytest.mark.parametrize("name", ["validation", "test"])
def test_split_tables_are_refused_outside_private_on_content_alone(
    tmp_path: Path, name: str
) -> None:
    """The point of ADR 009 item 3: take a table the script wrote and try to put it in the OLD open
    folder. write_table must refuse it because of its licence column, not because of the path.
    Before the fix the column is missing and this write succeeds."""
    write_labels(tmp_path)
    benchmark_split(["chemlex", "amide_coupling", FAMILY, str(tmp_path)])
    written = pq.read_table(BENCHMARK_DIR(tmp_path) / FAMILY / f"{name}.parquet")
    outside = tmp_path / "benchmarks" / "chemlex" / FAMILY / f"{name}.parquet"
    with pytest.raises(ValueError, match="private"):
        write_table(written, outside, root=tmp_path)
    assert not outside.exists()


def sides(
    validation_keys: tuple[str, str], test_keys: tuple[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-row validation and test frames holding only (acid, amine) flat keys, made up."""
    columns = ["key_a_flat", "key_b_flat"]
    return (
        pd.DataFrame([validation_keys], columns=columns),
        pd.DataFrame([test_keys], columns=columns),
    )


def test_shared_substrates_counts_an_acid_shared_alone() -> None:
    """Both sides hold acid 'acid1' but different amines: one shared acid, no shared amine. A counter
    wired to the amine column (key_a_flat and key_b_flat are easy to swap) would report 0 and 1."""
    validation, test = sides(("acid1", "amineV"), ("acid1", "amineT"))
    assert shared_substrates(validation, test) == {
        "shared_acids": 1,
        "shared_amines": 0,
    }


def test_shared_substrates_counts_an_amine_shared_alone() -> None:
    """Both sides hold amine 'amine1' but different acids: no shared acid, one shared amine."""
    validation, test = sides(("acidV", "amine1"), ("acidT", "amine1"))
    assert shared_substrates(validation, test) == {
        "shared_acids": 0,
        "shared_amines": 1,
    }


def test_shared_substrates_counts_an_acid_and_an_amine_shared_together() -> None:
    """The same pair on both sides shares its acid and its amine: one of each."""
    validation, test = sides(("acid1", "amine1"), ("acid1", "amine1"))
    assert shared_substrates(validation, test) == {
        "shared_acids": 1,
        "shared_amines": 1,
    }


def test_shared_substrates_does_not_match_an_acid_against_an_amine() -> None:
    """A key that is the validation acid and the test amine is no overlap: roles stay apart."""
    validation, test = sides(("key1", "amineV"), ("acidT", "key1"))
    assert shared_substrates(validation, test) == {
        "shared_acids": 0,
        "shared_amines": 0,
    }


def test_leaking_split_raises_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This test guards the guard, not the split: the real family_split cannot return overlapping
    sides (it excludes on flat keys), so it is replaced by one that returns the same acid on both
    sides. main must raise naming the count and leave no validation, test or report file."""
    write_labels(tmp_path)
    validation, test = sides(("leakedacid", "amineV"), ("leakedacid", "amineT"))
    for part in (validation, test):
        part["label"] = True
    monkeypatch.setattr(split_module, "family_split", lambda *_: (validation, test))
    with pytest.raises(ValueError, match="shared_acids.*1"):
        benchmark_split(["chemlex", "amide_coupling", FAMILY, str(tmp_path)])
    assert not (BENCHMARK_DIR(tmp_path) / FAMILY).exists()
