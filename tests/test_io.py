"""Tests for writing stage outputs: the licence guard and the stage report."""

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

import platepu.io as io_module
from platepu.io import load_paths, write_report, write_table

OPEN = pa.table({"record_id": ["orderly:1"], "licence": ["open"]})
LICENSED = pa.table({"record_id": ["pistachio:1"], "licence": ["licensed"]})


def test_open_rows_are_written_anywhere(tmp_path: Path) -> None:
    """Open data may go to any folder; missing folders are created."""
    path = tmp_path / "orderly" / "1_records.parquet"
    write_table(OPEN, path, root=tmp_path)
    assert path.exists()


def test_licensed_rows_outside_private_are_refused(tmp_path: Path) -> None:
    """Pistachio-derived rows may only be written under a private/ folder (CLAUDE.md §9).
    On macOS tmp_path itself lies under /private/var, so this also checks that only folders
    below the output root count."""
    with pytest.raises(ValueError, match="private"):
        write_table(
            LICENSED, tmp_path / "pistachio" / "1_records.parquet", root=tmp_path
        )
    assert not (tmp_path / "pistachio").exists()


def test_licensed_rows_under_private_are_written(tmp_path: Path) -> None:
    """At any depth under private/, licensed rows are allowed."""
    path = tmp_path / "private" / "pistachio" / "1_records.parquet"
    write_table(LICENSED, path, root=tmp_path)
    assert path.exists()


def test_report_records_counts_and_provenance(tmp_path: Path) -> None:
    """The report keeps the counts given and adds where and with what it was made."""
    path = tmp_path / "1_records.report.json"
    write_report(path, {"stage": "ingest", "rows_out": 5})
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["stage"] == "ingest"
    assert report["rows_out"] == 5
    assert {"git_commit", "versions", "written_at"} <= report.keys()
    assert "rdkit" in report["versions"]


def test_paths_file_expands_home(tmp_path: Path) -> None:
    """``~`` in a paths file means the user's home, as in configs/paths/local.yml."""
    path = tmp_path / "local.yml"
    path.write_text(
        "root: ~/derived\nrxnmapper_python: ~/envs/py\nmapper_batch_size: 16\n",
        encoding="utf-8",
    )
    paths = load_paths(path)
    assert paths.root == Path.home() / "derived"
    assert paths.mapper_batch_size == 16


def test_paths_file_without_root_stops(tmp_path: Path) -> None:
    """A missing key stops loading (KeyError); an unknown one too (TypeError)."""
    path = tmp_path / "local.yml"
    path.write_text("rxnmapper_python: ~/py\nmapper_batch_size: 16\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_paths(path)
    path.write_text(
        "root: ~/d\nrxnmapper_python: ~/py\nmapper_batch_size: 16\ngpu: true\n",
        encoding="utf-8",
    )
    with pytest.raises(TypeError):
        load_paths(path)


INTERNAL = pa.table({"container_id": ["7"], "licence": ["internal"]})


def test_internal_rows_outside_private_are_refused(tmp_path: Path) -> None:
    """ADR 004: rows from the group's inventory are 'internal' and stay under private/, like licensed."""
    with pytest.raises(ValueError, match="private"):
        write_table(
            INTERNAL, tmp_path / "library" / "amide_coupling.parquet", root=tmp_path
        )


def test_internal_rows_under_private_are_written(tmp_path: Path) -> None:
    """Under private/ the inventory library is written as before."""
    path = tmp_path / "private" / "library" / "amide_coupling.parquet"
    write_table(INTERNAL, path, root=tmp_path)
    assert path.exists()


# --- Holes found by data-guardian, 2026-10-22: tests written before the fix. ---------------------
# Hole 1: the private/ check looked at path parts without resolving ``..`` or symlinks.
# Hole 2: the licence check refused only the exact strings "licensed" and "internal".

LICENSED_AND_INTERNAL = pytest.mark.parametrize(
    "table", [INTERNAL, LICENSED], ids=["internal", "licensed"]
)


@LICENSED_AND_INTERNAL
def test_dotdot_out_of_private_is_refused(tmp_path: Path, table: pa.Table) -> None:
    """``private/../library`` has "private" in its parts but the file lands in ``library/``.
    Nothing may be written, neither at the literal nor at the resolved location."""
    path = tmp_path / "private" / ".." / "library" / "a.parquet"
    with pytest.raises(ValueError, match="private"):
        write_table(table, path, root=tmp_path)
    assert not (tmp_path / "library" / "a.parquet").exists()
    assert not path.exists()


def test_symlinked_private_pointing_outside_root_is_refused(tmp_path: Path) -> None:
    """``root/private`` is a symlink to a folder outside ``root``: the bytes would leave the
    output root. A deliberate symlink could be a legitimate mount, but the guard cannot tell
    a mount from an accident, so the safe rule is: the resolved path must stay inside the
    resolved root (and, there, under private/)."""
    root = tmp_path / "root"
    elsewhere = tmp_path / "elsewhere"
    root.mkdir()
    elsewhere.mkdir()
    (root / "private").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="private"):
        write_table(INTERNAL, root / "private" / "a.parquet", root=root)
    assert not (elsewhere / "a.parquet").exists()


def test_internal_rows_in_nested_private_folder_are_written(tmp_path: Path) -> None:
    """Control, must stay green: a plain private/sub/ path is fine."""
    path = tmp_path / "private" / "sub" / "a.parquet"
    write_table(INTERNAL, path, root=tmp_path)
    assert path.exists()


def test_internal_rows_outside_root_are_refused(tmp_path: Path) -> None:
    """Control, must stay green: a path outside ``root`` raises (today via ``relative_to``)."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "private" / "a.parquet"
    with pytest.raises(ValueError):
        write_table(INTERNAL, outside, root=root)
    assert not outside.exists()


def test_all_open_licence_is_written_outside_private(tmp_path: Path) -> None:
    """Control, must stay green: an allow-list still lets open data through."""
    path = tmp_path / "orderly" / "a.parquet"
    write_table(pa.table({"licence": ["open", "open"]}), path, root=tmp_path)
    assert path.exists()


# Policy chosen by Luka, 2026-10-22 (narrow fix, not a strict allow-list): outside private/, a
# table that HAS a licence column must hold exactly "open" in every row. Null, wrong case,
# stray whitespace, unknown words and mixes are refused.
NOT_EXACTLY_OPEN = [
    pytest.param(
        pa.table({"licence": pa.array([None], pa.string())}), id="null_licence"
    ),
    pytest.param(
        pa.table({"licence": pa.array(["open", None], pa.string())}),
        id="open_and_null_licence",
    ),
    pytest.param(pa.table({"licence": ["Internal"]}), id="wrong_case_Internal"),
    pytest.param(pa.table({"licence": ["internal "]}), id="trailing_space_internal"),
    pytest.param(pa.table({"licence": ["proprietary"]}), id="unrecognised_proprietary"),
]


@pytest.mark.parametrize("table", NOT_EXACTLY_OPEN)
def test_licence_not_exactly_open_is_refused_outside_private(
    tmp_path: Path, table: pa.Table
) -> None:
    """Outside private/ a licence column must hold exactly "open" in every row."""
    path = tmp_path / "library" / "a.parquet"
    with pytest.raises(ValueError, match="private"):
        write_table(table, path, root=tmp_path)
    assert not path.exists()


def test_table_without_a_licence_column_is_written_outside_private(
    tmp_path: Path,
) -> None:
    """Pins a KNOWN, RECORDED GAP (Luka's decision, 2026-10-22); do not tidy it into a refusal.

    The guard only judges a licence column that exists. These writers have none today: the five
    ``*.rejected.parquet`` writers, ``scripts/positives.py:72``, ``scripts/map_missing.py:32``,
    ``scripts/benchmark.py:103`` and ``scripts/benchmark_split.py:64`` (rejected tables, RXNMapper
    shards, benchmark labels). A strict allow-list would break all nine; giving them a licence
    column is separate work. Read the decision before changing this test."""
    path = tmp_path / "library" / "a.parquet"
    write_table(pa.table({"record_id": ["x"]}), path, root=tmp_path)
    assert path.exists()


def test_all_open_licence_is_written_under_private(tmp_path: Path) -> None:
    """Control: private/ refuses nothing, so the fix must not make it stricter than open data needs."""
    path = tmp_path / "private" / "orderly" / "a.parquet"
    write_table(pa.table({"licence": ["open", "open"]}), path, root=tmp_path)
    assert path.exists()


# --- ADR 009 (2026-10-22): Chemlex is noncommercial; one shared precedence. Tests before code. ----

NONCOMMERCIAL = pa.table({"record_id": ["chemlex:1"], "licence": ["noncommercial"]})


def test_noncommercial_rows_outside_private_are_refused(tmp_path: Path) -> None:
    """GREEN regression pin, not a RED test: the allow-list already refuses every value that is
    not "open", so Chemlex's new label needs no new rule (ADR 009 item 1)."""
    path = tmp_path / "chemlex" / "1_records.parquet"
    with pytest.raises(ValueError, match="private"):
        write_table(NONCOMMERCIAL, path, root=tmp_path)
    assert not path.exists()


def test_noncommercial_rows_under_private_are_written(tmp_path: Path) -> None:
    """GREEN regression pin: under private/ the Chemlex tables are written."""
    path = tmp_path / "private" / "chemlex" / "1_records.parquet"
    write_table(NONCOMMERCIAL, path, root=tmp_path)
    assert path.exists()


# Spelled out here, not imported: a missing name must fail the tests that use it, not hide the
# GREEN pins above by breaking the import of the whole file.
PRECEDENCE = ("internal", "licensed", "noncommercial", "open")


def strictest_licence(values: Iterable[str]) -> str:
    """Look the function up at call time, so a missing one is an AttributeError in that test only."""
    return str(io_module.strictest_licence(values))


def test_licence_precedence_is_strictest_first() -> None:
    """internal (the group's own) > licensed (Pistachio) > noncommercial (Chemlex) > open."""
    assert io_module.LICENCE_PRECEDENCE == (
        "internal",
        "licensed",
        "noncommercial",
        "open",
    )


@pytest.mark.parametrize("licence", PRECEDENCE)
def test_strictest_of_a_single_licence_is_itself(licence: str) -> None:
    """One value in, the same value out, for every licence in the precedence."""
    assert strictest_licence([licence]) == licence


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        pytest.param(["open", "noncommercial"], "noncommercial", id="open_and_chemlex"),
        pytest.param(["open", "licensed"], "licensed", id="open_and_pistachio"),
        pytest.param(
            ["noncommercial", "internal"], "internal", id="chemlex_and_inventory"
        ),
        pytest.param(
            ["open", "noncommercial", "internal"], "internal", id="three_way_mix"
        ),
        pytest.param(
            ["licensed", "noncommercial"], "licensed", id="pistachio_beats_chemlex"
        ),
        pytest.param(
            ["open", "open", "noncommercial", "open"], "noncommercial", id="repeats"
        ),
    ],
)
def test_strictest_licence_of_a_mix(values: list[str], expected: str) -> None:
    """A merged table carries the most restrictive licence of its inputs (ADR 009 item 5)."""
    assert strictest_licence(values) == expected


def test_strictest_licence_accepts_a_pandas_series() -> None:
    """The two callers hand it a groupby column, so any iterable of strings must work."""
    assert strictest_licence(pd.Series(["open", "noncommercial"])) == "noncommercial"


def test_unknown_licence_stops_strictest_licence_and_is_named() -> None:
    """ "cc_by" is not in the precedence. Guessing it is open would disarm the guard, so it raises,
    and the message names the value and keeps the words "unknown licence" the features test relies on."""
    with pytest.raises(ValueError, match="unknown licence.*cc_by"):
        strictest_licence(["open", "cc_by"])


def test_unknown_licence_is_refused_even_beside_a_stricter_one() -> None:
    """Strictest-first must not stop at "internal" and never look at the rest."""
    with pytest.raises(ValueError, match="cc_by"):
        strictest_licence(["internal", "cc_by"])


def test_empty_input_stops_strictest_licence() -> None:
    """Decision: no values is an error, not "open". A group without a licence is a table that lost
    its column; the weakest default would let it through the guard. Fail loudly (CLAUDE.md §8)."""
    with pytest.raises(ValueError):
        strictest_licence([])


def test_null_among_valid_licences_stops_strictest_licence() -> None:
    """A null is not a licence: it must not be skipped on the way to "internal" or "open"."""
    with pytest.raises(ValueError, match="unknown licence"):
        strictest_licence(["open", None])  # type: ignore[list-item]
