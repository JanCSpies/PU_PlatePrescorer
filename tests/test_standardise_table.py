"""Tests for stage 6 on the 12 rows mapped for real (clean -> classify -> standardise)."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from platepu.classify.engine import classify_table, load_class
from platepu.reactions.clean import clean_table, load_pipeline
from platepu.standardise.molecule import load_std_config
from platepu.standardise.table import standardise_table

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def rows() -> dict[str, dict[str, object]]:
    """Stage 6 rows of the fixture, by record_id."""
    cleaned, _ = clean_table(
        pq.read_table(ROOT / "tests" / "data" / "mapped_12rows.parquet"),
        load_pipeline(ROOT / "configs" / "clean.yml"),
    )
    classified, _ = classify_table(
        cleaned, [load_class(ROOT / "configs" / "classes" / "amide_coupling.yml")]
    )
    table = standardise_table(
        classified, load_std_config(ROOT / "configs" / "standardise.yml")
    )
    return {row["record_id"]: row for row in table.to_pylist()}


def test_both_hatu_drawings_give_one_condition(
    rows: dict[str, dict[str, object]],
) -> None:
    """chemlex:0 (O-form) and chemlex:1 (N-oxide): HATU + DIEA + DMF, PF6 dropped."""
    assert rows["chemlex:0"]["condition_key"] == rows["chemlex:1"]["condition_key"]
    assert len(rows["chemlex:0"]["condition_key"]) == 3


def test_condition_comes_from_the_mapped_reagents(
    rows: dict[str, dict[str, object]],
) -> None:
    """ORDerly 638744 (EDC, HOBt, Et3N, DMF; HCl dropped): roles from the mapping, not columns."""
    key = rows["orderly:638744"]["condition_key"]
    assert "CCN=C=NCCCN(C)C" in key and "CN(C)C=O" in key and "Cl" not in key


def test_substrates_have_both_keys(rows: dict[str, dict[str, object]]) -> None:
    """Stereo-aware and stereo-agnostic keys for each substrate; the version is recorded."""
    row = rows["chemlex:0"]
    assert row["key_a_flat"] == str(row["key_a_stereo"]).split("-")[0]
    assert row["std_version"] == "2"


def test_replicates_share_their_keys(rows: dict[str, dict[str, object]]) -> None:
    """chemlex:4 and :5 are one reaction measured twice: same substrates, same condition."""
    a, b = rows["chemlex:4"], rows["chemlex:5"]
    assert (a["key_a_stereo"], a["key_b_stereo"], a["condition_key"]) == (
        b["key_a_stereo"],
        b["key_b_stereo"],
        b["condition_key"],
    )


@pytest.mark.parametrize("substrate_a", ["*C(=O)O", "C(=O"])
def test_failed_substrate_stops_the_stage(substrate_a: str) -> None:
    """No classified substrate failed standardisation in full ORDerly or Chemlex (counted
    2026-10-20), so a failure means input we have not seen: stop with the record, never write a null
    substrate that could reach the positives (CLAUDE.md §8)."""
    classified = pa.table(
        {
            "record_id": ["orderly:1"],
            "licence": ["open"],
            "class": ["amide_coupling"],
            "substrate_a": [substrate_a],
            "substrate_b": ["CN"],
            "n_reactants": [2],
            "n_products": [1],
            "n_dynamic_bonds": [2],
            "clean_rsmi": ["CC(=O)O.CN>CN(C)C=O>CNC(C)=O"],
        }
    )
    with pytest.raises(ValueError, match="orderly:1"):
        standardise_table(
            classified, load_std_config(ROOT / "configs" / "standardise.yml")
        )
