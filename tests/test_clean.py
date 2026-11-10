"""Tests for stage 4 (rxnutils clean, configs/clean.yml) on 12 rows mapped for real by RXNMapper.

Fixture ``tests/data/mapped_12rows.parquet``: stage-3 rows mapped with RXNMapper 0.4.3. ORDerly: the
5 fixture reactions and ``orderly:267359`` (its product is also a reactant, so nothing is left), cut
on 2026-10-09 from the samples run. Chemlex layout: the 5 readable hand-written reactions of
``chemlex_like_6rows.parquet`` and ``chemlex:6``, mapped wrongly by hand (4 dynamic bonds), all built
on 2026-10-20 by ``scratch/build_chemlex_like_fixtures.py`` so no Chemlex data is in the repository.
"""

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from platepu.reactions.clean import clean_table, load_pipeline

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "data" / "mapped_12rows.parquet"


@pytest.fixture(scope="module")
def cleaned() -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    """Clean the fixture once: rows by record_id, and reject reasons by record_id."""
    table, rejected = clean_table(
        pq.read_table(FIXTURE), load_pipeline(ROOT / "configs" / "clean.yml")
    )
    rows = {row["record_id"]: row for row in table.to_pylist()}
    return rows, dict(
        zip(
            rejected["record_id"].to_pylist(),
            rejected["reason"].to_pylist(),
            strict=True,
        )
    )


def test_correct_amide_coupling_has_two_dynamic_bonds(
    cleaned: tuple[dict, dict],
) -> None:
    """Acid + amine → amide: C–O broken, C–N formed. HATU, DIPEA, DMF are reagents."""
    rows, _ = cleaned
    row = rows["chemlex:0"]
    assert (row["n_reactants"], row["n_products"], row["n_dynamic_bonds"]) == (2, 1, 2)
    assert "CN(C)C=O" in str(row["clean_rsmi"]).split(">")[1]


def test_mis_mapped_reaction_is_kept_with_its_count(cleaned: tuple[dict, dict]) -> None:
    """Clean never filters by dynamic bonds; the mis-mapped chemlex:6 stays, with 4."""
    rows, _ = cleaned
    assert rows["chemlex:6"]["n_dynamic_bonds"] == 4


def test_reaction_without_product_is_rejected(cleaned: tuple[dict, dict]) -> None:
    """orderly:267359: the product is also a reactant, so no product is left."""
    rows, rejected = cleaned
    assert "orderly:267359" not in rows
    assert rejected == {"orderly:267359": "no_product"}


def test_output_keeps_the_key_and_the_licence(cleaned: tuple[dict, dict]) -> None:
    """Every later stage needs record_id and licence; 11 of 12 rows go on."""
    rows, _ = cleaned
    assert len(rows) == 11
    assert all(row["licence"] == "open" for row in rows.values())
