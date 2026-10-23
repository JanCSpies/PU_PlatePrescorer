"""Tests for stage 2: the canonical unmapped reaction SMILES and its hash (mapping cache key)."""

from pathlib import Path

import pandas as pd

from platepu.ingest.chemlex import records_from_table
from platepu.ingest.record import ReactionRecord, records_to_table
from platepu.reactions.smiles import reaction_smiles, rsmi_table

DATA = Path(__file__).parent / "data"


def _record(
    reactants: tuple[str, ...],
    agents: tuple[str, ...] = (),
    solvents: tuple[str, ...] = (),
    products: tuple[str, ...] = ("CC(=O)NCc1ccccc1",),
    source_id: str = "1",
) -> ReactionRecord:
    """A minimal record; only the SMILES fields matter here."""
    return ReactionRecord(
        source="test",
        source_id=source_id,
        licence="open",
        reactants=reactants,
        agents=agents,
        solvents=solvents,
        products=products,
        source_rxn=None,
        temperature_c=None,
        reaction_time_h=None,
        yield_pct=None,
        yield_kind=None,
        year=None,
        source_document=None,
    )


def test_every_molecule_goes_left_of_the_arrow() -> None:
    """Agents and solvents join the reactants: RXNMapper decides roles (ADR 002)."""
    rsmi = reaction_smiles(_record(("CC(=O)O", "NCc1ccccc1"), ("CN(C)C=O",)), {})
    assert rsmi == "CC(=O)O.CN(C)C=O.NCc1ccccc1>>CC(=O)NCc1ccccc1"


def test_same_reaction_written_differently_gives_one_string() -> None:
    """Other atom order, other molecule order, other roles: one key, so one mapping."""
    a = _record(("OC(C)=O", "c1ccccc1CN"), solvents=("O=CN(C)C",))
    b = _record(("NCc1ccccc1",), agents=("CC(=O)O", "CN(C)C=O"))
    assert reaction_smiles(a, {}) == reaction_smiles(b, {})


def test_atom_maps_are_removed() -> None:
    """A source mapping must not change the cache key."""
    mapped = _record(("[CH3:1][C:2](=[O:3])O", "NCc1ccccc1"))
    plain = _record(("CC(=O)O", "NCc1ccccc1"))
    assert reaction_smiles(mapped, {}) == reaction_smiles(plain, {})


def test_unparsable_molecule_gives_none() -> None:
    """PyBrOP's PF6 written as a cation (as in Chemlex) cannot be read, so there is no reaction string."""
    assert reaction_smiles(_record(("CC(=O)O",), ("F[P+](F)(F)(F)(F)F",)), {}) is None


def test_table_has_hash_and_licence_and_rejects() -> None:
    """Six Chemlex-layout rows: PyBrOP (3) is rejected; the replicate pair (4, 5) shares one hash."""
    records = records_from_table(pd.read_parquet(DATA / "chemlex_like_6rows.parquet"))
    table, rejected = rsmi_table(records_to_table(records))
    assert table.column_names == ["record_id", "licence", "rsmi", "rsmi_hash"]
    assert rejected.to_pylist() == [{"record_id": "chemlex:3", "reason": "unparsable"}]
    hashes = dict(
        zip(table["record_id"].to_pylist(), table["rsmi_hash"].to_pylist(), strict=True)
    )
    assert hashes["chemlex:4"] == hashes["chemlex:5"]
    assert len(hashes["chemlex:0"]) == 40
