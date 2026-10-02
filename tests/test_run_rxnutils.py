"""Tests for the rxnutils input built by scripts/run_rxnutils.py."""

from run_rxnutils import rxnutils_input

from platepu.ingest.record import ReactionRecord


def record(source_id: str, mapped_rxn: str | None) -> ReactionRecord:
    """A minimal ORDerly record; only source_id and mapped_rxn matter here."""
    return ReactionRecord(
        source="orderly",
        source_id=source_id,
        licence="open",
        reactants=("CC(=O)O", "NCc1ccccc1"),
        agents=("CCN=C=NCCCN(C)C",),
        solvents=("CN(C)C=O",),
        products=("CC(=O)NCc1ccccc1",),
        mapped_rxn=mapped_rxn,
        temperature_c=None,
        reaction_time_h=None,
        yield_pct=None,
        year=None,
    )


def test_row_id_is_source_id_and_rsmi_is_mapped_reaction() -> None:
    """The join key back to agents/solvents is the record's source_id."""
    table = rxnutils_input([record("638744", "[CH3:1][OH:2]>>[CH3:1][Cl:3]")])
    assert table.to_dict(orient="records") == [
        {"row_id": "638744", "rsmi": "[CH3:1][OH:2]>>[CH3:1][Cl:3]"}
    ]


def test_record_without_mapped_reaction_is_left_out() -> None:
    """reactants2reagents needs atom mapping, so unmapped rows never reach rxnutils."""
    table = rxnutils_input(
        [record("1", None), record("2", "[CH3:1][OH:2]>>[CH3:1][Cl:3]")]
    )
    assert list(table.row_id) == ["2"]
    assert list(table.columns) == ["row_id", "rsmi"]
