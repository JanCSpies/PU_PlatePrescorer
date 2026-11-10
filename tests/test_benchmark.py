"""Tests for benchmark measurements and reaction labels, on the six hand-written Chemlex-layout rows."""

from pathlib import Path

import pandas as pd
import pytest

from platepu.benchmarks.labels import LabelRule
from platepu.benchmarks.measurements import measurements, reaction_labels
from platepu.classify.engine import fits_roles, load_class
from platepu.ingest.chemlex import records_from_table, splits_from_table
from platepu.ingest.record import records_to_table
from platepu.standardise.molecule import UNREADABLE, load_std_config

ROOT = Path(__file__).parents[1]
RULE = load_class(ROOT / "configs" / "classes" / "amide_coupling.yml")
CONFIG = load_std_config(ROOT / "configs" / "standardise.yml")
TABLE = pd.read_parquet(ROOT / "tests" / "data" / "chemlex_like_6rows.parquet")
SPLITS = splits_from_table(TABLE).assign(
    record_id=lambda t: "chemlex:" + t["source_id"]
)


@pytest.fixture(scope="module")
def measured() -> pd.DataFrame:
    """Measurements of the fixture rows."""
    return measurements(records_to_table(records_from_table(TABLE)), RULE, CONFIG)


def test_roles_come_from_the_columns(measured: pd.DataFrame) -> None:
    """Acid and amine are taken as given; all six fixture amines are true amines."""
    assert measured["in_class"].all()
    assert measured.set_index("record_id").loc["chemlex:0", "substrate_a"] == (
        "O=C(O)CCCc1ccccc1"
    )


def test_sulfonamide_does_not_fit_the_class() -> None:
    """Chemlex's sulfonamide 'amines' are outside amide_coupling (decided 2026-10-09)."""
    assert not fits_roles("CC(=O)O", "CC(C)(C)NS(=O)(=O)c1ccccc1", RULE)
    assert fits_roles("CC(=O)O", "NCc1ccccc1", RULE)


def test_unreadable_reagent_keeps_the_row(measured: pd.DataFrame) -> None:
    """PyBrOP's PF6 cannot be read: the row stays, with a condition that never matches."""
    assert measured.set_index("record_id").loc["chemlex:3", "condition"] == UNREADABLE


def test_replicates_become_one_labelled_reaction(measured: pd.DataFrame) -> None:
    """Rows 4 (31 %) and 5 (0 %): one reaction, inconsistent, no label; splits kept per reaction."""
    labels = reaction_labels(
        measured,
        SPLITS,
        ["one_unseen", "both_unseen"],
        LabelRule(20.0, 10.0),
        {"chemlex:4"},
    )
    pair = labels[labels["substrate_a"] == "O=C(O)c1ccc(Cl)cc1"].iloc[0]
    assert (pair["n_measurements"], pair["status"]) == (2, "inconsistent")
    assert pd.isna(pair["label"])
    assert pair["both_unseen"] == "test"
    assert pair["n_mapped_in_class"] == 1  # only row 4 was given as mapped in class
    assert len(labels) == 5


def test_disagreeing_splits_become_none(measured: pd.DataFrame) -> None:
    """If replicates disagree on a split, that split is unusable for the reaction, and flagged."""
    splits = SPLITS.copy()
    splits.loc[splits["record_id"] == "chemlex:5", "one_unseen"] = "train"
    labels = reaction_labels(
        measured, splits, ["one_unseen"], LabelRule(20.0, 10.0), set()
    )
    pair = labels[labels["substrate_a"] == "O=C(O)c1ccc(Cl)cc1"].iloc[0]
    assert pair["one_unseen"] is None and pair["split_conflict"]
    assert labels["split_conflict"].sum() == 1


def test_failed_substrate_stops_the_benchmark() -> None:
    """No Chemlex substrate fails standardisation (23,338 counted 2026-10-20). A failure would give a
    null key that pandas drops silently when grouping replicates, so it stops with the record."""
    broken = TABLE.assign(Acid=["*C(=O)O", *TABLE["Acid"][1:]])
    with pytest.raises(ValueError, match="chemlex:0"):
        measurements(records_to_table(records_from_table(broken)), RULE, CONFIG)


def test_measurements_carry_std_version(measured: pd.DataFrame) -> None:
    """Every measurement names the standardisation of its keys (CLAUDE.md section 6)."""
    assert set(measured["std_version"]) == {CONFIG.std_version}


def test_reaction_labels_carry_std_version(measured: pd.DataFrame) -> None:
    """Replicates collapse to one reaction, which keeps the one std_version."""
    labels = reaction_labels(
        measured, SPLITS, ["one_unseen"], LabelRule(20.0, 10.0), set()
    )
    assert set(labels["std_version"]) == {CONFIG.std_version}
