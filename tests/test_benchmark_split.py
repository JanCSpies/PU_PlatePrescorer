"""Tests for splitting benchmark labels of one condition family into validation and test.

Rule (agreed 2026-10-22): test = labelled family rows the benchmark marked "test"; validation =
every other labelled family row whose acid key and amine key both appear nowhere in the test side.
All rows are hand-written with made-up keys (no Chemlex row).
"""

import pandas as pd
import pytest

from platepu.benchmarks.split import family_split

HATU = frozenset({"A", "B", "C"})


def _rows(*rows: tuple[str, str, str, str | None, bool | None]) -> pd.DataFrame:
    """Rows of (condition, acid key, amine key, split value, label).

    The flat key is the part of the stereo key before the first "-", as the first InChIKey block is.
    """
    table = pd.DataFrame(
        rows,
        columns=["condition", "key_a_stereo", "key_b_stereo", "both_unseen", "label"],
    ).astype({"label": "boolean"})
    table["key_a_flat"] = table["key_a_stereo"].str.split("-").str[0]
    table["key_b_flat"] = table["key_b_stereo"].str.split("-").str[0]
    return table


LABELS = _rows(
    ("A + B + C", "acid1", "amine1", "train", True),
    ("A + B + C + D", "acid2", "amine2", "test", False),
    ("A + B", "acid3", "amine3", "train", True),
    ("A + B + C", "acid4", "amine4", None, True),
    ("A + B + C", "acid5", "amine5", "test", None),
)


def test_family_and_split_decide_the_set() -> None:
    """Row 0 (train) and row 3 (null split, substrates unseen in test) are validation; row 1 is
    test; row 2 is outside the family; row 4 has no label (inconsistent replicates).
    Before 2026-10-22 validation was only the train side, [0]."""
    val, test = family_split(LABELS, HATU, "both_unseen")
    assert list(val.index) == [0, 3]
    assert list(test.index) == [1]


def test_validation_excludes_every_test_substrate() -> None:
    """A row sharing only its amine with the test side, or only its acid, must not be validation."""
    labels = _rows(
        ("A + B + C", "acid_t", "amine_t", "test", True),
        ("A + B + C", "acid_x", "amine_t", "train", True),
        ("A + B + C", "acid_t", "amine_x", None, False),
        ("A + B + C", "acid_x", "amine_x", "train", True),
    )
    val, _ = family_split(labels, HATU, "both_unseen")
    assert list(val.index) == [3]


def test_validation_keeps_a_row_the_benchmark_discarded() -> None:
    """A labelled family row with a null split and no test substrate is validation now; the old
    rule (split == "train") dropped it."""
    labels = _rows(
        ("A + B + C", "acid_t", "amine_t", "test", True),
        ("A + B + C", "acid_d", "amine_d", None, False),
    )
    val, _ = family_split(labels, HATU, "both_unseen")
    assert list(val.index) == [1]


def test_test_side_is_unchanged_by_the_new_rule() -> None:
    """Test holds exactly the labelled family rows with split "test"."""
    _, test = family_split(LABELS, HATU, "both_unseen")
    assert list(test.index) == [1]
    assert set(test["both_unseen"]) == {"test"}


def test_validation_and_test_share_no_substrate() -> None:
    """The sealed test depends on this: no acid key and no amine key in both frames."""
    labels = _rows(
        ("A + B + C", "acid_t", "amine_t", "test", True),
        ("A + B + C", "acid_t", "amine_x", "train", True),
        ("A + B + C", "acid_x", "amine_t", None, False),
        ("A + B + C", "acid_x", "amine_x", "train", True),
        ("A + B + C", "acid_y", "amine_y", None, True),
    )
    val, test = family_split(labels, HATU, "both_unseen")
    assert not set(val["key_a_stereo"]) & set(test["key_a_stereo"])
    assert not set(val["key_b_stereo"]) & set(test["key_b_stereo"])
    assert not set(val["key_a_flat"]) & set(test["key_a_flat"])
    assert not set(val["key_b_flat"]) & set(test["key_b_flat"])
    assert len(val) == 2


def test_validation_excludes_a_stereoisomer_of_a_test_substrate() -> None:
    """L- and D-alanine share the first InChIKey block but differ in the stereo block, so a
    stereo-only exclusion lets the D form into validation next to the L form in test. Our Morgan
    fingerprints ignore chirality: both have the identical feature vector, so the model has then
    seen the test substrate and the test row is not held out. Same for the amine side."""
    l_form = "QNAYBMKLOCPYGJ-REOHCLBHSA-N"
    d_form = "QNAYBMKLOCPYGJ-UWTATZPHSA-N"
    l_amine = "BAWFJGJZGIEFAR-VIFPVBQESA-N"
    d_amine = "BAWFJGJZGIEFAR-SECBINFHSA-N"
    labels = _rows(
        ("A + B + C", l_form, l_amine, "test", True),
        ("A + B + C", d_form, "amine_x", "train", True),
        ("A + B + C", "acid_x", d_amine, "train", True),
        ("A + B + C", "acid_y", "amine_y", "train", True),
    )
    val, _ = family_split(labels, HATU, "both_unseen")
    assert list(val.index) == [3]


def test_rows_outside_the_family_or_without_a_label_are_dropped() -> None:
    """A row without all family members or without a label is in neither frame."""
    labels = _rows(
        ("A + B", "acid_x", "amine_x", "train", True),
        ("A + B + C", "acid_y", "amine_y", "train", None),
        ("A + B + C", "acid_z", "amine_z", "test", None),
    )
    val, test = family_split(labels, HATU, "both_unseen")
    assert val.empty
    assert test.empty


def test_validation_keeps_std_version() -> None:
    """The split only selects rows, so the std_version column stays."""
    val, _ = family_split(LABELS.assign(std_version="2"), HATU, "both_unseen")
    assert list(val["std_version"]) == ["2", "2"]


@pytest.mark.parametrize("spelling", ["Test", "TEST", "test ", None])
def test_split_column_without_the_word_test_stops_instead_of_filling_validation(
    spelling: str | None,
) -> None:
    """Only the exact value "test" marks the test side. With "Test" (or NaN) the test side came out
    empty, every labelled row became validation, nothing raised, and the seal looked intact because
    an empty test side shares no substrate with anything."""
    labels = _rows(
        ("A + B + C", "acid1", "amine1", spelling, True),
        ("A + B + C", "acid2", "amine2", spelling, False),
    )
    with pytest.raises(ValueError, match="test"):
        family_split(labels, HATU, "both_unseen")
