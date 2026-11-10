"""Tests for the unlabeled space U: every library acid x library amine, minus the positives."""

import pandas as pd

from platepu.library.unlabeled import unlabeled_pairs

LIBRARY = pd.DataFrame(
    {
        "key_stereo": ["A1", "A2", "N1", "N2", "N3"],
        "key_flat": ["a1", "a2", "n1", "n2", "n3"],
        "smiles": ["CC(=O)O", "OC(=O)c1ccccc1", "NC", "NCC", "NCCC"],
        "role": ["acid", "acid", "amine", "amine", "amine"],
    }
)
POSITIVES = pd.DataFrame({"key_a_stereo": ["A1"], "key_b_stereo": ["N2"]})


def test_every_pair_except_positives() -> None:
    """2 acids x 3 amines = 6 pairs; A1 + N2 is a positive, so 5 remain."""
    pairs = unlabeled_pairs(LIBRARY, POSITIVES, "acid", "amine")
    assert len(pairs) == 5
    assert ("A1", "N2") not in set(
        zip(pairs["key_a_stereo"], pairs["key_b_stereo"], strict=True)
    )


def test_columns_follow_the_class_roles() -> None:
    """Role A columns come from the acids, role B from the amines, like the positives table."""
    pairs = unlabeled_pairs(LIBRARY, POSITIVES, "acid", "amine")
    row = pairs.iloc[0]
    assert row["key_a_stereo"].startswith("A") and row["key_b_stereo"].startswith("N")
    assert {"substrate_a", "substrate_b", "key_a_flat", "key_b_flat"} <= set(
        pairs.columns
    )


def test_pairs_are_internal() -> None:
    """ADR 004: U is made of inventory compounds, so every pair is 'internal'."""
    pairs = unlabeled_pairs(LIBRARY, POSITIVES, "acid", "amine")
    assert set(pairs["licence"]) == {"internal"}
