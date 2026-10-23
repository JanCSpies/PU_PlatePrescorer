"""Tests for platepu.score.pairs.pair_features (docs/design/modeling.md, "Stage S - scores").

Substrate rows and keys are made up (open, hand-written, 2026-10-21); no library or Chemlex row.
"""

import pandas as pd
import pytest

from platepu.score.pairs import pair_features

STD = "std1"


def substrate_table() -> pd.DataFrame:
    """Two acids and one amine, as stage F would write them (only a few of its columns)."""
    return pd.DataFrame(
        {
            "key_stereo": ["A1", "A2", "N1"],
            "key_flat": ["a1", "a2", "n1"],
            "n_heavy_atoms": [4, 9, 7],
            "subclass": ["aryl", "alpha_vinyl", "aniline"],
            "std_version": [STD, STD, STD],
        }
    )


def pair_table() -> pd.DataFrame:
    """Three pairs in an order that is neither sorted by acid nor by amine; one extra column."""
    return pd.DataFrame(
        {
            "score_set": ["inventory", "validation", "inventory"],
            "key_b_stereo": ["N1", "N1", "N1"],
            "key_a_stereo": ["A2", "A1", "A1"],
        }
    )


def test_pair_features_prefixes_both_sides() -> None:
    """Pair (A2, N1): acid A2 has 9 heavy atoms and the amine N1 has 7, so a_n_heavy_atoms == 9 and
    b_n_heavy_atoms == 7; every substrate column except key_stereo appears twice, once per side."""
    paired = pair_features(pair_table(), substrate_table())
    first = paired.iloc[0]
    assert (first["a_n_heavy_atoms"], first["b_n_heavy_atoms"]) == (9, 7)
    assert (first["a_subclass"], first["b_subclass"]) == ("alpha_vinyl", "aniline")
    assert (first["a_key_flat"], first["b_key_flat"]) == ("a2", "n1")
    assert "a_key_stereo" not in paired.columns
    assert "key_stereo" not in paired.columns


def test_pair_features_keeps_the_pair_columns_and_order() -> None:
    """Three pairs in, three out, in the same order, with the three own columns untouched."""
    pairs = pair_table()
    paired = pair_features(pairs, substrate_table())
    assert len(paired) == 3
    pd.testing.assert_frame_equal(paired[list(pairs.columns)], pairs)


def test_missing_substrate_key_raises() -> None:
    """A pair whose amine key 'N9' is not in the feature table is input we have not seen: ValueError
    naming the key, never a row of NaN."""
    pairs = pair_table().assign(key_b_stereo=["N1", "N9", "N1"])
    with pytest.raises(ValueError, match="N9"):
        pair_features(pairs, substrate_table())


def test_mixed_std_version_raises() -> None:
    """Acid A1 standardised under std1 and amine N1 under std2: the two sides of one pair are not
    comparable, so ValueError."""
    substrates = substrate_table()
    substrates.loc[substrates["key_stereo"] == "N1", "std_version"] = "std2"
    with pytest.raises(ValueError, match="std_version"):
        pair_features(pair_table(), substrates)


def test_a_table_that_mixes_std_versions_raises() -> None:
    """Every pair is consistent (A1 and N1 both std1) but the table also holds A2 under std2: the
    table mixes versions, which the design also rejects, so ValueError."""
    substrates = substrate_table()
    substrates.loc[substrates["key_stereo"] == "A2", "std_version"] = "std2"
    pairs = pair_table().iloc[[1, 2]]  # A1 with N1 only
    with pytest.raises(ValueError, match="std_version"):
        pair_features(pairs, substrates)


def test_pairs_from_another_standardisation_raise() -> None:
    """Pairs standardised under std0 against features under std1 are not comparable (the run would
    stamp the wrong version on every row): ValueError naming std_version."""
    pairs = pair_table().assign(std_version="std0")
    with pytest.raises(ValueError, match="std_version"):
        pair_features(pairs, substrate_table())


def test_pairs_from_the_same_standardisation_pass() -> None:
    """Pairs carrying std_version std1, equal to the features', are scored and keep their column."""
    paired = pair_features(pair_table().assign(std_version=STD), substrate_table())
    assert len(paired) == 3
    assert set(paired["std_version"]) == {STD}


def test_pairs_without_a_std_version_column_still_work() -> None:
    """Not every caller has a std_version on its pairs (pair_table has none): no error, and the
    substrates' own std_version arrives as a_std_version and b_std_version."""
    paired = pair_features(pair_table(), substrate_table())
    assert "std_version" not in pair_table().columns
    assert set(paired["a_std_version"]) == {STD}
