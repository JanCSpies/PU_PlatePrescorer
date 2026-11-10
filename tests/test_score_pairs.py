"""Tests for platepu.score.pairs: pair_features, pair_matrix and substrate_index
(docs/design/modeling.md, "Stage S - scores").

Substrate rows and keys are made up (open, hand-written, 2026-10-21; the pair_matrix and
substrate_index cases added 2026-11-02, the first moved here from test_score_pu.py so that this file
imports nothing from the PU modules). No library or Chemlex row.
"""

import re

import numpy as np
import pandas as pd
import pytest

from platepu.score.pairs import pair_features, pair_matrix, substrate_index

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


# --- pair_matrix and substrate_index ------------------------------------------------------------

ACID_BITS = [1, 1, 0, 0]
AMINE_BITS = [0, 0, 1, 1]


def fingerprint_table() -> pd.DataFrame:
    """One acid and one amine with 4-bit blocks; descriptors are 99 so a leak is visible.

    ``morgan`` is twice ``morgan_centre``, so the block that was read shows in the values.
    """
    bits = {"A1": ACID_BITS, "N1": AMINE_BITS}
    return pd.DataFrame(
        {
            "key_stereo": list(bits),
            "morgan_centre": [np.array(b, dtype=np.uint8) for b in bits.values()],
            "morgan": [np.array(b, dtype=np.uint8) * 2 for b in bits.values()],
            "logp": [99, 99],
            "n_heavy_atoms": [99, 99],
            "std_version": [STD, STD],
        }
    )


def one_pair(acid: str = "A1", amine: str = "N1") -> pd.DataFrame:
    return pd.DataFrame({"key_a_stereo": [acid], "key_b_stereo": [amine]})


def test_pair_matrix_is_acid_bits_then_amine_bits() -> None:
    """The acid's block first, the amine's second; a second pair (N1, A1) swaps the halves."""
    pairs = pd.DataFrame({"key_a_stereo": ["A1", "N1"], "key_b_stereo": ["N1", "A1"]})
    matrix = pair_matrix(pairs, fingerprint_table(), "morgan_centre")
    assert matrix.tolist() == [ACID_BITS + AMINE_BITS, AMINE_BITS + ACID_BITS]


def test_pair_matrix_holds_no_descriptor_column() -> None:
    """The table carries logp and n_heavy_atoms (both 99); neither reaches the matrix. ADR 008
    item 1: hand-written chemistry lives in a baseline, never in the model."""
    matrix = pair_matrix(one_pair(), fingerprint_table(), "morgan_centre")
    assert matrix.shape == (1, 2 * len(ACID_BITS))
    assert 99 not in matrix


def test_pair_matrix_is_float64_not_float32() -> None:
    """float64 is pinned because float32 looks cheaper and is not: scikit-learn's histogram boosting
    copies a float32 matrix to float64 inside fit, so the peak is the sum of both (measured
    2026-10-30, one fit of 40,000 x 4,096: float32 input 2.81 GB peak, float64 input 2.16 GB).
    Values are unchanged either way, so only the dtype can show it."""
    matrix = pair_matrix(one_pair(), fingerprint_table(), "morgan_centre")
    assert matrix.dtype == np.float64


def test_pair_matrix_morgan_block_reads_the_morgan_column() -> None:
    """The fixture's morgan column is twice morgan_centre. Counts pass through unchanged (ADR 018)."""
    centre = pair_matrix(one_pair(), fingerprint_table(), "morgan_centre")
    assert (pair_matrix(one_pair(), fingerprint_table(), "morgan") == 2 * centre).all()


def test_pair_matrix_refuses_a_pair_whose_substrate_is_missing() -> None:
    with pytest.raises(ValueError, match="A_MISSING"):
        pair_matrix(one_pair(acid="A_MISSING"), fingerprint_table(), "morgan_centre")


@pytest.mark.parametrize(
    "block", ["descriptors", "descriptors + size", "size alone", "morgan_typo"]
)
def test_pair_matrix_refuses_a_non_fingerprint_block_by_name(block: str) -> None:
    with pytest.raises(ValueError, match=re.escape(block)):
        pair_matrix(one_pair(), fingerprint_table(), block)


def test_substrate_index_is_the_row_position_of_each_key() -> None:
    """Rows A2, A1, N1 in that order: positions 0, 1, 2, so the index follows the table's order,
    not the sorted order of the keys."""
    table = pd.DataFrame({"key_stereo": ["A2", "A1", "N1"]})
    assert substrate_index(table) == {"A2": 0, "A1": 1, "N1": 2}


def test_substrate_index_refuses_a_table_without_key_stereo() -> None:
    table = pd.DataFrame({"key_flat": ["a1"]})
    with pytest.raises(ValueError, match=r"lacks column\(s\) \['key_stereo'\]"):
        substrate_index(table)


def test_substrate_index_refuses_a_repeated_key_with_counts() -> None:
    """'dup' twice and 'one' once: 1 duplicated value on 2 rows. A dictionary would keep the last
    row of 'dup' and return a wrong feature row with no error, so this is refused."""
    table = pd.DataFrame({"key_stereo": ["dup", "one", "dup"]})
    with pytest.raises(ValueError, match=r"\(1 on 2 rows\)") as refusal:
        substrate_index(table)
    assert "dup" in str(refusal.value)


def test_substrate_index_counts_two_null_keys_as_one_duplicated_value() -> None:
    """A count by nunique() drops nulls and would say none duplicated; the null must be shown."""
    table = pd.DataFrame({"key_stereo": ["a", np.nan, np.nan]})
    with pytest.raises(ValueError, match=r"\(1 on 2 rows\)") as refusal:
        substrate_index(table)
    assert "nan" in str(refusal.value)
