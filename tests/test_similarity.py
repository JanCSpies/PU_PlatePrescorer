"""Tests for platepu/score/similarity.py, the project's one per-side substrate similarity:
``fingerprints`` (binarised count matrix plus key -> row index) and ``side_similarity`` (Tanimoto
between the distinct scored and the distinct positive substrates of one side).

The two functions were moved unchanged out of score/baselines.py, where they were private; the
oracle test (tests/test_score_baselines_oracle.py) guards the move, these tests pin the claims in
their docstrings, which no caller test states directly.

Written BEFORE the module exists (2026-10-30): until src/platepu/score/similarity.py is written
this file fails at collection with ModuleNotFoundError, and that is intended.

Every fixture is made up and worked out by hand; no file is read and no real row is used.
"""

import numpy as np
import pandas as pd
import pytest

from platepu.score.similarity import fingerprints, side_similarity

BLOCK = "morgan"
WIDTH = 6


def substrate_table(rows: dict[str, list[int]], block: str = BLOCK) -> pd.DataFrame:
    """A stage F-like frame: key_stereo plus one uint8 count vector per substrate."""
    return pd.DataFrame(
        {
            "key_stereo": list(rows),
            block: [np.array(counts, dtype=np.uint8) for counts in rows.values()],
        }
    )


def tanimoto_by_hand(left: list[int], right: list[int]) -> float:
    """Tanimoto of two 0/1 vectors written out as set arithmetic, independent of the code."""
    left_on = {i for i, bit in enumerate(left) if bit}
    right_on = {i for i, bit in enumerate(right) if bit}
    union = left_on | right_on
    return len(left_on & right_on) / len(union) if union else 0.0


def similarity_of(
    rows: dict[str, list[int]], scored: list[str], positive: list[str]
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    """Run side_similarity on 0/1 vectors given directly as bits."""
    index, bits = fingerprints(substrate_table(rows), BLOCK)
    return side_similarity(pd.Series(scored), pd.Series(positive), index, bits)


# ---- fingerprints ----------------------------------------------------------------------------


def test_count_of_three_becomes_one() -> None:
    """Tanimoto asks which environments occur, not how often: counts 3 and 255 both become 1.0.

    block_matrix never binarises (ADR 018); this caller does, so this is where it is pinned.
    """
    table = substrate_table({"acid": [3, 0, 1, 255, 0, 2]})
    _, bits = fingerprints(table, BLOCK)
    assert bits[0].tolist() == [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]


def test_fingerprints_stay_float32() -> None:
    """The matrix is float32 on purpose and must not be widened to float64.

    The module docstring records, as ADR 025's reasoning and not re-measured here, that float32
    rounding can surface as a negative eigenvalue in a positive-semidefiniteness check. This test
    measures nothing of that: it only asserts the dtype. Widening it changes a documented
    property, so it should cost a test change.
    """
    _, bits = fingerprints(substrate_table({"acid": [1, 0, 2, 0, 0, 0]}), BLOCK)
    assert bits.dtype == np.float32


def test_index_maps_each_key_to_its_own_row_in_table_order() -> None:
    rows = {
        "acid_b": [1, 0, 0, 0, 0, 0],
        "acid_a": [0, 1, 0, 0, 0, 0],
        "acid_c": [0, 0, 1, 0, 0, 0],
    }
    index, bits = fingerprints(substrate_table(rows), BLOCK)
    assert index == {"acid_b": 0, "acid_a": 1, "acid_c": 2}
    for key, counts in rows.items():
        assert bits[index[key]].tolist() == [float(count > 0) for count in counts]


def test_unknown_block_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="unknown feature block 'morgn'"):
        fingerprints(substrate_table({"acid": [1, 0, 0, 0, 0, 0]}), "morgn")


def test_known_block_without_its_column_is_refused_differently() -> None:
    """'morgan_centre' is a real block; this table was built without the column."""
    table = substrate_table({"acid": [1, 0, 0, 0, 0, 0]})  # only 'morgan'
    with pytest.raises(ValueError, match=r"lacks column\(s\) \['morgan_centre'\]"):
        fingerprints(table, "morgan_centre")


# ---- side_similarity -------------------------------------------------------------------------


def test_identical_fingerprints_give_one_and_disjoint_give_zero() -> None:
    """a = {0,1}, b = {0,1}, c = {2,3}.

    a vs b: shared 2, union 2 -> 1.0.  a vs c: shared 0, union 4 -> 0.0.
    """
    rows = {"a": [1, 1, 0, 0, 0, 0], "b": [1, 1, 0, 0, 0, 0], "c": [0, 0, 1, 1, 0, 0]}
    similarity, scored, positive = similarity_of(rows, ["a"], ["b", "c"])
    assert similarity[scored["a"], positive["b"]] == 1.0
    assert similarity[scored["a"], positive["c"]] == 0.0


def test_two_shared_bits_of_five_give_point_four() -> None:
    """a = {0,1,2,3}, b = {2,3,4}: shared = {2,3} = 2, union = {0,1,2,3,4} = 5, 2/5 = 0.4.

    0.4 is not representable in float32, so compare with the float32 nearest to 2/5, which is what
    the single division must return.
    """
    rows = {"a": [1, 1, 1, 1, 0, 0], "b": [0, 0, 1, 1, 1, 0]}
    similarity, scored, positive = similarity_of(rows, ["a"], ["b"])
    assert similarity[scored["a"], positive["b"]] == np.float32(0.4)


def test_two_empty_fingerprints_give_zero_not_nan() -> None:
    """Union 0 would be 0/0; the where=union > 0 guard must leave 0.0 in place.

    Reachable at d = 0: a substrate whose reacting centre contributes no environment.
    """
    rows = {"empty_a": [0, 0, 0, 0, 0, 0], "empty_b": [0, 0, 0, 0, 0, 0]}
    similarity, _, _ = similarity_of(rows, ["empty_a"], ["empty_b"])
    assert not np.isnan(similarity).any()
    assert similarity[0, 0] == 0.0


def test_empty_fingerprint_against_a_nonempty_one_is_zero() -> None:
    """Union 2 > 0, shared 0 -> 0.0 (the guard must not zero the wrong cells)."""
    rows = {"empty": [0, 0, 0, 0, 0, 0], "a": [1, 1, 0, 0, 0, 0]}
    similarity, scored, positive = similarity_of(rows, ["empty"], ["a"])
    assert similarity[scored["empty"], positive["a"]] == 0.0


def test_repeated_scored_key_gets_one_row() -> None:
    """Only distinct substrates are compared; the pair reduction is the caller's job.

    Scored 'a', 'b', 'a' -> two rows, and the row map sends 'a' to a single row.
    """
    rows = {"a": [1, 1, 0, 0, 0, 0], "b": [0, 1, 1, 0, 0, 0], "p": [1, 0, 0, 0, 0, 0]}
    similarity, scored, positive = similarity_of(rows, ["a", "b", "a"], ["p", "p"])
    assert similarity.shape == (2, 1)
    assert scored == {"a": 0, "b": 1}
    assert positive == {"p": 0}


def test_row_and_column_maps_address_the_right_tanimoto() -> None:
    """Every (scored, positive) cell equals the hand-written Tanimoto of those two substrates.

    The scored side is given out of table order with a repeat, so a map that followed table order
    instead of first appearance would put values in the wrong cells.
    """
    rows = {
        "s1": [1, 1, 0, 0, 0, 0],
        "s2": [0, 1, 1, 1, 0, 0],
        "p1": [1, 0, 0, 0, 0, 1],
        "p2": [0, 1, 1, 0, 0, 0],
        "p3": [0, 0, 0, 0, 0, 0],
    }
    scored_keys = ["s2", "s1", "s2"]
    positive_keys = ["p3", "p1", "p2"]
    similarity, scored, positive = similarity_of(rows, scored_keys, positive_keys)
    assert scored == {"s2": 0, "s1": 1}
    assert positive == {"p3": 0, "p1": 1, "p2": 2}
    for scored_key in scored:
        for positive_key in positive:
            expected = tanimoto_by_hand(rows[scored_key], rows[positive_key])
            assert similarity[
                scored[scored_key], positive[positive_key]
            ] == pytest.approx(expected, abs=1e-7)


def test_full_width_float32_shared_and_union_are_exact_integers() -> None:
    """With 2,048 bits, every step before the division is exact in float32.

    The docstring claims each dot product is an integer far below 2^24. Check it against int64:
    the float32 matrix product must equal the int64 one exactly, and the returned similarity must
    be the single correctly rounded float32 division of those integers, and nothing else.
    Random dense 0/1 vectors are the worst case. Counts are drawn from 0, 1, 2, so about 2/3 of the
    bits are on and two independent rows share about 2048 * (2/3)**2 = 910 (standard deviation
    about 22). The sentinel below asks for more than 700, the value a density of one third would
    give, so it fails if the fixture silently becomes sparse; it looks only at pairs of different
    rows, because a row against itself shares all its own bits whatever the density.
    """
    generator = np.random.default_rng(20261030)
    width = 2048
    counts = generator.integers(
        0, 3, size=(6, width)
    )  # 0, 1 or 2; about 2/3 of bits on
    rows = {f"s{i}": counts[i].tolist() for i in range(6)}
    index, bits = fingerprints(substrate_table(rows), BLOCK)
    assert bits.shape == (6, width)

    left_keys, right_keys = ["s0", "s1", "s2"], ["s3", "s4", "s5", "s0"]
    left = bits[[index[key] for key in left_keys]]
    right = bits[[index[key] for key in right_keys]]
    shared_int = left.astype(np.int64) @ right.astype(np.int64).T
    union_int = (
        left.astype(np.int64).sum(1)[:, None]
        + right.astype(np.int64).sum(1)[None, :]
        - shared_int
    )
    assert shared_int[:, :3].min() > 700  # columns s3..s5: different rows, dense enough
    assert ((left @ right.T) == shared_int).all()  # exact in float32, no rounding
    assert (
        left.sum(1)[:, None] + right.sum(1)[None, :] - left @ right.T == union_int
    ).all()

    similarity, scored, positive = side_similarity(
        pd.Series(left_keys), pd.Series(right_keys), index, bits
    )
    assert similarity.dtype == np.float32
    expected = shared_int.astype(np.float32) / union_int.astype(np.float32)
    assert (similarity == expected).all()
    assert similarity[scored["s0"], positive["s0"]] == 1.0
