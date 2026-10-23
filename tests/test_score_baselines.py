"""Tests for platepu.score.baselines B0 to B7 (docs/design/modeling.md, "Stage S - scores").

Every fixture is made up and worked out by hand on 2026-10-21. Fingerprints are bit sets chosen so
a Tanimoto is a fraction of small integers: |intersection| / |union| of the set bits (counts > 0).
    acids : A1 = {0,1,2,3}   A2 = {0,1,4,5}   A3 = {10,11,12,13}
            T(A1,A2) = 2/6 = 1/3, T(A1,A3) = T(A2,A3) = 0
    amines: N1 = {100..103}  N2 = {100,101,104,105}  N3 = {200..203}
            T(N1,N2) = 1/3, T(N1,N3) = T(N2,N3) = 0
A1 carries a count of 3 on bit 0 so that binarising the counts is exercised.
"""

import math

import numpy as np
import pandas as pd
import pytest

from platepu.score import baselines
from platepu.score.baselines import (
    additive_scores,
    nearest_pair_scores,
    popularity_scores,
    random_scores,
    rank_sum,
    size_scores,
    subclass_scores,
)

N_BITS = 2048
BITS = {
    "A1": [0, 1, 2, 3],
    "A2": [0, 1, 4, 5],
    "A3": [10, 11, 12, 13],
    "N1": [100, 101, 102, 103],
    "N2": [100, 101, 104, 105],
    "N3": [200, 201, 202, 203],
}


def fingerprint(key: str) -> np.ndarray:
    """Count vector with 1 on the listed bits (3 on bit 0 of A1)."""
    counts = np.zeros(N_BITS, dtype=np.uint8)
    counts[BITS[key]] = 1
    if key == "A1":
        counts[0] = 3
    return counts


def substrates() -> pd.DataFrame:
    """Stage F table of the six substrates (key_stereo, key_flat, morgan)."""
    keys = list(BITS)
    return pd.DataFrame(
        {
            "key_stereo": keys,
            "key_flat": [key.lower() for key in keys],
            "morgan": [fingerprint(key) for key in keys],
        }
    )


def positives(*pairs: tuple[str, str], n_reports: int = 1) -> pd.DataFrame:
    """Positive pairs, one row each, with stereo and flat keys."""
    return pd.DataFrame(
        {
            "family": "all",
            "key_a_stereo": [a for a, _ in pairs],
            "key_b_stereo": [b for _, b in pairs],
            "key_a_flat": [a.lower() for a, _ in pairs],
            "key_b_flat": [b.lower() for _, b in pairs],
            "n_reports": n_reports,
        }
    )


def paired(*pairs: tuple[str, str]) -> pd.DataFrame:
    """What pair_features returns for these pairs, built by hand (a_ and b_ columns)."""
    rows = []
    for a, b in pairs:
        rows.append(
            {
                "key_a_stereo": a,
                "key_b_stereo": b,
                "a_key_flat": a.lower(),
                "b_key_flat": b.lower(),
                "a_morgan": fingerprint(a),
                "b_morgan": fingerprint(b),
            }
        )
    return pd.DataFrame(rows)


def keyed(frame: pd.DataFrame, scores: np.ndarray) -> dict[tuple[str, str], float]:
    """Score of each pair by its two stereo keys."""
    return {
        (a, b): float(score)
        for a, b, score in zip(
            frame["key_a_stereo"], frame["key_b_stereo"], scores, strict=True
        )
    }


GRID = [(f"A{i}", f"N{j}") for i in (1, 2, 3) for j in (1, 2, 3)]


def test_random_score_of_a_pair_does_not_depend_on_row_order() -> None:
    """Nine pairs scored in grid order and in a shuffled order give the same score per pair, because
    the draw is made on the pairs sorted by (key_a_stereo, key_b_stereo)."""
    frame = paired(*GRID)
    shuffled = frame.sample(frac=1, random_state=3).reset_index(drop=True)
    first = keyed(frame, random_scores(frame, 7, "validation"))
    second = keyed(shuffled, random_scores(shuffled, 7, "validation"))
    assert first == second


def test_random_scores_are_uniform_floats_in_zero_one() -> None:
    """float64 in [0, 1), one per row, not all equal."""
    scores = random_scores(paired(*GRID), 7, "validation")
    assert scores.dtype == np.float64
    assert len(scores) == 9
    assert ((scores >= 0) & (scores < 1)).all()
    assert len(set(scores)) == 9


def test_random_differs_between_score_sets_with_one_seed() -> None:
    """The same nine pairs and seed 7 under the names validation and inventory must not share one
    stream; otherwise the two sets would be correlated."""
    frame = paired(*GRID)
    validation = random_scores(frame, 7, "validation")
    inventory = random_scores(frame, 7, "inventory")
    assert not np.array_equal(validation, inventory)


def test_random_validation_scores_do_not_change_when_the_other_set_grows() -> None:
    """Score validation, then score an inventory of 9 and one of 900 pairs, then validation again:
    identical. One stream over both sets would shift validation whenever U changes size."""
    validation = paired(*GRID[:4])
    before = random_scores(validation, 7, "validation")
    random_scores(paired(*GRID), 7, "inventory")
    big = pd.DataFrame(
        {"key_a_stereo": [f"X{i}" for i in range(900)], "key_b_stereo": "N1"}
    )
    random_scores(big, 7, "inventory")
    np.testing.assert_array_equal(before, random_scores(validation, 7, "validation"))


def test_random_seed_changes_the_scores() -> None:
    """Seeds 7 and 8 on the same set give different draws."""
    frame = paired(*GRID)
    assert not np.array_equal(
        random_scores(frame, 7, "validation"), random_scores(frame, 8, "validation")
    )


def test_popularity_sums_log1p_of_both_sides() -> None:
    """P: (a1,n1) x3 reports, (a1,n2) x1, (a2,n1) x2 (n_reports 3, 1, 2). Pair (A1,N1): acid a1 is in
    rows 1 and 2 -> 3 + 1 = 4; amine n1 is in rows 1 and 3 -> 3 + 2 = 5; score log1p(4) + log1p(5)."""
    positive_rows = positives(("A1", "N1"), ("A1", "N2"), ("A2", "N1"))
    positive_rows["n_reports"] = [3, 1, 2]
    scores = popularity_scores(paired(("A1", "N1")), positive_rows)
    assert scores.dtype == np.float64
    assert scores[0] == pytest.approx(math.log1p(4) + math.log1p(5))


def test_popularity_is_zero_for_two_unknown_substrates() -> None:
    """A3 and N3 are in no positive: log1p(0) + log1p(0) == 0, the tie that holds 68.6 % of the
    validation pairs."""
    scores = popularity_scores(paired(("A3", "N3")), positives(("A1", "N1")))
    assert scores[0] == 0.0


def test_nearest_pair_uses_the_minimum_not_the_mean() -> None:
    """P = {(A1,N1)}. Scored (A1,N2) shares the acid exactly (1) and has an amine at T(N1,N2) = 1/3.
    Minimum: 1/3 = 0.333. The mean would give (1 + 1/3) / 2 = 0.667, about halfway."""
    scores = nearest_pair_scores(
        paired(("A1", "N2")), positives(("A1", "N1")), substrates()
    )
    assert scores[0] == pytest.approx(1 / 3)


def test_a_pair_identical_to_a_positive_scores_one() -> None:
    """(A1,N1) against P = {(A1,N1)}: both sides 1, minimum 1. A1 has a count of 3 on bit 0, so this
    also shows the counts are binarised (a count-weighted Tanimoto of A1 with itself is still 1,
    but with A2 it would differ from 1/3)."""
    scores = nearest_pair_scores(
        paired(("A1", "N1")), positives(("A1", "N1")), substrates()
    )
    assert scores.dtype == np.float64
    assert scores[0] == pytest.approx(1.0)


def test_nearest_pair_binarises_the_counts() -> None:
    """Scored (A2,N1) against P = {(A1,N1)}: T(A2,A1) on the bits is 2/6 = 1/3 although A1 holds a
    count of 3 on bit 0 (a count Tanimoto would be 2/(3+1+...) = 2/8 = 0.25). Amine side is 1."""
    scores = nearest_pair_scores(
        paired(("A2", "N1")), positives(("A1", "N1")), substrates()
    )
    assert scores[0] == pytest.approx(1 / 3)


def test_nearest_pair_takes_the_maximum_over_positive_pairs() -> None:
    """Scored (A1,N2). P1 = (A1,N1) gives min(1, 1/3) = 1/3; P2 = (A1,N2) gives 1. Maximum: 1."""
    scores = nearest_pair_scores(
        paired(("A1", "N2")), positives(("A1", "N1"), ("A1", "N2")), substrates()
    )
    assert scores[0] == pytest.approx(1.0)


def test_nearest_pair_is_independent_of_the_chunk_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nine scored pairs against two P pairs with CHUNK = 1, 2, 4 and 5000 (the real value): the
    arrays are equal, so chunking is only a memory device."""
    frame = paired(*GRID)
    positive_rows = positives(("A1", "N1"), ("A2", "N2"))
    monkeypatch.setattr(baselines, "CHUNK", 5000)
    reference = nearest_pair_scores(frame, positive_rows, substrates())
    for chunk in (1, 2, 4):
        monkeypatch.setattr(baselines, "CHUNK", chunk)
        np.testing.assert_array_equal(
            nearest_pair_scores(frame, positive_rows, substrates()), reference
        )


def test_additive_cannot_see_the_pairing() -> None:
    """P1 = {(A1,N1),(A2,N2)} and P2 = {(A1,N2),(A2,N1)} hold the same acids and amines, paired
    differently. B3 uses the substrates only, so it is identical (scored (A1,N1) -> 1 + 1 = 2).
    B2 does see the pairing: min(1,1) = 1 under P1, but under P2 max(min(1,1/3), min(1/3,1)) = 1/3."""
    frame = paired(*GRID)
    pairing_1 = positives(("A1", "N1"), ("A2", "N2"))
    pairing_2 = positives(("A1", "N2"), ("A2", "N1"))
    np.testing.assert_array_equal(
        additive_scores(frame, pairing_1, substrates()),
        additive_scores(frame, pairing_2, substrates()),
    )
    one = nearest_pair_scores(paired(("A1", "N1")), pairing_1, substrates())
    two = nearest_pair_scores(paired(("A1", "N1")), pairing_2, substrates())
    assert (one[0], two[0]) == (pytest.approx(1.0), pytest.approx(1 / 3))


def test_additive_sums_two_perfect_matches_to_two() -> None:
    """(A1,N1) with A1 and N1 both among the P substrates: 1 + 1 = 2."""
    scores = additive_scores(
        paired(("A1", "N1")), positives(("A1", "N1"), ("A2", "N2")), substrates()
    )
    assert scores[0] == pytest.approx(2.0)


def test_additive_takes_the_single_nearest_substrate_per_side() -> None:
    """Scored (A3,N2), P substrates {A1,A2} x {N1,N2}: A3 has T = 0 to both, N2 has T = 1 to N2
    itself (the nearest of N1 at 1/3 and N2 at 1; k = 1). Score 0 + 1 = 1."""
    scores = additive_scores(
        paired(("A3", "N2")), positives(("A1", "N1"), ("A2", "N2")), substrates()
    )
    assert scores[0] == pytest.approx(1.0)


def test_subclass_scores_every_acid_of_one_amine_the_same() -> None:
    """Amine side: three acids of different subclass paired with one aniline get the aniline value
    0.5; a primary aliphatic amine gets 0.8."""
    frame = pd.DataFrame(
        {
            "a_subclass": ["aryl", "alpha_vinyl", "other", "aryl"],
            "b_subclass": ["aniline", "aniline", "aniline", "aliphatic_primary"],
        }
    )
    table = {
        "aniline": 0.5,
        "aliphatic_primary": 0.8,
        "aryl": 0.1,
        "alpha_vinyl": 0.2,
        "other": 0.0,
    }
    scores = subclass_scores(frame, "b", table)
    assert scores.dtype == np.float64
    assert scores.tolist() == [0.5, 0.5, 0.5, 0.8]


def test_subclass_missing_from_the_config_raises() -> None:
    """The amine subclass 'other' is absent from the table; it must be listed explicitly, so
    ValueError naming it."""
    frame = pd.DataFrame({"b_subclass": ["aniline", "other"]})
    with pytest.raises(ValueError, match="other"):
        subclass_scores(frame, "b", {"aniline": 0.5})


def test_rank_sum_ignores_the_scale_of_a_part() -> None:
    """Parts [1,2,3] (ranks 1,2,3) and [30,10,20] (ranks 3,1,2) sum to [4,3,5]; multiplying the
    second by 100 leaves the ranks, hence the result, unchanged."""
    first, second = np.array([1.0, 2.0, 3.0]), np.array([30.0, 10.0, 20.0])
    np.testing.assert_array_equal(rank_sum(first, second), [4.0, 3.0, 5.0])
    np.testing.assert_array_equal(
        rank_sum(first, second * 100), rank_sum(first, second)
    )


def test_rank_sum_gives_tied_values_one_rank() -> None:
    """[1,1,2] has average ranks [1.5,1.5,3]; with [0,1,2] (ranks 1,2,3) the sum is [2.5,3.5,6]. The
    two tied rows share a rank in the first part, they differ only through the second."""
    result = rank_sum(np.array([1.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(result, [2.5, 3.5, 6.0])


def test_size_score_prefers_the_smaller_pair() -> None:
    """Pair 1 has 10 + 5 = 15 heavy atoms, pair 2 has 20 + 8 = 28: scores -15 and -28, so the smaller
    pair scores higher."""
    frame = pd.DataFrame({"a_n_heavy_atoms": [10, 20], "b_n_heavy_atoms": [5, 8]})
    scores = size_scores(frame)
    assert scores.dtype == np.float64
    assert scores.tolist() == [-15.0, -28.0]


def test_nearest_pair_with_no_positives_fails_loudly() -> None:
    """An empty positives frame (a typo in the family name selects no row) must raise ValueError
    naming the positives. Unguarded, B2 would not index past the end: columns["a"] is simply empty,
    the block is (n, 0), and numpy's max over it raises a bare zero-size reduction ValueError."""
    empty = positives(("A1", "N1")).iloc[:0]
    with pytest.raises(ValueError, match="positives"):
        nearest_pair_scores(paired(("A1", "N1")), empty, substrates())


def test_additive_with_no_positives_fails_loudly() -> None:
    """Same input for B3: ValueError naming the positives. Unguarded, B3 raises the same bare
    zero-size reduction ValueError from numpy as B2 (similarity.max over zero columns)."""
    empty = positives(("A1", "N1")).iloc[:0]
    with pytest.raises(ValueError, match="positives"):
        additive_scores(paired(("A1", "N1")), empty, substrates())
