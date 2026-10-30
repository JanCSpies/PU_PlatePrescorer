"""Reference-oracle tests for B2 and B3 of platepu.score.baselines, written 2026-10-22.

Why they exist: B2 (``nearest_pair_scores``) and B3 (``additive_scores``) produced the numbers in
ADR 008 and docs/design/modeling.md, and a full S1 run reproduced its 2,872,100 score rows byte for
byte. A memory rewrite of the B2 inner loop must not move one bit of them. The functions below are
frozen copies of the code of that day; the tests compare with ``np.array_equal``, never
``allclose``, because a changed tie-break or a float32 rounding would not show in a tolerance.

The fixtures are made up. Fingerprints are random bit sets of 2,048 bits, so Tanimoto values are
fractions such as 3/17 that float32 and float64 round differently; hand-sized pattern pools give
the heavy ties. Nothing here is real or licensed data.
"""

import numpy as np
import pandas as pd
import pytest

from platepu.score import baselines
from platepu.score.baselines import MORGAN, additive_scores, nearest_pair_scores

N_BITS = 2048


def _bits_matrix(substrates: pd.DataFrame) -> tuple[dict[str, int], np.ndarray]:
    """Frozen copy of ``_fingerprints`` as of 2026-10-22."""
    index = {key: row for row, key in enumerate(substrates["key_stereo"])}
    counts = np.stack([np.asarray(fingerprint) for fingerprint in substrates["morgan"]])
    return index, (counts > 0).astype(np.float32)


def _tanimoto_float32(
    scored: pd.Series, positive: pd.Series, index: dict[str, int], bits: np.ndarray
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    """Frozen copy of ``_side_similarity`` as of 2026-10-22 (float32 maths, widened at the end)."""
    scored_keys, positive_keys = pd.unique(scored), pd.unique(positive)
    left = bits[[index[key] for key in scored_keys]]
    right = bits[[index[key] for key in positive_keys]]
    shared = left @ right.T
    union = left.sum(1)[:, None] + right.sum(1)[None, :] - shared
    similarity = np.divide(shared, union, out=np.zeros_like(shared), where=union > 0)
    return (
        similarity.astype(np.float64),
        {key: row for row, key in enumerate(scored_keys)},
        {key: column for column, key in enumerate(positive_keys)},
    )


def b2_reference_2026_10_22(
    paired: pd.DataFrame, positives: pd.DataFrame, substrates: pd.DataFrame
) -> np.ndarray:
    """B2 as it was on 2026-10-22, in one block with no CHUNK.

    It exists to prove that a rewrite is value-identical. Do not tidy, speed up or share code with
    ``baselines``: its whole worth is that it stays the same while the real function changes.
    """
    index, bits = _bits_matrix(substrates)
    rows, columns, similarity = {}, {}, {}
    for side, key in (("a", "key_a_stereo"), ("b", "key_b_stereo")):
        similarity[side], scored, positive = _tanimoto_float32(
            paired[key], positives[key], index, bits
        )
        rows[side] = paired[key].map(scored).to_numpy()
        columns[side] = positives[key].map(positive).to_numpy()
    acid = similarity["a"][rows["a"]][:, columns["a"]]
    amine = similarity["b"][rows["b"]][:, columns["b"]]
    return np.minimum(acid, amine).max(axis=1)


def b3_reference_2026_10_22(
    paired: pd.DataFrame, positives: pd.DataFrame, substrates: pd.DataFrame
) -> np.ndarray:
    """B3 as it was on 2026-10-22: the two per-side maxima are added in float64.

    The accumulation must stay float64. Each per-side value is a float32 number widened to float64;
    adding them in float64 is exact enough to be the sum of two floats, while adding them in float32
    rounds the sum to 24 bits and moves the last digit of most scores.
    """
    index, bits = _bits_matrix(substrates)
    total = np.zeros(len(paired), dtype=np.float64)
    for key in ("key_a_stereo", "key_b_stereo"):
        similarity, scored, _ = _tanimoto_float32(
            paired[key], positives[key], index, bits
        )
        total += similarity.max(axis=1)[paired[key].map(scored).to_numpy()]
    return total


def _substrates(
    acid_bits: list[np.ndarray], amine_bits: list[np.ndarray]
) -> pd.DataFrame:
    """Stage F table: keys ``a0, a1, ...`` and ``n0, n1, ...`` with the given set bits."""
    keys, counts = [], []
    for prefix, group in (("a", acid_bits), ("n", amine_bits)):
        for number, on_bits in enumerate(group):
            vector = np.zeros(N_BITS, dtype=np.uint8)
            vector[on_bits] = 2  # a count of 2 is binarised to 1
            keys.append(f"{prefix}{number}")
            counts.append(vector)
    return pd.DataFrame({"key_stereo": keys, "morgan": counts})


def _pairs(*pairs: tuple[int, int]) -> pd.DataFrame:
    """Rows with the two stereo keys, from (acid number, amine number)."""
    return pd.DataFrame(
        {
            "key_a_stereo": [f"a{a}" for a, _ in pairs],
            "key_b_stereo": [f"n{n}" for _, n in pairs],
        }
    )


def _random_bits(
    generator: np.random.Generator, count: int, size: int = 40, pool: int = 120
) -> list[np.ndarray]:
    """``count`` random bit sets of up to ``size`` bits drawn from the first ``pool`` bits."""
    return [generator.choice(pool, size=size, replace=False) for _ in range(count)]


def _all_pairs(n_acids: int, n_amines: int) -> pd.DataFrame:
    return _pairs(*[(a, n) for a in range(n_acids) for n in range(n_amines)])


def _assert_b2_identical(
    paired: pd.DataFrame, positives: pd.DataFrame, substrates: pd.DataFrame
) -> np.ndarray:
    actual = nearest_pair_scores(paired, positives, substrates, MORGAN)
    expected = b2_reference_2026_10_22(paired, positives, substrates)
    assert actual.dtype == np.float64
    assert np.array_equal(actual, expected)
    return actual


def test_b2_random_fingerprints_equal_the_oracle_bit_for_bit() -> None:
    generator = np.random.default_rng(2026)
    substrates = _substrates(_random_bits(generator, 9), _random_bits(generator, 8))
    positive_rows = _pairs(
        *[(int(a), int(n)) for a, n in generator.integers(0, 8, (15, 2))]
    )
    scores = _assert_b2_identical(_all_pairs(9, 8), positive_rows, substrates)
    # Guard on the fixture itself: several distinct, non-dyadic values, so rounding is exercised.
    assert len(set(scores.tolist())) > 5


def test_b2_heavy_ties_pick_the_same_values() -> None:
    """Three fingerprint patterns per side, 60 scored pairs: nearly every score is a tie."""
    patterns_a = [np.arange(0, 6), np.arange(3, 9), np.arange(50, 56)]
    patterns_n = [np.arange(100, 104), np.arange(102, 106), np.arange(300, 304)]
    acids = [patterns_a[number % 3] for number in range(6)]
    amines = [patterns_n[number % 3] for number in range(5)]
    substrates = _substrates(acids, amines)
    positive_rows = _pairs((0, 0), (1, 1), (2, 2), (3, 4))
    scores = _assert_b2_identical(_all_pairs(6, 5), positive_rows, substrates)
    assert len(set(scores.tolist())) < 8 < len(scores)


def test_b2_remainder_block_of_one_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """Seven scored pairs with CHUNK = 3 give blocks of 3, 3 and 1."""
    generator = np.random.default_rng(7)
    substrates = _substrates(_random_bits(generator, 4), _random_bits(generator, 4))
    paired = _pairs((0, 0), (1, 1), (2, 2), (3, 3), (0, 3), (3, 0), (1, 2))
    positive_rows = _pairs((0, 1), (2, 3), (3, 3))
    monkeypatch.setattr(baselines, "CHUNK", 3)
    _assert_b2_identical(paired, positive_rows, substrates)


def test_b2_single_scored_pair_and_single_positive_pair() -> None:
    generator = np.random.default_rng(1)
    substrates = _substrates(_random_bits(generator, 3), _random_bits(generator, 3))
    _assert_b2_identical(_pairs((1, 2)), _pairs((0, 1), (2, 2)), substrates)
    _assert_b2_identical(_all_pairs(3, 3), _pairs((2, 1)), substrates)
    scores = _assert_b2_identical(_pairs((0, 0)), _pairs((0, 0)), substrates)
    assert scores[0] == 1.0


def test_b2_positive_pairs_sharing_an_acid_or_an_amine() -> None:
    generator = np.random.default_rng(3)
    substrates = _substrates(_random_bits(generator, 5), _random_bits(generator, 5))
    shared_acid = _pairs((0, 0), (0, 1), (0, 2), (3, 4))
    shared_amine = _pairs((0, 1), (1, 1), (2, 1), (4, 3))
    both = _pairs((0, 1), (0, 2), (1, 1), (2, 1), (3, 3))
    for positive_rows in (shared_acid, shared_amine, both):
        _assert_b2_identical(_all_pairs(5, 5), positive_rows, substrates)


def test_b2_acid_in_no_positive_pair_but_its_amine_is_and_the_reverse() -> None:
    """Scored (a4, n0): a4 is in no positive pair, n0 is. Scored (a0, n4): the reverse."""
    generator = np.random.default_rng(4)
    substrates = _substrates(_random_bits(generator, 5), _random_bits(generator, 5))
    positive_rows = _pairs((0, 0), (1, 1), (2, 2))
    paired = _pairs((4, 0), (0, 4), (4, 4), (0, 0))
    _assert_b2_identical(paired, positive_rows, substrates)


def test_b2_substrates_with_no_bits_give_zero_not_nan() -> None:
    """An empty fingerprint against an empty one has union 0; the guarded divide returns 0."""
    generator = np.random.default_rng(5)
    acids = _random_bits(generator, 3) + [np.array([], dtype=int)] * 2
    amines = _random_bits(generator, 3) + [np.array([], dtype=int)] * 2
    substrates = _substrates(acids, amines)
    positive_rows = _pairs((0, 0), (3, 3), (1, 4), (4, 2))
    scores = _assert_b2_identical(_all_pairs(5, 5), positive_rows, substrates)
    assert not np.isnan(scores).any()
    both_empty = nearest_pair_scores(_pairs((3, 3)), _pairs((4, 4)), substrates, MORGAN)
    assert both_empty[0] == 0.0


def test_b2_identical_fingerprints_give_duplicate_similarity_columns() -> None:
    generator = np.random.default_rng(6)
    acid_bits = _random_bits(generator, 3)
    amine_bits = _random_bits(generator, 3)
    substrates = _substrates(
        [*acid_bits, acid_bits[0], acid_bits[0]], [*amine_bits, amine_bits[1]]
    )
    positive_rows = _pairs((0, 0), (3, 1), (4, 3), (1, 2))
    _assert_b2_identical(_all_pairs(5, 4), positive_rows, substrates)


def test_b2_returns_float64() -> None:
    generator = np.random.default_rng(8)
    substrates = _substrates(_random_bits(generator, 2), _random_bits(generator, 2))
    scores = nearest_pair_scores(_all_pairs(2, 2), _pairs((0, 0)), substrates, MORGAN)
    assert scores.dtype == np.float64


def test_b3_random_fingerprints_equal_the_oracle_bit_for_bit() -> None:
    generator = np.random.default_rng(2026)
    substrates = _substrates(_random_bits(generator, 9), _random_bits(generator, 8))
    positive_rows = _pairs(
        *[(int(a), int(n)) for a, n in generator.integers(0, 8, (15, 2))]
    )
    paired = _all_pairs(9, 8)
    actual = additive_scores(paired, positive_rows, substrates)
    assert actual.dtype == np.float64
    assert np.array_equal(
        actual, b3_reference_2026_10_22(paired, positive_rows, substrates)
    )


def test_b3_float64_accumulation_differs_from_a_float32_sum() -> None:
    """Shows the B3 test has teeth: adding the two per-side maxima in float32 would change scores."""
    generator = np.random.default_rng(2026)
    substrates = _substrates(_random_bits(generator, 9), _random_bits(generator, 8))
    positive_rows = _pairs(
        *[(int(a), int(n)) for a, n in generator.integers(0, 8, (15, 2))]
    )
    paired = _all_pairs(9, 8)
    index, bits = _bits_matrix(substrates)
    total32 = np.zeros(len(paired), dtype=np.float32)
    for key in ("key_a_stereo", "key_b_stereo"):
        similarity, scored, _ = _tanimoto_float32(
            paired[key], positive_rows[key], index, bits
        )
        total32 += similarity.astype(np.float32).max(axis=1)[
            paired[key].map(scored).to_numpy()
        ]
    actual = additive_scores(paired, positive_rows, substrates)
    assert not np.array_equal(actual, total32.astype(np.float64))


def test_nearest_pair_scores_unchanged_after_the_refactor() -> None:
    """B2 and B8 on six hand-made substrates, values hard-coded.

    Taken from ``nearest_pair_scores`` on 2026-10-28, before ``features.blocks`` existed. Acids
    a0 {0,1,2,3}, a1 {0,1,4,5}, a2 {10..13}; amines n0 {100..103}, n1 {100,101,104,105},
    n2 {200..203}. On ``morgan`` T(a0,a1) = T(n0,n1) = 2/6; on ``morgan_centre`` a1 keeps three of
    a0's bits {0,1,2,6} and n1 three of n0's {100,101,102,104}, so both are 3/5. The values are the
    float32 Tanimoto widened to float64, which is what B2 returns (see ``similarity.side_similarity``), so
    ``array_equal`` and not ``allclose``: a move in the last digit is a change.
    Grid order a0n0, a0n1, a0n2, a1n0, a1n1, a1n2, a2n0, a2n1, a2n2; positives a0n0, a1n1, a2n2.
    """
    substrates = _substrates(
        [np.array([0, 1, 2, 3]), np.array([0, 1, 4, 5]), np.arange(10, 14)],
        [np.arange(100, 104), np.array([100, 101, 104, 105]), np.arange(200, 204)],
    )
    centre_bits = {"a1": [0, 1, 2, 6], "n1": [100, 101, 102, 104]}
    centre = []
    for key, counts in zip(substrates["key_stereo"], substrates["morgan"], strict=True):
        vector = np.zeros(N_BITS, dtype=np.uint8)
        vector[centre_bits.get(key, np.flatnonzero(counts))] = 1
        centre.append(vector)
    substrates["morgan_centre"] = centre
    third, three_fifths = 0.3333333432674408, 0.6000000238418579
    expected = {
        "morgan": [1.0, third, 0.0, third, 1.0, 0.0, 0.0, 0.0, 1.0],
        "morgan_centre": [
            1.0,
            three_fifths,
            0.0,
            three_fifths,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    }
    for block, values in expected.items():
        actual = nearest_pair_scores(
            _all_pairs(3, 3), _pairs((0, 0), (1, 1), (2, 2)), substrates, block
        )
        assert actual.dtype == np.float64
        assert np.array_equal(actual, np.array(values)), block
