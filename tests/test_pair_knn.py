"""Tests for scripts/counts/count_pair_knn.py, the pair-level K-nearest-neighbour probe that asks
whether our two unlabeled spaces (the literature-built U and the lab inventory U) are separable by
a plain fingerprint K-NN, as Chen et al. asked of the DUD-E decoys.

Every fixture is made up and worked out by hand on 2026-10-29; no real row is used.

The probe's one trap: a positive pair compared with the pool of pairs finds itself at similarity 1.0
on both sides, while an unlabeled pair cannot (reported pairs are subtracted from U), so a probe
without leave-one-out separates the classes perfectly with no chemistry. Most tests below exist to
make that impossible. Similarity 1.0 on ONE side is correct and must survive: an unlabeled pair's
acid really is a published acid.

Interface these tests pin (all in scripts/counts/count_pair_knn.py):
    pair_similarity(labelled, substrates, block) -> (n, n) float array, min of the two sides'
        Tanimoto on binarised counts, diagonal included (a pair against itself is 1.0 there).
    neighbour_indices(labelled, substrates, block, k) -> (n, k) int array of row positions, the
        pair itself never among them, ties broken by the lower row position.
    pair_knn_auc(labelled, substrates, block, ks) -> {k: ROC-AUC}, the score of a pair being the
        fraction of its k neighbours labelled 1; keys in the order of ``ks``.
    balanced_sample(labelled, per_class, seed) -> frame with ``per_class`` rows of each label.
``labelled`` has columns key_a_stereo, key_b_stereo, label (1 = positive, 0 = unlabeled);
``substrates`` has key_stereo and a fingerprint column named by ``block``.

The neighbour pool is the frame passed in: the caller draws a balanced sample (judgement recorded
in the report of 2026-10-29), so the probe itself never scores against positives only.
"""

import numpy as np
import pandas as pd
import pytest

from platepu.score.pair_knn import (
    balanced_sample,
    neighbour_indices,
    pair_knn_auc,
    pair_similarity,
)

N_BITS = 2048
BLOCK = "morgan"


def substrates_from(bit_sets: dict[str, set[int]]) -> pd.DataFrame:
    """Stage F style table: one binary count vector per key."""
    vectors = []
    for bits in bit_sets.values():
        counts = np.zeros(N_BITS, dtype=np.uint8)
        counts[sorted(bits)] = 1
        vectors.append(counts)
    return pd.DataFrame({"key_stereo": list(bit_sets), BLOCK: vectors})


def labelled_from(*rows: tuple[str, str, int]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["key_a_stereo", "key_b_stereo", "label"])


# Two groups of two pairs. Inside a group the acids and the amines are one bit apart (inter 3,
# union 5, Tanimoto 3/5); across groups they share no bit, so the cross similarity is 0.
TWINS = substrates_from(
    {
        "A1": {0, 1, 2, 3},
        "A1x": {0, 1, 2, 4},
        "N1": {100, 101, 102, 103},
        "N1x": {100, 101, 102, 104},
        "A2": {10, 11, 12, 13},
        "A2x": {10, 11, 12, 14},
        "N2": {110, 111, 112, 113},
        "N2x": {110, 111, 112, 114},
    }
)
# Each positive has an unlabeled twin that is its nearest neighbour and carries the other label.
TWIN_PAIRS = labelled_from(
    ("A1", "N1", 1), ("A1x", "N1x", 0), ("A2", "N2", 1), ("A2x", "N2x", 0)
)


def test_a_pair_is_never_its_own_neighbour() -> None:
    """Each pair is 1.0 against itself, yet none lists itself, at k = 1 and at the widest k."""
    for k in (1, 3):
        neighbours = neighbour_indices(TWIN_PAIRS, TWINS, BLOCK, k)
        assert neighbours.shape == (4, k)
        for row, found in enumerate(neighbours):
            assert row not in found


def test_a_pair_finds_its_twin_not_itself() -> None:
    """The nearest neighbour of each pair is the other member of its group (0.6 against 1.0 for self)."""
    neighbours = neighbour_indices(TWIN_PAIRS, TWINS, BLOCK, 1)
    assert neighbours[:, 0].tolist() == [1, 0, 3, 2]


def test_a_mixed_fixture_does_not_give_a_perfect_auc() -> None:
    """Behavioural version. Every nearest neighbour carries the opposite label, so the honest k = 1
    AUC is 0.0. A probe that lets a pair match itself would score each positive 1 and each
    unlabeled 0, an AUC of 1.0, without any chemistry."""
    aucs = pair_knn_auc(TWIN_PAIRS, TWINS, BLOCK, (1,))
    assert aucs[1] != 1.0
    assert aucs[1] == 0.0


def test_one_sided_exact_similarity_is_kept() -> None:
    """The unlabeled pair (A1, N1x) shares its acid exactly with the positive (A1, N1) and its
    amine at 3/5, so it scores min(1.0, 0.6) = 0.6 and that positive is its nearest neighbour.
    Excluding every exact substrate match would drop it and this would fail."""
    substrates = substrates_from(
        {
            "A1": {0, 1, 2, 3},
            "N1": {100, 101, 102, 103},
            "N1x": {100, 101, 102, 104},
            "A9": {50, 51, 52, 53},
            "N9": {150, 151, 152, 153},
        }
    )
    frame = labelled_from(("A1", "N1", 1), ("A1", "N1x", 0), ("A9", "N9", 0))
    similarity = pair_similarity(frame, substrates, BLOCK)
    assert similarity[1, 0] == pytest.approx(0.6)
    assert neighbour_indices(frame, substrates, BLOCK, 1)[1, 0] == 0


def test_the_same_pair_at_two_row_positions_is_each_others_neighbour() -> None:
    """Self is excluded by row position, not by similarity value, so a pair listed twice finds its
    twin at 1.0. This documents the design choice rather than guarding live data: P has one row per
    pair and U's reported pairs are subtracted on flat keys, so it cannot arise today.
    """
    frame = labelled_from(("A1", "N1", 1), ("A1", "N1", 0), ("A2", "N2", 1))
    neighbours = neighbour_indices(frame, TWINS, BLOCK, 1)
    assert neighbours[0, 0] == 1
    assert neighbours[1, 0] == 0


def test_pair_similarity_is_the_minimum_of_the_sides() -> None:
    """Acids {0,1,2,3} and {0,1,4,5}: 2/6 = 1/3. Amines {100,101,102,103} and {100,101,102,104}:
    3/5. Minimum 1/3; the mean would be 7/15 and the product 1/5."""
    substrates = substrates_from(
        {
            "A1": {0, 1, 2, 3},
            "A2": {0, 1, 4, 5},
            "N1": {100, 101, 102, 103},
            "N2": {100, 101, 102, 104},
        }
    )
    frame = labelled_from(("A1", "N1", 1), ("A2", "N2", 0))
    similarity = pair_similarity(frame, substrates, BLOCK)
    assert similarity[0, 1] == pytest.approx(1 / 3)
    assert similarity[1, 0] == pytest.approx(1 / 3)
    assert similarity[0, 0] == pytest.approx(1.0)


def test_k_sweep_returns_one_auc_per_k() -> None:
    """One float in [0, 1] per requested k, keys in the order given."""
    aucs = pair_knn_auc(TWIN_PAIRS, TWINS, BLOCK, (3, 1, 2))
    assert list(aucs) == [3, 1, 2]
    assert all(0.0 <= value <= 1.0 for value in aucs.values())


def test_k_not_below_the_pool_size_is_refused() -> None:
    """With the pair itself left out a pool of 4 has 3 neighbours; k = 4 would have to invent one."""
    with pytest.raises(ValueError, match="k"):
        neighbour_indices(TWIN_PAIRS, TWINS, BLOCK, 4)


def clusters() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Four positives and four unlabeled in two separated clusters. Inside a cluster the acids are
    10-bit sets missing one bit each (Tanimoto 8/10 = 0.8), likewise the amines; across clusters
    nothing is shared."""
    bit_sets: dict[str, set[int]] = {}
    rows = []
    for name, label, acid_base, amine_base in (
        ("P", 1, 0, 100),
        ("U", 0, 200, 300),
    ):
        for i in range(4):
            bit_sets[f"{name}a{i}"] = set(range(acid_base, acid_base + 10)) - {
                acid_base + i
            }
            bit_sets[f"{name}n{i}"] = set(range(amine_base, amine_base + 10)) - {
                amine_base + i
            }
            rows.append((f"{name}a{i}", f"{name}n{i}", label))
    return labelled_from(*rows), substrates_from(bit_sets)


def test_two_clusters_give_their_best_auc_at_k_one() -> None:
    """Positive control for Chen's signature. Two separated clusters, one per class: k = 1 (and
    k = 3, the cluster size less one) is perfect; k = 7 votes over everything and falls to 0.0.
    Without this, 'k = 1 won' on real data has no calibration."""
    frame, substrates = clusters()
    aucs = pair_knn_auc(frame, substrates, BLOCK, (1, 3, 7))
    assert aucs[1] == 1.0
    assert aucs[1] == max(aucs.values())
    assert aucs[7] < aucs[1]


def gradient() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Twelve pairs on a line. Pair t has acid and amine windows of 100 bits starting at position
    p_t, so Tanimoto falls smoothly with distance. Labels run from mostly 1 on the left to mostly 0
    on the right with flips in between, so one nearest neighbour is a noisy witness. Checked
    against a reference implementation under 200 row permutations: k = 1, 3, 5 give
    0.586, 0.600, 0.686 every time, so the order does not depend on how ties fall."""
    positions = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66]
    labels = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    bit_sets: dict[str, set[int]] = {}
    rows = []
    for t, (position, label) in enumerate(zip(positions, labels, strict=True)):
        bit_sets[f"A{t}"] = set(range(position, position + 100))
        bit_sets[f"N{t}"] = set(range(1000 + position, 1000 + position + 100))
        rows.append((f"A{t}", f"N{t}", label))
    return labelled_from(*rows), substrates_from(bit_sets)


def test_a_gradient_does_not_favour_k_one() -> None:
    """Negative control. With overlapping classes a larger k does at least as well as k = 1, and
    nothing is perfect. The fixture is a deterministic 12-pair line, not a random draw, so it is
    not flaky."""
    frame, substrates = gradient()
    aucs = pair_knn_auc(frame, substrates, BLOCK, (1, 3, 5))
    assert aucs[3] >= aucs[1]
    assert aucs[5] >= aucs[1]
    assert max(aucs.values()) < 1.0


def tied_pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Five pairs with one shared acid and amines of 7 of 8 bits (6/8 = 0.75 to each other), so
    every off-diagonal similarity is the same 0.75 and every neighbour choice is a tie."""
    bit_sets: dict[str, set[int]] = {"A": {0, 1, 2, 3}}
    rows = []
    for i in range(5):
        bit_sets[f"N{i}"] = set(range(100, 108)) - {100 + i}
        rows.append(("A", f"N{i}", i % 2))
    return labelled_from(*rows), substrates_from(bit_sets)


def test_ties_in_similarity_are_handled_deterministically() -> None:
    """Two runs agree, and the tie rule is the lower row position among those that are not the pair
    itself: row 0 takes rows 1, 2; row 4 takes rows 0, 1."""
    frame, substrates = tied_pairs()
    first = neighbour_indices(frame, substrates, BLOCK, 2)
    second = neighbour_indices(frame, substrates, BLOCK, 2)
    assert (first == second).all()
    assert first[0].tolist() == [1, 2]
    assert first[4].tolist() == [0, 1]
    assert pair_knn_auc(frame, substrates, BLOCK, (1, 2)) == pair_knn_auc(
        frame, substrates, BLOCK, (1, 2)
    )


def test_balanced_sample_has_equal_classes_and_is_seeded() -> None:
    """Ten positives and thirty unlabeled, five of each drawn: 5 and 5, rows taken from the input,
    the same seed giving the same rows and another seed (here) a different draw."""
    frame = labelled_from(*[(f"a{i}", f"n{i}", 1 if i < 10 else 0) for i in range(40)])
    sample = balanced_sample(frame, 5, 0)
    assert sample["label"].value_counts().to_dict() == {1: 5, 0: 5}
    assert set(sample["key_a_stereo"]) <= set(frame["key_a_stereo"])
    again = balanced_sample(frame, 5, 0)
    assert sample["key_a_stereo"].tolist() == again["key_a_stereo"].tolist()
    other = balanced_sample(frame, 5, 1)
    assert sample["key_a_stereo"].tolist() != other["key_a_stereo"].tolist()
