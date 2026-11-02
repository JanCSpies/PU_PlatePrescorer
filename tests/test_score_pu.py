"""Tests for stage S2a, the spy step (docs/design/modeling.md "Stage S2"; ADR 013 item 3, ADR 016,
ADR 024, ADR 026): hide some known positives inside U, fit P against U, put the threshold where about
95 % of the hidden positives (the spies) score above it, and call the U rows below it reliable
negatives.

Written BEFORE the module exists (2026-10-30): until src/platepu/score/pu.py is written this file
fails at collection with ModuleNotFoundError, and that is intended.

Every fixture is made up and worked out by hand; no file is read except the YAML the config tests
write into ``tmp_path``, and no real row is used.

What these tests do and do not show (ADR 026). The step is a **heuristic with a measured weakness**:
on the real data the reliable-negative set is depleted of the 1,121 literature negatives (3.9 %
caught while calling 14.3 % of U reliable negatives, lift 0.27). The planted test below proves the
*mechanics* -- that the code returns the rows the threshold says it should, on a case built to be
separable. It says nothing about whether the reliable negatives are trustworthy on real data, and
neither does the realised spy recall: that is recall on *reported* positives, and the hidden ones
are the unreported ones (ADR 014 item 1).

The tie rule, written once here and implemented by ``spy_threshold``:

    n spies, spy_recall r  ->  k = ceil(r * n), computed so that float noise cannot add one
    (0.95 * 20 is 19.0, not 19.000...1);
    threshold = the k-th LARGEST spy score;
    a spy is recalled when its score is >= threshold; a U row is a reliable negative only when its
    score is < threshold.

So the realised recall is never below the configured one: at least k spies sit at or above the
threshold, and every spy tied with the k-th is counted as well.
"""

import ast
import inspect
import itertools
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import platepu.score.pu as pu_module
from platepu.features.blocks import FINGERPRINT_BLOCKS, MODEL_DESCRIPTORS
from platepu.score.pu import (
    reliable_negatives,
    spy_labels,
    spy_threshold,
)
from platepu.score.pu_audit import selection_shift
from platepu.score.pu_config import PuConfig, load_pu_config

STD_VERSION = "std1"
PAIR_KEYS = ["key_a_stereo", "key_b_stereo", "key_a_flat", "key_b_flat"]

# Two kinds of substrate. "P-like" ones carry bits 0-1 (acids) and 2-3 (amines); "odd" ones carry
# bits 4-5 and 6-7, so a pair of odd substrates shares no bit with any P-like pair. Within a kind
# the fingerprint is IDENTICAL, so every pair of the same kind has the same feature vector and the
# fitted score of a kind is one number, not a spread; that is what lets a set be compared exactly.
P_LIKE_ACID = [1, 1, 0, 0, 0, 0, 0, 0]
P_LIKE_AMINE = [0, 0, 1, 1, 0, 0, 0, 0]
ODD_ACID = [0, 0, 0, 0, 1, 1, 0, 0]
ODD_AMINE = [0, 0, 0, 0, 0, 0, 1, 1]

CONFIG = PuConfig(
    model="pair_hgb",
    feature_block="morgan_centre",
    p_width="all",
    unlabeled="unlabeled_literature_k10",
    spy_fraction=0.25,
    spy_recall=0.95,
    spy_scoring="out_of_fold",
    n_folds=5,
    learner="HistGradientBoostingClassifier",
    learner_params={"max_iter": 30, "early_stopping": False, "min_samples_leaf": 1},
    seed=7,
)


def substrate_rows(
    role: str, prefix: str, count: int, bits: list[int], heavy: int
) -> list[dict[str, object]]:
    """``count`` substrates sharing one fingerprint; descriptors are 99 so a leak is visible."""
    row: dict[str, object] = {name: 99 for name in MODEL_DESCRIPTORS}
    return [
        {
            **row,
            "key_stereo": f"{prefix}{i}",
            "role": role,
            "n_heavy_atoms": heavy,
            "morgan_centre": np.array(bits, dtype=np.uint8),
            # A different block with different numbers, so a wrong block shows in the values.
            "morgan": np.array(bits, dtype=np.uint8) * 2,
            "morgan_near": np.array(bits, dtype=np.uint8) * 3,
            "std_version": STD_VERSION,
        }
        for i in range(count)
    ]


SUBSTRATES = pd.DataFrame(
    substrate_rows("acid", "PA", 10, P_LIKE_ACID, 20)
    + substrate_rows("amine", "PN", 10, P_LIKE_AMINE, 10)
    + substrate_rows("acid", "OA", 4, ODD_ACID, 10)
    + substrate_rows("amine", "ON", 4, ODD_AMINE, 5)
)


def pair_table(pairs: list[tuple[str, str]], **extra: object) -> pd.DataFrame:
    """Pair keys in the layout of stages P and U; the flat key is the stereo key in lower case."""
    return pd.DataFrame(
        {
            "key_a_stereo": [a for a, _ in pairs],
            "key_b_stereo": [b for _, b in pairs],
            "key_a_flat": [a.lower() for a, _ in pairs],
            "key_b_flat": [b.lower() for _, b in pairs],
            **{name: [value] * len(pairs) for name, value in extra.items()},
        }
    )


P_LIKE_PAIRS = list(
    itertools.product([f"PA{i}" for i in range(10)], [f"PN{i}" for i in range(10)])
)
ODD_PAIRS = list(
    itertools.product([f"OA{i}" for i in range(4)], [f"ON{i}" for i in range(4)])
)
# 60 positives, all P-like. 0.25 of them is 15 spies and leaves 45 labelled positive.
POSITIVES = pair_table(
    P_LIKE_PAIRS[:60],
    family="all",
    n_reports=1,
    std_version=STD_VERSION,
    licence="open",
)
# U: 10 P-like pairs nobody reported (indistinguishable from P) and 10 ODD pairs. The odd ones are
# the only rows that look different, so they are what a working step must find.
PLANTED = ODD_PAIRS[:10]
UNLABELED = pair_table(
    P_LIKE_PAIRS[60:70] + PLANTED,
    n_draws=1,
    std_version=STD_VERSION,
    licence="open",
)
PLANTED_KEYS = {(a, b) for a, b in PLANTED}


def run(**changes: object) -> tuple[pd.DataFrame, dict]:
    """``reliable_negatives`` on the fixture with some config fields replaced."""
    return reliable_negatives(
        POSITIVES, UNLABELED, SUBSTRATES, replace(CONFIG, **changes)
    )


def keys_of(table: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(table["key_a_stereo"], table["key_b_stereo"], strict=True))


# --- spy_threshold: the tie rule, computable by hand -------------------------------------------


@pytest.mark.parametrize(
    ("n_spies", "expected_threshold"), [(20, 2.0), (40, 3.0), (100, 6.0)]
)
def test_threshold_is_the_kth_largest_spy_score(
    n_spies: int, expected_threshold: float
) -> None:
    """Scores are 1, 2, ..., n in a shuffled order, all distinct.

    n = 20, recall 0.95: k = ceil(0.95 * 20) = 19. The 19th largest of 1..20 is 20 - 19 + 1 = 2, so
    the threshold is 2.0, the spy scoring 1.0 falls below it, and 19 of 20 are recalled.
    n = 40: k = 38, threshold 3.0. n = 100: k = 95, threshold 6.0.
    0.95 * n is exactly an integer in all three, so a rule that computes ceil on the raw float
    product and lets 19.000000000000004 become 20 returns the minimum instead and fails here.
    """
    scores = np.arange(1, n_spies + 1, dtype=np.float64)
    shuffled = np.random.default_rng(0).permutation(scores)
    assert spy_threshold(shuffled, 0.95) == expected_threshold


def test_threshold_does_not_reorder_its_input() -> None:
    """The caller's array is its own; sorting it in place would reorder the table's rows."""
    scores = np.array([0.9, 0.1, 0.5, 0.3])
    spy_threshold(scores, 0.5)
    assert scores.tolist() == [0.9, 0.1, 0.5, 0.3]


def test_exact_tie_at_the_threshold_recalls_every_tied_spy() -> None:
    """Scores 0.1, 0.2, 0.2, 0.9 and recall 0.5: k = ceil(0.5 * 4) = 2, the 2nd largest is 0.2.

    Two spies score exactly 0.2. The rule is 'recalled when score >= threshold', so BOTH are
    recalled and three of four are at or above the threshold: the realised recall is 0.75, above
    the configured 0.5. A tie is never split; the threshold does not move to break it. A U row that
    also scores exactly 0.2 is therefore not a reliable negative (see the table consistency test).
    """
    scores = np.array([0.1, 0.2, 0.2, 0.9])
    threshold = spy_threshold(scores, 0.5)
    assert threshold == 0.2
    assert (scores >= threshold).sum() == 3


def test_all_spies_tied_gives_that_score() -> None:
    """Five spies that all score 0.4: k = 5, the 5th largest is 0.4, all five are recalled."""
    assert spy_threshold(np.full(5, 0.4), 0.95) == 0.4


def test_few_spies_threshold_is_the_lowest() -> None:
    """Three spies, recall 0.95: ceil(2.85) = 3, so the threshold is the lowest score.

    With so few spies 95 % cannot be reached without recalling all of them; the rule rounds the
    count of recalled spies UP, never down, so the realised recall is 1.0.
    """
    assert spy_threshold(np.array([0.7, 0.2, 0.5]), 0.95) == 0.2


def test_one_spy_threshold_is_its_score() -> None:
    assert spy_threshold(np.array([0.37]), 0.95) == 0.37


def test_realised_recall_can_equal_the_configured_recall() -> None:
    """With distinct scores and n * recall a whole number, exactly k spies are at or above the
    k-th largest, so the realised recall is k / n = recall: 19/20, 38/40, 95/100 are all 0.95."""
    for n_spies in (20, 40, 100):
        scores = np.arange(1, n_spies + 1, dtype=np.float64)
        threshold = spy_threshold(scores, 0.95)
        assert (scores >= threshold).mean() == pytest.approx(0.95)


def test_realised_recall_is_above_configured_when_n_times_recall_is_not_whole() -> None:
    """Ten distinct spies, recall 0.95: k = ceil(9.5) = 10, so all ten are recalled (1.0, not
    0.95). The report carries both numbers because this is allowed to happen."""
    scores = np.arange(1, 11, dtype=np.float64)
    threshold = spy_threshold(scores, 0.95)
    assert (scores >= threshold).mean() == 1.0


@pytest.mark.parametrize("recall", [0.0, 1.0, -0.1, 1.5])
def test_threshold_refuses_recall_outside_zero_one(recall: float) -> None:
    with pytest.raises(ValueError, match="spy_recall"):
        spy_threshold(np.array([0.1, 0.2]), recall)


def test_threshold_refuses_no_spies() -> None:
    """No spies means no threshold; guessing one would put 'reliable' on arbitrary rows."""
    with pytest.raises(ValueError, match="spies"):
        spy_threshold(np.array([]), 0.95)


# --- the planted case: the step finds what it was built to find --------------------------------


@pytest.mark.parametrize("block", ["morgan_centre", "morgan", "morgan_near"])
def test_planted_odd_pairs_are_exactly_the_reliable_negatives(block: str) -> None:
    """Every P-like pair (positive, spy or unreported U row) has one identical feature vector, so
    in-sample they all get one score s1; the ten odd pairs share no bit with P and get s0 < s1.
    The threshold is s1 (all spies tie at s1), 'below' is strict, so the reliable negatives are the
    ten odd pairs and nothing else. Set equality, not a count.

    Proves the mechanics on a case built to be separable. It does not show the reliable negatives
    are trustworthy on real data (module docstring, ADR 026).
    """
    table, _ = run(spy_scoring="in_sample", feature_block=block)
    reliable = table[table["reliable_negative"]]
    assert keys_of(reliable) == PLANTED_KEYS


def test_planted_odd_pairs_are_reliable_out_of_fold_too() -> None:
    """Out of fold the P-like score moves a little from fold to fold, so which unreported P-like
    rows fall under the threshold is not fixed and equality would be an accident of the seed. What
    is fixed: an odd pair is labelled unlabeled in every training fold and shares no bit with a
    positive, so it scores far below any spy. The planted rows are a subset of the reliable ones."""
    table, _ = run(spy_scoring="out_of_fold")
    reliable = table[table["reliable_negative"]]
    assert PLANTED_KEYS <= keys_of(reliable)


def test_table_is_consistent_with_its_own_threshold() -> None:
    """From the table alone: threshold = spy_threshold over the spy rows' stored scores, a U row is
    reliable exactly when its stored score is strictly below it, and no spy is ever flagged. So a
    U row that ties the threshold is NOT reliable, and the stored float32 scores are the numbers
    the threshold was cut from (a float64 threshold between two float32 scores would break this)."""
    table, _ = run(spy_scoring="in_sample")
    spies = table[table["in_p"]]
    unlabeled = table[~table["in_p"]]
    threshold = spy_threshold(spies["spy_score"].to_numpy(), CONFIG.spy_recall)
    assert (table["threshold"] == threshold).all()
    assert (
        unlabeled["reliable_negative"] == (unlabeled["spy_score"] < threshold)
    ).all()
    assert not spies["reliable_negative"].any()


# --- features: the fingerprint and nothing else ------------------------------------------------


@pytest.mark.parametrize("block", ["descriptors", "descriptors + size"])
def test_reliable_negatives_refuses_a_descriptor_block_by_name(block: str) -> None:
    """A descriptor block is not a fingerprint, so the PU model may not see it."""
    with pytest.raises(ValueError, match=re.escape(block)):
        run(feature_block=block)


def test_exactly_three_feature_blocks_are_legal() -> None:
    """Three fingerprint blocks, no more: the whole molecule, the reacting atom alone (distance 0)
    and the atoms within the class's distance of it (ADR 030). Named here so that adding a fourth
    is a decision that changes this line, not an absence of failure."""
    assert set(FINGERPRINT_BLOCKS) == {"morgan", "morgan_centre", "morgan_near"}
    assert len(FINGERPRINT_BLOCKS) == 3


@pytest.mark.parametrize("block", ["morgan", "morgan_centre", "morgan_near"])
def test_pu_config_accepts_each_of_the_three_fingerprint_blocks(block: str) -> None:
    table, _ = run(feature_block=block)
    assert set(table["feature_block"]) == {block}


def test_a_fingerprint_block_missing_from_the_feature_table_names_the_column() -> None:
    """morgan_near is a legal name; a table built before it existed lacks the column. That is a
    different mistake from a misspelt name, and the message says which one it is."""
    old_table = SUBSTRATES.drop(columns=["morgan_near"])
    with pytest.raises(ValueError, match=r"lacks column\(s\) \['morgan_near'\]"):
        reliable_negatives(
            POSITIVES,
            UNLABELED,
            old_table,
            replace(CONFIG, feature_block="morgan_near"),
        )


# --- the spy draw and the labels: a spy stays in U ---------------------------------------------


def test_spies_are_a_fraction_of_p_and_are_labelled_unlabeled() -> None:
    """60 positives, spy_fraction 0.25: 15 spies. Each is a known positive (in_p = True) that is
    labelled 0 in training, like a U row. The other 45 positives are labelled 1, and the 20 U rows
    are in_p = False and labelled 0. The whole method rests on the spy looking unlabeled."""
    labels = spy_labels(POSITIVES, UNLABELED, CONFIG)
    spies = labels[labels["in_p"] & (labels["label"] == 0)]
    assert len(spies) == 15
    assert keys_of(spies) <= keys_of(POSITIVES)
    assert (labels["in_p"] & (labels["label"] == 1)).sum() == 45
    unlabeled = labels[~labels["in_p"]]
    assert len(unlabeled) == 20
    assert (unlabeled["label"] == 0).all()


def test_three_positives_at_a_quarter_round_down_to_no_spies() -> None:
    """floor(0.25 * 3) = floor(0.75) = 0 spies. 0.25 is inside (0, 1), so the config check passes;
    spy_labels must refuse before any fit, naming spy_fraction and the 3 positives."""
    with pytest.raises(ValueError, match=r"spy_fraction 0\.25 of 3 positives"):
        spy_labels(POSITIVES.head(3), UNLABELED, CONFIG)


def test_spy_rows_are_in_the_output_marked_in_p() -> None:
    """The output has one row per U pair and per spy: 20 + 15, and the 15 in_p rows are P pairs."""
    table, _ = run()
    assert len(table) == 20 + 15
    spies = table[table["in_p"]]
    assert len(spies) == 15
    assert keys_of(spies) <= keys_of(POSITIVES)
    assert keys_of(table[~table["in_p"]]) == keys_of(UNLABELED)


def test_no_spy_is_called_a_reliable_negative() -> None:
    """A spy is a known positive; flagging it would put a reported reaction in the negatives."""
    table, _ = run()
    assert not table.loc[table["in_p"], "reliable_negative"].any()


def test_a_different_seed_draws_different_spies() -> None:
    """The seed reaches the spy draw: 15 of 60 at random coincide with probability far below 1e-9."""
    first = spy_labels(POSITIVES, UNLABELED, CONFIG)
    second = spy_labels(POSITIVES, UNLABELED, replace(CONFIG, seed=8))
    assert keys_of(first[first["in_p"] & (first["label"] == 0)]) != keys_of(
        second[second["in_p"] & (second["label"] == 0)]
    )


def test_a_different_seed_changes_the_folds() -> None:
    """The seed reaches the folds. (It cannot be observed reaching the learner: without row or
    feature subsampling and with early stopping off, HistGradientBoosting is deterministic whatever
    its random_state, so the learner's seed is passed on trust and read in review.)"""
    first, _ = run(seed=7)
    second, _ = run(seed=8)
    assert not first["fold"].equals(second["fold"])


# --- determinism and the table's columns -------------------------------------------------------


@pytest.mark.parametrize("scoring", ["out_of_fold", "in_sample"])
def test_two_runs_with_the_same_config_are_identical(scoring: str) -> None:
    first, first_report = run(spy_scoring=scoring)
    second, second_report = run(spy_scoring=scoring)
    pd.testing.assert_frame_equal(first, second)
    assert first_report == second_report


def test_columns_and_dtypes_follow_the_design() -> None:
    table, _ = run()
    assert list(table.columns) == [
        *PAIR_KEYS,
        "in_p",
        "weight",
        "fold",
        "spy_score",
        "reliable_negative",
        "threshold",
        "model",
        "feature_block",
        "seed",
        "config_sha256",
        "std_version",
        "licence",
    ]
    assert table["spy_score"].dtype == np.float32
    assert table["in_p"].dtype == bool
    assert table["weight"].dtype == np.float64
    assert table["reliable_negative"].dtype == bool


def test_out_of_fold_rows_all_carry_a_fold_in_range() -> None:
    table, _ = run(spy_scoring="out_of_fold")
    assert set(table["fold"]) <= set(range(CONFIG.n_folds))


def test_provenance_columns_are_copied_and_the_licence_is_open() -> None:
    """The fixture's P and U are open, so the output is open (and may be written outside private/)."""
    table, _ = run()
    assert set(table["licence"]) == {"open"}
    assert set(table["std_version"]) == {STD_VERSION}
    assert set(table["model"]) == {CONFIG.model}
    assert set(table["feature_block"]) == {CONFIG.feature_block}
    assert set(table["seed"]) == {CONFIG.seed}


def test_config_hash_follows_the_config() -> None:
    first, _ = run(seed=7)
    again, _ = run(seed=7)
    other, _ = run(seed=8)
    digest = first["config_sha256"].iloc[0]
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert (first["config_sha256"] == digest).all()
    assert again["config_sha256"].iloc[0] == digest
    assert other["config_sha256"].iloc[0] != digest


# --- the report: the audit record --------------------------------------------------------------


def test_report_carries_the_audit_numbers() -> None:
    """Keys present and sane, not their values. The last group is the one that matters most: U was
    built to match P's marginals, the threshold breaks that match on purpose, and how far it breaks
    it is what a later reader will want to know."""
    table, report = run()
    assert report["n_spies"] == 15
    assert report["spy_recall_configured"] == 0.95
    assert 0.95 <= report["spy_recall_realised"] <= 1.0
    for key in ("threshold_out_of_fold", "threshold_in_sample"):
        assert 0.0 <= report[key] <= 1.0
    for key in ("n_reliable_negatives_out_of_fold", "n_reliable_negatives_in_sample"):
        assert 0 <= report[key] <= 20
    assert report["n_reliable_negatives"] == int(table["reliable_negative"].sum())
    assert report["reliable_negative_share"] == pytest.approx(
        report["n_reliable_negatives"] / 20
    )
    for key in ("spy_score_deciles", "unlabeled_score_deciles"):
        assert len(report[key]) == 9
        assert report[key] == sorted(report[key])
    shift = report["selection_shift"]
    for key in (
        "marginal_tv_a",
        "marginal_tv_b",
        "mean_heavy_atoms_reliable",
        "mean_heavy_atoms_unlabeled",
        "single_partner_share_reliable",
        "single_partner_share_unlabeled",
    ):
        assert key in shift
    assert 0.0 <= shift["marginal_tv_a"] <= 1.0
    assert 0.0 <= shift["marginal_tv_b"] <= 1.0
    assert 0.0 <= shift["single_partner_share_reliable"] <= 1.0
    assert 0.0 <= shift["single_partner_share_unlabeled"] <= 1.0


def test_selection_shift_values_on_the_planted_case() -> None:
    """Hand-computed on the fixture, in-sample, where the reliable negatives are the 10 odd pairs
    OA0 x ON0..3, OA1 x ON0..3, OA2 x ON0..1 (the planted test shows this) and U is 10 pairs
    PA6 x PN0..9 plus those same 10.

    Heavy atoms: odd pair 10 + 5 = 15; P-like pair 20 + 10 = 30; U (10 * 30 + 10 * 15) / 20 = 22.5.
    Acid side: reliable OA0 .4, OA1 .4, OA2 .2; U PA6 .5, OA0 .2, OA1 .2, OA2 .1;
    TV = 0.5 * (.5 + .2 + .2 + .1) = 0.5. Amine side: reliable ON0 .3, ON1 .3, ON2 .2, ON3 .2;
    U PN0..9 .05 each (.5 together), ON0 .15, ON1 .15, ON2 .1, ON3 .1; TV = 0.5 * (.5 + .5) = 0.5.
    Single partner: every reliable substrate has 2 to 4 pairs, so 0 of 7; in U the ten PN are in one
    pair each, of 18 distinct substrates (PA6, OA0-2, PN0-9, ON0-3), so 10 / 18.
    """
    _, report = run(spy_scoring="in_sample")
    shift = report["selection_shift"]
    assert shift == pytest.approx(
        {
            "marginal_tv_a": 0.5,
            "marginal_tv_b": 0.5,
            "mean_heavy_atoms_reliable": 15.0,
            "mean_heavy_atoms_unlabeled": 22.5,
            "single_partner_share_reliable": 0.0,
            "single_partner_share_unlabeled": 10 / 18,
        }
    )


SHIFT_SUBSTRATES = pd.DataFrame(
    {
        "key_stereo": ["A1", "A2", "B1", "B2"],
        "n_heavy_atoms": [10, 20, 5, 15],
    }
)
SHIFT_UNLABELED = pair_table([("A1", "B1"), ("A1", "B2"), ("A2", "B1"), ("A2", "B2")])


def test_selection_shift_small_hand_computed_values() -> None:
    """Reliable = (A1,B1), (A1,B2); U = all four pairs of A1, A2 x B1, B2.

    TV acids: reliable A1 1.0; U A1 .5, A2 .5; 0.5 * (.5 + .5) = 0.5. TV amines: .5/.5 on both,
    so 0. Mean pair size: reliable (10+5, 10+15) -> 20; U (15, 25, 25, 35) -> 25.
    Single partner: reliable A1 in 2 pairs, B1 and B2 in 1 each -> 2/3; U every substrate in 2 -> 0.
    """
    reliable = SHIFT_UNLABELED.head(2)
    assert selection_shift(
        reliable, SHIFT_UNLABELED, SHIFT_SUBSTRATES
    ) == pytest.approx(
        {
            "marginal_tv_a": 0.5,
            "marginal_tv_b": 0.0,
            "mean_heavy_atoms_reliable": 20.0,
            "mean_heavy_atoms_unlabeled": 25.0,
            "single_partner_share_reliable": 2 / 3,
            "single_partner_share_unlabeled": 0.0,
        }
    )


def test_selection_shift_of_no_reliable_negatives_is_none_not_nan() -> None:
    """An empty reliable set describes nothing: its three values are None, which json.dumps writes
    as null (a NaN would be invalid JSON). U's own values are still computed."""
    shift = selection_shift(SHIFT_UNLABELED.head(0), SHIFT_UNLABELED, SHIFT_SUBSTRATES)
    assert shift["marginal_tv_a"] is None
    assert shift["marginal_tv_b"] is None
    assert shift["mean_heavy_atoms_reliable"] is None
    assert shift["single_partner_share_reliable"] is None
    assert shift["mean_heavy_atoms_unlabeled"] == 25.0
    assert shift["single_partner_share_unlabeled"] == 0.0
    assert json.loads(json.dumps(shift))["marginal_tv_a"] is None


def test_report_realised_recall_is_counted_from_the_table() -> None:
    """The realised recall is the share of spy rows at or above the threshold, as the rule says."""
    table, report = run()
    spies = table[table["in_p"]]
    expected = float((spies["spy_score"] >= spies["threshold"]).mean())
    assert report["spy_recall_realised"] == pytest.approx(expected)


@pytest.mark.parametrize("scoring", ["out_of_fold", "in_sample"])
def test_table_threshold_is_the_configured_scorings_threshold(scoring: str) -> None:
    """The report holds both thresholds whichever scoring is configured (ADR 016: the cost of
    PAYN's in-sample choice is read as a number); the table holds the configured one."""
    table, report = run(spy_scoring=scoring)
    assert table["threshold"].iloc[0] == pytest.approx(report[f"threshold_{scoring}"])


def test_report_is_plain_json() -> None:
    """The report is written to a .report.json; a numpy scalar in it would fail there, late."""
    _, report = run()
    json.dumps(report)


# --- refusals ----------------------------------------------------------------------------------


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.5])
def test_spy_fraction_outside_the_open_interval_is_refused(fraction: float) -> None:
    """0 gives no spies, 1 leaves no positive to learn from; both would be degenerate, not wrong."""
    with pytest.raises(ValueError, match="spy_fraction"):
        run(spy_fraction=fraction)


@pytest.mark.parametrize("recall", [0, 1, -0.1, 1.5])
def test_spy_recall_outside_the_open_interval_is_refused(recall: float) -> None:
    with pytest.raises(ValueError, match="spy_recall"):
        run(spy_recall=recall)


@pytest.mark.parametrize("folds", [1, 0, -2])
def test_fewer_than_two_folds_is_refused(folds: int) -> None:
    with pytest.raises(ValueError, match="n_folds"):
        run(n_folds=folds)


@pytest.mark.parametrize("scoring", ["oof", "out-of-fold", "", "both"])
def test_spy_scoring_other_than_the_two_is_refused_by_name(scoring: str) -> None:
    with pytest.raises(ValueError, match="spy_scoring"):
        run(spy_scoring=scoring)


def test_both_spy_scorings_are_accepted() -> None:
    for scoring in ("out_of_fold", "in_sample"):
        table, _ = run(spy_scoring=scoring)
        assert len(table) == 35


def test_learner_is_a_recorded_value_not_a_branch() -> None:
    with pytest.raises(ValueError, match="learner"):
        run(learner="CatBoostClassifier")


@pytest.mark.parametrize(
    "params", [{"max_iter": 30}, {"max_iter": 30, "early_stopping": True}]
)
def test_early_stopping_must_be_written_out_as_false(params: dict) -> None:
    """ADR 016 item 3: above 10,000 rows scikit-learn turns it on by itself and carves a validation
    split with its own seed. A config that leaves it out, or turns it on, is refused."""
    with pytest.raises(ValueError, match="early_stopping"):
        run(learner_params=params)


def test_learner_params_without_max_iter_is_refused() -> None:
    """The design fixes an explicit max_iter, so the number of boosting rounds is in the config."""
    with pytest.raises(ValueError, match="max_iter"):
        run(learner_params={"early_stopping": False})


# --- the config file ---------------------------------------------------------------------------

CONFIG_YAML = """\
model: pair_hgb
feature_block: morgan_centre
p_width: all
unlabeled: unlabeled_literature_k10
spy_fraction: 0.25
spy_recall: 0.95
spy_scoring: out_of_fold
n_folds: 5
learner: HistGradientBoostingClassifier
learner_params:
  max_iter: 30
  early_stopping: false
  min_samples_leaf: 1
seed: 7
"""


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "pu_fixture.yml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_pu_config_reads_every_key(tmp_path: Path) -> None:
    assert load_pu_config(write_config(tmp_path, CONFIG_YAML)) == CONFIG


@pytest.mark.parametrize(
    "missing", ["spy_fraction", "n_folds", "learner_params", "seed"]
)
def test_load_pu_config_names_the_file_for_a_missing_key(
    tmp_path: Path, missing: str
) -> None:
    lines = [line for line in CONFIG_YAML.splitlines() if not line.startswith(missing)]
    if missing == "learner_params":
        # Its indented children would be orphaned and the file would not be YAML at all.
        lines = [line for line in lines if not line.startswith("  ")]
    path = write_config(tmp_path, "\n".join(lines))
    with pytest.raises(ValueError, match=re.escape(str(path))):
        load_pu_config(path)


def test_load_pu_config_refuses_an_unknown_key(tmp_path: Path) -> None:
    """A typo beside the correct key used to be dropped, and the run differed from the file."""
    path = write_config(tmp_path, CONFIG_YAML + "spy_recal: 0.90\n")
    with pytest.raises(ValueError, match=re.escape(str(path))) as error:
        load_pu_config(path)
    assert "spy_recal" in str(error.value)


@pytest.mark.parametrize(
    ("good_line", "bad_line"),
    [
        ("model: pair_hgb", "model:"),
        ("n_folds: 5", "n_folds: 7.9"),
        ("seed: 7", "seed: true"),
    ],
)
def test_load_pu_config_refuses_a_blank_or_wrongly_typed_scalar(
    tmp_path: Path, good_line: str, bad_line: str
) -> None:
    """model: (YAML null) used to become the string "None" on every score row; n_folds: 7.9 used to
    run 7 folds; seed: true used to be the seed 1 because bool is an int."""
    path = write_config(tmp_path, CONFIG_YAML.replace(good_line, bad_line))
    with pytest.raises(ValueError, match=re.escape(str(path))):
        load_pu_config(path)


@pytest.mark.parametrize(
    "bad_line",
    ["n_folds: five", "spy_fraction: lots", "learner_params: 3", "seed: [1, 2]"],
)
def test_load_pu_config_names_the_file_for_a_wrongly_typed_key(
    tmp_path: Path, bad_line: str
) -> None:
    key = bad_line.split(":")[0]
    lines = [
        bad_line if line.startswith(key) else line for line in CONFIG_YAML.splitlines()
    ]
    # Drop the nested learner_params lines when the key itself is replaced by a scalar.
    if key == "learner_params":
        lines = [line for line in lines if not line.startswith("  ")]
    path = write_config(tmp_path, "\n".join(lines))
    with pytest.raises(ValueError, match=re.escape(str(path))):
        load_pu_config(path)


def test_load_pu_config_reads_morgan_near_from_a_file(tmp_path: Path) -> None:
    text = CONFIG_YAML.replace(
        "feature_block: morgan_centre", "feature_block: morgan_near"
    )
    config = load_pu_config(write_config(tmp_path, text))
    assert config == replace(CONFIG, feature_block="morgan_near")
    table, _ = reliable_negatives(POSITIVES, UNLABELED, SUBSTRATES, config)
    assert set(table["feature_block"]) == {"morgan_near"}


def test_the_repository_s2a_config_for_morgan_near_loads() -> None:
    config = load_pu_config(
        Path(__file__).parents[1] / "configs" / "models" / "pu_spy_morgan_near.yml"
    )
    assert config.feature_block == "morgan_near"
    assert config.model == "pair_hgb"


def test_load_pu_config_refuses_a_descriptor_block_in_the_file(tmp_path: Path) -> None:
    text = CONFIG_YAML.replace(
        "feature_block: morgan_centre", "feature_block: descriptors"
    )
    with pytest.raises(ValueError, match="descriptors"):
        reliable_negatives(
            POSITIVES,
            UNLABELED,
            SUBSTRATES,
            load_pu_config(write_config(tmp_path, text)),
        )


# --- n_draws is the sampling weight (2026-11-02) -----------------------------------------------
#
# Stage U writes one row per distinct pair and the number of times it was drawn in ``n_draws``
# (102,200 pairs carrying 110,585 draws). The draw is the unit everywhere else: the stage U gate
# expands by ``n_draws``, and the measurement ADR 026 rests on fitted with sample_weight = n_draws
# for U and 1 for P. So the fit must weigh a U row by ``n_draws`` and a P row (a spy included, which
# is a reported positive and not a draw) by 1.


def unlabeled_with_draws(n_draws: list[int]) -> pd.DataFrame:
    """The fixture's 20 U pairs with their own ``n_draws``, in the same order."""
    return UNLABELED.assign(n_draws=n_draws)


def run_with(unlabeled: pd.DataFrame, **changes: object) -> pd.DataFrame:
    table, _ = reliable_negatives(
        POSITIVES, unlabeled, SUBSTRATES, replace(CONFIG, **changes)
    )
    return table


def test_one_heavily_drawn_p_like_pair_lowers_every_p_like_in_sample_score() -> None:
    """All 70 P-like rows share one feature vector: 45 labelled 1, 15 spies and 10 unreported U
    rows labelled 0. In sample the score is the weighted share of positives among them.
    Weights all 1: 45 / 70 = 0.643 (the learner returns 0.639). Give ONE unreported P-like U row
    n_draws = 50: 45 / (70 + 49) = 0.378 (the learner returns 0.377). The threshold is the minimum
    spy score, and every spy is P-like, so it moves from about 0.64 to about 0.38. A fit that
    ignores n_draws leaves it at 0.64 whatever n_draws is.

    The learner does not land exactly on the weighted share at max_iter 30, so the figures are
    pinned to within 0.02 and the ordering (0.38 is below 0.5, 0.64 above) is the real claim.
    """
    draws = [50] + [
        1
    ] * 19  # row 0 is a P-like U pair (PA6 x PN0); rows 10-19 are the odd pairs
    table = run_with(unlabeled_with_draws(draws), spy_scoring="in_sample")
    p_like_scores = table.loc[table["in_p"], "spy_score"]
    assert p_like_scores.to_numpy() == pytest.approx(0.378, abs=0.02)
    assert table["threshold"].iloc[0] == pytest.approx(0.378, abs=0.02)


def test_unit_draws_leave_the_p_like_score_at_the_unweighted_share() -> None:
    """The control for the test above: with every n_draws 1 the score is 45 / 70 = 0.643 (0.639
    from the learner), so the 0.378 above comes from the weight and from nothing else."""
    table = run_with(unlabeled_with_draws([1] * 20), spy_scoring="in_sample")
    assert table["threshold"].iloc[0] == pytest.approx(0.643, abs=0.02)


def test_weight_column_is_one_for_p_and_n_draws_for_u() -> None:
    """A spy is a reported positive, not a draw, so it weighs 1.0 whatever U's n_draws are."""
    draws = list(range(1, 21))
    table = run_with(unlabeled_with_draws(draws))
    assert table["weight"].dtype == np.float64
    assert (table.loc[table["in_p"], "weight"] == 1.0).all()
    unlabeled_rows = table[~table["in_p"]]
    assert unlabeled_rows["weight"].tolist() == [float(n) for n in draws]
    assert keys_of(unlabeled_rows) == keys_of(UNLABELED)


def test_spy_labels_carries_the_weight() -> None:
    labels = spy_labels(POSITIVES, unlabeled_with_draws([3] * 20), CONFIG)
    assert (labels.loc[labels["in_p"], "weight"] == 1.0).all()
    assert (labels.loc[~labels["in_p"], "weight"] == 3.0).all()


def test_unlabeled_without_n_draws_is_refused_by_name() -> None:
    """The weight is required input; defaulting it to 1 is the defect being fixed."""
    with pytest.raises(ValueError, match="n_draws"):
        run_with(UNLABELED.drop(columns="n_draws"))


@pytest.mark.parametrize("bad_draws", [0, -1])
def test_non_positive_n_draws_is_refused(bad_draws: int) -> None:
    """A zero-weight row would be in the table but not in the fit; no design describes that."""
    draws = [bad_draws] + [1] * 19
    with pytest.raises(ValueError, match="n_draws"):
        run_with(unlabeled_with_draws(draws))


def test_fit_predict_weights_two_identical_rows_with_opposite_labels() -> None:
    """Two identical feature rows, one labelled 1 and one labelled 0. Weights (1, 9): the weighted
    share of positives is 1 / 10, so the probability is 0.1, not the unweighted 0.5. Swapped
    weights (9, 1) give 0.9. Weights are the fourth positional argument, the learner the fifth."""
    from platepu.score.pu_fit import Learner, fit_predict

    learner = Learner(params=CONFIG.learner_params, seed=CONFIG.seed)
    features = np.zeros((2, 1))
    target = np.array([1, 0])
    scored = features[:1]
    light_positive = fit_predict(
        features, target, scored, np.array([1.0, 9.0]), learner
    )
    heavy_positive = fit_predict(
        features, target, scored, np.array([9.0, 1.0]), learner
    )
    assert light_positive[0] == pytest.approx(0.1, abs=0.01)
    assert heavy_positive[0] == pytest.approx(0.9, abs=0.01)


# --- no benchmark label is read ----------------------------------------------------------------


def test_reliable_negatives_takes_no_label_argument() -> None:
    """The design: S2 reads no label and computes no metric against one. A function that is never
    handed a label cannot read it; this pins the signature."""
    parameters = list(inspect.signature(reliable_negatives).parameters)
    assert parameters == ["positives", "unlabeled", "substrates", "config"]


def test_module_imports_no_benchmark_and_reads_no_table() -> None:
    """Only a signature can be asserted for the argument; for the module, its imports and calls.
    It imports nothing from a benchmark module and calls no table reader (chemistry functions
    never open files, CLAUDE.md section 5; reading is the script's job). This cannot prove no label
    is read through some other path, only that none is imported or opened here."""
    source = inspect.getsource(pu_module)
    tree = ast.parse(source)
    imported = [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ] + [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not [name for name in imported if "benchmark" in name]
    called = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not called & {"read_parquet", "read_csv", "read_table", "open"}


def test_config_has_exactly_the_designed_keys() -> None:
    assert set(PuConfig.__dataclass_fields__) == {
        "model",
        "feature_block",
        "p_width",
        "unlabeled",
        "spy_fraction",
        "spy_recall",
        "spy_scoring",
        "n_folds",
        "learner",
        "learner_params",
        "seed",
    }
