"""Tests for stage U, the literature-built unlabeled space (docs/design/modeling.md, ADRs 013-016).

U is drawn from the substrates of the literature positives P, never from the inventory. Each
counter-pair replaces ONE side of a positive pair, from the same heavy-atom bin, weighted by P
frequency, so U keeps P's per-role popularity (ADR 013) and P's size anti-correlation (ADR 015).

The subtraction set is the **reported** table (every pair the class reported, labelled positive,
negative or unlabeled; ``build_reported``), not P alone: the yield cut-off moved 16,762 reported
pairs out of P (1,121 known negatives, 15,641 with no usable yield, counted 2026-10-28) and U must
still not draw them. The draw pools and frequency weights stay P's.

Cases are hand-written (counted 2026-10-23 in the design, written 2026-10-28), all open, no Chemlex
and no inventory row. Keys are made-up short strings except where the case is an InChIKey pattern
(L-/D-alanine as in ``test_benchmark_split.py``; benzoic acid and its 13C and d5 isotopologues,
two of the five leaks found on 2026-10-23). The flat key of a made-up key is the part before
the first "-", as the first InChIKey block is.

Section 10 (written 2026-10-29, before the code) pins the second sampling rule,
``size_stratified_both_sides``: the same U with no anchor, so that "does the model read the anchor?"
is an ablation of one thing. The one-side rule is pinned there to the draw, so the comparison
cannot drift.

The scripts' command line is pinned here too: ``unlabeled_literature.py CLASS ROOT --config YAML``.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import unlabeled_literature as script_module
from unlabeled_literature import main as unlabeled_script

from platepu.positives.unlabeled import (
    MATCH,
    UnlabeledConfig,
    load_unlabeled_config,
    unlabeled_pairs,
)

STD = "std1"
ONE_SIDE = "size_stratified_one_side"
BOTH_SIDES = "size_stratified_both_sides"
OUTPUT_COLUMNS = [
    "key_a_stereo",
    "key_b_stereo",
    "key_a_flat",
    "key_b_flat",
    "substrate_a",
    "substrate_b",
    "n_draws",
    "licence",
    "std_version",
]
L_ALANINE = "QNAYBMKLOCPYGJ-REOHCLBHSA-N"
D_ALANINE = "QNAYBMKLOCPYGJ-UWTATZPHSA-N"
BENZOIC_ACID = "WPYMKLBDIGXBTP-UHFFFAOYSA-N"
BENZOIC_ACID_13C = "WPYMKLBDIGXBTP-PTQBSOBMSA-N"
BENZOIC_ACID_D5 = "WPYMKLBDIGXBTP-RALIUCGRSA-N"


def make_config(**changes: object) -> UnlabeledConfig:
    """The design's configuration (k = 10) with ``changes`` applied.

    The two gate values are part of the configuration, not constants of the stage: they are the
    ones in ``configs/models/unlabeled_literature_k10.yml`` and stay overridable, because
    ``UnlabeledConfig`` has no defaults (a default would let a config omit the gate silently).
    """
    settings: dict[str, object] = {
        "k": 10,
        "seed": 7,
        "size_bins": 1,
        "p_width": "all",
        "subtract_p_on": "key_flat",
        "match": ONE_SIDE,
        "gate_rank_auc_tolerance": 0.02,
        "gate_marginal_tv_multiple": 2.0,
        "gate_null_draws": 200,
    }
    return UnlabeledConfig(**{**settings, **changes})  # type: ignore[arg-type]


def flat(key: str) -> str:
    """The first InChIKey block; for a made-up key, everything before the first "-"."""
    return key.split("-")[0]


def positives_table(
    pairs: list[tuple[str, str]], licence: str = "open", std_version: str = STD
) -> pd.DataFrame:
    """A positives table in the layout of ``build_positives``: one row per pair, family "all"."""
    return pd.DataFrame(
        {
            "class": "amide_coupling",
            "family": "all",
            "key_a_stereo": [a for a, _ in pairs],
            "key_b_stereo": [b for _, b in pairs],
            "key_a_flat": [flat(a) for a, _ in pairs],
            "key_b_flat": [flat(b) for _, b in pairs],
            "substrate_a": [f"smiles_{a}" for a, _ in pairs],
            "substrate_b": [f"smiles_{b}" for _, b in pairs],
            "licence": licence,
            "std_version": std_version,
        }
    )


def reported_of(positives: pd.DataFrame, *others: tuple[str, str, str]) -> pd.DataFrame:
    """The reported table matching ``positives`` in the layout of ``build_reported``.

    Every pair of ``positives`` is a ``positive`` row; each ``(acid, amine, label)`` of ``others``
    is a further reported pair with that label (``negative`` or ``unlabeled``), absent from P.
    """
    extra = pd.DataFrame(others, columns=["key_a_stereo", "key_b_stereo", "label"])
    pairs = positives[["key_a_stereo", "key_b_stereo"]].assign(label="positive")
    pairs = pd.concat([pairs, extra], ignore_index=True)
    return pairs.assign(
        key_a_flat=pairs["key_a_stereo"].map(flat),
        key_b_flat=pairs["key_b_stereo"].map(flat),
        substrate_a="smiles_" + pairs["key_a_stereo"],
        substrate_b="smiles_" + pairs["key_b_stereo"],
        n_reports=1,
        max_yield=np.where(pairs["label"] == "positive", 80.0, np.nan),
        licence="open",
        std_version=STD,
    )[
        [
            "key_a_stereo",
            "key_b_stereo",
            "key_a_flat",
            "key_b_flat",
            "substrate_a",
            "substrate_b",
            "n_reports",
            "max_yield",
            "label",
            "licence",
            "std_version",
        ]
    ]


def substrate_table(sizes: dict[str, int]) -> pd.DataFrame:
    """The feature table's columns stage U reads: key and heavy-atom count of each substrate."""
    return pd.DataFrame(
        {
            "key_stereo": list(sizes),
            "key_flat": [flat(key) for key in sizes],
            "smiles": [f"smiles_{key}" for key in sizes],
            "role": "acid",
            "n_heavy_atoms": list(sizes.values()),
            "licence": "open",
            "std_version": STD,
        }
    )


def pair_set(table: pd.DataFrame) -> set[tuple[str, str]]:
    """The (acid key, amine key) pairs of a table, stereo keys."""
    return set(zip(table["key_a_stereo"], table["key_b_stereo"], strict=True))


def p_frequency(positives: pd.DataFrame, column: str) -> pd.Series:
    """Number of positive pairs each substrate is in, as a fraction of all pairs."""
    return positives[column].value_counts() / len(positives)


def u_frequency(unlabeled: pd.DataFrame, column: str) -> pd.Series:
    """Share of U's draws that carry each substrate (``n_draws`` is the weight)."""
    weight = unlabeled.groupby(column)["n_draws"].sum()
    return weight / weight.sum()


def auc(score: np.ndarray, is_positive: np.ndarray) -> float:
    """ROC-AUC by ranks (ties get their average rank); 0.5 means no separation."""
    ranks = pd.Series(score).rank().to_numpy()
    n_pos = int(is_positive.sum())
    n_neg = len(score) - n_pos
    return float((ranks[is_positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# Lopsided popularity (all one size, so the size rule is out of the way): acid HOT_A is in 10 of
# 40 pairs, amine HOT_N in 8; the rest of the acids are reported once, most amines once or twice.
HOT_A, HOT_N = "hotacid", "hotamine"
LOPSIDED = (
    [(HOT_A, f"n{i}") for i in range(10)]
    + [(f"a{i}", HOT_N) for i in range(8)]
    + [(f"a{i}", f"m{i % 12}") for i in range(8, 30)]
)
LOPSIDED_P = positives_table(LOPSIDED)
LOPSIDED_SUBSTRATES = substrate_table(
    {key: 10 for key in {*(a for a, _ in LOPSIDED), *(b for _, b in LOPSIDED)}}
)


def uniform_unlabeled(positives: pd.DataFrame, n_pairs: int, seed: int) -> pd.DataFrame:
    """A U of the naive kind: both sides drawn uniformly from the distinct substrates of P, known
    positives dropped, ``n_pairs`` rows kept (a repeated pair is a repeated row, as in U)."""
    rng = np.random.default_rng(seed)
    drawn = pd.DataFrame(
        {
            "key_a_stereo": rng.choice(positives["key_a_stereo"].unique(), 3 * n_pairs),
            "key_b_stereo": rng.choice(positives["key_b_stereo"].unique(), 3 * n_pairs),
        }
    )
    known = pair_set(positives)
    is_new = [pair not in known for pair in pair_set_list(drawn)]
    return drawn[is_new].head(n_pairs).assign(n_draws=1)


def pair_set_list(table: pd.DataFrame) -> list[tuple[str, str]]:
    """The (acid key, amine key) pairs of a table in row order."""
    return list(zip(table["key_a_stereo"], table["key_b_stereo"], strict=True))


# ---------------------------------------------------------------------- 1. one side, same bin


def test_a_counter_pair_replaces_one_side_from_the_same_size_bin() -> None:
    """Two bins per role. P pairs each big acid with a small amine and the reverse. The only
    one-side replacements from the same bin are A3/A4 with N1/N2 and A1/A2 with N3/N4. A fresh
    pair on both sides, or a replacement from the other bin, would give a big-big or small-small
    pair. Without this the literature's size anti-correlation (-0.388) is lost."""
    positives = positives_table(
        [("A3", "N1"), ("A4", "N2"), ("A1", "N3"), ("A2", "N4")]
    )
    sizes = {"A1": 5, "A2": 6, "A3": 30, "A4": 31, "N1": 5, "N2": 6, "N3": 30, "N4": 31}
    unlabeled, _ = unlabeled_pairs(
        positives,
        reported_of(positives),
        substrate_table(sizes),
        make_config(size_bins=2, k=20),
    )
    assert pair_set(unlabeled) == {
        ("A3", "N2"),
        ("A4", "N1"),
        ("A1", "N4"),
        ("A2", "N3"),
    }


def test_a_replacement_never_comes_from_outside_p() -> None:
    """The feature table also holds inventory substrates (stage F merges them in). They are the
    same size as the P substrates here, so only 'draw from P's own substrates' keeps them out of
    U, which must be built from the open literature alone (ADR 008 item 4)."""
    positives = positives_table([("A1", "N1"), ("A2", "N2"), ("A3", "N3")])
    sizes = {
        "A1": 8,
        "A2": 8,
        "A3": 8,
        "N1": 8,
        "N2": 8,
        "N3": 8,
        "LIB_A": 8,
        "LIB_N": 8,
    }
    substrates = substrate_table(sizes).assign(
        licence=lambda t: np.where(
            t["key_stereo"].str.startswith("LIB"), "internal", "open"
        )
    )
    unlabeled, _ = unlabeled_pairs(
        positives, reported_of(positives), substrates, make_config(k=50)
    )
    keys = set(unlabeled["key_a_stereo"]) | set(unlabeled["key_b_stereo"])
    assert keys <= {"A1", "A2", "A3", "N1", "N2", "N3"}


# ---------------------------------------------------------------------- 2. marginals match P


def test_each_substrates_u_share_matches_its_p_share() -> None:
    """The hot acid is in 25 % of P's pairs, the hot amine in 20 %; a uniform sampler would give
    them 1/31 and 1/23. U keeps those shares to within 25 % relative, per role. Not tighter: a
    popular substrate is also in the most known pairs, so subtracting P takes slightly more of its
    draws (a reference sampler read 0.21 and 0.18 against 0.25 and 0.20 on this small, dense P)."""
    unlabeled, _ = unlabeled_pairs(
        LOPSIDED_P, reported_of(LOPSIDED_P), LOPSIDED_SUBSTRATES, make_config(k=100)
    )
    p_acid = p_frequency(LOPSIDED_P, "key_a_stereo")
    p_amine = p_frequency(LOPSIDED_P, "key_b_stereo")
    u_acid = u_frequency(unlabeled, "key_a_stereo")
    u_amine = u_frequency(unlabeled, "key_b_stereo")
    assert u_acid[HOT_A] == pytest.approx(p_acid[HOT_A], rel=0.25)
    assert u_amine[HOT_N] == pytest.approx(p_amine[HOT_N], rel=0.25)


def test_total_variation_between_p_and_u_marginals_is_small() -> None:
    """All substrates at once, both roles: half the summed absolute difference of the shares."""
    unlabeled, _ = unlabeled_pairs(
        LOPSIDED_P, reported_of(LOPSIDED_P), LOPSIDED_SUBSTRATES, make_config(k=100)
    )
    for column in ("key_a_stereo", "key_b_stereo"):
        p_share = p_frequency(LOPSIDED_P, column)
        u_share = u_frequency(unlabeled, column).reindex(p_share.index, fill_value=0.0)
        assert (p_share - u_share).abs().sum() / 2 < 0.10


def test_the_number_of_draws_is_k_times_p_and_rejects_are_not_replaced() -> None:
    """No top-up loop (design, 'No top-up loop'): kept draws never exceed k x |P|."""
    unlabeled, _ = unlabeled_pairs(
        LOPSIDED_P, reported_of(LOPSIDED_P), LOPSIDED_SUBSTRATES, make_config(k=10)
    )
    assert 0 < unlabeled["n_draws"].sum() <= 10 * len(LOPSIDED_P)


# ---------------------------------------------------------------------- 3. subtract P on flat


def test_l_and_d_alanine_pair_is_subtracted_from_u_on_the_flat_key() -> None:
    """P holds (L-ala, n1) and (D-ala, n2). Every one-side counter-pair, (D-ala, n1), (L-ala, n2),
    has the flat pair key of a positive though its stereo pair is new. Our fingerprints ignore
    chirality, so keeping them would teach the model that a published reaction is unlabeled.
    Subtracting on the stereo key would leave all four in U."""
    positives = positives_table([(L_ALANINE, "n1-X"), (D_ALANINE, "n2-X")])
    substrates = substrate_table({L_ALANINE: 6, D_ALANINE: 6, "n1-X": 6, "n2-X": 6})
    unlabeled, rejected = unlabeled_pairs(
        positives, reported_of(positives), substrates, make_config()
    )
    assert unlabeled.empty
    assert set(rejected["reason"]) == {"pair_reported_flat"}
    assert pair_set(rejected) >= {(D_ALANINE, "n1-X"), (L_ALANINE, "n2-X")}


@pytest.mark.parametrize("isotopologue", [BENZOIC_ACID_13C, BENZOIC_ACID_D5])
def test_isotopologue_of_a_positive_acid_is_subtracted_on_the_flat_key(
    isotopologue: str,
) -> None:
    """13C- and d5-benzoic acid have the connectivity block of benzoic acid and differ only in the
    second block, so they are the same input to the model. Two of the five leaks of 2026-10-23."""
    positives = positives_table([(BENZOIC_ACID, "n1-X"), (isotopologue, "n2-X")])
    substrates = substrate_table(
        {BENZOIC_ACID: 9, isotopologue: 9, "n1-X": 6, "n2-X": 6}
    )
    unlabeled, rejected = unlabeled_pairs(
        positives, reported_of(positives), substrates, make_config()
    )
    assert unlabeled.empty
    assert (BENZOIC_ACID, "n2-X") in pair_set(rejected)


def test_no_pair_in_u_shares_its_flat_key_with_a_positive() -> None:
    """The same rule as a property of a whole run on the lopsided P: zero overlap on flat keys."""
    unlabeled, _ = unlabeled_pairs(
        LOPSIDED_P, reported_of(LOPSIDED_P), LOPSIDED_SUBSTRATES, make_config()
    )
    flat_u = set(zip(unlabeled["key_a_flat"], unlabeled["key_b_flat"], strict=True))
    flat_p = set(zip(LOPSIDED_P["key_a_flat"], LOPSIDED_P["key_b_flat"], strict=True))
    assert not flat_u & flat_p


def test_rejected_pairs_are_the_known_positives_and_nothing_else() -> None:
    """The one reason is pair_reported_flat (the set is no longer P); a key is never under both roles."""
    _, rejected = unlabeled_pairs(
        LOPSIDED_P, reported_of(LOPSIDED_P), LOPSIDED_SUBSTRATES, make_config(k=50)
    )
    assert len(rejected) > 0
    assert set(rejected["reason"]) == {"pair_reported_flat"}


# ---------------------------------------------------------------------- 3b. subtract everything reported
#
# The yield cut-off moved pairs out of P into ``reported.parquet``. They are still reported
# reactions, far more likely to work than a random pair, so U must not draw them (added 2026-10-28).
# With P = (A1, N1), (A2, N2) and one size bin, the draws are (A1, N1), (A1, N2), (A2, N1), (A2, N2)
# and k = 20 makes every one of them certain to occur.

TWO_P = positives_table([("A1", "N1"), ("A2", "N2")])
TWO_SIZES = {"A1": 8, "A2": 8, "N1": 8, "N2": 8, "A9": 8, "N9": 8}


def run_two_p(*others: tuple[str, str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage U on ``TWO_P`` with ``others`` added to its reported table."""
    return unlabeled_pairs(
        TWO_P,
        reported_of(TWO_P, *others),
        substrate_table(TWO_SIZES),
        make_config(k=20),
    )


def test_draw_of_a_reported_negative_pair_is_rejected() -> None:
    """(A1, N2) is a known negative: reported, below the cut-off, not in P. Drawing it would hand
    the model a reaction the literature says failed as an unlabeled one."""
    unlabeled, rejected = run_two_p(("A1", "N2", "negative"))
    assert ("A1", "N2") not in pair_set(unlabeled)
    held = rejected[
        (rejected["key_a_stereo"] == "A1") & (rejected["key_b_stereo"] == "N2")
    ]
    assert len(held) > 0
    assert set(held["reason"]) == {"pair_reported_flat"}


def test_draw_of_a_reported_unlabeled_pair_is_rejected() -> None:
    """(A1, N2) was reported with no usable yield (15,641 such pairs). It left P with the cut-off
    but is still a reported reaction, so a draw of it is the contamination to prevent."""
    unlabeled, rejected = run_two_p(("A1", "N2", "unlabeled"))
    assert ("A1", "N2") not in pair_set(unlabeled)
    held = rejected[
        (rejected["key_a_stereo"] == "A1") & (rejected["key_b_stereo"] == "N2")
    ]
    assert len(held) > 0
    assert set(held["reason"]) == {"pair_reported_flat"}


def test_draw_of_a_positive_pair_is_still_rejected() -> None:
    """(A1, N2) is in P as well as reported; the old behaviour holds. The control draw (A2, N1)
    is neither, and stays."""
    positives = positives_table([("A1", "N1"), ("A2", "N2"), ("A1", "N2")])
    unlabeled, rejected = unlabeled_pairs(
        positives,
        reported_of(positives),
        substrate_table(TWO_SIZES),
        make_config(k=20),
    )
    assert ("A1", "N2") not in pair_set(unlabeled)
    assert ("A1", "N2") in pair_set(rejected)
    assert set(rejected["reason"]) == {"pair_reported_flat"}


def test_draw_of_an_unreported_pair_is_kept() -> None:
    """The control: with only P reported, (A1, N2) and (A2, N1) are in U. Without it the three
    tests above pass if everything is rejected."""
    unlabeled, _ = run_two_p()
    assert pair_set(unlabeled) == {("A1", "N2"), ("A2", "N1")}


def test_pools_are_built_from_positives_only() -> None:
    """A9 is only in a negative row of the reported table and N9 only in an unlabeled one, both
    the size of every P substrate. If the pools came from the reported table they would be drawn
    as replacements; U's marginals are matched to P's, so they must never appear."""
    unlabeled, rejected = run_two_p(("A9", "N1", "negative"), ("A1", "N9", "unlabeled"))
    assert pair_set(unlabeled) == {("A1", "N2"), ("A2", "N1")}
    seen = set(rejected["key_a_stereo"]) | set(rejected["key_b_stereo"])
    assert not seen & {"A9", "N9"}


def test_subtraction_is_on_flat_keys() -> None:
    """P holds (L-ala, n1) and (benzoic, n2). The reported table also holds (D-ala, n2) as an
    unlabeled pair. The draw (L-ala, n2) has a new stereo pair but the flat key of that reported
    one, so it is rejected (ADR 013 item 2); (benzoic, n1) is the control that stays."""
    positives = positives_table([(L_ALANINE, "n1-X"), (BENZOIC_ACID, "n2-X")])
    substrates = substrate_table({L_ALANINE: 6, BENZOIC_ACID: 6, "n1-X": 6, "n2-X": 6})
    reported = reported_of(positives, (D_ALANINE, "n2-X", "unlabeled"))
    unlabeled, rejected = unlabeled_pairs(
        positives, reported, substrates, make_config(k=20)
    )
    assert (L_ALANINE, "n2-X") not in pair_set(unlabeled)
    assert (L_ALANINE, "n2-X") in pair_set(rejected)
    assert (BENZOIC_ACID, "n1-X") in pair_set(unlabeled)


def test_reported_missing_a_positive_raises() -> None:
    """Every pair of P is reported by construction. A reported table without two of them is from
    another run or another cut-off, and U would silently subtract less than it should."""
    reported = reported_of(LOPSIDED_P).iloc[2:]
    with pytest.raises(ValueError, match=r"missing 2 of the 40 positive pairs"):
        unlabeled_pairs(LOPSIDED_P, reported, LOPSIDED_SUBSTRATES, make_config())


# ---------------------------------------------------------------------- 4. no self-dedup


def test_a_pair_drawn_twice_is_one_row_with_n_draws_two_or_more() -> None:
    """P has two pairs, so only (A1, N2) and (A2, N1) can be drawn and 20 draws land on them.
    They are kept once each with their count. Deduplicating U would pull popular substrates
    below their P frequency and reopen the popularity shortcut in reverse (ADR 015 item 2)."""
    positives = positives_table([("A1", "N1"), ("A2", "N2")])
    substrates = substrate_table({"A1": 8, "A2": 8, "N1": 8, "N2": 8})
    unlabeled, _ = unlabeled_pairs(
        positives, reported_of(positives), substrates, make_config(k=10)
    )
    assert pair_set(unlabeled) == {("A1", "N2"), ("A2", "N1")}
    assert not unlabeled.duplicated(["key_a_stereo", "key_b_stereo"]).any()
    assert unlabeled["n_draws"].min() >= 2
    assert unlabeled["n_draws"].dtype == np.int16


# ---------------------------------------------------------------------- 5. determinism


def test_same_seed_gives_an_identical_frame() -> None:
    first, first_rejected = unlabeled_pairs(
        LOPSIDED_P, reported_of(LOPSIDED_P), LOPSIDED_SUBSTRATES, make_config(seed=3)
    )
    second, second_rejected = unlabeled_pairs(
        LOPSIDED_P, reported_of(LOPSIDED_P), LOPSIDED_SUBSTRATES, make_config(seed=3)
    )
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_rejected, second_rejected)


def test_another_seed_changes_the_rows_but_not_the_marginal_match() -> None:
    """Seed is the only source of randomness; the hot acid share holds to 25 % for both seeds."""
    frames = [
        unlabeled_pairs(
            LOPSIDED_P,
            reported_of(LOPSIDED_P),
            LOPSIDED_SUBSTRATES,
            make_config(k=100, seed=seed),
        )[0]
        for seed in (1, 2)
    ]
    assert not frames[0].equals(frames[1])
    p_share = p_frequency(LOPSIDED_P, "key_a_stereo")[HOT_A]
    for frame in frames:
        assert u_frequency(frame, "key_a_stereo")[HOT_A] == pytest.approx(
            p_share, rel=0.25
        )


# ---------------------------------------------------------------------- 6. loud failures


def test_a_positive_of_another_family_never_reaches_u() -> None:
    """Only rows with family == p_width are the positives: neither the other family's pair nor its
    substrates may appear in U. (Its pair is in the class-wide reported table, so it is subtracted
    there; with no pool it can never be drawn.)"""
    other = positives_table([("hatu_a", "hatu_n")]).assign(family="HATU")
    positives = pd.concat([LOPSIDED_P, other], ignore_index=True)
    substrates = substrate_table(
        {**dict(zip(LOPSIDED_SUBSTRATES["key_stereo"], [10] * len(LOPSIDED_SUBSTRATES))),
         "hatu_a": 10, "hatu_n": 10}
    )  # fmt: skip
    unlabeled, rejected = unlabeled_pairs(
        positives, reported_of(positives), substrates, make_config()
    )
    keys = set(unlabeled["key_a_stereo"]) | set(unlabeled["key_b_stereo"])
    assert not keys & {"hatu_a", "hatu_n"}
    assert not {"hatu_a", "hatu_n"} & (
        set(rejected["key_a_stereo"]) | set(rejected["key_b_stereo"])
    )


def test_mixed_std_version_in_positives_raises() -> None:
    """Keys of two standardisations would not join; the other stages stop the same way."""
    positives = pd.concat(
        [
            positives_table([("a1", "n1")]),
            positives_table([("a2", "n2")], std_version="std2"),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="std_version"):
        unlabeled_pairs(
            positives, reported_of(positives), LOPSIDED_SUBSTRATES, make_config()
        )


def test_a_match_rule_that_is_not_implemented_raises() -> None:
    """Independent draws are known to carry a 0.640 size shortcut (ADR 015); no flag for them."""
    with pytest.raises(ValueError, match="independent"):
        unlabeled_pairs(
            LOPSIDED_P,
            reported_of(LOPSIDED_P),
            LOPSIDED_SUBSTRATES,
            make_config(match="independent"),
        )


def test_a_positive_substrate_missing_from_the_feature_table_raises() -> None:
    """Without its heavy-atom count a substrate has no bin; dropping it would silently thin P."""
    substrates = LOPSIDED_SUBSTRATES[LOPSIDED_SUBSTRATES["key_stereo"] != HOT_A]
    with pytest.raises(ValueError, match=HOT_A):
        unlabeled_pairs(LOPSIDED_P, reported_of(LOPSIDED_P), substrates, make_config())


def test_pools_refuses_a_single_size() -> None:
    """Every acid has 10 heavy atoms, so the quantile edges collapse and no bin is left; the error
    names the acid column instead of the bare ``KeyError: nan`` it ended in."""
    positives = positives_table([(f"a{i}", f"n{i}") for i in range(5)])
    sizes = {f"a{i}": 10 for i in range(5)} | {f"n{i}": 5 + i for i in range(5)}
    with pytest.raises(ValueError, match="key_a_stereo.*binned"):
        unlabeled_pairs(
            positives,
            reported_of(positives),
            substrate_table(sizes),
            make_config(size_bins=10),
        )


def test_pools_refuses_a_missing_heavy_atom_count() -> None:
    """The amine is in the feature table but its heavy-atom count is NaN; the missing-substrate
    check does not see it, and the error must name the amine column."""
    positives = positives_table([(f"a{i}", f"n{i}") for i in range(5)])
    sizes = {f"a{i}": 5 + i for i in range(5)} | {f"n{i}": 5 + i for i in range(5)}
    substrates = substrate_table(sizes)
    substrates["n_heavy_atoms"] = substrates["n_heavy_atoms"].astype(float)
    substrates.loc[substrates["key_stereo"] == "n2", "n_heavy_atoms"] = np.nan
    with pytest.raises(ValueError, match="key_b_stereo.*binned"):
        unlabeled_pairs(
            positives, reported_of(positives), substrates, make_config(size_bins=2)
        )


def test_unlabeled_refuses_stereo_subtraction() -> None:
    """Subtraction is on the flat key only; a config asking for another key must not run on flat
    keys anyway."""
    with pytest.raises(ValueError, match="key_stereo.*key_flat"):
        unlabeled_pairs(
            LOPSIDED_P,
            reported_of(LOPSIDED_P),
            LOPSIDED_SUBSTRATES,
            make_config(subtract_p_on="key_stereo"),
        )


def test_unlabeled_accepts_flat_subtraction() -> None:
    """The production value still runs."""
    unlabeled, _ = unlabeled_pairs(
        LOPSIDED_P,
        reported_of(LOPSIDED_P),
        LOPSIDED_SUBSTRATES,
        make_config(subtract_p_on="key_flat"),
    )
    assert len(unlabeled) > 0


def test_load_unlabeled_config_reads_the_yaml(tmp_path: Path) -> None:
    path = tmp_path / "k10.yml"
    path.write_text(
        "k: 10\nseed: 20261023\nsize_bins: 10\np_width: all\n"
        f"subtract_p_on: key_flat\nmatch: {ONE_SIDE}\n"
        "gate_rank_auc_tolerance: 0.02\ngate_marginal_tv_multiple: 2.0\n"
        "gate_null_draws: 200\n",
        encoding="utf-8",
    )
    assert load_unlabeled_config(path) == UnlabeledConfig(
        k=10, seed=20261023, size_bins=10, p_width="all",
        subtract_p_on="key_flat", match=ONE_SIDE,
        gate_rank_auc_tolerance=0.02, gate_marginal_tv_multiple=2.0,
        gate_null_draws=200,
    )  # fmt: skip


# ---------------------------------------------------------------------- 7. output shape


def test_output_columns_and_open_licence() -> None:
    unlabeled, rejected = unlabeled_pairs(
        LOPSIDED_P, reported_of(LOPSIDED_P), LOPSIDED_SUBSTRATES, make_config()
    )
    assert list(unlabeled.columns) == OUTPUT_COLUMNS
    assert set(unlabeled["licence"]) == {"open"}
    assert set(unlabeled["std_version"]) == {STD}
    assert {"key_a_stereo", "key_b_stereo", "reason"} <= set(rejected.columns)


def test_licence_comes_from_the_positives_not_from_the_feature_row() -> None:
    """Stage F marks a substrate 'internal' as soon as the inventory holds it too. U is built
    from the open positives alone, so the same substrate stays 'open' here, or the table could
    not be written outside private/ and the publishable training side would vanish."""
    substrates = LOPSIDED_SUBSTRATES.assign(licence="internal")
    unlabeled, _ = unlabeled_pairs(
        LOPSIDED_P, reported_of(LOPSIDED_P), substrates, make_config()
    )
    assert set(unlabeled["licence"]) == {"open"}


def test_licence_is_the_strictest_of_the_positives() -> None:
    """A noncommercial positive makes U noncommercial, via io.strictest_licence."""
    positives = pd.concat(
        [LOPSIDED_P.iloc[:-1], LOPSIDED_P.iloc[[-1]].assign(licence="noncommercial")],
        ignore_index=True,
    )
    unlabeled, _ = unlabeled_pairs(
        positives, reported_of(positives), LOPSIDED_SUBSTRATES, make_config()
    )
    assert set(unlabeled["licence"]) == {"noncommercial"}


def test_script_writes_u_outside_private(tmp_path: Path) -> None:
    """Every substrate is from the open positives, so the table is publishable: <root>/unlabeled/,
    never <root>/private/unlabeled/. The feature table is read from private/ (it holds library
    rows), but none of them reach U. The report carries counts that add up and the config hash."""
    (tmp_path / "positives" / "amide_coupling").mkdir(parents=True)
    LOPSIDED_P.to_parquet(
        tmp_path / "positives" / "amide_coupling" / "positives.parquet"
    )
    reported_of(LOPSIDED_P).to_parquet(
        tmp_path / "positives" / "amide_coupling" / "reported.parquet"
    )
    features = tmp_path / "private" / "features" / "amide_coupling"
    features.mkdir(parents=True)
    LOPSIDED_SUBSTRATES.assign(licence="internal").to_parquet(
        features / "substrates.parquet"
    )
    config = tmp_path / "k10.yml"
    # This test pins paths and counts, not the match, so the tolerances are opened here. The gate
    # has its own tests in section 8.
    config.write_text(
        config_text(gate_rank_auc_tolerance=0.5, gate_marginal_tv_multiple=100.0),
        encoding="utf-8",
    )

    unlabeled_script(["amide_coupling", str(tmp_path), "--config", str(config)])

    out = tmp_path / "unlabeled" / "amide_coupling"
    table = pd.read_parquet(out / "k10.parquet")
    rejected = pd.read_parquet(out / "k10.rejected.parquet")
    report = json.loads((out / "k10.report.json").read_text(encoding="utf-8"))
    assert list(table.columns) == OUTPUT_COLUMNS
    assert not (tmp_path / "private" / "unlabeled").exists()
    assert report["draws"]["requested"] == 10 * len(LOPSIDED_P)
    assert report["draws"]["out"] == int(table["n_draws"].sum())
    assert (
        report["draws"]["requested"]
        == report["draws"]["out"] + report["draws"]["rejected"]["pair_reported_flat"]
    )
    assert report["draws"]["rejected"]["pair_reported_flat"] == len(rejected) > 0
    assert str(config) in report["configs"]


# ---------------------------------------------------------------------- probes (the reason U exists)


def popularity_score(frame: pd.DataFrame, positives: pd.DataFrame) -> np.ndarray:
    """Each side's number of P pairs, multiplied: all a classifier would see of 'how often reported'."""
    acid_count = positives["key_a_stereo"].value_counts()
    amine_count = positives["key_b_stereo"].value_counts()
    return (
        frame["key_a_stereo"].map(acid_count).fillna(0).to_numpy()
        * frame["key_b_stereo"].map(amine_count).fillna(0).to_numpy()
    )


def test_popularity_probe_separates_p_from_uniform_u_but_not_from_matched_u() -> None:
    """Both halves. A score from each side's P frequency alone must beat 0.5 clearly on a uniform
    U (positive control: the probe can fail) and sit near 0.5 on the matched U (the stage works).
    Either alone proves nothing: a broken probe also returns 0.5. Design: 0.892 against 0.511.

    This pins the SAMPLER on a 40-pair fixture, where the joint structure is negligible. It is not
    the gate on the full run: there a probe seeing both sides reads about 0.6 against the matched
    U by construction (a U matching P in every joint respect would be P), and only the one-sided
    marginal reads 0.5. The marginal is the gate; the joint probes are reported (ADR 017)."""
    matched, _ = unlabeled_pairs(
        LOPSIDED_P, reported_of(LOPSIDED_P), LOPSIDED_SUBSTRATES, make_config(k=50)
    )
    matched = matched.loc[matched.index.repeat(matched["n_draws"])]
    uniform = uniform_unlabeled(LOPSIDED_P, len(matched), seed=0)

    def probe(u: pd.DataFrame) -> float:
        score = np.concatenate(
            [popularity_score(LOPSIDED_P, LOPSIDED_P), popularity_score(u, LOPSIDED_P)]
        )
        label = np.arange(len(score)) < len(LOPSIDED_P)
        return auc(score, label)

    assert probe(uniform) > 0.70
    assert abs(probe(matched) - 0.5) < 0.15


# Disassortative P: acid i (size 5 + i) is reported with amines of size 24 - i and its neighbours,
# the shape of "a complex fragment is coupled to a simple building block".
ACID_SIZE = {f"A{i}": 5 + i for i in range(20)}
AMINE_SIZE = {f"N{j}": 5 + j for j in range(20)}
BAND = [
    (f"A{i}", f"N{j}") for i in range(20) for j in range(20) if abs(i + j - 19) <= 1
]
BAND_P = positives_table(BAND)
BAND_SUBSTRATES = substrate_table({**ACID_SIZE, **AMINE_SIZE})


def size_correlation(frame: pd.DataFrame) -> float:
    """Correlation of the two sides' heavy-atom counts over the rows of ``frame``."""
    size_a = frame["key_a_stereo"].map(ACID_SIZE)
    size_b = frame["key_b_stereo"].map(AMINE_SIZE)
    return float(np.corrcoef(size_a, size_b)[0, 1])


def test_joint_size_probe_u_keeps_the_negative_size_correlation_of_p() -> None:
    """The two heavy-atom counts as two features, not their sum (the sum is blind to this).
    P is strongly anti-correlated; independent draws give about 0, which a classifier on the
    two counts separates from P at 0.640. Matched U must stay on P's side of zero. The
    independent U is the positive control: it must read near 0 or this test cannot fail.

    Like the popularity probe, this pins the sampler on a small hand-made P. On the real run the
    joint probes are diagnostics, reported and never pass/fail (ADR 017): size is stratified, the
    degree axis stays visible, and the joint size probe reads about 0.54 there, not 0.5."""
    matched, _ = unlabeled_pairs(
        BAND_P, reported_of(BAND_P), BAND_SUBSTRATES, make_config(k=20, size_bins=5)
    )
    matched = matched.loc[matched.index.repeat(matched["n_draws"])]
    independent = uniform_unlabeled(BAND_P, len(matched), seed=0)
    r_p = size_correlation(BAND_P)
    assert r_p < -0.9
    assert size_correlation(independent) > r_p / 2
    assert size_correlation(matched) < r_p / 2


# ---------------------------------------------------------------------- 8. the gate (ADR 017)
#
# The marginal match is a gate, not a measurement: the script raises past the tolerances in the
# config, before it writes anything. Added 2026-10-28 after data-guardian found the first version
# computed total variation and compared it with nothing.


def config_text(**changes: object) -> str:
    """A stage U YAML as ``make_config`` would read it, with ``changes`` applied."""
    settings = {
        "k": 10,
        "seed": 7,
        "size_bins": 1,
        "p_width": "all",
        "subtract_p_on": "key_flat",
        "match": ONE_SIDE,
        "gate_rank_auc_tolerance": 0.02,
        "gate_marginal_tv_multiple": 2.0,
        "gate_null_draws": 200,
        **changes,
    }
    return "".join(f"{name}: {value}\n" for name, value in settings.items())


def write_inputs(root: Path, **changes: object) -> Path:
    """The lopsided P and its feature table under ``root`` as stage U reads them; the config path."""
    (root / "positives" / "amide_coupling").mkdir(parents=True)
    LOPSIDED_P.to_parquet(root / "positives" / "amide_coupling" / "positives.parquet")
    reported_of(LOPSIDED_P).to_parquet(
        root / "positives" / "amide_coupling" / "reported.parquet"
    )
    features = root / "private" / "features" / "amide_coupling"
    features.mkdir(parents=True)
    LOPSIDED_SUBSTRATES.assign(licence="internal").to_parquet(
        features / "substrates.parquet"
    )
    config = root / "k10.yml"
    config.write_text(config_text(**changes), encoding="utf-8")
    return config


def run_script(root: Path, config: Path) -> None:
    unlabeled_script(["amide_coupling", str(root), "--config", str(config)])


def measured_gate(config: UnlabeledConfig) -> dict[str, dict[str, float]]:
    """What the script measures on the lopsided P for ``config``, computed outside ``main``."""
    unlabeled, _ = unlabeled_pairs(
        LOPSIDED_P, reported_of(LOPSIDED_P), LOPSIDED_SUBSTRATES, config
    )
    return {
        "acid": script_module.marginals(LOPSIDED_P, unlabeled, "key_a_stereo", config),
        "amine": script_module.marginals(LOPSIDED_P, unlabeled, "key_b_stereo", config),
    }


def test_a_uniform_u_trips_the_gate_and_nothing_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure the stage exists to prevent, with the config's own tolerances: the sampler is
    replaced by the naive uniform draw, and the run must refuse. Nothing may exist afterwards, not
    the table, not the rejects, not the report, not even the folder."""
    config = write_inputs(tmp_path)

    def uniform_sampler(
        positives: pd.DataFrame,
        reported: pd.DataFrame,
        substrates: pd.DataFrame,
        settings: UnlabeledConfig,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return uniform_unlabeled(positives, 400, seed=0), pd.DataFrame()

    monkeypatch.setattr(script_module, "unlabeled_pairs", uniform_sampler)
    with pytest.raises(ValueError, match="nothing was written"):
        run_script(tmp_path, config)
    assert not (tmp_path / "unlabeled").exists()


@pytest.mark.parametrize(
    "tolerance", ["gate_rank_auc_tolerance", "gate_marginal_tv_multiple"]
)
def test_a_vanishing_tolerance_trips_the_gate_before_any_file_appears(
    tmp_path: Path, tolerance: str
) -> None:
    """A correct run measures a small but non-zero distance on both numbers, so a tolerance of
    1e-9 must refuse it (zero is refused earlier, by the loader). The other tolerance is wide open,
    so only this one can be the cause."""
    wide = {"gate_rank_auc_tolerance": 1.0, "gate_marginal_tv_multiple": 1000.0}
    config = write_inputs(tmp_path, **{**wide, tolerance: 1e-9})
    with pytest.raises(ValueError, match="nothing was written"):
        run_script(tmp_path, config)
    assert not (tmp_path / "unlabeled").exists()


def test_a_tolerance_the_sampler_comfortably_meets_lets_the_run_through(
    tmp_path: Path,
) -> None:
    """Positive control for the two tests above: the same inputs with wide tolerances write the
    table, so the raise there is the gate and not a broken fixture."""
    config = write_inputs(
        tmp_path, gate_rank_auc_tolerance=1.0, gate_marginal_tv_multiple=1000.0
    )
    run_script(tmp_path, config)
    assert (tmp_path / "unlabeled" / "amide_coupling" / "k10.parquet").exists()


def test_the_error_names_the_role_the_measured_value_the_multiple_and_the_null(
    tmp_path: Path,
) -> None:
    """An operator must be able to act on it. The multiple sits between the two roles' ratios of
    measured total variation to null, so exactly the larger-ratio role trips; the message names it
    and quotes its value, the multiple and the null."""
    gate = measured_gate(make_config())
    ratio = {
        role: v["total_variation"] / v["null_total_variation"]
        for role, v in gate.items()
    }
    assert ratio["acid"] != ratio["amine"]
    worse = max(ratio, key=lambda role: ratio[role])
    multiple = (ratio["acid"] + ratio["amine"]) / 2
    config = write_inputs(
        tmp_path, gate_rank_auc_tolerance=1.0, gate_marginal_tv_multiple=multiple
    )
    with pytest.raises(ValueError) as raised:
        run_script(tmp_path, config)
    message = str(raised.value)
    assert worse in message
    assert str(gate[worse]["total_variation"]) in message
    assert str(gate[worse]["null_total_variation"]) in message
    assert str(multiple) in message
    assert "total variation" in message


def test_the_error_names_the_role_and_the_measured_rank_auc(tmp_path: Path) -> None:
    """The same for the rank-AUC: the message gives the role, the value, and the tolerance."""
    gate = measured_gate(make_config())
    drift = {role: abs(values["rank_auc"] - 0.5) for role, values in gate.items()}
    assert drift["acid"] != drift["amine"]
    worse = max(drift, key=lambda role: drift[role])
    config = write_inputs(
        tmp_path,
        gate_marginal_tv_multiple=1000.0,
        gate_rank_auc_tolerance=(drift["acid"] + drift["amine"]) / 2,
    )
    with pytest.raises(ValueError) as raised:
        run_script(tmp_path, config)
    message = str(raised.value)
    assert worse in message
    assert str(gate[worse]["rank_auc"]) in message
    assert "rank-AUC" in message


@pytest.mark.parametrize(
    "missing",
    ["gate_rank_auc_tolerance", "gate_marginal_tv_multiple", "gate_null_draws"],
)
def test_load_unlabeled_config_raises_when_a_gate_key_is_missing(
    tmp_path: Path, missing: str
) -> None:
    """No default: a config that omits the gate must not get one silently (CLAUDE.md section 5)."""
    path = tmp_path / "k10.yml"
    lines = config_text().splitlines()
    path.write_text(
        "\n".join(line for line in lines if not line.startswith(missing)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=missing):
        load_unlabeled_config(path)


def test_the_report_carries_the_gate_with_rank_auc_and_total_variation_per_role(
    tmp_path: Path,
) -> None:
    """Provenance for the numbers quoted in ADR 017: under ``gate``, per role, the rank-AUC and the
    total variation, equal to what the sampler measures, and each inside its tolerance. The old
    ``marginals`` key and its substrate-valued ``largest_on`` are gone."""
    config = write_inputs(
        tmp_path, gate_rank_auc_tolerance=0.5, gate_marginal_tv_multiple=1000.0
    )
    run_script(tmp_path, config)
    report = json.loads(
        (tmp_path / "unlabeled" / "amide_coupling" / "k10.report.json").read_text(
            encoding="utf-8"
        )
    )
    expected = measured_gate(
        make_config(gate_rank_auc_tolerance=0.5, gate_marginal_tv_multiple=1000.0)
    )
    assert set(report["gate"]) == {"acid", "amine"}
    for role in ("acid", "amine"):
        assert report["gate"][role] == expected[role]
        assert report["gate"][role]["rank_auc_tolerance"] == 0.5
        assert report["gate"][role]["total_variation_allowed"] == pytest.approx(
            1000.0 * expected[role]["null_total_variation"], rel=1e-3
        )
    assert "marginals" not in report
    assert "largest_on" not in json.dumps(report)


@pytest.mark.parametrize(
    "key", ["gate_rank_auc_tolerance", "gate_marginal_tv_multiple"]
)
@pytest.mark.parametrize("value", [0, -0.5])
def test_load_unlabeled_config_refuses_a_non_positive_tolerance(
    tmp_path: Path, key: str, value: float
) -> None:
    """Zero or negative would reject every run with a message that blames the sampler."""
    path = tmp_path / "k10.yml"
    path.write_text(config_text(**{key: value}), encoding="utf-8")
    with pytest.raises(ValueError, match=key):
        load_unlabeled_config(path)


# ---------------------------------------------------------------------- 9. the null


def test_the_null_falls_as_p_grows_so_the_gate_scales() -> None:
    """The point of a null over an absolute number. Twenty equally frequent substrates: the total
    variation a perfect sampler still shows is sampling noise and falls like 1/sqrt(n), so 100
    times the draws gives about a tenth of the floor. An absolute 0.10 would be slack at the large
    size and tighter than a perfect sampler at the small one."""
    shares = np.full(20, 1 / 20)
    small = script_module.null_total_variation(shares, 400, 200, seed=7)
    large = script_module.null_total_variation(shares, 40_000, 200, seed=7)
    assert large < small / 5


def test_the_allowed_distance_is_the_multiple_times_the_null() -> None:
    """Doubling the multiple doubles what is allowed, the null being the same stream."""
    unlabeled, _ = unlabeled_pairs(
        LOPSIDED_P, reported_of(LOPSIDED_P), LOPSIDED_SUBSTRATES, make_config()
    )
    once = script_module.marginals(LOPSIDED_P, unlabeled, "key_a_stereo", make_config())
    twice = script_module.marginals(
        LOPSIDED_P,
        unlabeled,
        "key_a_stereo",
        make_config(gate_marginal_tv_multiple=4.0),
    )
    assert twice["null_total_variation"] == once["null_total_variation"]
    assert twice["total_variation_allowed"] == pytest.approx(
        2 * once["total_variation_allowed"], abs=1e-4
    )


def test_the_sampler_sits_below_its_own_null_on_a_sparse_p() -> None:
    """The strongest statement about this sampler. A perfect i.i.d. draw from P's frequencies still
    misses them by the null; the sampler beats that, because its anchor side is not random: each
    positive pair contributes its own anchor exactly k/2 times. Less sampling noise than a
    multinomial, by design and not by luck.

    The P has the real shape, mostly substrates reported once plus a few popular ones (the real
    one is 67 % single-pair). On the dense 40-pair LOPSIDED P it does NOT hold (0.0545 against a
    null of 0.0461 at k = 100): there the known pairs are a large share of the cross-product, and
    subtracting them takes extra draws from exactly the popular substrates."""
    pairs = (
        [(f"hot{i % 3}", f"n{i}") for i in range(30)]
        + [(f"a{i}", f"hotn{i % 3}") for i in range(30)]
        + [(f"a{i + 100}", f"n{i + 100}") for i in range(140)]
    )
    positives = positives_table(pairs)
    substrates = substrate_table(
        {key: 10 for key in {*(a for a, _ in pairs), *(b for _, b in pairs)}}
    )
    config = make_config()
    unlabeled, _ = unlabeled_pairs(
        positives, reported_of(positives), substrates, config
    )
    for key in ("key_a_stereo", "key_b_stereo"):
        gate = script_module.marginals(positives, unlabeled, key, config)
        assert gate["total_variation"] < gate["null_total_variation"]


def test_the_null_is_reproducible_from_the_seed_and_depends_on_it() -> None:
    """The floor is itself a draw, so a gate that moved between runs would be a coin. The same
    seed gives the same floor; another seed gives another."""
    shares = np.full(20, 1 / 20)
    first = script_module.null_total_variation(shares, 400, 200, seed=7)
    again = script_module.null_total_variation(shares, 400, 200, seed=7)
    other = script_module.null_total_variation(shares, 400, 200, seed=8)
    assert first == again
    assert first != other


def test_gate_null_draws_changes_the_floor_and_a_run_repeats_it(tmp_path: Path) -> None:
    """``gate_null_draws`` is read from the config (not a constant): 5 draws and 200 give different
    floors, and the same config twice stamps the same floor in its report."""
    unlabeled, _ = unlabeled_pairs(
        LOPSIDED_P, reported_of(LOPSIDED_P), LOPSIDED_SUBSTRATES, make_config()
    )

    def floor(config: UnlabeledConfig) -> float:
        return script_module.marginals(LOPSIDED_P, unlabeled, "key_a_stereo", config)[
            "null_total_variation"
        ]

    assert floor(make_config()) == floor(make_config())
    assert floor(make_config(gate_null_draws=5)) != floor(make_config())


# ---------------------------------------------------------------------- 10. the rule without an anchor
#
# Stage U keeps one side of a published pair as an anchor. The ablation asks whether a model reads
# that anchor instead of the chemistry (the failure Chen et al. documented for the DUD-E decoys), so
# the second rule must differ from the first in that one respect and no other: both sides are
# replaced on every draw, each from the size bin of the side it replaces, weighted by P frequency;
# pools, subtraction of everything reported, rejects and table are unchanged. Written 2026-10-29.

# Two size bins per role, and each P pair joins a template acid to an amine of the other role's bin
# so that the two rules reach DIFFERENT pairs. Acids: S = {aS1, aS2}, B = {aB1, aB2}; amines the
# same. Templates: (aS1,bB1) (aS2,bS1) (aB1,bB2) (aB2,bS2). Every substrate is in P once, so every
# pool is uniform. Replacing one side stays inside the anchor's other bin, replacing both reaches the
# four pairs below that no one-side draw can make:
#   from (aS1,bB1): (aS2,bB2)   from (aS2,bS1): (aS1,bS2)
#   from (aB1,bB2): (aB2,bB1)   from (aB2,bS2): (aB1,bS1)
CROSSED = [("aS1", "bB1"), ("aS2", "bS1"), ("aB1", "bB2"), ("aB2", "bS2")]
CROSSED_P = positives_table(CROSSED)
CROSSED_SIZES = {
    "aS1": 5, "aS2": 6, "aB1": 30, "aB2": 31,
    "bS1": 5, "bS2": 6, "bB1": 30, "bB2": 31,
}  # fmt: skip
CROSSED_SUBSTRATES = substrate_table(CROSSED_SIZES)
# Reachable by replacing both sides and by no single-side replacement.
ONLY_BY_REPLACING_BOTH = {
    ("aS2", "bB2"),
    ("aS1", "bS2"),
    ("aB2", "bB1"),
    ("aB1", "bS1"),
}
# Reachable by replacing one side within its bin (the one-side rule's whole output).
ONE_SIDE_REACHES = {
    ("aS1", "bB2"),
    ("aS2", "bB1"),
    ("aS2", "bS2"),
    ("aS1", "bS1"),
    ("aB1", "bB1"),
    ("aB2", "bB2"),
    ("aB2", "bS1"),
    ("aB1", "bS2"),
}


def crossed_unlabeled(match: str, **changes: object) -> pd.DataFrame:
    """U of the CROSSED case under ``match``; k = 40 by default, so a pair that a template reaches
    with chance 1/4 per draw is missed with chance (3/4)^40, about 1e-5, and the fixed seed makes
    the outcome one fixed table."""
    settings = {"k": 40, "size_bins": 2, "match": match, **changes}
    unlabeled, _ = unlabeled_pairs(
        CROSSED_P,
        reported_of(CROSSED_P),
        CROSSED_SUBSTRATES,
        make_config(**settings),
    )
    return unlabeled


def test_the_implemented_match_rules_are_exactly_the_two_named_in_the_configs() -> None:
    """``MATCH`` is the module's own list of rules, imported here so the tests cannot drift from
    it. The names are the YAML contract, so they are also written out above."""
    assert set(MATCH) == {ONE_SIDE, BOTH_SIDES}


def test_both_sides_rule_replaces_both_substrates() -> None:
    """The ablation itself. The four pairs in ONLY_BY_REPLACING_BOTH share no substrate with the
    template they come from, and no one-side draw can make them (shown on the same case under the
    one-side rule). If both-sides U lacked them it would still be anchored and the experiment
    would measure nothing.

    A replacement is drawn from a pool that contains the substrate it replaces, so a single draw
    keeps a side by chance (1/2 here); 'never keeps either' is not the property, 'reaches what an
    anchor cannot' is."""
    both = crossed_unlabeled(BOTH_SIDES)
    one = crossed_unlabeled(ONE_SIDE)
    assert ONLY_BY_REPLACING_BOTH <= pair_set(both)
    assert pair_set(one).isdisjoint(ONLY_BY_REPLACING_BOTH)


def test_both_sides_replacements_come_from_their_own_decile() -> None:
    """The acid's replacement is from the acid's bin and the amine's from the amine's. U is then
    exactly the 12 pairs of the two bins of each template (4 reached by one side, 4 by both, the
    positives are subtracted): a replacement from one shared pool, or from the wrong role's bin,
    would put an aB or bS substrate beside a template that has none, and the size structure of P
    (a small acid with a big amine) would be lost."""
    both = crossed_unlabeled(BOTH_SIDES)
    assert pair_set(both) == ONE_SIDE_REACHES | ONLY_BY_REPLACING_BOTH


def test_both_sides_draws_both_pools_on_every_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """There is no draw % 2 parity for this rule: every draw calls the acid pool with the
    template's acid and the amine pool with its amine, 2 x k x |P| calls and k x |P| for the
    one-side rule. Counted at ``_draw`` because the output cannot say which side was replaced."""
    from platepu.positives import unlabeled as module

    calls: list[str] = []
    real_draw = module._draw

    def counting_draw(pools: object, key: str, generator: object) -> str:
        calls.append(key)
        return real_draw(pools, key, generator)  # type: ignore[arg-type]

    monkeypatch.setattr(module, "_draw", counting_draw)
    crossed_unlabeled(BOTH_SIDES, k=6)
    assert len(calls) == 2 * 6 * len(CROSSED)
    assert sorted(set(calls)) == sorted(CROSSED_SIZES)
    calls.clear()
    crossed_unlabeled(ONE_SIDE, k=6)
    assert len(calls) == 6 * len(CROSSED)


# The real shape of P (67 % of substrates reported once, a few popular): three hot acids and three
# hot amines, 140 singleton pairs, as in test_the_sampler_sits_below_its_own_null_on_a_sparse_p.
SPARSE = (
    [(f"hot{i % 3}", f"n{i}") for i in range(30)]
    + [(f"a{i}", f"hotn{i % 3}") for i in range(30)]
    + [(f"a{i + 100}", f"n{i + 100}") for i in range(140)]
)
SPARSE_P = positives_table(SPARSE)
SPARSE_SUBSTRATES = substrate_table(
    {key: 10 for key in {*(a for a, _ in SPARSE), *(b for _, b in SPARSE)}}
)


def test_both_sides_keeps_the_per_role_marginals() -> None:
    """Each role's U frequencies still track its P frequencies. This matters because the stage
    gate (ADR 017) measures only per-role marginals: its silence on the both-sides rule is a
    property of the draw, not an accident, only if this holds.

    The tolerance is not a number of mine. Drawing every side from P's frequencies is a multinomial
    with P's shares, which is what ``null_total_variation`` simulates, and its 99th percentile is
    the distance a perfect sampler shows from sampling noise alone; the gate allows
    ``gate_marginal_tv_multiple`` (2.0) times it, scaled to this P and this number of draws. The
    both-sides rule has no anchor to beat that noise (the one-side rule sits below it), so it is
    held to the gate's own line, not to a fixed 0.05. The rank-AUC tolerance of 0.02 is not asserted:
    with 200 positives the AUC of a perfect sampler has a standard error of about 0.02 itself, so
    it would test the fixture size and not the sampler (it is the real 28,158-pair P that meets
    it)."""
    config = make_config(match=BOTH_SIDES)
    unlabeled, _ = unlabeled_pairs(
        SPARSE_P, reported_of(SPARSE_P), SPARSE_SUBSTRATES, config
    )
    for key in ("key_a_stereo", "key_b_stereo"):
        gate = script_module.marginals(SPARSE_P, unlabeled, key, config)
        assert gate["total_variation"] < gate["total_variation_allowed"]


def test_both_sides_is_reproducible_from_the_seed() -> None:
    first, first_rejected = unlabeled_pairs(
        LOPSIDED_P,
        reported_of(LOPSIDED_P),
        LOPSIDED_SUBSTRATES,
        make_config(match=BOTH_SIDES, seed=3),
    )
    second, second_rejected = unlabeled_pairs(
        LOPSIDED_P,
        reported_of(LOPSIDED_P),
        LOPSIDED_SUBSTRATES,
        make_config(match=BOTH_SIDES, seed=3),
    )
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_rejected, second_rejected)


def test_both_sides_with_another_seed_gives_another_frame() -> None:
    frames = [
        unlabeled_pairs(
            LOPSIDED_P,
            reported_of(LOPSIDED_P),
            LOPSIDED_SUBSTRATES,
            make_config(match=BOTH_SIDES, seed=seed),
        )[0]
        for seed in (1, 2)
    ]
    assert not frames[0].equals(frames[1])


def test_an_unimplemented_match_rule_lists_what_is_implemented() -> None:
    """A reader learns the options from the error: both rule names are in it, with the refused
    value. (The test above, for 'independent', stays: the refusal itself is unchanged.)"""
    with pytest.raises(ValueError, match="not implemented") as error:
        unlabeled_pairs(
            LOPSIDED_P,
            reported_of(LOPSIDED_P),
            LOPSIDED_SUBSTRATES,
            make_config(match="uniform"),
        )
    message = str(error.value)
    assert "'uniform'" in message
    assert ONE_SIDE in message
    assert BOTH_SIDES in message


# The one-side rule on CROSSED, k = 8, seed 11, size_bins = 2, counted on 2026-10-29 from the code
# as it stood BEFORE the second rule was added: (acid, amine, n_draws) of the 7 kept pairs, 12 kept
# draws and 20 rejected (the 32 = 8 x 4 draws). Hard-coded so that adding the second rule cannot
# quietly change the thing it is compared against.
ONE_SIDE_ORACLE = [
    ("aB1", "bS2", 2),
    ("aB2", "bB2", 2),
    ("aB2", "bS1", 3),
    ("aS1", "bB2", 1),
    ("aS1", "bS1", 1),
    ("aS2", "bB1", 1),
    ("aS2", "bS2", 2),
]


def test_the_one_side_rule_is_unchanged() -> None:
    unlabeled, rejected = unlabeled_pairs(
        CROSSED_P,
        reported_of(CROSSED_P),
        CROSSED_SUBSTRATES,
        make_config(match=ONE_SIDE, k=8, seed=11, size_bins=2),
    )
    kept = list(
        zip(
            unlabeled["key_a_stereo"],
            unlabeled["key_b_stereo"],
            unlabeled["n_draws"].astype(int),
            strict=True,
        )
    )
    assert kept == ONE_SIDE_ORACLE
    assert len(rejected) == 20


@pytest.mark.parametrize("match", [ONE_SIDE, BOTH_SIDES])
def test_a_pair_drawn_more_than_int16_can_hold_raises(match: str) -> None:
    """Two disjoint positives, one bin: the pair (a0, b1) is half of all one-side draws and a
    quarter of both-sides draws from each template, 40,000 of 160,000 either way, past the 32,767
    that ``n_draws`` (int16) holds. Cast silently it wraps negative and U loses weight where it is
    heaviest; it must raise, naming the pair and the count. Both rules, because the cast is
    shared."""
    positives = positives_table([("a0", "b0"), ("a1", "b1")])
    substrates = substrate_table({"a0": 8, "a1": 8, "b0": 8, "b1": 8})
    with pytest.raises(ValueError) as error:
        unlabeled_pairs(
            positives,
            reported_of(positives),
            substrates,
            make_config(match=match, k=80_000),
        )
    message = str(error.value)
    assert "a0" in message
    assert "b1" in message
    assert any(int(n) > 32_767 for n in re.findall(r"\d+", message))
