"""Tests for stage U, the literature-built unlabeled space (docs/design/modeling.md, ADRs 013-016).

U is drawn from the substrates of the literature positives P, never from the inventory. Each
counter-pair replaces ONE side of a positive pair, from the same heavy-atom bin, weighted by P
frequency, so U keeps P's per-role popularity (ADR 013) and P's size anti-correlation (ADR 015).

Cases are hand-written (counted 2026-10-23 in the design, written 2026-10-28), all open, no Chemlex
and no inventory row. Keys are made-up short strings except where the case is an InChIKey pattern
(L-/D-alanine as in ``test_benchmark_split.py``; benzoic acid and its 13C and d5 isotopologues,
two of the five leaks found on 2026-10-23). The flat key of a made-up key is the part before
the first "-", as the first InChIKey block is.

The scripts' command line is pinned here too: ``unlabeled_literature.py CLASS ROOT --config YAML``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import unlabeled_literature as script_module
from unlabeled_literature import main as unlabeled_script

from platepu.positives.unlabeled import (
    UnlabeledConfig,
    load_unlabeled_config,
    unlabeled_pairs,
)

STD = "std1"
MATCH = "size_stratified_one_side"
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
        "match": MATCH,
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
        positives, substrate_table(sizes), make_config(size_bins=2, k=20)
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
    unlabeled, _ = unlabeled_pairs(positives, substrates, make_config(k=50))
    keys = set(unlabeled["key_a_stereo"]) | set(unlabeled["key_b_stereo"])
    assert keys <= {"A1", "A2", "A3", "N1", "N2", "N3"}


# ---------------------------------------------------------------------- 2. marginals match P


def test_each_substrates_u_share_matches_its_p_share() -> None:
    """The hot acid is in 25 % of P's pairs, the hot amine in 20 %; a uniform sampler would give
    them 1/31 and 1/23. U keeps those shares to within 25 % relative, per role. Not tighter: a
    popular substrate is also in the most known pairs, so subtracting P takes slightly more of its
    draws (a reference sampler read 0.21 and 0.18 against 0.25 and 0.20 on this small, dense P)."""
    unlabeled, _ = unlabeled_pairs(LOPSIDED_P, LOPSIDED_SUBSTRATES, make_config(k=100))
    p_acid = p_frequency(LOPSIDED_P, "key_a_stereo")
    p_amine = p_frequency(LOPSIDED_P, "key_b_stereo")
    u_acid = u_frequency(unlabeled, "key_a_stereo")
    u_amine = u_frequency(unlabeled, "key_b_stereo")
    assert u_acid[HOT_A] == pytest.approx(p_acid[HOT_A], rel=0.25)
    assert u_amine[HOT_N] == pytest.approx(p_amine[HOT_N], rel=0.25)


def test_total_variation_between_p_and_u_marginals_is_small() -> None:
    """All substrates at once, both roles: half the summed absolute difference of the shares."""
    unlabeled, _ = unlabeled_pairs(LOPSIDED_P, LOPSIDED_SUBSTRATES, make_config(k=100))
    for column in ("key_a_stereo", "key_b_stereo"):
        p_share = p_frequency(LOPSIDED_P, column)
        u_share = u_frequency(unlabeled, column).reindex(p_share.index, fill_value=0.0)
        assert (p_share - u_share).abs().sum() / 2 < 0.10


def test_the_number_of_draws_is_k_times_p_and_rejects_are_not_replaced() -> None:
    """No top-up loop (design, 'No top-up loop'): kept draws never exceed k x |P|."""
    unlabeled, _ = unlabeled_pairs(LOPSIDED_P, LOPSIDED_SUBSTRATES, make_config(k=10))
    assert 0 < unlabeled["n_draws"].sum() <= 10 * len(LOPSIDED_P)


# ---------------------------------------------------------------------- 3. subtract P on flat


def test_l_and_d_alanine_pair_is_subtracted_from_u_on_the_flat_key() -> None:
    """P holds (L-ala, n1) and (D-ala, n2). Every one-side counter-pair, (D-ala, n1), (L-ala, n2),
    has the flat pair key of a positive though its stereo pair is new. Our fingerprints ignore
    chirality, so keeping them would teach the model that a published reaction is unlabeled.
    Subtracting on the stereo key would leave all four in U."""
    positives = positives_table([(L_ALANINE, "n1-X"), (D_ALANINE, "n2-X")])
    substrates = substrate_table({L_ALANINE: 6, D_ALANINE: 6, "n1-X": 6, "n2-X": 6})
    unlabeled, rejected = unlabeled_pairs(positives, substrates, make_config())
    assert unlabeled.empty
    assert set(rejected["reason"]) == {"pair_in_p_flat"}
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
    unlabeled, rejected = unlabeled_pairs(positives, substrates, make_config())
    assert unlabeled.empty
    assert (BENZOIC_ACID, "n2-X") in pair_set(rejected)


def test_no_pair_in_u_shares_its_flat_key_with_a_positive() -> None:
    """The same rule as a property of a whole run on the lopsided P: zero overlap on flat keys."""
    unlabeled, _ = unlabeled_pairs(LOPSIDED_P, LOPSIDED_SUBSTRATES, make_config())
    flat_u = set(zip(unlabeled["key_a_flat"], unlabeled["key_b_flat"], strict=True))
    flat_p = set(zip(LOPSIDED_P["key_a_flat"], LOPSIDED_P["key_b_flat"], strict=True))
    assert not flat_u & flat_p


def test_rejected_pairs_are_the_known_positives_and_nothing_else() -> None:
    """The one reason in the design is pair_in_p_flat; a key is never under both roles."""
    _, rejected = unlabeled_pairs(LOPSIDED_P, LOPSIDED_SUBSTRATES, make_config(k=50))
    assert len(rejected) > 0
    assert set(rejected["reason"]) == {"pair_in_p_flat"}


# ---------------------------------------------------------------------- 4. no self-dedup


def test_a_pair_drawn_twice_is_one_row_with_n_draws_two_or_more() -> None:
    """P has two pairs, so only (A1, N2) and (A2, N1) can be drawn and 20 draws land on them.
    They are kept once each with their count. Deduplicating U would pull popular substrates
    below their P frequency and reopen the popularity shortcut in reverse (ADR 015 item 2)."""
    positives = positives_table([("A1", "N1"), ("A2", "N2")])
    substrates = substrate_table({"A1": 8, "A2": 8, "N1": 8, "N2": 8})
    unlabeled, _ = unlabeled_pairs(positives, substrates, make_config(k=10))
    assert pair_set(unlabeled) == {("A1", "N2"), ("A2", "N1")}
    assert not unlabeled.duplicated(["key_a_stereo", "key_b_stereo"]).any()
    assert unlabeled["n_draws"].min() >= 2
    assert unlabeled["n_draws"].dtype == np.int16


# ---------------------------------------------------------------------- 5. determinism


def test_same_seed_gives_an_identical_frame() -> None:
    first, first_rejected = unlabeled_pairs(
        LOPSIDED_P, LOPSIDED_SUBSTRATES, make_config(seed=3)
    )
    second, second_rejected = unlabeled_pairs(
        LOPSIDED_P, LOPSIDED_SUBSTRATES, make_config(seed=3)
    )
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_rejected, second_rejected)


def test_another_seed_changes_the_rows_but_not_the_marginal_match() -> None:
    """Seed is the only source of randomness; the hot acid share holds to 25 % for both seeds."""
    frames = [
        unlabeled_pairs(LOPSIDED_P, LOPSIDED_SUBSTRATES, make_config(k=100, seed=seed))[
            0
        ]
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
    substrates may appear in U, and its pair is not subtracted from it either."""
    other = positives_table([("hatu_a", "hatu_n")]).assign(family="HATU")
    positives = pd.concat([LOPSIDED_P, other], ignore_index=True)
    substrates = substrate_table(
        {**dict(zip(LOPSIDED_SUBSTRATES["key_stereo"], [10] * len(LOPSIDED_SUBSTRATES))),
         "hatu_a": 10, "hatu_n": 10}
    )  # fmt: skip
    unlabeled, rejected = unlabeled_pairs(positives, substrates, make_config())
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
        unlabeled_pairs(positives, LOPSIDED_SUBSTRATES, make_config())


def test_a_match_rule_that_is_not_implemented_raises() -> None:
    """Independent draws are known to carry a 0.640 size shortcut (ADR 015); no flag for them."""
    with pytest.raises(ValueError, match="independent"):
        unlabeled_pairs(
            LOPSIDED_P, LOPSIDED_SUBSTRATES, make_config(match="independent")
        )


def test_a_positive_substrate_missing_from_the_feature_table_raises() -> None:
    """Without its heavy-atom count a substrate has no bin; dropping it would silently thin P."""
    substrates = LOPSIDED_SUBSTRATES[LOPSIDED_SUBSTRATES["key_stereo"] != HOT_A]
    with pytest.raises(ValueError, match=HOT_A):
        unlabeled_pairs(LOPSIDED_P, substrates, make_config())


def test_load_unlabeled_config_reads_the_yaml(tmp_path: Path) -> None:
    path = tmp_path / "k10.yml"
    path.write_text(
        "k: 10\nseed: 20261023\nsize_bins: 10\np_width: all\n"
        f"subtract_p_on: key_flat\nmatch: {MATCH}\n"
        "gate_rank_auc_tolerance: 0.02\ngate_marginal_tv_multiple: 2.0\n"
        "gate_null_draws: 200\n",
        encoding="utf-8",
    )
    assert load_unlabeled_config(path) == UnlabeledConfig(
        k=10, seed=20261023, size_bins=10, p_width="all",
        subtract_p_on="key_flat", match=MATCH,
        gate_rank_auc_tolerance=0.02, gate_marginal_tv_multiple=2.0,
        gate_null_draws=200,
    )  # fmt: skip


# ---------------------------------------------------------------------- 7. output shape


def test_output_columns_and_open_licence() -> None:
    unlabeled, rejected = unlabeled_pairs(
        LOPSIDED_P, LOPSIDED_SUBSTRATES, make_config()
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
    unlabeled, _ = unlabeled_pairs(LOPSIDED_P, substrates, make_config())
    assert set(unlabeled["licence"]) == {"open"}


def test_licence_is_the_strictest_of_the_positives() -> None:
    """A noncommercial positive makes U noncommercial, via io.strictest_licence."""
    positives = pd.concat(
        [LOPSIDED_P.iloc[:-1], LOPSIDED_P.iloc[[-1]].assign(licence="noncommercial")],
        ignore_index=True,
    )
    unlabeled, _ = unlabeled_pairs(positives, LOPSIDED_SUBSTRATES, make_config())
    assert set(unlabeled["licence"]) == {"noncommercial"}


def test_script_writes_u_outside_private(tmp_path: Path) -> None:
    """Every substrate is from the open positives, so the table is publishable: <root>/unlabeled/,
    never <root>/private/unlabeled/. The feature table is read from private/ (it holds library
    rows), but none of them reach U. The report carries counts that add up and the config hash."""
    (tmp_path / "positives" / "amide_coupling").mkdir(parents=True)
    LOPSIDED_P.to_parquet(
        tmp_path / "positives" / "amide_coupling" / "positives.parquet"
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
        == report["draws"]["out"] + report["draws"]["rejected"]["pair_in_p_flat"]
    )
    assert report["draws"]["rejected"]["pair_in_p_flat"] == len(rejected) > 0
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
    matched, _ = unlabeled_pairs(LOPSIDED_P, LOPSIDED_SUBSTRATES, make_config(k=50))
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
        BAND_P, BAND_SUBSTRATES, make_config(k=20, size_bins=5)
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
        "match": MATCH,
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
    unlabeled, _ = unlabeled_pairs(LOPSIDED_P, LOPSIDED_SUBSTRATES, config)
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
        positives: pd.DataFrame, substrates: pd.DataFrame, settings: UnlabeledConfig
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
    unlabeled, _ = unlabeled_pairs(LOPSIDED_P, LOPSIDED_SUBSTRATES, make_config())
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
    unlabeled, _ = unlabeled_pairs(positives, substrates, config)
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
    unlabeled, _ = unlabeled_pairs(LOPSIDED_P, LOPSIDED_SUBSTRATES, make_config())

    def floor(config: UnlabeledConfig) -> float:
        return script_module.marginals(LOPSIDED_P, unlabeled, "key_a_stereo", config)[
            "null_total_variation"
        ]

    assert floor(make_config()) == floor(make_config())
    assert floor(make_config(gate_null_draws=5)) != floor(make_config())
