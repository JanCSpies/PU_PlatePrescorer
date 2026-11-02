"""Tests for stage S2b, the final fit and the two arms (docs/design/modeling.md "S2b -- the final fit
and the two arms"; ADR 028, ADR 029): fit P against the negative class once and score the validation
and inventory pairs.

Written BEFORE the module exists (2026-11-02): until ``PuScoreConfig`` is added to
src/platepu/score/pu_config.py and src/platepu/score/pu_scores.py is written, this file fails at
collection with ImportError, and that is intended.

The spy arm trains P against the reliable negatives of an S2a table; the naive arm trains P against
every row of U. They differ in one thing, where the negative class comes from, and ``negatives`` in
the config picks the arm (a stem = spy, null = naive).

Every fixture is made up and worked out by hand; no file is read except the YAML the config tests
write into ``tmp_path``. The builders ``pair_table`` and ``substrate_rows`` are rewritten here in a
smaller form, not imported, because ``tests/`` is not a package.

The planted case. Two kinds of substrate: "P-like" ones carry bits 0-3 and "odd" ones bits 4-7, and
within a kind the fingerprint is identical, so every pair of one kind has one feature vector and one
fitted score. That is what makes the tie structure and the Spearman value computable by hand:

    scored set "validation", five pairs, heavy atoms of the pair (acid + amine):
        (PA0,PN0) 30, (PA4,PN0) 32, (PA5,PN0) 34      P-like, score high
        (OA0,ON0) 10, (OA1,ON0) 12                    odd, score low
    two distinct scores, the larger tie is 3.
    Spearman of score against heavy atoms: ranks of heavy are 1..5, ranks of score are
    1.5, 1.5, 4, 4, 4 (the odd pairs first), so
        sum of products of centred ranks = 3 + 1.5 + 0 + 1 + 2 = 7.5,
        variance sums are 10 and 7.5, r = 7.5 / sqrt(75) = sqrt(3) / 2 = 0.8660254.
    scored set "inventory", four pairs, heavy 30, 34 (P-like) and 10, 12 (odd):
        two distinct scores, the larger tie is 2, ranks 1..4 against 1.5, 1.5, 3.5, 3.5,
        r = 4 / sqrt(20) = 0.8944272.

Where the report's keys are not fixed by the design, the names below are my choice (the test author's)
and are flagged for the code author; see ``SET_REPORT_KEYS``.
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
from sklearn.ensemble import HistGradientBoostingClassifier

import platepu.score.pu_scores as pu_scores_module
from platepu.score.pu_audit import score_deciles
from platepu.score.pu_config import (
    PuConfig,
    PuScoreConfig,
    config_sha256,
    load_pu_score_config,
)
from platepu.score.pu_scores import (
    check_negatives_pointer,
    pair_scores,
    training_rows,
)

STD_VERSION = "std1"
PAIR_KEYS = ["key_a_stereo", "key_b_stereo", "key_a_flat", "key_b_flat"]
LEARNER_PARAMS = {"max_iter": 30, "early_stopping": False, "min_samples_leaf": 1}

P_LIKE_ACID = [1, 1, 0, 0, 0, 0, 0, 0]
P_LIKE_AMINE = [0, 0, 1, 1, 0, 0, 0, 0]
ODD_ACID = [0, 0, 0, 0, 1, 1, 0, 0]
ODD_AMINE = [0, 0, 0, 0, 0, 0, 1, 1]

SPY_CONFIG = PuScoreConfig(
    model="pair_hgb_spy",
    feature_block="morgan_centre",
    p_width="all",
    unlabeled="unlabeled_literature_k10",
    negatives="pu_spy_morgan_centre",
    learner="HistGradientBoostingClassifier",
    learner_params=LEARNER_PARAMS,
    seed=7,
)
NAIVE_CONFIG = replace(SPY_CONFIG, model="pair_hgb_naive", negatives=None)

# The S2a config the spy arm points at. It must agree with SPY_CONFIG on the five shared keys.
S2A_CONFIG = PuConfig(
    model="pair_hgb",
    feature_block="morgan_centre",
    p_width="all",
    unlabeled="unlabeled_literature_k10",
    spy_fraction=0.2,
    spy_recall=0.95,
    spy_scoring="out_of_fold",
    n_folds=5,
    learner="HistGradientBoostingClassifier",
    learner_params=LEARNER_PARAMS,
    seed=7,
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


def substrate_rows(
    role: str, keys: list[str], bits: list[int], heavy: list[int]
) -> list[dict[str, object]]:
    return [
        {
            "key_stereo": key,
            "role": role,
            "n_heavy_atoms": atoms,
            "morgan_centre": np.array(bits, dtype=np.uint8),
            "morgan": np.array(bits, dtype=np.uint8) * 2,
            "morgan_near": np.array(bits, dtype=np.uint8) * 3,
            "std_version": STD_VERSION,
        }
        for key, atoms in zip(keys, heavy, strict=True)
    ]


# Heavy atoms are what the Spearman value depends on: see the module docstring for the pairs.
SUBSTRATES = pd.DataFrame(
    substrate_rows(
        "acid",
        ["PA0", "PA1", "PA2", "PA3", "PA4", "PA5"],
        P_LIKE_ACID,
        [20, 20, 20, 20, 22, 24],
    )
    + substrate_rows(
        "amine", ["PN0", "PN1", "PN2", "PN3"], P_LIKE_AMINE, [10, 10, 10, 10]
    )
    + substrate_rows("acid", ["OA0", "OA1", "OA2", "OA3"], ODD_ACID, [6, 8, 6, 6])
    + substrate_rows("amine", ["ON0", "ON1", "ON2", "ON3"], ODD_AMINE, [4, 4, 4, 4])
)

# 12 positives: PA0..PA3 x PN0..PN2.
POSITIVE_PAIRS = list(
    itertools.product(["PA0", "PA1", "PA2", "PA3"], ["PN0", "PN1", "PN2"])
)
# 12 negatives: OA0, OA2, OA3 x ON0..ON3. OA1 is left out on purpose, so (OA1,ON0) is not in training.
NEGATIVE_PAIRS = list(
    itertools.product(["OA0", "OA2", "OA3"], ["ON0", "ON1", "ON2", "ON3"])
)
TRAIN = pd.concat(
    [
        pair_table(POSITIVE_PAIRS, target=1, weight=1.0),
        pair_table(NEGATIVE_PAIRS, target=0, weight=2.0),
    ],
    ignore_index=True,
)

VALIDATION = pair_table(
    [
        ("PA0", "PN0"),
        ("PA4", "PN0"),
        ("PA5", "PN0"),
        ("OA0", "ON0"),
        ("OA1", "ON0"),
    ]
)
# (PA4,PN0) is the other stereoisomer of PA1: same flat acid key as the training pair (PA1,PN0).
VALIDATION.loc[1, "key_a_flat"] = "pa1"
INVENTORY = pair_table([("PA1", "PN3"), ("PA5", "PN1"), ("OA3", "ON3"), ("OA1", "ON1")])
SCORED = {"validation": VALIDATION, "inventory": INVENTORY}

POSITIVE_SIDE = {"validation": [0, 1, 2], "inventory": [0, 1]}
NEGATIVE_SIDE = {"validation": [3, 4], "inventory": [2, 3]}

SET_REPORT_KEYS = {
    "score_deciles",
    "n_distinct_scores",
    "largest_tie",
    "spearman_heavy_atoms",
    "n_in_training_stereo",
    "n_in_training_flat",
}


def run(
    config: PuScoreConfig = NAIVE_CONFIG,
) -> tuple[dict[str, np.ndarray], dict]:
    return pair_scores(TRAIN, SCORED, SUBSTRATES, config)


# --- the config record and its file ------------------------------------------------------------

SCORE_YAML = """\
model: pair_hgb_spy
feature_block: morgan_centre
p_width: all
unlabeled: unlabeled_literature_k10
negatives: pu_spy_morgan_centre
learner: HistGradientBoostingClassifier
learner_params:
  max_iter: 30
  early_stopping: false
  min_samples_leaf: 1
seed: 7
"""


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "pu_score_fixture.yml"
    path.write_text(text, encoding="utf-8")
    return path


def test_spy_arm_config_loads_with_its_s2a_stem(tmp_path: Path) -> None:
    assert load_pu_score_config(write_config(tmp_path, SCORE_YAML)) == SPY_CONFIG


def test_naive_arm_config_loads_with_negatives_null(tmp_path: Path) -> None:
    text = SCORE_YAML.replace("pair_hgb_spy", "pair_hgb_naive").replace(
        "negatives: pu_spy_morgan_centre", "negatives:"
    )
    config = load_pu_score_config(write_config(tmp_path, text))
    assert config.negatives is None
    assert config == NAIVE_CONFIG


def test_score_config_has_exactly_the_eight_designed_keys() -> None:
    assert set(PuScoreConfig.__dataclass_fields__) == {
        "model",
        "feature_block",
        "p_width",
        "unlabeled",
        "negatives",
        "learner",
        "learner_params",
        "seed",
    }


@pytest.mark.parametrize(
    "spy_key",
    ["spy_fraction: 0.2", "spy_recall: 0.95", "spy_scoring: out_of_fold", "n_folds: 5"],
)
def test_spy_settings_are_an_unknown_key_in_an_s2b_file(
    tmp_path: Path, spy_key: str
) -> None:
    """ADR 028 decision 1: a naive-arm file carrying spy_fraction would claim spies were drawn."""
    path = write_config(tmp_path, SCORE_YAML + spy_key + "\n")
    with pytest.raises(ValueError, match=re.escape(str(path))) as error:
        load_pu_score_config(path)
    assert spy_key.split(":")[0] in str(error.value)


@pytest.mark.parametrize("bad_model", ["pair_hgb", "pair_hgb_spi", "pair_hgb_naive_"])
def test_model_must_name_an_arm(tmp_path: Path, bad_model: str) -> None:
    path = write_config(tmp_path, SCORE_YAML.replace("pair_hgb_spy", bad_model))
    with pytest.raises(ValueError, match=re.escape(str(path))) as error:
        load_pu_score_config(path)
    assert bad_model in str(error.value)


def test_empty_string_negatives_is_not_the_naive_arm(tmp_path: Path) -> None:
    """Null means 'no table'; an empty string would be a stem that names nothing."""
    path = write_config(
        tmp_path,
        SCORE_YAML.replace("negatives: pu_spy_morgan_centre", 'negatives: ""'),
    )
    with pytest.raises(ValueError, match=re.escape(str(path))):
        load_pu_score_config(path)


@pytest.mark.parametrize(
    ("good_line", "bad_line"),
    [
        ("feature_block: morgan_centre", "feature_block: amine_subclasses"),
        ("learner: HistGradientBoostingClassifier", "learner: RandomForest"),
        ("  max_iter: 30", "  max_iter: null"),
        ("  early_stopping: false", "  early_stopping: true"),
        ("seed: 7", "seed: 7.9"),
        ("p_width: all", "p_width:"),
        ("unlabeled: unlabeled_literature_k10", "unlabeled: ' '"),
    ],
)
def test_score_config_refuses_what_pu_config_refuses(
    tmp_path: Path, good_line: str, bad_line: str
) -> None:
    path = write_config(tmp_path, SCORE_YAML.replace(good_line, bad_line))
    with pytest.raises(ValueError, match=re.escape(str(path))):
        load_pu_score_config(path)


def test_config_sha256_accepts_both_records_and_sees_the_arm() -> None:
    spy_hash = config_sha256(SPY_CONFIG)
    assert re.fullmatch(r"[0-9a-f]{64}", spy_hash)
    assert config_sha256(S2A_CONFIG) != spy_hash
    assert config_sha256(replace(SPY_CONFIG)) == spy_hash
    assert config_sha256(NAIVE_CONFIG) != spy_hash


# --- the spy arm's pointer is checked, not trusted ---------------------------------------------


def negatives_with_hash(*hashes: str) -> pd.DataFrame:
    """An S2a table reduced to what the check reads: one row per hash given."""
    return pair_table([("OA0", "ON0")] * len(hashes)).assign(config_sha256=list(hashes))


def test_pointer_to_a_matching_s2a_config_and_table_passes() -> None:
    table = negatives_with_hash(*[config_sha256(S2A_CONFIG)] * 3)
    check_negatives_pointer(SPY_CONFIG, S2A_CONFIG, table)


def disagree(config: PuConfig, key: str) -> PuConfig:
    """``config`` with one shared key changed. ``object.__setattr__`` because ``PuConfig`` refuses a
    foreign learner or a missing max_iter where it is built, and the check must still catch it.
    This is a test-only construction of a config the class would otherwise refuse, not a supported
    path."""
    changed = {
        "feature_block": "morgan",
        "p_width": "hatu",
        "unlabeled": "unlabeled_literature_k5",
        "learner": "RandomForestClassifier",
        "learner_params": {**LEARNER_PARAMS, "max_iter": 99},
    }[key]
    copy = replace(config)
    object.__setattr__(copy, key, changed)
    return copy


@pytest.mark.parametrize(
    "key", ["feature_block", "p_width", "unlabeled", "learner", "learner_params"]
)
def test_pointer_to_an_s2a_config_that_disagrees_names_the_key(key: str) -> None:
    spy_config = disagree(S2A_CONFIG, key)
    table = negatives_with_hash(config_sha256(spy_config))
    with pytest.raises(ValueError, match=key):
        check_negatives_pointer(SPY_CONFIG, spy_config, table)


def test_negatives_cut_under_another_config_are_refused() -> None:
    """A table cut under `morgan` must never be fitted under `morgan_centre`: its hash is another's."""
    other_hash = config_sha256(replace(S2A_CONFIG, feature_block="morgan"))
    with pytest.raises(ValueError, match="config_sha256"):
        check_negatives_pointer(SPY_CONFIG, S2A_CONFIG, negatives_with_hash(other_hash))


def test_negatives_with_two_distinct_hashes_are_refused() -> None:
    right = config_sha256(S2A_CONFIG)
    table = negatives_with_hash(right, right, "0" * 64)
    with pytest.raises(ValueError, match="config_sha256"):
        check_negatives_pointer(SPY_CONFIG, S2A_CONFIG, table)


# --- training_rows ------------------------------------------------------------------------------

TRAINING_COLUMNS = [*PAIR_KEYS, "target", "weight"]

# Four positives, two of which (PA2,PN0) and (PA3,PN0) were spies in S2a and rejoin P.
FIXTURE_POSITIVES = pair_table(
    [("PA0", "PN0"), ("PA1", "PN0"), ("PA2", "PN0"), ("PA3", "PN0")],
    family="all",
    n_reports=1,
    std_version=STD_VERSION,
    licence="open",
)


def s2a_table() -> pd.DataFrame:
    """Six rows of an S2a table: two reliable negatives (weights 1 and 3), one U row not flagged
    (weight 2), and the two spies (in_p, weight 1, never a reliable negative)."""
    rows = [
        ("OA0", "ON0", False, True, 1.0),
        ("OA2", "ON1", False, True, 3.0),
        ("OA3", "ON2", False, False, 2.0),
        ("PA2", "PN0", True, False, 1.0),
        ("PA3", "PN0", True, False, 1.0),
    ]
    table = pair_table([(a, b) for a, b, *_ in rows])
    table["in_p"] = [row[2] for row in rows]
    table["reliable_negative"] = [row[3] for row in rows]
    table["weight"] = [row[4] for row in rows]
    return table.assign(config_sha256=config_sha256(S2A_CONFIG))


def records(table: pd.DataFrame) -> list[tuple[str, str, int, float]]:
    return sorted(
        zip(
            table["key_a_stereo"],
            table["key_b_stereo"],
            table["target"].astype(int),
            table["weight"].astype(float),
            strict=True,
        )
    )


def test_spy_arm_training_rows_are_p_plus_the_reliable_negatives_with_their_own_weight() -> (
    None
):
    s2a = s2a_table()
    train = training_rows(FIXTURE_POSITIVES, s2a, SPY_CONFIG)
    assert list(train.columns) == TRAINING_COLUMNS
    assert records(train) == [
        ("OA0", "ON0", 0, 1.0),
        ("OA2", "ON1", 0, 3.0),
        ("PA0", "PN0", 1, 1.0),
        ("PA1", "PN0", 1, 1.0),
        ("PA2", "PN0", 1, 1.0),
        ("PA3", "PN0", 1, 1.0),
    ]


def test_spy_arm_leaves_out_the_unflagged_u_row_and_never_doubles_a_spy() -> None:
    s2a = s2a_table()
    train = training_rows(FIXTURE_POSITIVES, s2a, SPY_CONFIG)
    keys = list(zip(train["key_a_stereo"], train["key_b_stereo"], strict=True))
    assert ("OA3", "ON2") not in keys
    assert len(keys) == len(set(keys)) == 6


def test_naive_arm_training_rows_are_p_plus_every_u_row_weighted_by_n_draws() -> None:
    unlabeled = pair_table(
        [("OA0", "ON0"), ("OA2", "ON1"), ("OA3", "ON2")],
        std_version=STD_VERSION,
        licence="open",
    ).assign(n_draws=[1, 7, 2])
    train = training_rows(FIXTURE_POSITIVES, unlabeled, NAIVE_CONFIG)
    assert list(train.columns) == TRAINING_COLUMNS
    assert records(train) == [
        ("OA0", "ON0", 0, 1.0),
        ("OA2", "ON1", 0, 7.0),
        ("OA3", "ON2", 0, 2.0),
        ("PA0", "PN0", 1, 1.0),
        ("PA1", "PN0", 1, 1.0),
        ("PA2", "PN0", 1, 1.0),
        ("PA3", "PN0", 1, 1.0),
    ]


@pytest.mark.parametrize("config", [SPY_CONFIG, NAIVE_CONFIG])
def test_a_positive_that_is_also_a_negative_raises_naming_the_key(
    config: PuScoreConfig,
) -> None:
    clash = pair_table(
        [("PA1", "PN0")], weight=1.0, n_draws=1, in_p=False, reliable_negative=True
    )
    with pytest.raises(ValueError, match="PA1"):
        training_rows(FIXTURE_POSITIVES, clash, config)


# --- pair_scores: one fit, and the planted case -----------------------------------------------


def test_every_pair_on_the_positive_side_outscores_every_pair_on_the_negative_side() -> (
    None
):
    scores, _ = run()
    positive_side = np.concatenate([scores[n][POSITIVE_SIDE[n]] for n in SCORED])
    negative_side = np.concatenate([scores[n][NEGATIVE_SIDE[n]] for n in SCORED])
    assert positive_side.min() > negative_side.max()


def test_spy_arm_config_fits_the_same_way() -> None:
    """The arm is a property of the rows, not of the fit: the same train frame gives the same scores."""
    naive, _ = run(NAIVE_CONFIG)
    spy, _ = run(SPY_CONFIG)
    assert np.array_equal(naive["validation"], spy["validation"])


def test_scored_sets_come_back_as_separate_float32_arrays_in_the_callers_order() -> (
    None
):
    scores, _ = run()
    assert list(scores) == ["validation", "inventory"]
    assert scores["validation"].shape == (5,)
    assert scores["inventory"].shape == (4,)
    assert all(array.dtype == np.float32 for array in scores.values())
    reordered, _ = pair_scores(
        TRAIN,
        {"inventory": INVENTORY, "validation": VALIDATION},
        SUBSTRATES,
        NAIVE_CONFIG,
    )
    assert list(reordered) == ["inventory", "validation"]
    assert np.array_equal(reordered["validation"], scores["validation"])


def test_pair_scores_fits_once_for_all_scored_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    original = HistGradientBoostingClassifier.fit

    def counting_fit(
        self: HistGradientBoostingClassifier, *args: object, **kwargs: object
    ):  # type: ignore[no-untyped-def]
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", counting_fit)
    run()
    assert len(calls) == 1


def test_same_config_gives_identical_scores_and_reports() -> None:
    first_scores, first_report = run()
    second_scores, second_report = run()
    for name in SCORED:
        assert np.array_equal(first_scores[name], second_scores[name])
    # json.dumps also pins that the report can be written as .report.json without a converter.
    assert json.dumps(first_report, sort_keys=True) == json.dumps(
        second_report, sort_keys=True
    )


def test_pair_with_a_substrate_missing_from_the_feature_table_raises() -> None:
    lost = pair_table([("PA0", "PN9")])
    with pytest.raises(ValueError, match="PN9"):
        pair_scores(TRAIN, {"validation": lost}, SUBSTRATES, NAIVE_CONFIG)


# --- the report ----------------------------------------------------------------------------------


def test_naive_report_counts_rows_weights_and_draws() -> None:
    """12 positives of weight 1; 12 negatives of weight 2, so a weight sum of 24.
    Negatives per positive (the way ADR 024, 026 and modeling.md quote 1:9.7) is
    12/12 = 1.0 by rows and 24/12 = 2.0 by weight; the negative weight sum is 24 in both arms."""
    _, report = run(NAIVE_CONFIG)
    assert report["n_positives"] == 12
    assert report["n_negatives"] == 12
    assert report["negative_source"] == "unlabeled"
    assert report["negatives_per_positive_rows"] == pytest.approx(1.0)
    assert report["negatives_per_positive_weight"] == pytest.approx(2.0)
    assert report["negative_weight_sum"] == pytest.approx(24.0)
    assert "n_draws_sum" not in report


def test_spy_report_names_reliable_negatives_as_its_source() -> None:
    _, report = run(SPY_CONFIG)
    assert report["negative_source"] == "reliable_negatives"
    assert report["n_positives"] == 12
    assert report["n_negatives"] == 12
    assert report["negatives_per_positive_rows"] == pytest.approx(1.0)
    assert report["negatives_per_positive_weight"] == pytest.approx(2.0)
    assert report["negative_weight_sum"] == pytest.approx(24.0)


def test_each_scored_set_reports_the_designed_keys() -> None:
    _, report = run()
    assert set(report["scored"]) == {"validation", "inventory"}
    for name in SCORED:
        assert SET_REPORT_KEYS <= set(report["scored"][name])


def test_tie_structure_is_the_planted_two_scores() -> None:
    """Validation: three P-like pairs share one score and two odd pairs another, so 2 distinct and
    the largest tie is 3. Inventory: 2 and 2."""
    _, report = run()
    assert report["scored"]["validation"]["n_distinct_scores"] == 2
    assert report["scored"]["validation"]["largest_tie"] == 3
    assert report["scored"]["inventory"]["n_distinct_scores"] == 2
    assert report["scored"]["inventory"]["largest_tie"] == 2


def test_spearman_of_score_against_heavy_atoms_is_computed_by_hand() -> None:
    """sqrt(3)/2 on validation and 4/sqrt(20) on inventory; the derivation is in the module docstring."""
    _, report = run()
    assert report["scored"]["validation"]["spearman_heavy_atoms"] == pytest.approx(
        np.sqrt(3) / 2
    )
    assert report["scored"]["inventory"]["spearman_heavy_atoms"] == pytest.approx(
        4 / np.sqrt(20)
    )


def test_scored_pairs_found_in_training_are_counted_on_stereo_and_on_flat_keys() -> (
    None
):
    """Validation: (PA0,PN0) is a training positive and (OA0,ON0) a training negative, so 2 on
    stereo keys. (PA4,PN0) is a stereoisomer of PA1 with the same flat keys as the training pair
    (PA1,PN0), so flat is 3. (PA5,PN0), (OA1,ON0) are in neither.
    Inventory: (OA3,ON3) is a training negative, 1 on both; (PA1,PN3), (PA5,PN1), (OA1,ON1) are not."""
    _, report = run()
    validation = report["scored"]["validation"]
    inventory = report["scored"]["inventory"]
    assert validation["n_in_training_stereo"] == 2
    assert validation["n_in_training_flat"] == 3
    assert inventory["n_in_training_stereo"] == 1
    assert inventory["n_in_training_flat"] == 1


def test_score_deciles_are_pu_audits_nine_cut_points() -> None:
    """One name, one meaning: the same ``score_deciles`` the S2a report uses, nine ascending floats."""
    scores, report = run()
    for name in SCORED:
        deciles = report["scored"][name]["score_deciles"]
        assert len(deciles) == 9
        assert deciles == score_deciles(scores[name])
        assert deciles == sorted(deciles)


# --- no label, no metric -----------------------------------------------------------------------


def test_signatures_take_no_label_argument() -> None:
    assert list(inspect.signature(training_rows).parameters) == [
        "positives",
        "negatives",
        "config",
    ]
    assert list(inspect.signature(pair_scores).parameters) == [
        "train",
        "scored",
        "substrates",
        "config",
    ]


def test_module_imports_no_benchmark_reads_no_table_and_computes_no_auc() -> None:
    """The validation read is a separate step by design (modeling.md, 'S2b must not'). Only imports
    and names are checked: this cannot prove no label is read through some other path."""
    tree = ast.parse(inspect.getsource(pu_scores_module))
    imported = [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ] + [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    imported += [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    ]
    assert not [name for name in imported if "benchmark" in name]
    assert not [name for name in imported if "roc_auc" in name]
    called = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not called & {"read_parquet", "read_csv", "read_table", "open"}
    assert not [name for name in called if "roc_auc" in name]


# --- morgan_near (ADR 030, 2026-11-02): three fingerprint blocks are legal ----------------------


@pytest.mark.parametrize("block", ["morgan", "morgan_centre", "morgan_near"])
def test_score_config_accepts_each_of_the_three_fingerprint_blocks(block: str) -> None:
    assert replace(SPY_CONFIG, feature_block=block).feature_block == block


def test_score_config_reads_morgan_near_from_a_file(tmp_path: Path) -> None:
    text = SCORE_YAML.replace(
        "feature_block: morgan_centre", "feature_block: morgan_near"
    )
    config = load_pu_score_config(write_config(tmp_path, text))
    assert config == replace(SPY_CONFIG, feature_block="morgan_near")


@pytest.mark.parametrize(
    ("stem", "arm", "negatives"),
    [
        ("pu_score_spy_morgan_near", "spy", "pu_spy_morgan_near"),
        ("pu_score_naive_morgan_near", "naive", None),
    ],
)
def test_the_repository_s2b_configs_for_morgan_near_load(
    stem: str, arm: str, negatives: str | None
) -> None:
    config = load_pu_score_config(
        Path(__file__).parents[1] / "configs" / "models" / f"{stem}.yml"
    )
    assert config.feature_block == "morgan_near"
    assert config.model == f"pair_hgb_{arm}"
    assert config.negatives == negatives


def test_pair_scores_runs_on_the_morgan_near_block() -> None:
    """The planted case is separable in every block, so the two score levels are the same."""
    scores, _ = run(replace(NAIVE_CONFIG, feature_block="morgan_near"))
    assert len(set(scores["validation"].tolist())) == 2
