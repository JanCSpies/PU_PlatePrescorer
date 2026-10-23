"""Tests for the test-set labels built from HTE conversions, on a table written by hand.

Rule (decided 2026-10-09, ``configs/benchmarks/chemlex.yml``): replicates of one reaction are kept only
when all are 0 % or all are at least 10 %; anything else is inconsistent and leaves the test set.
A kept reaction gets the mean conversion, and it is positive when that mean is at least 20 %.

Groups in the table (acid, amine → conversions):
- A (0, 0)        → all zero        → mean 0,  negative
- B (30, 50)      → all ≥ 10 %      → mean 40, positive
- C (12, 16)      → all ≥ 10 %      → mean 14, negative
- D (0, 31)       → inconsistent: once it worked, once not at all
- E (0, 2.5)      → inconsistent: not all 0, not all ≥ 10
- F (5, 15)       → inconsistent: not all 0, not all ≥ 10
- G (25)          → single          → positive
- H (0)           → single          → negative
- I (15, 85, 20)  → all ≥ 10 %      → mean 40, positive
"""

from pathlib import Path

import pandas as pd
import pytest

from platepu.benchmarks.labels import LabelRule, label_reactions, load_label_rule

RULE = LabelRule(positive_min_pct=20.0, replicate_min_pct=10.0)
KEYS = ["acid", "amine"]
CONFIG = Path(__file__).parents[1] / "configs" / "benchmarks" / "chemlex.yml"

GROUPS: dict[str, list[float]] = {
    "A": [0.0, 0.0],
    "B": [30.0, 50.0],
    "C": [12.0, 16.0],
    "D": [0.0, 31.0],
    "E": [0.0, 2.5],
    "F": [5.0, 15.0],
    "G": [25.0],
    "H": [0.0],
    "I": [15.0, 85.0, 20.0],
}

# group → (status, mean conversion, label); None = not labelled, leaves the test set
EXPECTED: dict[str, tuple[str, float, bool | None]] = {
    "A": ("all_zero", 0.0, False),
    "B": ("all_above_min", 40.0, True),
    "C": ("all_above_min", 14.0, False),
    "D": ("inconsistent", 15.5, None),
    "E": ("inconsistent", 1.25, None),
    "F": ("inconsistent", 10.0, None),
    "G": ("single", 25.0, True),
    "H": ("single", 0.0, False),
    "I": ("all_above_min", 40.0, True),
}


@pytest.fixture
def labelled() -> pd.DataFrame:
    """Label the hand-written table; one acid per group, all with the same amine."""
    rows = [
        {"acid": group, "amine": "n1", "conversion": value}
        for group, values in GROUPS.items()
        for value in values
    ]
    return label_reactions(pd.DataFrame(rows), KEYS, RULE).set_index("acid")


def test_one_row_per_reaction(labelled: pd.DataFrame) -> None:
    """Replicates collapse to one row each, and every reaction is still there."""
    assert sorted(labelled.index) == sorted(GROUPS)
    assert labelled.loc["I", "n_measurements"] == 3


@pytest.mark.parametrize("group", sorted(EXPECTED))
def test_status_mean_and_label(labelled: pd.DataFrame, group: str) -> None:
    """Each reaction gets the status, mean conversion and label worked out in the docstring."""
    status, mean, label = EXPECTED[group]
    row = labelled.loc[group]
    assert row["status"] == status
    assert row["conversion_mean"] == pytest.approx(mean)
    if label is None:
        assert pd.isna(row["label"])
    else:
        assert row["label"] == label


def test_boundaries_count_as_reached() -> None:
    """10 % counts as 'at least 10 %', and a mean of exactly 20 % is positive."""
    table = pd.DataFrame(
        {"acid": ["a", "a"], "amine": ["n", "n"], "conversion": [10.0, 30.0]}
    )
    row = label_reactions(table, KEYS, RULE).iloc[0]
    assert row["status"] == "all_above_min"
    assert row["label"]


def test_keys_decide_what_counts_as_a_replicate() -> None:
    """The same pair under two reagent strings: two single reactions with the reagent key,
    one inconsistent reaction without it. This is how both HATU drawings become one condition."""
    table = pd.DataFrame(
        {
            "acid": ["a", "a"],
            "amine": ["n", "n"],
            "reagents": ["HATU O-form", "HATU N-oxide"],
            "conversion": [0.0, 31.0],
        }
    )
    by_string = label_reactions(table, ["acid", "amine", "reagents"], RULE)
    as_one = label_reactions(table, KEYS, RULE)
    assert list(by_string["status"]) == ["single", "single"]
    assert list(as_one["status"]) == ["inconsistent"]


def test_config_holds_the_agreed_rule() -> None:
    """The rule in configs/benchmarks/chemlex.yml is the one decided on 2026-10-09."""
    assert load_label_rule(CONFIG) == RULE
