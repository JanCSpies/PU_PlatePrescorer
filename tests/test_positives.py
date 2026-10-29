"""Tests for stage 7a, positives, on a hand-written table of standardised reactions.

Rows (acid + amine, condition, dynamic bonds, source):
- r1  A + B, HATU+DIPEA+DMF,       2, orderly, yield 80, 2010  -> P (family and class level)
- r2  A + B, HATU+DIPEA+DMF+DMAP,  2, uspto,   yield 30, 2005  -> same pair as r1 (extra reagent allowed)
- r3  C + D, HATU+DIPEA+DMF,       4, orderly, yield 75         -> rejected: dynamic_bonds
- r4  succinic acid + B, EDC+DMF,  2, orderly, yield 75         -> rejected: multiple_groups (two COOH)
- r5  E + F, EDC+DMF,              2, orderly, yield 75         -> class level only
- r6  E + G, EDC+DMF,              2, pistachio (licensed), yield 75 -> licensed pair

Every row carries a yield well above the production cut-off, so these tests do not depend on it.
"""

import dataclasses
from pathlib import Path

import pandas as pd
import pytest
from positives import sizes

from platepu.classify.engine import load_class
from platepu.io import strictest_licence
from platepu.positives.build import build_positives, load_positives_config
from platepu.standardise.molecule import load_std_config, standardize

ROOT = Path(__file__).parents[1]
CLASS = ROOT / "configs" / "classes" / "amide_coupling.yml"
STD = load_std_config(ROOT / "configs" / "standardise.yml")
RULE = load_class(CLASS)
CONFIG = load_positives_config(CLASS, STD)

HATU = ["CCN(C(C)C)C(C)C", "CN(C)C(n1n[n+]([O-])c2ncccc21)=[N+](C)C", "CN(C)C=O"]
EDC = ["CCN=C=NCCCN(C)C", "CN(C)C=O"]
A, B, C, D = "CC(=O)O", "NCc1ccccc1", "OC(=O)c1ccccc1", "CNC"
E, F, G, SUCCINIC = "OC(=O)CC1CC1", "NC1CCCC1", "NCCO", "OC(=O)CCC(=O)O"


def _row(
    rid: str, acid: str, amine: str, cond: list[str], dyn: int, **extra: object
) -> dict[str, object]:
    """One standardised reaction with keys computed like stage 6."""
    a, b = standardize(acid, "substrate", STD), standardize(amine, "substrate", STD)
    return {
        "record_id": rid,
        "source": extra.get("source", "orderly"),
        "licence": extra.get(
            "licence", "licensed" if extra.get("source") == "pistachio" else "open"
        ),
        "class": "amide_coupling",
        "std_version": STD.std_version,
        "substrate_a": a.smiles,
        "key_a_stereo": a.key_stereo,
        "key_a_flat": a.key_flat,
        "substrate_b": b.smiles,
        "key_b_stereo": b.key_stereo,
        "key_b_flat": b.key_flat,
        "condition_key": cond,
        "n_reactants": 2,
        "n_products": 1,
        "n_dynamic_bonds": dyn,
        "yield_pct": extra.get("yield_pct"),
        "year": extra.get("year"),
    }


REACTIONS = pd.DataFrame(
    [
        _row("orderly:1", A, B, HATU, 2, yield_pct=80.0, year=2010),
        _row(
            "uspto:1",
            A,
            B,
            [*HATU, "CN(C)c1ccncc1"],
            2,
            source="uspto",
            yield_pct=30.0,
            year=2005,
        ),
        _row("orderly:3", C, D, HATU, 4, yield_pct=75.0),
        _row("orderly:4", SUCCINIC, B, EDC, 2, yield_pct=75.0),
        _row("orderly:5", E, F, EDC, 2, yield_pct=75.0),
        _row("pistachio:6", E, G, EDC, 2, source="pistachio", yield_pct=75.0),
    ]
)
BENCHMARK = pd.DataFrame(
    {
        "key_a_flat": [standardize(A, "substrate", STD).key_flat],
        "key_b_flat": [standardize(B, "substrate", STD).key_flat],
    }
)


Built = tuple[pd.DataFrame, pd.DataFrame]


@pytest.fixture(scope="module")
def built() -> Built:
    """Positives and rejects of the hand-written table (the leakage counts are tested apart)."""
    return build_positives(REACTIONS, RULE, CONFIG, BENCHMARK)[:2]


def test_family_takes_extra_reagents(built: Built) -> None:
    """HATU_DIPEA_DMF holds the A + B pair, reported twice (one report adds DMAP)."""
    positives, _ = built
    family = positives[positives["family"] == "HATU_DIPEA_DMF"]
    assert len(family) == 1
    pair = family.iloc[0]
    assert (
        pair["n_reports"],
        pair["n_sources"],
        pair["first_year"],
        pair["max_yield"],
    ) == (2, 2, 2005, 80.0)


def test_class_level_has_every_kept_pair(
    built: Built,
) -> None:
    """'all' = any condition: A+B, E+F, E+G."""
    positives, _ = built
    assert (positives["family"] == "all").sum() == 3


def test_rejects_have_reasons(built: Built) -> None:
    """Too many dynamic bonds, and a substrate with two groups of its role."""
    _, rejected = built
    assert dict(zip(rejected["record_id"], rejected["reason"], strict=True)) == {
        "orderly:3": "dynamic_bonds",
        "orderly:4": "multiple_groups",
    }


def test_licensed_source_makes_the_pair_licensed(
    built: Built,
) -> None:
    """A pair reported by Pistachio is licensed; open pairs stay open."""
    positives, _ = built
    licence = positives[positives["family"] == "all"].set_index("substrate_b")[
        "licence"
    ]
    assert licence["NCCO"] == "licensed"
    assert licence["NCc1ccccc1"] == "open"


def pair_licence(*licences: str) -> str:
    """Licence of the one A + B pair reported once under each given licence."""
    rows = pd.DataFrame(
        [
            _row(f"orderly:{i}", A, B, HATU, 2, licence=x, yield_pct=75.0)
            for i, x in enumerate(licences)
        ]
    )
    positives = build_positives(rows, RULE, CONFIG, BENCHMARK)[0]
    return str(positives[positives["family"] == "all"]["licence"].iloc[0])


def test_noncommercial_report_makes_the_pair_noncommercial() -> None:
    """ADR 009 item 5: an open and a noncommercial report of one pair merge to noncommercial; the
    old rule (licensed or else open) relabelled it open."""
    assert pair_licence("open", "noncommercial") == "noncommercial"


def test_licensed_report_beats_noncommercial() -> None:
    """Strictest wins: Pistachio (licensed) is stricter than Chemlex (noncommercial)."""
    assert pair_licence("noncommercial", "licensed", "open") == "licensed"


def test_internal_report_beats_everything() -> None:
    """The group's own data is the strictest label."""
    assert pair_licence("open", "noncommercial", "internal") == "internal"


def test_unknown_licence_in_a_report_stops_the_build() -> None:
    """A value outside the precedence is an error, not 'open' by default."""
    with pytest.raises(ValueError, match="cc_by"):
        pair_licence("open", "cc_by")


def test_positives_carry_no_benchmark_derived_column() -> None:
    """ADR 009: a per-row flag 'this substrate is in Chemlex' would sit in an open table."""
    result = build_positives(REACTIONS, RULE, CONFIG, BENCHMARK)
    assert len(result) == 3
    assert not {"leak_pairs_one_side", "leak_pairs_both_sides"} & set(result[0].columns)
    assert set(result[2]) == {"leak_pairs_one_side", "leak_pairs_both_sides"}


def _leak_counts(benchmark: pd.DataFrame) -> dict[str, int]:
    """Leakage counts of three class-level pairs A+B, E+F, C+G (no family matches the condition)."""
    rows = pd.DataFrame(
        [
            _row("orderly:1", A, B, ["CCO"], 2, yield_pct=75.0),
            _row("orderly:2", E, F, ["CCO"], 2, yield_pct=75.0),
            _row("orderly:3", C, G, ["CCO"], 2, yield_pct=75.0),
        ]
    )
    return build_positives(rows, RULE, CONFIG, benchmark)[2]


def test_leakage_counts_one_pair_and_three_substrates() -> None:
    """A+B is a benchmark pair (both sides); E+F shares only the acid E, C+G only the amine G."""
    key = lambda smiles: standardize(smiles, "substrate", STD).key_flat
    benchmark = pd.DataFrame(
        {
            "key_a_flat": [key(A), key(E), "made_up_acid"],
            "key_b_flat": [key(B), key(D), key(G)],
        }
    )
    assert _leak_counts(benchmark) == {
        "leak_pairs_one_side": 3,
        "leak_pairs_both_sides": 1,
    }


NO_OVERLAP = pd.DataFrame(
    {"key_a_flat": ["made_up_acid"], "key_b_flat": ["made_up_amine"]}
)


def test_empty_benchmark_stops_the_build_instead_of_reporting_zero_leakage() -> None:
    """No --benchmark-labels, or an empty label file, gave {leak_pairs_one_side: 0, leak_pairs_both_sides: 0}: the
    report then said 'no overlap with Chemlex', indistinguishable from 'checked, none found', and
    CLAUDE.md §6 requires that number. (Until 2026-10-22 this case was pinned as returning zeros.)"""
    empty = pd.DataFrame(columns=["key_a_flat", "key_b_flat"])
    with pytest.raises(ValueError, match="benchmark"):
        build_positives(REACTIONS, RULE, CONFIG, empty)


def test_benchmark_without_overlap_reports_zero_leakage() -> None:
    """Control: a benchmark that was supplied and shares nothing is a real 'checked, none found'."""
    assert _leak_counts(NO_OVERLAP) == {
        "leak_pairs_one_side": 0,
        "leak_pairs_both_sides": 0,
    }
    positives = build_positives(REACTIONS, RULE, CONFIG, NO_OVERLAP)[0]
    assert not {"leak_pairs_one_side", "leak_pairs_both_sides"} & set(positives.columns)


def test_open_positives_keep_licence_open(built: Built) -> None:
    """Regression pin (ADR 009): open reports give an open pair; strictest_licence merges them."""
    positives, _ = built
    open_pair = positives[positives["family"] == "HATU_DIPEA_DMF"]
    assert set(open_pair["licence"]) == {"open"}
    assert strictest_licence(["open", "noncommercial"]) == "noncommercial"


def test_positives_carry_std_version(built: Built) -> None:
    """CLAUDE.md section 6: every row carries std_version; each pair takes its reactions' one value."""
    positives, _ = built
    assert set(positives["std_version"]) == {STD.std_version}


def test_mixed_std_version_fails_loudly() -> None:
    """Reactions standardised with two versions must not be pooled into one positives table."""
    mixed = REACTIONS.copy()
    mixed.loc[0, "std_version"] = "other"
    with pytest.raises(ValueError, match="std_version"):
        build_positives(mixed, RULE, CONFIG, BENCHMARK)


def _yield_rows() -> pd.DataFrame:
    """Three class-level pairs: A+B yield 80, E+F yield 5, C+G yield unknown (ORDerly without yield_000)."""
    return pd.DataFrame(
        [
            _row("orderly:1", A, B, ["CCO"], 2, yield_pct=80.0),
            _row("orderly:2", E, F, ["CCO"], 2, yield_pct=5.0),
            _row("orderly:3", C, G, ["CCO"], 2),
        ]
    )


def _cut_at(minimum: float | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = dataclasses.replace(CONFIG, min_yield_pct=minimum)
    return build_positives(_yield_rows(), RULE, config, NO_OVERLAP)[:2]


def test_unknown_yield_is_rejected_by_a_yield_cutoff() -> None:
    """With a cut-off of 10 % only the 80 % pair may stay. The unknown-yield reaction passed every
    cut-off (notna() & (< minimum)), so 'pairs with yield at least 10 %' counted it: 2, not 1.
    Decision: a reaction that cannot show it reaches the cut-off is rejected as ``no_yield``,
    apart from ``below_min_yield``, so the rejects report says how many were dropped for want of a yield."""
    positives, rejected = _cut_at(10.0)
    assert list(positives["substrate_a"]) == [standardize(A, "substrate", STD).smiles]
    assert dict(zip(rejected["record_id"], rejected["reason"], strict=True)) == {
        "orderly:2": "below_min_yield",
        "orderly:3": "no_yield",
    }


def test_unknown_yield_is_kept_when_there_is_no_cutoff() -> None:
    """Control: min_yield_pct null means no cut-off, so all three pairs stay and nothing is rejected."""
    positives, rejected = _cut_at(None)
    assert len(positives) == 3
    assert rejected.empty


def test_sizes_by_yield_cutoff_count_only_known_yields() -> None:
    """The report's min_yield_10 / min_yield_20 are the numbers that choose the cut-off; each must be
    the pairs with a known yield of at least that much (1 here), min_yield_0 all of them (3)."""
    assert sizes(_yield_rows(), RULE, CONFIG, NO_OVERLAP) == {
        "min_yield_0": {"all": 3},
        "min_yield_10": {"all": 1},
        "min_yield_20": {"all": 1},
    }


# --- yield rule (agreed 2026-10-28): usable only in (0, 100] ------------------------------------
# A yield outside the interval is evidence of neither success nor failure. Zero is outside it on
# purpose: 6 rows in the class, and a branch for them would be a branch for a case the data lacks.
# The rows are hand-written; the real-data counts (11,396 / 1,125 / 15,637) belong to the
# end-to-end run, not here.

CUTOFF_20 = dataclasses.replace(CONFIG, min_yield_pct=20.0)
NO_CUTOFF = dataclasses.replace(CONFIG, min_yield_pct=None)


def _reports(pair: tuple[str, str], *yields: float | None) -> list[dict[str, object]]:
    """One reaction per yield of the same acid + amine pair (class level, condition ethanol)."""
    acid, amine = pair
    return [
        _row(f"orderly:{acid[:6]}{amine[:6]}{i}", acid, amine, ["CCO"], 2, yield_pct=y)
        for i, y in enumerate(yields)
    ]


def _reasons(config: object, *yields: float | None) -> list[object]:
    """Reject reasons of A + B reported once per yield, in order."""
    rows = pd.DataFrame(_reports((A, B), *yields))
    rejected = build_positives(rows, RULE, config, NO_OVERLAP)[1]  # type: ignore[arg-type]
    by_id = dict(zip(rejected["record_id"], rejected["reason"], strict=True))
    return [by_id.get(record_id, "") for record_id in rows["record_id"]]


def _labels(config: object, *reactions: list[dict[str, object]]) -> dict[str, str]:
    """Label of each pair in ``reported``, keyed by its acid SMILES (one pair per acid here)."""
    from platepu.positives.build import build_reported

    rows = pd.DataFrame([row for group in reactions for row in group])
    reported = build_reported(rows, RULE, config)  # type: ignore[arg-type]
    assert not reported.duplicated(["key_a_stereo", "key_b_stereo"]).any()
    return dict(zip(reported["substrate_a"], reported["label"], strict=True))


def _smiles(acid: str) -> str:
    return standardize(acid, "substrate", STD).smiles  # type: ignore[return-value]


def test_yield_of_exactly_20_is_positive() -> None:
    """The lower boundary is inclusive: 20 % with a cut-off of 20 is kept and labelled positive."""
    assert _reasons(CUTOFF_20, 20.0) == [""]
    assert _labels(CUTOFF_20, _reports((A, B), 20.0)) == {_smiles(A): "positive"}


def test_yield_of_exactly_100_is_usable() -> None:
    """The upper boundary is inclusive: 100 % is a real yield, kept and positive."""
    assert _reasons(CUTOFF_20, 100.0) == [""]
    assert _labels(CUTOFF_20, _reports((A, B), 100.0)) == {_smiles(A): "positive"}


def test_yield_just_above_100_is_out_of_range() -> None:
    """100.1 % cannot be a yield; it is rejected with its own reason, not kept."""
    assert _reasons(CUTOFF_20, 100.1) == ["yield_out_of_range"]


def test_yield_of_zero_is_out_of_range_not_negative() -> None:
    """Luka's decision: a reported 0 % is outside (0, 100], so unusable, and not a negative."""
    assert _reasons(CUTOFF_20, 0.0) == ["yield_out_of_range"]
    assert _labels(CUTOFF_20, _reports((A, B), 0.0)) == {_smiles(A): "unlabeled"}


def test_out_of_range_beats_below_min_yield() -> None:
    """np.select takes the first branch that applies. 150 is not below 20 anyway, but -5 is, and
    the reason must still be yield_out_of_range: the branch goes before below_min_yield."""
    assert _reasons(CUTOFF_20, 150.0, -5.0) == ["yield_out_of_range"] * 2


def test_out_of_range_applies_without_a_cutoff() -> None:
    """An impossible yield is impossible with min_yield_pct = None as well; 56,524,596 is not kept."""
    assert _reasons(NO_CUTOFF, 56524596.0, 0.0, 50.0) == [
        "yield_out_of_range",
        "yield_out_of_range",
        "",
    ]


def test_pair_reported_below_and_above_the_cutoff_is_positive() -> None:
    """15 % and 60 % on one pair: the pair survives on its best report, so it is positive."""
    assert _labels(CUTOFF_20, _reports((A, B), 15.0, 60.0)) == {_smiles(A): "positive"}


def test_pair_rescued_only_by_an_impossible_yield_is_not_positive() -> None:
    """15 % and 150 %: the 150 is unusable, so nothing shows success and the 15 is not 'every
    report usable' -- the pair is unlabeled, neither positive nor negative (447 pairs in the data)."""
    assert _labels(CUTOFF_20, _reports((A, B), 15.0, 150.0)) == {
        _smiles(A): "unlabeled"
    }


def test_pair_with_every_report_below_the_cutoff_is_negative() -> None:
    """Two usable reports, 5 % and 15 %, both below 20: a known negative."""
    assert _labels(CUTOFF_20, _reports((A, B), 5.0, 15.0)) == {_smiles(A): "negative"}


def test_pair_with_no_yield_is_unlabeled() -> None:
    """Nobody recorded a yield: no evidence either way. A single missing report also disqualifies
    a negative (5 % and unknown)."""
    assert _labels(CUTOFF_20, _reports((A, B), None)) == {_smiles(A): "unlabeled"}
    assert _labels(CUTOFF_20, _reports((E, F), 5.0, None)) == {_smiles(E): "unlabeled"}


def test_every_reported_pair_gets_exactly_one_label() -> None:
    """Four pairs -- positive, negative, unlabeled (no yield), unlabeled (only 150) -- each appear
    once with one label: nothing dropped, nothing double-counted, and the labels add up to the pairs."""
    from platepu.positives.build import build_reported

    rows = pd.DataFrame(
        [
            *_reports((A, B), 80.0),
            *_reports((E, F), 5.0),
            *_reports((C, G), None),
            *_reports((A, F), 150.0),
        ]
    )
    reported = build_reported(rows, RULE, CUTOFF_20)
    assert len(reported) == 4
    assert reported["label"].value_counts().to_dict() == {
        "positive": 1,
        "negative": 1,
        "unlabeled": 2,
    }


def test_structurally_rejected_pair_is_absent_from_reported() -> None:
    """Succinic acid has two COOH (multiple_groups): the pair never reaches the labelled table,
    whatever its yield."""
    labels = _labels(CUTOFF_20, _reports((A, B), 80.0), _reports((SUCCINIC, B), 80.0))
    assert labels == {_smiles(A): "positive"}
