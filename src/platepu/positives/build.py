"""Positives per class and condition family (docs/design/rebuild.md, Stage 7a; ADR 002).

Input: standardised reactions of one or more sources (stage 6) with ``source``, ``yield_pct`` and
``year`` joined from stage 1. Filters, in order: dynamic bonds not allowed by the class, not exactly
one product, not exactly two contributing reactants, a substrate without exactly one group of its
own role or with any group of the other role, a yield outside ``(0, 100]``, yield below the
cut-off, no yield at all. Then one row per (class, family, substrate pair)
with provenance; ``family = "all"`` is the class level (any condition). Leakage against the
benchmark keys (stereo-agnostic) is **counted and returned**, never written onto a row: a per-row
boolean derived from a CC BY-NC-ND benchmark has no place in a table licensed ``open`` (ADR 010).

``build_reported`` writes the second table the cut-off makes necessary: every pair that survives
the structural filters, whatever its yields, labelled ``positive``, ``negative`` or ``unlabeled``.
Stage U subtracts all of it, and the spy step reads the negatives.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from platepu.classify.engine import ClassRule
from platepu.classify.groups import group_counts
from platepu.io import strictest_licence
from platepu.standardise.molecule import StdConfig, in_family, standardize

PAIR = ["class", "family", "key_a_stereo", "key_b_stereo"]
# The filters that decide whether a reaction belongs to the class at all, before any yield is read.
STRUCTURAL_REASONS = (
    "dynamic_bonds",
    "n_products",
    "extra_reactant",
    "multiple_groups",
)
# A yield is evidence only inside this interval. Outside it the number says nothing either way:
# above 100 it is not a yield, and a reported 0 is far more often an empty field than a measurement.
MIN_USABLE_YIELD = 0.0
MAX_USABLE_YIELD = 100.0


@dataclass(frozen=True)
class PositivesConfig:
    """Positives settings of one class.

    Attributes:
        families: Family name → standardised reagents a condition key must all contain.
        min_yield_pct: Yield cut-off, or None for no cut-off (reactions without a yield are kept).
        single_reactive_group: Apply the single-reactive-group filter to both substrates.
    """

    families: Mapping[str, frozenset[str]]
    min_yield_pct: float | None
    single_reactive_group: bool


def load_positives_config(class_path: Path, std: StdConfig) -> PositivesConfig:
    """Read ``condition_families`` and ``positives`` from a class YAML; family SMILES are standardised."""
    config = yaml.safe_load(class_path.read_text(encoding="utf-8"))
    families = {
        name: frozenset(_agent(smiles, std) for smiles in members)
        for name, members in config["condition_families"].items()
    }
    settings = config["positives"]
    minimum = settings["min_yield_pct"]
    return PositivesConfig(
        families=families,
        min_yield_pct=float(minimum) if minimum is not None else None,
        single_reactive_group=bool(settings["single_reactive_group"]),
    )


def build_positives(
    reactions: pd.DataFrame,
    rule: ClassRule,
    config: PositivesConfig,
    benchmark: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Filter, assign families, deduplicate pairs and count the benchmark overlap.

    Args:
        reactions: Stage-6 rows of ``rule``'s class with ``source``, ``yield_pct`` and ``year``.
        rule: The class (allowed dynamic bonds, role SMARTS).
        config: The class's positives settings.
        benchmark: Benchmark pairs as ``key_a_flat``, ``key_b_flat``.

    Returns:
        The positives (one row per class, family and pair), the rejected reactions with a reason,
        and how many **distinct pairs** overlap ``benchmark``, as ``leak_substrate`` (one side is in
        it) and ``leak_pair`` (both are). The counts are returned rather than written onto the rows:
        a per-row boolean saying "this substrate also occurs in Chemlex" is a fact derived from a
        CC BY-NC-ND dataset, and it has no place in a table licensed ``open`` (ADR 010). The count
        is aggregate and cannot be inverted to a substrate, so it stays, because CLAUDE.md §6 asks
        for the overlap to be reported.

    Raises:
        ValueError: The reactions come from more than one ``std_version``; their keys would not
            agree, so pairs could split or merge silently.
    """
    versions = sorted(set(reactions["std_version"]))
    if len(versions) > 1:
        raise ValueError(f"reactions mix std_version values {versions}")
    reason = _reject_reasons(reactions, rule, config)
    kept = reactions[reason == ""]
    per_family = [kept.assign(family="all")]
    for name, members in config.families.items():
        member = kept["condition_key"].map(lambda key, m=members: in_family(key, m))
        per_family.append(kept[member].assign(family=name))
    pairs = _pairs(pd.concat(per_family, ignore_index=True))
    leakage = _count_leakage(pairs, benchmark)
    rejected = pd.DataFrame({"record_id": reactions["record_id"], "reason": reason})
    kept_rejected = rejected[rejected["reason"] != ""].reset_index(drop=True)
    return pairs, kept_rejected, leakage


def build_reported(
    reactions: pd.DataFrame, rule: ClassRule, config: PositivesConfig
) -> pd.DataFrame:
    """Every pair reported for the class, labelled by what its yields support.

    Once a cut-off is set, ``build_positives`` writes P alone and the other reported pairs have no
    home. They are needed twice: stage U must subtract **every** reported pair from the unlabeled
    draw, not only the ones that became P, or it draws known reactions as if nothing were known
    about them; and the spy step reads the known negatives, the only real ones the literature gives.

    Args:
        reactions: Stage-6 rows of ``rule``'s class, as ``build_positives`` takes them.
        rule: The class (allowed dynamic bonds, role SMARTS).
        config: The class's positives settings; ``min_yield_pct`` has to be set.

    Returns:
        One row per ``(key_a_stereo, key_b_stereo)`` surviving the **structural** filters, whatever
        its yields, with ``label``:

        - ``positive``: at least one report with a usable yield of at least the cut-off;
        - ``negative``: every report has a usable yield and every one is below the cut-off;
        - ``unlabeled``: anything else.

        A negative is strict on purpose. One missing or out-of-range report disqualifies it, because
        a yield nobody recorded, or one reading 56,524,596 %, is not evidence that the pair failed.
        ``max_yield`` is the largest **usable** yield, so the values the label ignores cannot come
        back through the column.

    Raises:
        ValueError: ``config.min_yield_pct`` is None, which leaves every label undefined.
    """
    if config.min_yield_pct is None:
        raise ValueError(
            "a labelled table needs min_yield_pct, and the class config leaves it null"
        )
    structural = np.select(
        _structural_masks(reactions, rule, config), STRUCTURAL_REASONS, default=""
    )
    kept = reactions[structural == ""].copy()
    kept["usable_yield"] = kept["yield_pct"].where(_usable(kept["yield_pct"]))
    reported = (
        kept.groupby(["key_a_stereo", "key_b_stereo"], sort=False)
        .agg(
            key_a_flat=("key_a_flat", "first"),
            key_b_flat=("key_b_flat", "first"),
            substrate_a=("substrate_a", "first"),
            substrate_b=("substrate_b", "first"),
            n_reports=("record_id", "size"),
            n_usable=("usable_yield", "count"),
            max_yield=("usable_yield", "max"),
            licence=("licence", strictest_licence),
            std_version=("std_version", "first"),
        )
        .reset_index()
    )
    positive = reported["max_yield"] >= config.min_yield_pct
    every_report_readable = reported["n_usable"] == reported["n_reports"]
    reported["label"] = np.select(
        [positive, every_report_readable & ~positive],
        ["positive", "negative"],
        default="unlabeled",
    )
    return reported.drop(columns=["n_usable"])


def _usable(yield_pct: pd.Series) -> pd.Series:
    """Whether each yield lies in ``(0, 100]``, the range in which it means anything."""
    return (yield_pct > MIN_USABLE_YIELD) & (yield_pct <= MAX_USABLE_YIELD)


def _structural_masks(
    reactions: pd.DataFrame, rule: ClassRule, config: PositivesConfig
) -> list[pd.Series | np.ndarray]:
    """One mask per entry of ``STRUCTURAL_REASONS``, in that order."""
    counts: dict[
        str, tuple[int, int]
    ] = {}  # molecules repeat a lot: count each once per call
    groups = [
        config.single_reactive_group and not _single_group(a, b, rule, counts)
        for a, b in zip(reactions["substrate_a"], reactions["substrate_b"], strict=True)
    ]
    return [
        ~reactions["n_dynamic_bonds"].isin(rule.allowed_dynamic_bonds),
        reactions["n_products"] != 1,
        reactions["n_reactants"] != 2,
        np.array(groups, dtype=bool),
    ]


def _reject_reasons(
    reactions: pd.DataFrame, rule: ClassRule, config: PositivesConfig
) -> np.ndarray:
    """The first reason that applies to each reaction, or "" when it is kept."""
    yields = reactions["yield_pct"]
    minimum = config.min_yield_pct
    never = pd.Series(False, index=reactions.index)
    # A reaction with no yield used to survive every cut-off, because the filter was
    # ``notna() & (yield < minimum)``. "Pairs with a yield of at least 20 %" then also held every
    # reaction whose yield nobody recorded -- and those counts are what will choose the cut-off.
    return np.select(
        [
            *_structural_masks(reactions, rule, config),
            # Before the cut-off, not after it: 56,524,596 % is not "at least 20 %", it is a mass
            # that landed in the percent field. Appending this branch below ``below_min_yield``
            # would let every such reaction through as a positive. 709 reactions of this class
            # carry a yield above 100 and 6 carry a 0, counted 2026-10-28.
            yields.notna() & ~_usable(yields),
            yields.notna() & _usable(yields) & (yields < minimum)
            if minimum is not None
            else never,
            yields.isna() if minimum is not None else never,
        ],
        [
            *STRUCTURAL_REASONS,
            "yield_out_of_range",
            "below_min_yield",
            "no_yield",
        ],
        default="",
    )


def _single_group(
    substrate_a: str,
    substrate_b: str,
    rule: ClassRule,
    counts: dict[str, tuple[int, int]],
) -> bool:
    """Each substrate has exactly one group of its own role and none of the other role."""
    return _counts(substrate_a, rule, counts) == (1, 0) and _counts(
        substrate_b, rule, counts
    ) == (0, 1)


def _counts(
    smiles: str, rule: ClassRule, counts: dict[str, tuple[int, int]]
) -> tuple[int, int]:
    """(groups of role A, groups of role B) in one molecule, remembered in ``counts``."""
    if smiles not in counts:
        counts[smiles] = group_counts(smiles, rule)
    return counts[smiles]


def _pairs(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (class, family, pair) with where and how often it was reported."""
    return (
        rows.groupby(PAIR, sort=False)
        .agg(
            substrate_a=("substrate_a", "first"),
            substrate_b=("substrate_b", "first"),
            key_a_flat=("key_a_flat", "first"),
            key_b_flat=("key_b_flat", "first"),
            n_reports=("record_id", "size"),
            n_sources=("source", "nunique"),
            sources=("source", lambda s: sorted(set(s))),
            record_ids=("record_id", list),
            first_year=("year", "min"),
            max_yield=("yield_pct", "max"),
            licence=("licence", strictest_licence),
            std_version=(
                "std_version",
                "first",
            ),  # one value, checked in build_positives
        )
        .reset_index()
    )


def _count_leakage(pairs: pd.DataFrame, benchmark: pd.DataFrame) -> dict[str, int]:
    """How many distinct pairs overlap ``benchmark``: one side (``leak_substrate``), both (``leak_pair``).

    Counted on the ``all`` family only, which holds every kept pair exactly once. Counting every row
    would count a pair once per condition family it belongs to, so the old figure was pair-rows and
    not pairs, which is not a number anybody could interpret.
    """
    if benchmark.empty:
        raise ValueError(
            "no benchmark pairs to count the overlap against: zero would be indistinguishable "
            "from 'checked, none found', and CLAUDE.md §6 asks for the overlap to be reported"
        )
    keys_a, keys_b = set(benchmark["key_a_flat"]), set(benchmark["key_b_flat"])
    both = set(zip(benchmark["key_a_flat"], benchmark["key_b_flat"], strict=True))
    distinct = pairs[pairs["family"] == "all"]
    one_side = distinct["key_a_flat"].isin(keys_a) | distinct["key_b_flat"].isin(keys_b)
    return {
        "leak_substrate": int(one_side.sum()),
        "leak_pair": sum(
            pair in both
            for pair in zip(distinct["key_a_flat"], distinct["key_b_flat"], strict=True)
        ),
    }


def _agent(smiles: str, std: StdConfig) -> str:
    """Standardised agent SMILES of a family member; a typo stops loading."""
    result = standardize(smiles, "agent", std).smiles
    if result is None:
        raise ValueError(f"unparsable SMILES in condition_families: {smiles}")
    return result
