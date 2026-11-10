"""Benchmark measurements and their reaction-level labels (docs/design/rebuild.md, Stage 7b).

Roles come from the benchmark's own columns (``reactants[0]``, ``reactants[1]``), the class is checked
with the role SMARTS on those molecules, and molecules are standardised exactly like the positives,
so keys match. The mapping is used only as a check (``n_mapped_in_class``); it never removes a row.
"""

from collections.abc import Collection, Sequence

import pandas as pd
import pyarrow as pa

from platepu.benchmarks.labels import LabelRule, label_reactions
from platepu.classify.engine import ClassRule, fits_roles
from platepu.ingest.record import table_to_records
from platepu.standardise.molecule import (
    StdConfig,
    condition_key,
    condition_text,
    required,
    standardize,
)

REACTION = ["key_a_stereo", "key_b_stereo", "condition"]


def measurements(records: pa.Table, rule: ClassRule, config: StdConfig) -> pd.DataFrame:
    """One row per measured record: standardised roles, condition, conversion and class fit."""
    rows = []
    for record in table_to_records(records):
        if len(record.reactants) != 2:
            raise ValueError(
                f"{record.record_id}: a benchmark record needs exactly two reactants"
            )
        a = required(
            standardize(record.reactants[0], "substrate", config), record.record_id
        )
        b = required(
            standardize(record.reactants[1], "substrate", config), record.record_id
        )
        key = condition_key(record.agents, record.solvents, config)
        rows.append(
            {
                "record_id": record.record_id,
                "substrate_a": a.smiles,
                "key_a_stereo": a.key_stereo,
                "key_a_flat": a.key_flat,
                "substrate_b": b.smiles,
                "key_b_stereo": b.key_stereo,
                "key_b_flat": b.key_flat,
                "condition": condition_text(key),  # unreadable: kept, never matched
                "conversion": record.yield_pct,
                "in_class": fits_roles(*record.reactants, rule),
                "std_version": config.std_version,
            }
        )
    return pd.DataFrame(rows)


def reaction_labels(
    measured: pd.DataFrame,
    splits: pd.DataFrame,
    split_columns: Sequence[str],
    rule: LabelRule,
    mapped_in_class: Collection[str],
) -> pd.DataFrame:
    """Label each in-class reaction (same substrates and condition) from its measurements.

    Args:
        measured: Output of ``measurements``.
        splits: ``source_id``-keyed split labels of the benchmark (``record_id`` column).
        split_columns: Splits that must agree across one reaction's measurements.
        rule: The benchmark's label rule.
        mapped_in_class: record_ids the mapping also put in this class (stage 5), for the check.

    A reaction whose measurements disagree on a split gets None for that split (unusable) and
    ``split_conflict = True``. Seen in Chemlex: acarbose written as two SMILES of one molecule was
    treated as two amines, one in train and one in test of ``one_unseen`` (4 reactions, 2026-10-09).
    """
    rows = measured[measured["in_class"]].merge(
        splits[["record_id", *split_columns]],
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    rows["mapped_in_class"] = rows["record_id"].isin(set(mapped_in_class))
    grouped = rows.groupby(REACTION, sort=False)
    conflict = grouped[list(split_columns)].nunique(dropna=False).gt(1)
    first = grouped.agg(
        substrate_a=("substrate_a", "first"),
        key_a_flat=("key_a_flat", "first"),
        substrate_b=("substrate_b", "first"),
        key_b_flat=("key_b_flat", "first"),
        n_mapped_in_class=("mapped_in_class", "sum"),
        std_version=("std_version", "first"),  # one config per measurements() call
        **{column: (column, "first") for column in split_columns},
    )
    for column in split_columns:
        first[column] = first[column].mask(conflict[column], None)
    first["split_conflict"] = conflict.any(axis=1)
    labels = label_reactions(rows, REACTION, rule)
    return labels.merge(first.reset_index(), on=REACTION, validate="one_to_one")
