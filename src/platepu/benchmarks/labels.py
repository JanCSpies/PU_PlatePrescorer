"""Test labels from HTE conversions: one label per reaction, replicates combined by a fixed rule.

Rule (docs/design/rebuild.md, Stage 7b; values in configs/benchmarks/<name>.yml, ``label_rule``): replicates are kept
only when all are 0 % or all are at least ``replicate_min_pct``; anything else is inconsistent and
gets no label. A kept reaction is positive when its mean conversion is at least
``positive_min_pct``. Applies to LC-MS conversions only, never to isolated yields.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class LabelRule:
    """The two cut-offs of the labelling rule, in percent conversion.

    Attributes:
        positive_min_pct: A reaction is positive when its mean conversion is at least this.
        replicate_min_pct: Replicates are kept when all are 0 % or all are at least this.
    """

    positive_min_pct: float
    replicate_min_pct: float


def load_label_rule(path: Path) -> LabelRule:
    """Read ``label_rule`` from a benchmark YAML; it must hold exactly the two ``LabelRule`` keys."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))["label_rule"]
    return LabelRule(**{key: float(value) for key, value in config.items()})


def label_reactions(
    table: pd.DataFrame, keys: Sequence[str], rule: LabelRule
) -> pd.DataFrame:
    """Combine the measurements of each reaction into one row with a status and a label.

    Args:
        table: One row per measurement, with the ``keys`` columns and ``conversion`` (percent).
        keys: Columns that identify a reaction; rows sharing them are replicates.
        rule: The cut-offs.

    Returns:
        One row per reaction: the ``keys``, ``n_measurements``, ``conversion_mean``, ``status``
        (``single``, ``all_zero``, ``all_above_min`` or ``inconsistent``) and ``label``
        (nullable boolean; missing for ``inconsistent``).
    """
    grouped = table.groupby(list(keys), sort=False)["conversion"]
    reactions = grouped.agg(
        n_measurements="size", conversion_mean="mean", low="min", high="max"
    ).reset_index()
    reactions["status"] = np.select(
        [
            reactions["n_measurements"] == 1,
            reactions["high"] == 0,
            reactions["low"] >= rule.replicate_min_pct,
        ],
        ["single", "all_zero", "all_above_min"],
        default="inconsistent",
    )
    positive = (reactions["conversion_mean"] >= rule.positive_min_pct).astype("boolean")
    reactions["label"] = positive.mask(reactions["status"] == "inconsistent")
    return reactions.drop(columns=["low", "high"])
