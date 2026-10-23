"""Stage 4, clean: rxnutils actions from ``configs/clean.yml`` on the mapped reactions.

Roles come from here for every source (ADR 002): after ``reactants2reagents`` every molecule that
gives no atom to the product is a reagent. The CGR dynamic-bond count is recorded, never filtered on;
the positives stage filters it (benchmarks never are).
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

# Importing these modules registers rxnutils' actions by name, as its own runner does.
import rxnutils.pipeline.actions.reaction_mod
import rxnutils.pipeline.actions.reaction_props  # noqa: F401
import yaml
from rxnutils.pipeline.base import create_action

from platepu.io import REJECTED_SCHEMA

KEEP = [
    "record_id",
    "licence",
    "rsmi_hash",
    "clean_rsmi",
    "n_reactants",
    "n_reagents",
    "n_products",
    "n_unmapped_product_atoms",
    "n_widow_product_atoms",
    "n_dynamic_bonds",
]


def load_pipeline(path: Path) -> dict[str, dict[str, Any]]:
    """The ordered rxnutils actions and their options from a YAML file."""
    pipeline: dict[str, dict[str, Any]] = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )
    return pipeline


def clean_table(
    mapped: pa.Table, pipeline: Mapping[str, Mapping[str, Any]]
) -> tuple[pa.Table, pa.Table]:
    """Run the actions on stage 3's table; reject reactions with nothing left to classify.

    Reasons, first that applies: ``no_product``, ``no_reactant``, ``no_cgr``.
    """
    data = mapped.to_pandas()
    for name, options in pipeline.items():
        data = create_action(name, **options)(data)
    reason = np.select(
        [
            data["n_products"].eq(0),
            data["n_reactants"].eq(0),
            data["n_dynamic_bonds"].isna(),
        ],
        ["no_product", "no_reactant", "no_cgr"],
        default="",
    )
    ok = reason == ""
    kept = data.loc[ok, KEEP].astype({"n_dynamic_bonds": "int64"})
    rejected = pd.DataFrame(
        {"record_id": data.loc[~ok, "record_id"], "reason": reason[~ok]}
    )
    return (
        pa.Table.from_pandas(kept, preserve_index=False),
        pa.Table.from_pandas(rejected, schema=REJECTED_SCHEMA, preserve_index=False),
    )
