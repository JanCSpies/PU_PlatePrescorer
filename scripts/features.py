"""Stage F: one row of features per substrate (docs/design/modeling.md, "Stage F — features").

Reads the positives (``<root>/positives/<class>/positives.parquet``, every family), the library
(``<root>/private/library/<class>.parquet``) and a benchmark validation file. Writes
``<root>/private/features/<class>/substrates.parquet`` (under private/: it holds internal library
rows, ADR 004) and ``.report.json`` with descriptor ranges per input, the domain-shift record.

Usage:
    uv run python scripts/features.py amide_coupling ROOT --validation VALIDATION.parquet
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from platepu.classify.engine import load_class
from platepu.features.substrate import (
    DESCRIPTORS,
    SOURCES,
    load_feature_config,
    substrate_table,
)
from platepu.io import write_report, write_table


def to_arrow(table: pd.DataFrame, widths: dict[str, int]) -> pa.Table:
    """The table with each fingerprint column as a fixed-size list of uint8."""
    arrow = pa.Table.from_pandas(table.drop(columns=list(widths)), preserve_index=False)
    for column, n_bits in widths.items():
        counts = np.concatenate(table[column].to_list())
        arrow = arrow.append_column(
            column,
            pa.FixedSizeListArray.from_arrays(
                pa.array(counts, type=pa.uint8()), n_bits
            ),
        )
    return arrow


def by_input(table: pd.DataFrame) -> dict[str, object]:
    """Per input and role: keys, subclass counts and min / median / max of each descriptor."""
    report: dict[str, object] = {}
    for source in SOURCES:
        for role, rows in table[table[f"in_{source}"]].groupby("role"):
            stats = rows[list(DESCRIPTORS)].agg(["min", "median", "max"])
            report[f"{source}_{role}"] = {
                "keys": len(rows),
                "subclasses": rows["subclass"].value_counts().to_dict(),
                "descriptors": {
                    name: [float(value) for value in stats[name]]
                    for name in DESCRIPTORS
                },
            }
    return report


def overlaps(table: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Per role: keys seen in two inputs."""
    report = {}
    for role, rows in table.groupby("role"):
        report[str(role)] = {
            f"{first}_{second}": int((rows[f"in_{first}"] & rows[f"in_{second}"]).sum())
            for first, second in (
                ("positives", "library"),
                ("positives", "validation"),
                ("library", "validation"),
            )
        }
    return report


def morgan_summary(table: pd.DataFrame, column: str) -> dict[str, float | int]:
    """Largest count, mean number of set bits per substrate, bits never set, of one Morgan block."""
    counts = np.stack(table[column].to_list())
    return {
        "max_count": int(counts.max()),
        "mean_nonzero_bits": round(float((counts > 0).sum(axis=1).mean()), 2),
        "bits_never_set": int((counts.sum(axis=0) == 0).sum()),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Build the feature table of one class and its report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name")
    parser.add_argument("root", type=Path)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args(argv)
    # benchmark_split.py writes test.parquet next to it; the test side is read once, at the end
    if args.validation.name != "validation.parquet":
        raise ValueError(
            f"--validation must be a validation.parquet: {args.validation}"
        )

    class_path = Path(f"configs/classes/{args.name}.yml")
    config_path = Path("configs/features.yml")
    rule, config = load_class(class_path), load_feature_config(config_path)
    positives = pd.read_parquet(
        args.root / "positives" / args.name / "positives.parquet"
    )
    library = pd.read_parquet(
        args.root / "private" / "library" / f"{args.name}.parquet"
    )
    validation = pd.read_parquet(args.validation)
    table = substrate_table(positives, library, validation, rule, config)
    out = args.root / "private" / "features" / args.name / "substrates.parquet"
    widths = {"morgan": config.morgan_bits, "morgan_centre": config.centre_bits}
    write_table(to_arrow(table, widths), out, root=args.root)

    report: dict[str, object] = {
        "stage": "features",
        "class": args.name,
        "validation": str(args.validation),
        "rows_in": {
            "positives": len(positives),
            "library": len(library),
            "validation": len(validation),
        },
        "rows_out": table["role"].value_counts().to_dict(),
        "std_version": sorted(set(table["std_version"])),
        "keys_with_several_smiles": sorted(
            table.loc[table["n_smiles"] > 1, "key_stereo"]
        ),
        "overlaps": overlaps(table),
        "by_input": by_input(table),
        "morgan": morgan_summary(table, "morgan"),
        "morgan_centre": morgan_summary(table, "morgan_centre"),
    }
    write_report(out.with_suffix(".report.json"), report, [class_path, config_path])
    print(f"{args.name}: {len(table):,} substrates -> {out}")


if __name__ == "__main__":
    main()
