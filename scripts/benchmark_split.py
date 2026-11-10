"""Validation and test sets of one benchmark for one class family (docs/design/modeling.md).

Reads ``<root>/private/benchmarks/<name>/labels.parquet``; writes ``<family>/validation.parquet`` and
``<family>/test.parquet`` next to it. The test file is read once, in the final evaluation only.

Usage:
    uv run python scripts/benchmark_split.py chemlex amide_coupling HATU_DIPEA_DMF ROOT
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa

from platepu.benchmarks.split import family_split
from platepu.io import write_report, write_table
from platepu.positives.build import load_positives_config
from platepu.standardise.molecule import load_std_config


def shared_substrates(validation: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    """How many flat substrate keys the two sides share, per role.

    Recounted from the two frames rather than taken from ``family_split``, so a future change to
    the split cannot make its own seal look intact. The caller checks it **before** writing, not
    after: a leak recorded in a report nobody re-reads is not a guard, and once both files are on
    disk the next stage trains on them.
    """
    return {
        f"shared_{role}": len(set(validation[column]) & set(test[column]))
        for role, column in (("acids", "key_a_flat"), ("amines", "key_b_flat"))
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Split the benchmark's labelled reactions of one family into validation and test files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("benchmark")
    parser.add_argument("name", help="class name, e.g. amide_coupling")
    parser.add_argument("family", help="condition family, e.g. HATU_DIPEA_DMF")
    parser.add_argument("root", type=Path)
    parser.add_argument("--split-column", default="both_unseen")
    parser.add_argument(
        "--standardise", type=Path, default=Path("configs/standardise.yml")
    )
    args = parser.parse_args(argv)

    class_path = Path(f"configs/classes/{args.name}.yml")
    families = load_positives_config(
        class_path, load_std_config(args.standardise)
    ).families
    folder = args.root / "private" / "benchmarks" / args.benchmark
    labels = pd.read_parquet(folder / "labels.parquet")
    validation, test = family_split(labels, families[args.family], args.split_column)
    shared = shared_substrates(validation, test)
    if any(shared.values()):
        raise ValueError(
            f"the split leaks a test substrate into validation, so nothing was written: {shared}"
        )
    out = folder / args.family
    for name, part in (("validation", validation), ("test", test)):
        write_table(
            pa.Table.from_pandas(part, preserve_index=False),
            out / f"{name}.parquet",
            root=args.root,
        )
    write_report(
        out / "split.report.json",
        {
            "stage": "benchmark_split",
            "benchmark": args.benchmark,
            "family": args.family,
            "split_column": args.split_column,
            "labels_in": len(labels),
            "validation": len(validation),
            "validation_positive_share": round(float(validation["label"].mean()), 4),
            "validation_acids": validation["key_a_flat"].nunique(),
            "validation_amines": validation["key_b_flat"].nunique(),
            "test": len(test),
            **shared,
        },
        configs=[
            class_path,
            Path(f"configs/benchmarks/{args.benchmark}.yml"),
            args.standardise,
        ],
    )
    print(
        f"{args.benchmark} {args.family}: validation {len(validation):,}, test {len(test):,}"
    )


if __name__ == "__main__":
    main()
