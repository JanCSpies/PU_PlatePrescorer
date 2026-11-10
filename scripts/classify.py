"""Stage 5, classify: match one source's cleaned reactions against the class YAMLs.

Reads ``4_clean.parquet``; writes ``5_classified.parquet`` (one row per reaction and class match),
``5_classified.rejected.parquet`` (``no_formed_bond``, ``no_class``) and ``5_classified.report.json``.

Usage:
    uv run python scripts/classify.py SOURCE ROOT [--classes configs/classes/*.yml]
"""

import argparse
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from platepu.classify.engine import classify_table, load_class
from platepu.io import source_dir, write_report, write_table


def dynamic_bonds_per_class(table: pa.Table) -> dict[str, dict[str, int]]:
    """For each class, how many matched reactions have each dynamic-bond count."""
    counts: dict[str, Counter[str]] = {}
    for name, n_bonds in zip(
        table["class"].to_pylist(), table["n_dynamic_bonds"].to_pylist(), strict=True
    ):
        counts.setdefault(name, Counter())[str(n_bonds)] += 1
    return {name: dict(sorted(by_count.items())) for name, by_count in counts.items()}


def main(argv: Sequence[str] | None = None) -> None:
    """Classify the source's reactions and write the stage files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("root", type=Path, help="output root, e.g. Datasets/derived")
    parser.add_argument(
        "--classes",
        type=Path,
        nargs="+",
        default=sorted(Path("configs/classes").glob("*.yml")),
    )
    args = parser.parse_args(argv)
    if not args.classes:  # e.g. run outside the repository: every row would be no_class
        parser.error(
            "no class files found; run from the repository root or pass --classes"
        )

    start = time.perf_counter()
    folder = source_dir(args.root, args.source)
    cleaned = pq.read_table(folder / "4_clean.parquet")
    table, rejected = classify_table(
        cleaned, [load_class(path) for path in args.classes]
    )
    write_table(table, folder / "5_classified.parquet", root=args.root)
    write_table(rejected, folder / "5_classified.rejected.parquet", root=args.root)
    per_class = Counter(table["class"].to_pylist())
    bonds = dynamic_bonds_per_class(table)
    write_report(
        folder / "5_classified.report.json",
        {
            "stage": "classify",
            "source": args.source,
            "rows_in": cleaned.num_rows,
            "rows_out": table.num_rows,  # one row per (reaction, class match)
            "reactions_matched": len(set(table["record_id"].to_pylist())),
            "per_class": dict(per_class),
            "dynamic_bonds_per_class": bonds,
            "rejected": dict(Counter(rejected["reason"].to_pylist())),
            "seconds": round(time.perf_counter() - start, 1),
        },
        configs=list(args.classes),
    )
    print(f"{args.source}: {dict(per_class)}, {rejected.num_rows:,} rejected")


if __name__ == "__main__":
    main()
