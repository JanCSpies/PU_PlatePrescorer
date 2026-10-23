"""Stage 4, clean: rxnutils actions (configs/clean.yml) on one source's mapped reactions.

Reads ``3_mapped.parquet``; writes ``4_clean.parquet``, ``4_clean.rejected.parquet`` and
``4_clean.report.json`` (with the dynamic-bond counts) in the source's folder.

Usage:
    uv run python scripts/clean.py SOURCE ROOT [--pipeline configs/clean.yml]
"""

import argparse
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq

from platepu.io import source_dir, write_report, write_table
from platepu.reactions.clean import clean_table, load_pipeline


def main(argv: Sequence[str] | None = None) -> None:
    """Clean the source's mapped reactions and write the stage files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("root", type=Path, help="output root, e.g. Datasets/derived")
    parser.add_argument("--pipeline", type=Path, default=Path("configs/clean.yml"))
    args = parser.parse_args(argv)

    start = time.perf_counter()
    folder = source_dir(args.root, args.source)
    mapped = pq.read_table(folder / "3_mapped.parquet")
    table, rejected = clean_table(mapped, load_pipeline(args.pipeline))
    write_table(table, folder / "4_clean.parquet", root=args.root)
    write_table(rejected, folder / "4_clean.rejected.parquet", root=args.root)
    bonds = Counter(table["n_dynamic_bonds"].to_pylist())
    write_report(
        folder / "4_clean.report.json",
        {
            "stage": "clean",
            "source": args.source,
            "rows_in": mapped.num_rows,
            "rows_out": table.num_rows,
            "rejected": dict(Counter(rejected["reason"].to_pylist())),
            "dynamic_bonds": {str(k): bonds[k] for k in sorted(bonds)},
            "n_reactants": dict(
                sorted(Counter(table["n_reactants"].to_pylist()).items())
            ),
            "seconds": round(time.perf_counter() - start, 1),
        },
        configs=[args.pipeline],
    )
    print(f"{args.source}: {table.num_rows:,} clean, {rejected.num_rows:,} rejected")


if __name__ == "__main__":
    main()
