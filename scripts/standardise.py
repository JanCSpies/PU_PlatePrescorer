"""Stage 6, standardise: standardised substrates, product and condition key for one source.

Reads ``5_classified.parquet``; writes ``6_standardised.parquet`` and ``6_standardised.report.json``.

Usage:
    uv run python scripts/standardise.py SOURCE ROOT [--config configs/standardise.yml]
"""

import argparse
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq

from platepu.io import source_dir, write_report, write_table
from platepu.standardise.molecule import condition_text, load_std_config
from platepu.standardise.table import standardise_table


def main(argv: Sequence[str] | None = None) -> None:
    """Standardise the source's classified reactions and write the stage files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("root", type=Path, help="output root, e.g. Datasets/derived")
    parser.add_argument("--config", type=Path, default=Path("configs/standardise.yml"))
    args = parser.parse_args(argv)

    start = time.perf_counter()
    folder = source_dir(args.root, args.source)
    classified = pq.read_table(folder / "5_classified.parquet")
    table = standardise_table(classified, load_std_config(args.config))
    write_table(table, folder / "6_standardised.parquet", root=args.root)
    conditions = Counter(map(condition_text, table["condition_key"].to_pylist()))
    write_report(
        folder / "6_standardised.report.json",
        {
            "stage": "standardise",
            "source": args.source,
            "rows_in": classified.num_rows,
            "rows_out": table.num_rows,
            "no_condition_key": table["condition_key"].null_count,
            "top_conditions": dict(conditions.most_common(10)),
            "seconds": round(time.perf_counter() - start, 1),
        },
        configs=[args.config],
    )
    print(f"{args.source}: {table.num_rows:,} standardised")


if __name__ == "__main__":
    main()
