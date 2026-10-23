"""Stage 1, ingest: read one source file into the common record table (docs/design/rebuild.md).

Writes ``1_records.parquet``, ``1_records.rejected.parquet`` and ``1_records.report.json`` to
``<root>/<source>/`` (licensed sources: ``<root>/private/<source>/``).

Usage:
    uv run python scripts/ingest.py SOURCE INPUT ROOT
    uv run python scripts/ingest.py orderly .../orderly_condition_train.parquet .../Datasets/derived
"""

import argparse
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

from platepu.ingest.chemlex import read_chemlex
from platepu.ingest.orderly import read_orderly
from platepu.ingest.record import (
    ReactionRecord,
    RejectedRow,
    records_to_table,
    rejected_to_table,
)
from platepu.io import source_dir, write_report, write_table

Adapter = Callable[[Path], tuple[list[ReactionRecord], list[RejectedRow]]]

# The only source-specific part of the pipeline (ADR 002): one reader per source.
ADAPTERS: dict[str, Adapter] = {
    "orderly": read_orderly,
    "chemlex": read_chemlex,
}


def main(argv: Sequence[str] | None = None) -> None:
    """Read the source, then write the record table, the rejected rows and the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", choices=sorted(ADAPTERS))
    parser.add_argument("input", type=Path, help="the source file, read in place")
    parser.add_argument("root", type=Path, help="output root, e.g. Datasets/derived")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    records, rejected = ADAPTERS[args.source](args.input)
    out = source_dir(args.root, args.source)
    write_table(records_to_table(records), out / "1_records.parquet", root=args.root)
    write_table(
        rejected_to_table(rejected), out / "1_records.rejected.parquet", root=args.root
    )
    stat = args.input.stat()
    write_report(
        out / "1_records.report.json",
        {
            "stage": "ingest",
            "source": args.source,
            "input": str(args.input),
            "input_bytes": stat.st_size,
            "input_modified": stat.st_mtime,
            "rows_out": len(records),
            "rejected": dict(Counter(row.reason for row in rejected)),
            "seconds": round(time.perf_counter() - start, 1),
        },
    )
    print(
        f"{args.source}: {len(records):,} records, {len(rejected):,} rejected -> {out}"
    )


if __name__ == "__main__":
    main()
