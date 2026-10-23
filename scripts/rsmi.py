"""Stage 2, rsmi: one canonical unmapped reaction SMILES and hash per record (docs/design/rebuild.md).

Reads ``1_records.parquet`` and writes ``2_rsmi.parquet``, ``2_rsmi.rejected.parquet`` and
``2_rsmi.report.json`` in the source's folder.

Usage:
    uv run python scripts/rsmi.py SOURCE ROOT
"""

import argparse
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq

from platepu.io import source_dir, write_report, write_table
from platepu.reactions.smiles import rsmi_table


def main(argv: Sequence[str] | None = None) -> None:
    """Read the source's records, build the reaction strings and write the stage files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="source name, e.g. orderly")
    parser.add_argument("root", type=Path, help="output root, e.g. Datasets/derived")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    folder = source_dir(args.root, args.source)
    records = pq.read_table(folder / "1_records.parquet")
    table, rejected = rsmi_table(records)
    write_table(table, folder / "2_rsmi.parquet", root=args.root)
    write_table(rejected, folder / "2_rsmi.rejected.parquet", root=args.root)
    write_report(
        folder / "2_rsmi.report.json",
        {
            "stage": "rsmi",
            "source": args.source,
            "rows_in": records.num_rows,
            "rows_out": table.num_rows,
            "unique_reactions": len(set(table["rsmi_hash"].to_pylist())),
            "rejected": dict(Counter(rejected["reason"].to_pylist())),
            "seconds": round(time.perf_counter() - start, 1),
        },
    )
    print(
        f"{args.source}: {table.num_rows:,} reactions, {rejected.num_rows:,} rejected"
    )


if __name__ == "__main__":
    main()
