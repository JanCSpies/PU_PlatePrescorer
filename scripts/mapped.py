"""Stage 3, mapped: attach the cached RXNMapper mapping to every reaction of one source.

Reads ``2_rsmi.parquet`` and the mapping cache; writes ``3_mapped.parquet``,
``3_mapped.rejected.parquet`` (``mapping_failed``) and ``3_mapped.report.json``. Stops if a reaction
was never mapped (run map_missing.py and map_shard.py first).

Usage:
    uv run python scripts/mapped.py SOURCE --paths configs/paths/local.yml
"""

import argparse
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq

from platepu.io import load_paths, source_dir, write_report, write_table
from platepu.reactions.mapping import cache_dirs, join_mapped, mapper_tag, read_cache


def main(argv: Sequence[str] | None = None) -> None:
    """Join the source's reactions with the cache and write the stage files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("--paths", type=Path, required=True)
    args = parser.parse_args(argv)

    paths = load_paths(args.paths)
    tag = mapper_tag(paths.rxnmapper_python)
    folder = source_dir(paths.root, args.source)
    stage2 = pq.read_table(folder / "2_rsmi.parquet")
    table, rejected = join_mapped(stage2, read_cache(cache_dirs(paths.root, tag)))
    write_table(table, folder / "3_mapped.parquet", root=paths.root)
    write_table(rejected, folder / "3_mapped.rejected.parquet", root=paths.root)
    write_report(
        folder / "3_mapped.report.json",
        {
            "stage": "mapped",
            "source": args.source,
            "mapper": tag,
            "rows_in": stage2.num_rows,
            "rows_out": table.num_rows,
            "rejected": dict(Counter(rejected["reason"].to_pylist())),
        },
        configs=[args.paths],
    )
    print(f"{args.source}: {table.num_rows:,} mapped, {rejected.num_rows:,} failed")


if __name__ == "__main__":
    main()
