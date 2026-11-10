"""Mapping, step 1: write every reaction not yet in the mapping cache as shards to map.

Reads ``2_rsmi.parquet`` of each given source, skips hashes already cached, and writes shards to
``<cache>/todo/shard-NNNNN.parquet``. A reaction that any open source has goes to the open cache;
reactions only licensed sources have go to the private cache. Map each shard with map_shard.py
(on the cluster: one array task per shard).

Usage:
    uv run python scripts/map_missing.py SOURCE [SOURCE ...] --paths configs/paths/local.yml
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from platepu.io import PRIVATE_SOURCES, load_paths, source_dir, write_table
from platepu.reactions.mapping import cache_dirs, mapper_tag, read_cache, to_map


def write_shards(
    items: list[tuple[str, str]], folder: Path, size: int, root: Path
) -> int:
    """Write ``(rsmi_hash, rsmi)`` pairs as shards of at most ``size`` rows; return the count."""
    shards = [items[start : start + size] for start in range(0, len(items), size)]
    for number, shard in enumerate(shards):
        table = pa.table(
            {"rsmi_hash": [h for h, _ in shard], "rsmi": [r for _, r in shard]}
        )
        write_table(table, folder / "todo" / f"shard-{number:05d}.parquet", root=root)
    return len(shards)


def main(argv: Sequence[str] | None = None) -> None:
    """Collect the missing reactions of the given sources and write them as shards."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=10_000)
    args = parser.parse_args(argv)

    paths = load_paths(args.paths)
    open_dir, private_dir = cache_dirs(paths.root, mapper_tag(paths.rxnmapper_python))
    cache = read_cache([open_dir, private_dir])
    tables = {
        s: pq.read_table(source_dir(paths.root, s) / "2_rsmi.parquet")
        for s in args.sources
    }
    open_items = to_map(
        [t for s, t in tables.items() if s not in PRIVATE_SOURCES], cache
    )
    licensed = [t for s, t in tables.items() if s in PRIVATE_SOURCES]
    taken = {h for h, _ in open_items}
    private_items = [(h, r) for h, r in to_map(licensed, cache) if h not in taken]

    n_open = write_shards(open_items, open_dir, args.shard_size, paths.root)
    n_private = write_shards(private_items, private_dir, args.shard_size, paths.root)
    print(f"to map: {len(open_items):,} open in {n_open} shards -> {open_dir / 'todo'}")
    print(
        f"to map: {len(private_items):,} licensed in {n_private} shards -> {private_dir / 'todo'}"
    )


if __name__ == "__main__":
    main()
