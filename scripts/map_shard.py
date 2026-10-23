"""Mapping, step 2: map one shard with RXNMapper and add it to the cache as a new part file.

``<cache>/todo/shard-NNNNN.parquet`` becomes ``<cache>/part-NNNNN.parquet``; the shard is removed
afterwards, so a rerun maps only what is left. On the cluster, run one array task per shard.

Usage:
    uv run python scripts/map_shard.py SHARD --paths configs/paths/local.yml
"""

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq

from platepu.io import load_paths, write_table
from platepu.reactions.mapping import map_reactions, rxnmapper

# A shard this size or larger is checked for a suspicious failure share; below it, one honest
# failure would trip any threshold.
MIN_ROWS_FOR_FAILURE_CHECK = 20
FAILED_SHARE = 0.05


def main(argv: Sequence[str] | None = None) -> None:
    """Map the shard's reactions and write them next to the todo folder as a part file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("shard", type=Path)
    parser.add_argument("--paths", type=Path, required=True)
    args = parser.parse_args(argv)

    paths = load_paths(args.paths)
    start = time.perf_counter()
    shard = pq.read_table(args.shard).to_pylist()
    mapper = rxnmapper(paths.rxnmapper_python, paths.mapper_batch_size)
    table = map_reactions([(row["rsmi_hash"], row["rsmi"]) for row in shard], mapper)
    part = args.shard.parent.parent / args.shard.name.replace("shard-", "part-")
    # Shards are numbered from zero on every run, so a second source would otherwise replace the
    # first source's part and silently lose every mapping in it.
    if part.exists():
        raise FileExistsError(
            f"{part} already holds mappings from an earlier run; "
            "number this run's shards after the highest existing part instead of overwriting"
        )
    failed = table["mapped_rsmi"].null_count
    # rxnutils reports an environment failure -- out of memory, a token limit, a tokenizer error --
    # the same way it reports a reaction it cannot map: the input back, confidence 0. Caching those
    # makes the loss permanent, because `to_map` never retries a hash it finds. The worst real part
    # of the 63 in the open cache failed at 0.03 % (625,539 reactions, median 0), so 5 % is about
    # 150x the worst honest shard; the row floor keeps a tiny final shard from blocking forever.
    if (
        table.num_rows >= MIN_ROWS_FOR_FAILURE_CHECK
        and failed > FAILED_SHARE * table.num_rows
    ):
        raise RuntimeError(
            f"{failed:,} of {table.num_rows:,} reactions in {args.shard.name} failed to map, "
            f"above {FAILED_SHARE:.0%}; nothing was cached and the shard was kept. "
            "Check the RXNMapper environment before re-running."
        )
    write_table(table, part, root=paths.root)
    args.shard.unlink()
    seconds = time.perf_counter() - start
    print(f"{part.name}: {table.num_rows:,} mapped, {failed:,} failed, {seconds:.0f} s")


if __name__ == "__main__":
    main()
