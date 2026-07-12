"""CLI: parse one reaction source into ``<source>_parsed.parquet`` (class + conditions + dedup key).

Stage 1 of the reaction x condition grouping (``PROSPECTIVE_REACTION_CONDITION_GROUPING.md``). Runs
one source at a time so each can be a separate, courteous SLURM job.

Usage:
    uv run python -m scripts_JSP.rxngroup_parse --source pistachio --workers 32
    uv run python -m scripts_JSP.rxngroup_parse --source orderly --limit 10000   # smoke test
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from pubench.rxngroup.pipeline import parse_source
from pubench.rxngroup.sources import SOURCES

OUT_DIR = Path("datasets/processed/private/rxn_groups")


def main() -> None:
    """Parse the chosen source and write its parsed-reaction parquet."""
    parser = argparse.ArgumentParser(description="Parse one reaction source for grouping.")
    parser.add_argument("--source", required=True, choices=sorted(SOURCES), help="Source key.")
    parser.add_argument("--path", type=Path, default=None, help="Override the default source path.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory.")
    parser.add_argument("--workers", type=int, default=16, help="Worker processes.")
    parser.add_argument("--chunk-size", type=int, default=5_000, help="Reactions per worker task.")
    parser.add_argument("--limit", type=int, default=None, help="Cap reactions (smoke test).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s"
    )
    reader, default_path = SOURCES[args.source]
    path = args.path or Path(default_path)
    out_path = args.out_dir / f"{args.source}_parsed.parquet"

    logging.info(
        "parsing source=%s path=%s -> %s (workers=%s, limit=%s)",
        args.source,
        path,
        out_path,
        args.workers,
        args.limit,
    )
    t0 = time.perf_counter()
    written = parse_source(
        reader(path, limit=args.limit),  # type: ignore[operator]
        out_path,
        workers=args.workers,
        chunk_size=args.chunk_size,
    )
    dt = time.perf_counter() - t0
    rate = written / dt if dt else 0.0
    logging.info("wrote %s rows in %.1fs (%.0f rows/s) -> %s", f"{written:,}", dt, rate, out_path)


if __name__ == "__main__":
    main()
