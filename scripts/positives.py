"""Stage 7a, positives: literature pairs of one class from one or more sources (docs/design/rebuild.md).

Reads each source's ``6_standardised.parquet`` (yield and year from ``1_records.parquet``) and the
benchmark labels, which are read only to count the overlap (ADR 009). Writes ``<root>/positives/<class>/positives.parquet`` from the
open sources; if a private source is given (ADR 009), also ``<root>/private/positives/<class>/``
from all of them.

Usage:
    uv run python scripts/positives.py amide_coupling ROOT --sources orderly \
        --benchmark-labels ROOT/private/benchmarks/chemlex/labels.parquet
"""

import argparse
import dataclasses
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa

from platepu.classify.engine import ClassRule, load_class
from platepu.io import PRIVATE_SOURCES, source_dir, write_report, write_table
from platepu.positives.build import (
    PositivesConfig,
    build_positives,
    load_positives_config,
)
from platepu.standardise.molecule import load_std_config


def reactions(root: Path, sources: Sequence[str], name: str) -> pd.DataFrame:
    """Stage-6 rows of one class from the sources, with source, yield and year from stage 1."""
    frames = []
    for source in sources:
        folder = source_dir(root, source)
        std = pd.read_parquet(folder / "6_standardised.parquet")
        records = pd.read_parquet(
            folder / "1_records.parquet",
            columns=["record_id", "source", "yield_pct", "year"],
        )
        frames.append(
            std[std["class"] == name].merge(
                records, on="record_id", validate="many_to_one"
            )
        )
    return pd.concat(frames, ignore_index=True)


def sizes(
    rows: pd.DataFrame, rule: ClassRule, config: PositivesConfig, bench: pd.DataFrame
) -> dict[str, dict[str, int]]:
    """Pairs per family at yield cut-offs none, 10 and 20 %, for choosing ``min_yield_pct``."""
    out = {}
    for cut in (None, 10.0, 20.0):
        pairs = build_positives(
            rows, rule, dataclasses.replace(config, min_yield_pct=cut), bench
        )[0]
        out[f"min_yield_{cut or 0:g}"] = dict(Counter(pairs["family"]))
    return out


def write(
    pairs: pd.DataFrame, rejected: pd.DataFrame, out: Path, root: Path
) -> dict[str, object]:
    """Write one positives folder's tables; return their counts for the report."""
    write_table(
        pa.Table.from_pandas(pairs, preserve_index=False),
        out / "positives.parquet",
        root=root,
    )
    write_table(
        pa.Table.from_pandas(rejected, preserve_index=False),
        out / "positives.rejected.parquet",
        root=root,
    )
    return {
        "pairs_per_family": dict(Counter(pairs["family"])),
        "rejected": dict(Counter(rejected["reason"])),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Build and write the class's positives, open and (if needed) private."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="class name, e.g. amide_coupling")
    parser.add_argument("root", type=Path)
    parser.add_argument("--sources", nargs="+", required=True)
    # Required, not optional. With no labels the leakage count came out 0/0, which reads as
    # "no overlap" and is indistinguishable from "never checked" (CLAUDE.md §6).
    parser.add_argument("--benchmark-labels", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--standardise", type=Path, default=Path("configs/standardise.yml")
    )
    args = parser.parse_args(argv)

    start = time.perf_counter()
    class_path = Path(f"configs/classes/{args.name}.yml")
    rule = load_class(class_path)
    config = load_positives_config(class_path, load_std_config(args.standardise))
    bench = pd.concat(
        [
            pd.read_parquet(p, columns=["key_a_flat", "key_b_flat"])
            for p in args.benchmark_labels
        ]
        or [pd.DataFrame(columns=["key_a_flat", "key_b_flat"])]
    )
    open_sources = [s for s in args.sources if s not in PRIVATE_SOURCES]
    runs = [(open_sources, args.root / "positives" / args.name)]
    if (
        open_sources != args.sources
    ):  # a private source was given: also all sources together, under private/
        runs.append((args.sources, args.root / "private" / "positives" / args.name))
    for sources, out in runs:
        rows = reactions(args.root, sources, args.name)
        pairs, rejected, leakage = build_positives(rows, rule, config, bench)
        report = {
            "stage": "positives",
            "class": args.name,
            "sources": sources,
            "reactions_in": len(rows),
            "sizes_by_yield_cutoff": sizes(rows, rule, config, bench),
            **write(pairs, rejected, out, args.root),
            **leakage,
            "seconds": round(time.perf_counter() - start, 1),
        }
        configs = [class_path, args.standardise]
        write_report(out / "positives.report.json", report, configs=configs)
        print(f"{args.name} {sources}: {dict(Counter(pairs['family']))}")


if __name__ == "__main__":
    main()
