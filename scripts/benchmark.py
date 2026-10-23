"""Stage 7b, benchmark: reaction-level test labels for one HTE benchmark (docs/design/rebuild.md).

Reads the benchmark's ``1_records.parquet`` (roles from its own columns), its ``5_classified.parquet``
(mapping check only) and the raw file (splits); writes ``<root>/private/benchmarks/<name>/labels.parquet`` and
``labels.report.json``.

Usage:
    uv run python scripts/benchmark.py chemlex ROOT RAW_FILE [--config configs/benchmarks/chemlex.yml]
"""

import argparse
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from platepu.benchmarks.labels import load_label_rule
from platepu.benchmarks.measurements import measurements, reaction_labels
from platepu.classify.engine import load_class
from platepu.ingest.chemlex import load_chemlex, splits_from_table
from platepu.io import source_dir, write_report, write_table
from platepu.standardise.molecule import load_std_config

# Benchmark-specific split readers, like the adapters in ingest.py: raw file -> splits with record_id.
SPLITS: dict[str, Callable[[Path], pd.DataFrame]] = {
    "chemlex": lambda path: splits_from_table(load_chemlex(path)).assign(
        record_id=lambda t: "chemlex:" + t["source_id"]
    ),
}


def summary(labels: pd.DataFrame, measured: pd.DataFrame) -> dict[str, object]:
    """Counts for the report: rows, class fit, statuses, positives, both-unseen test, mapping check."""
    labelled = labels[labels["label"].notna()]
    test = (
        labelled[labelled.get("both_unseen") == "test"]
        if "both_unseen" in labels
        else labelled[:0]
    )
    # Every share below averages over the rows **outside** the test side. Removing the test-only
    # mean on 2026-10-22 was not enough: `positive_share` and `top_conditions` still averaged over
    # every labelled row, so a test-side outcome was published per condition and overall. This is a
    # counting path, not the final evaluation (CLAUDE.md §6: never tune on TEST).
    outside = labelled.drop(test.index)
    top = outside.groupby("condition").agg(
        n=("label", "size"), positive=("label", "mean")
    )
    return {
        "rows": len(measured),
        "rows_in_class": int(measured["in_class"].sum()),
        "reactions": len(labels),
        "status": dict(Counter(labels["status"])),
        "labelled": len(labelled),
        "labelled_outside_test": len(outside),
        "positive_share_outside_test": round(float(outside["label"].mean()), 4)
        if len(outside)
        else None,
        "both_unseen_test": len(test),
        "top_conditions_outside_test": {
            cond: {"reactions": int(r.n), "positive_share": round(float(r.positive), 4)}
            for cond, r in top.sort_values("n", ascending=False).head(5).iterrows()
        },
        "split_conflicts": int(labels["split_conflict"].sum()),
        "reactions_not_all_mapped_in_class": int(
            (labels["n_mapped_in_class"] < labels["n_measurements"]).sum()
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Build the benchmark's labels and write them with a report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", choices=sorted(SPLITS))
    parser.add_argument("root", type=Path, help="output root, e.g. Datasets/derived")
    parser.add_argument(
        "raw", type=Path, help="the benchmark's raw file (for its splits)"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--standardise", type=Path, default=Path("configs/standardise.yml")
    )
    args = parser.parse_args(argv)

    start = time.perf_counter()
    config_path = args.config or Path(f"configs/benchmarks/{args.name}.yml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    class_path = Path(f"configs/classes/{config['class']}.yml")
    folder = source_dir(args.root, args.name)
    measured = measurements(
        pq.read_table(folder / "1_records.parquet"),
        load_class(class_path),
        load_std_config(args.standardise),
    )
    classified = pq.read_table(folder / "5_classified.parquet").to_pandas()
    mapped = classified.loc[classified["class"] == config["class"], "record_id"]
    labels = reaction_labels(
        measured,
        SPLITS[args.name](args.raw),
        config["reaction_splits"],
        load_label_rule(config_path),
        set(mapped),
    )
    labels = labels.assign(licence=config["licence"])
    out = args.root / "private" / "benchmarks" / args.name
    write_table(
        pa.Table.from_pandas(labels, preserve_index=False),
        out / "labels.parquet",
        root=args.root,
    )
    report = {"stage": "benchmark", "benchmark": args.name, "class": config["class"]}
    report |= summary(labels, measured) | {
        "seconds": round(time.perf_counter() - start, 1)
    }
    write_report(
        out / "labels.report.json",
        report,
        configs=[config_path, class_path, args.standardise],
    )
    print(
        f"{args.name}: {report['labelled']:,} labelled reactions, {report['positive_share_outside_test']:.1%} positive"
    )


if __name__ == "__main__":
    main()
