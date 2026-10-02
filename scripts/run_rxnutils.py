"""Run ORDerly through the rxnutils standardisation pipeline (exploration, not package code).

Reads ORDerly with the package adapter (``platepu.ingest.orderly.read_orderly``), writes the
tab-separated input that rxnutils expects, and runs the pipeline in ``configs/rxnutils_std.yml``.

rxnutils only gets ``row_id`` and the atom-mapped reaction SMILES. Agents and solvents are not
passed as separate fields: later steps join them back from the ``ReactionRecord``s by ``row_id``
(= ORDerly's row index), never rebuild them from the reaction SMILES.

Usage:
    uv run python scripts/run_rxnutils.py ORDERLY_PARQUET OUT_DIR [--max-workers N]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from platepu.ingest.orderly import read_orderly
from platepu.ingest.record import ReactionRecord

PIPELINE = Path(__file__).resolve().parent.parent / "configs" / "rxnutils_std.yml"


def rxnutils_input(records: list[ReactionRecord]) -> pd.DataFrame:
    """Build the rxnutils input table: one row per record that has an atom-mapped reaction.

    Args:
        records: Records from a source adapter.

    Returns:
        A table with columns ``row_id`` (the record's ``source_id``) and ``rsmi`` (its
        ``mapped_rxn``). Records without a mapped reaction are left out, because
        ``reactants2reagents`` needs atom mapping.
    """
    rows = [
        {"row_id": record.source_id, "rsmi": record.mapped_rxn}
        for record in records
        if record.mapped_rxn
    ]
    return pd.DataFrame(rows, columns=["row_id", "rsmi"])


def run_pipeline(data: Path, output: Path, max_workers: int) -> float:
    """Run the rxnutils pipeline on a TSV file and return the wall-clock time in seconds."""
    command = [
        sys.executable, "-m", "rxnutils.pipeline.runner",
        "--pipeline", str(PIPELINE),
        "--data", str(data),
        "--output", str(output),
        "--max-workers", str(max_workers),
        "--no-intermediates",
    ]  # fmt: skip
    start = time.perf_counter()
    subprocess.run(command, check=True)
    return time.perf_counter() - start


def main() -> None:
    """Read ORDerly, write the rxnutils input, run the pipeline and print the counts."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="ORDerly condition parquet file")
    parser.add_argument(
        "out_dir", type=Path, help="output folder (use a git-ignored one)"
    )
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()

    records, rejected = read_orderly(args.path)
    table = rxnutils_input(records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = args.out_dir / "rxnutils_in.tsv"
    table.to_csv(data, sep="\t", index=False)
    print(f"records: {len(records):,} | rejected by the adapter: {len(rejected):,}")
    print(f"without a mapped reaction (left out): {len(records) - len(table):,}")
    print(f"rxnutils input rows: {len(table):,} -> {data}")

    seconds = run_pipeline(data, args.out_dir / "rxnutils_out.tsv", args.max_workers)
    print(f"rxnutils pipeline: {seconds:.1f} s with {args.max_workers} worker(s)")


if __name__ == "__main__":
    main()
