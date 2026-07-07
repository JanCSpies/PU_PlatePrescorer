"""CLI: split the curated BH-HTE dataset into test / literature / dropped.

Usage (from repo root):
    uv run python scripts/01_split_raw_dataset.py

Reads ``datasets/raw/BH_HTE_Curated_Dataset_v20260219.csv`` and writes:
    datasets/processed/test_jnj_hte_2024.{csv,parquet}
    datasets/processed/literature_hte.{csv,parquet}
    datasets/processed/dropped_ahneman_dreher_doyle_2018.{csv,parquet}
    datasets/processed/split_report.json

The split is keyed on the ``Source`` column; see
``src/bhpu/data/splitting.py`` for the rationale and invariants.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from bhpu.data import RawDatasetSplitter

# Resolve paths relative to the repository root (this file lives in scripts/).
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
# The curated Neves BH-HTE file lives under datasets/raw/Neves/ (a colleague
# reorganised datasets/raw/ into one folder per data source).
RAW_CSV: Path = REPO_ROOT / "datasets" / "raw" / "Neves" / "BH_HTE_Curated_Dataset_v20260219.csv"
OUTPUT_DIR: Path = REPO_ROOT / "datasets" / "processed"
REPORT_PATH: Path = OUTPUT_DIR / "split_report.json"


def main() -> None:
    console = Console()
    splitter = RawDatasetSplitter(raw_csv_path=RAW_CSV)

    console.print(f"[bold]Loading[/bold] {RAW_CSV.relative_to(REPO_ROOT)} ...")
    frame = splitter.load()

    console.print("[bold]Splitting[/bold] by Source (validating invariants) ...")
    partitions = splitter.split(frame)

    written = splitter.write(partitions, OUTPUT_DIR)
    report = splitter.report(frame, partitions)
    REPORT_PATH.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    _print_summary(console, report)
    console.print("\n[bold green]Wrote:[/bold green]")
    for name, path in written.items():
        console.print(f"  {name:11s} -> {path.relative_to(REPO_ROOT)}")
    console.print(f"  {'report':11s} -> {REPORT_PATH.relative_to(REPO_ROOT)}")


def _print_summary(console: Console, report) -> None:  # noqa: ANN001 (rich table view)
    table = Table(title="BH-HTE split summary")
    table.add_column("partition", style="cyan")
    table.add_column("rows", justify="right")
    table.add_column("label 0", justify="right")
    table.add_column("label 1", justify="right")
    table.add_column("sources")
    for part in report.partitions:
        table.add_row(
            part.name,
            str(part.n_rows),
            str(part.label_balance.get("0", 0)),
            str(part.label_balance.get("1", 0)),
            ", ".join(sorted(part.sources)),
        )
    console.print(table)
    console.print(f"raw rows: {report.raw_n_rows}  sha256: {report.raw_sha256[:16]}...")


if __name__ == "__main__":
    main()
