"""Lab substrate library for one reaction class, from the group's inventory export (private).

Writes ``<root>/private/library/<class>.parquet`` (one row per usable compound), ``.rejected.parquet``
(containers with a reason) and ``.report.json``.

Usage:
    uv run python scripts/library.py amide_coupling INVENTORY.xlsx ROOT
"""

import argparse
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa

from platepu.classify.engine import load_class
from platepu.io import write_report, write_table
from platepu.library.inventory import library_table, read_inventory
from platepu.standardise.molecule import load_std_config


def main(argv: Sequence[str] | None = None) -> None:
    """Read the inventory, keep compounds that fit one role of the class, write the library."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="class name, e.g. amide_coupling")
    parser.add_argument("inventory", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--standardise", type=Path, default=Path("configs/standardise.yml")
    )
    args = parser.parse_args(argv)

    start = time.perf_counter()
    class_path = Path(f"configs/classes/{args.name}.yml")
    inventory = read_inventory(args.inventory)
    compounds, rejected = library_table(
        inventory, load_class(class_path), load_std_config(args.standardise)
    )
    out = args.root / "private" / "library" / args.name
    write_table(
        pa.Table.from_pandas(compounds, preserve_index=False),
        out.with_suffix(".parquet"),
        root=args.root,
    )
    write_table(
        pa.Table.from_pandas(rejected, preserve_index=False),
        out.with_suffix(".rejected.parquet"),
        root=args.root,
    )
    write_report(
        out.with_suffix(".report.json"),
        {
            "stage": "library",
            "class": args.name,
            "containers": len(inventory),
            "compounds_per_role": dict(Counter(compounds["role"])),
            "containers_per_role": {
                r: int(n)
                for r, n in compounds.groupby("role")["n_containers"].sum().items()
            },
            "rejected": dict(Counter(rejected["reason"])),
            "seconds": round(time.perf_counter() - start, 1),
        },
        configs=[class_path, args.standardise],
    )
    print(
        f"{args.name}: {dict(Counter(compounds['role']))} compounds, {len(rejected):,} containers rejected"
    )


if __name__ == "__main__":
    main()
