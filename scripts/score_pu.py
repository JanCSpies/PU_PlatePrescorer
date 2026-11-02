"""Stage S2a: cut the reliable negatives for one PU config (docs/design/modeling.md "Stage S2").

Reads the positives at the config's width, the stage U table the config names and the stage F
features, and writes one row per unlabeled pair and per spy with its score and its flag, plus the
audit report that is the point of the sub-step. No metric is computed here (that is stage E).

**This is the one stage that cannot read a label even by accident.** It opens no benchmark file at
all -- not a validation split, not a labels table, and never a `test.parquet` -- so there is no
argument through which one could be handed to it, and `tests/test_score_pu_script.py` runs it in a
tree where no such file exists.

Usage:
    uv run python scripts/score_pu.py amide_coupling ROOT --config configs/models/pu_spy_morgan_centre.yml
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa

from platepu.features.blocks import SIZE
from platepu.io import read_table, write_report, write_table
from platepu.score.pu import reliable_negatives
from platepu.score.pu_config import load_pu_config

PAIR_KEYS = ["key_a_stereo", "key_b_stereo", "key_a_flat", "key_b_flat"]
# Explicit column lists, as stage S1 reads its inputs: a column the step did not ask for cannot
# arrive. `family` is read only to select the config's width, and is not carried into the output.
POSITIVE_COLUMNS = [*PAIR_KEYS, "family", "std_version", "licence"]
# `n_draws` is the draw weight every fit needs (ADR 029); `score/pu.py` refuses a table without it.
UNLABELED_COLUMNS = [*PAIR_KEYS, "n_draws", "std_version", "licence"]


def main(argv: Sequence[str] | None = None) -> None:
    """Cut the reliable negatives under one config and write the table and its report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name")
    parser.add_argument("root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)

    config = load_pu_config(args.config)
    positives = read_table(
        args.root / "positives" / args.name / "positives.parquet",
        POSITIVE_COLUMNS,
        "the positives table",
    )
    positives = positives[positives["family"] == config.p_width]
    unlabeled = read_table(
        args.root / "unlabeled" / args.name / f"{config.unlabeled}.parquet",
        UNLABELED_COLUMNS,
        f"the unlabeled table the config's `unlabeled` key names ({config.unlabeled!r})",
    )
    substrates = read_table(
        args.root / "private" / "features" / args.name / "substrates.parquet",
        ["key_stereo", config.feature_block, SIZE, "std_version"],
        f"the feature table, read for block {config.feature_block!r}",
    )

    negatives, report = reliable_negatives(positives, unlabeled, substrates, config)
    out = args.root / "pu" / args.name / f"{args.config.stem}.negatives.parquet"
    report |= {
        "stage": "score_pu",
        "class": args.name,
        "config": str(args.config),
        "p_width": config.p_width,
        "unlabeled": config.unlabeled,
        "rows_in": {
            "positives": len(positives),
            "unlabeled": len(unlabeled),
            "substrates": len(substrates),
        },
        "rows_out": len(negatives),
        # One value by construction: `pair_matrix` refuses a mixed std_version across the inputs.
        "std_version": min(set(negatives["std_version"])),
    }
    # The licence travels with the rows and `write_table` enforces the path from it: the moment a
    # licensed positive enters P this run **stops**, because `pu/` sits outside `private/`. That is
    # a refusal and not a reroute on purpose (CLAUDE.md §8): P is 100 % open today (counted
    # 2026-11-02), so a branch for the other case would be a branch for a case the data does not
    # have, and the day Pistachio enters P the path is a decision for a human to take.
    write_table(
        pa.Table.from_pandas(negatives, preserve_index=False), out, root=args.root
    )
    write_report(out.with_suffix(".report.json"), report, [args.config])
    print(
        f"{args.name} {args.config.stem}: {len(negatives):,} rows, "
        f"{report['n_reliable_negatives']:,} reliable negatives -> {out}"
    )


if __name__ == "__main__":
    main()
