"""Stage S1: score every validation and inventory pair with the baselines B0 to B7.

Reads the stage F features, the positives at both widths, the unlabeled inventory pairs and a
benchmark validation file, and writes one table under ``private/`` with a score per pair and model.
No metric is computed here (that is stage E) and no label is read: the validation file is opened
with an explicit column list, and any file not named ``validation.parquet`` is refused.

Usage:
    uv run python scripts/score_baselines.py amide_coupling HATU_DIPEA_DMF ROOT --validation V.parquet
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from platepu.io import sha256, strictest_licence, write_report, write_table
from platepu.score.baselines import (
    BaselinesConfig,
    additive_scores,
    load_baselines_config,
    nearest_pair_scores,
    popularity_scores,
    random_scores,
    rank_sum,
    size_scores,
    subclass_scores,
)
from platepu.score.pairs import pair_features

MODEL_CONFIG = Path("configs/models/baselines.yml")
# The only columns stage S may see: a label or a conversion never enters the process. Both scored
# sets are read with an explicit list, so no column arrives that the baselines did not ask for.
PAIR_COLUMNS = ["key_a_stereo", "key_b_stereo", "key_a_flat", "key_b_flat", "licence"]
# The inventory table carries no std_version, so ``pair_features`` cannot check its standardisation
# against the features' (it still refuses a feature table that mixes versions). Stamping one belongs
# to stage U, which writes that table, not here.
VALIDATION_COLUMNS = [*PAIR_COLUMNS, "std_version"]
COLUMNS = [
    "score_set",
    "key_a_stereo",
    "key_b_stereo",
    "score",
    "model",
    "p_width",
    "seed",
    "config_sha256",
    "std_version",
    "licence",
]
# A scored set's licence is the strictest its own pairs carry, never a value written here: the
# validation pairs were labelled "open" until 2026-10-22 although Chemlex is CC BY-NC-ND (ADR 009).


def score_one_set(
    paired: pd.DataFrame,
    positives: dict[str, pd.DataFrame],
    substrates: pd.DataFrame,
    config: BaselinesConfig,
    score_set: str,
) -> list[tuple[str, str | None, np.ndarray]]:
    """(model, P width, scores) for every baseline on one scored set.

    Only B2 and B3 read the positives as pairs, so only they are run at both widths; the others get
    a null width. Ranks are taken inside the set, because a score is comparable only there.
    """
    rows: list[tuple[str, str | None, np.ndarray]] = [
        ("B0", None, random_scores(paired, config.seed, score_set)),
        ("B1", None, popularity_scores(paired, positives["all"])),
    ]
    for width, pairs in positives.items():
        rows.append(("B2", width, nearest_pair_scores(paired, pairs, substrates)))
        rows.append(("B3", width, additive_scores(paired, pairs, substrates)))
    subclass = subclass_scores(paired, "b", config.amine_subclass_scores)
    sterics = -paired["b_n_heavy_at_2_bonds"].to_numpy(dtype=np.float64)
    electronics = -paired["b_n_ewg_on_ring"].to_numpy(dtype=np.float64)
    rows += [
        ("B4", None, subclass),
        ("B5", None, rank_sum(subclass, sterics)),
        ("B6", None, rank_sum(subclass, sterics, electronics)),
        ("B7", None, size_scores(paired)),
    ]
    return rows


def ties(scores: np.ndarray) -> dict[str, int]:
    """How coarse a baseline is: distinct values, and how many pairs share the commonest one."""
    counts = pd.Series(scores).value_counts()
    return {"n_distinct_scores": len(counts), "largest_tie": int(counts.iloc[0])}


def main(argv: Sequence[str] | None = None) -> None:
    """Score both sets with every baseline and write one table and its report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name")
    parser.add_argument("family")
    parser.add_argument("root", type=Path)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.validation.name != "validation.parquet":
        raise ValueError(
            f"--validation must be a validation.parquet: {args.validation}"
        )

    config = load_baselines_config(MODEL_CONFIG)
    substrates = pd.read_parquet(
        args.root / "private" / "features" / args.name / "substrates.parquet"
    )
    everything = pd.read_parquet(
        args.root / "positives" / args.name / "positives.parquet"
    )
    positives = {
        args.family: everything[everything["family"] == args.family],
        "all": everything[everything["family"] == "all"],
    }
    sets = {
        "validation": pd.read_parquet(args.validation, columns=VALIDATION_COLUMNS),
        "inventory": pd.read_parquet(
            args.root / "private" / "unlabeled" / args.name / f"{args.family}.parquet",
            columns=PAIR_COLUMNS,
        ),
    }

    licences = {
        name: strictest_licence(pairs["licence"]) for name, pairs in sets.items()
    }
    std_version = min(set(substrates["std_version"]))
    config_sha256 = sha256(MODEL_CONFIG)
    tables, structure = [], {}
    for score_set, pairs in sets.items():
        paired = pair_features(pairs, substrates)
        for model, width, scores in score_one_set(
            paired, positives, substrates, config, score_set
        ):
            tables.append(
                pd.DataFrame(
                    {
                        "score_set": score_set,
                        "key_a_stereo": paired["key_a_stereo"].to_numpy(),
                        "key_b_stereo": paired["key_b_stereo"].to_numpy(),
                        "score": scores.astype(np.float32),
                        "model": model,
                        "p_width": width,
                        "seed": config.seed,
                        "config_sha256": config_sha256,
                        "std_version": std_version,
                        "licence": licences[score_set],
                    }
                )
            )
            structure[f"{model}|{width}|{score_set}"] = ties(scores)
    scored = pd.concat(tables, ignore_index=True)[COLUMNS]

    out = (
        args.root / "private" / "scores" / args.name / args.family / "baselines.parquet"
    )
    report: dict[str, object] = {
        "stage": "score_baselines",
        "class": args.name,
        "family": args.family,
        "validation": str(args.validation),
        "seed": config.seed,
        "rows_in": {name: len(pairs) for name, pairs in sets.items()},
        "positives_pairs": {width: len(pairs) for width, pairs in positives.items()},
        "rows_out": len(scored),
        "std_version": std_version,
        "ties": structure,
    }
    write_table(pa.Table.from_pandas(scored, preserve_index=False), out, root=args.root)
    write_report(out.with_suffix(".report.json"), report, [MODEL_CONFIG])
    print(f"{args.name} {args.family}: {len(scored):,} scores -> {out}")


if __name__ == "__main__":
    main()
