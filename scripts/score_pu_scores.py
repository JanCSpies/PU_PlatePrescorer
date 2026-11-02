"""Stage S2b: score every validation and inventory pair with one PU model (one arm, one block).

Reads the stage F features, the positives at the config's width, the negative class the config's
``negatives`` key points at -- an S2a table for the spy arm, the whole stage U table for the naive
one -- and the two scored sets, and writes stage S1's layout unchanged so stage E reads one table.

No metric is computed here and no label is read: the validation file is opened with an explicit
column list, any file not named ``validation.parquet`` is refused, and no `test.parquet` is ever
touched. The validation ROC-AUC is a separate read, as it is for stage S1.

Usage:
    uv run python scripts/score_pu_scores.py amide_coupling HATU_DIPEA_DMF ROOT \
        --config configs/models/pu_score_spy_morgan_centre.yml --validation V.parquet
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from platepu.features.blocks import SIZE
from platepu.io import read_table, strictest_licence, write_report, write_table
from platepu.score.pu_config import (
    PuScoreConfig,
    config_sha256,
    load_pu_config,
    load_pu_score_config,
)
from platepu.score.pu_scores import (
    check_negatives_pointer,
    negative_source,
    pair_scores,
    training_rows,
)

# Repo-relative, resolved from the working directory and never from this file's location, exactly as
# stage S1 reads `configs/models/baselines.yml`: a run is reproducible from the repository it is run
# in, and a test can put a fixture config tree in front of it.
CONFIGS = Path("configs/models")
PAIR_KEYS = ["key_a_stereo", "key_b_stereo", "key_a_flat", "key_b_flat"]
PAIR_COLUMNS = [*PAIR_KEYS, "licence"]
# The inventory carries no `std_version` (stage U writes that table and does not stamp one), so it
# is read without one and stays out of the agreement check below.
VALIDATION_COLUMNS = [*PAIR_COLUMNS, "std_version"]
POSITIVE_COLUMNS = [*PAIR_KEYS, "family", "std_version", "licence"]
NEGATIVE_COLUMNS = [
    *PAIR_KEYS,
    "reliable_negative",
    "weight",
    "config_sha256",
    "std_version",
    "licence",
]
UNLABELED_COLUMNS = [*PAIR_KEYS, "n_draws", "std_version", "licence"]
# Stage S1's eleven columns in stage S1's order (docs/design/modeling.md, "Stage S — scores").
COLUMNS = [
    "score_set",
    "key_a_stereo",
    "key_b_stereo",
    "score",
    "model",
    "feature_block",
    "p_width",
    "seed",
    "config_sha256",
    "std_version",
    "licence",
]


def one_std_version(tables: dict[str, pd.DataFrame]) -> str:
    """The single standardisation every input agrees on, or a ValueError naming the disagreement.

    Checked here and not in the library: ``training_rows`` keeps only the keys, the target and the
    weight, so ``pair_matrix`` never sees a ``std_version`` to compare. A key written under one
    standardisation can be absent from, or mean something else in, a table built under another
    (CLAUDE.md §6).
    """
    found = {name: sorted(set(table["std_version"])) for name, table in tables.items()}
    versions = {version for values in found.values() for version in values}
    if len(versions) != 1:
        raise ValueError(f"the inputs disagree on std_version: {found}")
    return versions.pop()


def negative_class(
    name: str, root: Path, config: PuScoreConfig
) -> tuple[pd.DataFrame, str | None]:
    """The negative class and the S2a hash behind it, which is the arm (ADR 028 item 2).

    The spy arm's pointer is **checked and not trusted**: the S2a config it names must agree on every
    shared key and must be the one the table was actually cut under, because a selection made for
    another block is not a selection for this one.
    """
    if negative_source(config) == "unlabeled":
        return read_table(
            root / "unlabeled" / name / f"{config.unlabeled}.parquet",
            UNLABELED_COLUMNS,
            f"the unlabeled table the config's `unlabeled` key names ({config.unlabeled!r})",
        ), None
    negatives = read_table(
        root / "pu" / name / f"{config.negatives}.negatives.parquet",
        NEGATIVE_COLUMNS,
        f"the negatives table the config's `negatives` key names ({config.negatives!r}); "
        f"run scripts/score_pu.py for that config first",
    )
    spy_path = CONFIGS / f"{config.negatives}.yml"
    if not spy_path.exists():
        raise ValueError(
            f"the config's `negatives` key names {config.negatives!r}, but there is no stage S2a "
            f"config at {spy_path}"
        )
    spy_config = load_pu_config(spy_path)
    check_negatives_pointer(config, spy_config, negatives)
    return negatives, config_sha256(spy_config)


def scored_licences(
    scored: dict[str, pd.DataFrame],
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
) -> dict[str, str]:
    """One licence per scored set, widened by every table the fit read.

    Per set and not one for the whole table: every score comes from a model fitted on all of P and
    all of the negative class, so an open validation row's score is derived from every licensed row
    the fit saw -- but one value for the whole table would promote the validation rows to the
    inventory's ``internal`` and lose a true fact about them.

    **Known gap, recorded rather than closed** (``docs/design/modeling.md``, S2b "Licence"): the
    feature table's own licences are not in this merge. It is incomplete rather than wrong today,
    because the fit sees only fingerprint counts of molecules that are in an open corpus, and the
    ``internal`` and ``noncommercial`` stamps on 535 and 16 of those feature rows record
    co-occurrence with the lab inventory or with validation, which never reaches the fit. It becomes
    wrong the day a substrate whose structure comes only from a private source enters P or U.
    """
    trained_on = [*positives["licence"], *negatives["licence"]]
    return {
        name: strictest_licence([*pairs["licence"], *trained_on])
        for name, pairs in scored.items()
    }


def score_table(
    scored: dict[str, pd.DataFrame],
    scores: dict[str, np.ndarray],
    licences: dict[str, str],
    std_version: str,
    config: PuScoreConfig,
) -> pd.DataFrame:
    """Stage S1's eleven columns in stage S1's order, one block of rows per scored set.

    The layout is S1's unchanged so that stage E reads one table for baselines and PU models alike;
    ``feature_block`` is never null here, because a PU model always reads one, and there is no rank
    column, because a rank is derived and its meaning depends on a tie convention.
    """
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "score_set": name,
                    "key_a_stereo": pairs["key_a_stereo"].to_numpy(),
                    "key_b_stereo": pairs["key_b_stereo"].to_numpy(),
                    "score": scores[name],
                    "model": config.model,
                    "feature_block": config.feature_block,
                    "p_width": config.p_width,
                    "seed": config.seed,
                    "config_sha256": config_sha256(config),
                    "std_version": std_version,
                    "licence": licences[name],
                }
            )
            for name, pairs in scored.items()
        ],
        ignore_index=True,
    )[COLUMNS]


@dataclass(frozen=True)
class Inputs:
    """Every table this stage reads, and the two facts derived while reading them.

    A record rather than five return values, so the reading can live in one function without
    `main` unpacking a tuple nobody can name. `negatives_key` is the name of the *table* read --
    `negatives` for the spy arm, `unlabeled` for the naive one -- which is deliberately not
    `negative_source`'s word for the rows used: the spy arm reads the whole S2a table and keeps only
    its flagged rows, and calling that row count `reliable_negatives` would say there are ten of
    them where there are four.
    """

    config: PuScoreConfig
    substrates: pd.DataFrame
    positives: pd.DataFrame
    negatives: pd.DataFrame
    negatives_key: str
    scored: dict[str, pd.DataFrame]
    spy_hash: str | None
    std_version: str


def read_inputs(args: argparse.Namespace) -> Inputs:
    """All of this stage's reading, each table with an explicit column list.

    The column lists are as much the point as the paths: a stage that names its columns cannot be
    handed a benchmark label it never asked for, which is how stage S keeps one out of the process.
    """
    config = load_pu_score_config(args.config)
    substrates = read_table(
        args.root / "private" / "features" / args.name / "substrates.parquet",
        ["key_stereo", config.feature_block, SIZE, "std_version"],
        f"the feature table, read for block {config.feature_block!r}",
    )
    positives = read_table(
        args.root / "positives" / args.name / "positives.parquet",
        POSITIVE_COLUMNS,
        "the positives table",
    )
    positives = positives[positives["family"] == config.p_width]
    negatives, spy_hash = negative_class(args.name, args.root, config)
    negatives_key = "unlabeled" if config.negatives is None else "negatives"
    scored = {
        "validation": read_table(
            args.validation, VALIDATION_COLUMNS, "the validation split"
        ),
        "inventory": read_table(
            args.root / "private" / "unlabeled" / args.name / f"{args.family}.parquet",
            PAIR_COLUMNS,
            f"the inventory pairs of family {args.family!r}",
        ),
    }
    return Inputs(
        config=config,
        substrates=substrates,
        positives=positives,
        negatives=negatives,
        negatives_key=negatives_key,
        scored=scored,
        spy_hash=spy_hash,
        std_version=one_std_version(
            {
                "positives": positives,
                negatives_key: negatives,
                "substrates": substrates,
                "validation": scored["validation"],
            }
        ),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Fit one arm on one block, score both sets and write the table and its report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name")
    parser.add_argument("family")
    parser.add_argument("root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.validation.name != "validation.parquet":
        raise ValueError(
            f"--validation must be a validation.parquet: {args.validation}"
        )

    inputs = read_inputs(args)
    config, substrates, positives = inputs.config, inputs.substrates, inputs.positives
    negatives, scored = inputs.negatives, inputs.scored

    train = training_rows(positives, negatives, config)
    scores, report = pair_scores(train, scored, substrates, config)
    licences = scored_licences(scored, positives, negatives)
    table = score_table(scored, scores, licences, inputs.std_version, config)
    out = (
        args.root
        / "private"
        / "scores"
        / args.name
        / args.family
        / f"{args.config.stem}.parquet"
    )
    report |= {
        "stage": "score_pu_scores",
        "class": args.name,
        "family": args.family,
        "config": str(args.config),
        "negatives": config.negatives,
        "spy_config_sha256": inputs.spy_hash,
        "validation": str(args.validation),
        "rows_in": {
            "positives": len(positives),
            inputs.negatives_key: len(negatives),
            "substrates": len(substrates),
            **{name: len(pairs) for name, pairs in scored.items()},
        },
        "rows_out": len(table),
        "std_version": inputs.std_version,
    }
    write_table(pa.Table.from_pandas(table, preserve_index=False), out, root=args.root)
    write_report(out.with_suffix(".report.json"), report, [args.config])
    print(
        f"{args.name} {args.family} {args.config.stem}: {len(table):,} scores "
        f"({report['n_positives']:,} P against {report['n_negatives']:,} "
        f"{report['negative_source']}) -> {out}"
    )


if __name__ == "__main__":
    main()
