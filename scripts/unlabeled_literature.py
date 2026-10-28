"""Stage U: draw the literature-built unlabeled space (docs/design/modeling.md).

Reads the class's positives and the stage F feature table; writes ``<root>/unlabeled/<class>/``.
The table is **outside** ``private/``: every substrate comes from the open positives, so the whole
training side of the model is publishable and only the scoring side is internal.

Usage:
    uv run python scripts/unlabeled_literature.py amide_coupling ROOT --config configs/models/u.yml
"""

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from platepu.io import write_report, write_table
from platepu.positives.unlabeled import (
    UnlabeledConfig,
    load_unlabeled_config,
    unlabeled_pairs,
)


def rank_auc(score: np.ndarray, positive: np.ndarray) -> float:
    """ROC-AUC of a one-dimensional score, ties sharing their average rank.

    Ranked directly and not fitted. On a single monotone feature with these class sizes a
    classifier has nothing to learn and reports its own noise: on the first real draw it returned
    0.377 where the rank-AUC is 0.5036 (ADR 017).
    """
    ranks = pd.Series(score).rank().to_numpy()
    n_positive = int(positive.sum())
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * (len(score) - n_positive))
    )


def null_total_variation(shares: np.ndarray, size: int, draws: int, seed: int) -> float:
    """The total variation a **perfect** sampler would still show, at the 99th percentile.

    Drawing ``size`` items from ``shares`` leaves a distance from ``shares`` by sampling noise
    alone, and that floor grows as P shrinks: the real 28,158-pair draw sits near 0.05 while a
    correct 40-pair one reads about 0.105. Gating on an absolute number would therefore false-alarm
    on a small class and go slack on a big one, so the gate is a multiple of this.
    """
    generator = np.random.default_rng(seed)
    counts = generator.multinomial(size, shares, size=draws)
    distances = np.abs(counts / size - shares).sum(axis=1) / 2
    return float(np.percentile(distances, 99))


def marginals(
    positives: pd.DataFrame, unlabeled: pd.DataFrame, key: str, config: UnlabeledConfig
) -> dict[str, float]:
    """How closely one role's U frequencies follow its P frequencies, and the rank-AUC of that side.

    These two numbers are the stage's **gate** (ADR 017). ``main`` raises past the tolerances in the
    config rather than reporting them and carrying on; the joint probes are diagnostics and live in
    ``scripts/counts/count_u_probes.py``, which keeps a model out of the pipeline stage.
    """
    in_p = positives[key].value_counts()
    in_u = unlabeled.groupby(key)["n_draws"].sum()
    share_p = in_p / in_p.sum()
    share_u = in_u.reindex(share_p.index).fillna(0) / in_u.sum()
    deviation = (share_u - share_p).abs()
    drawn = unlabeled.loc[unlabeled.index.repeat(unlabeled["n_draws"])]
    score = np.r_[
        positives[key].map(in_p).to_numpy(float), drawn[key].map(in_p).to_numpy(float)
    ]
    is_positive = np.r_[
        np.ones(len(positives), dtype=bool), np.zeros(len(drawn), dtype=bool)
    ]
    floor = null_total_variation(
        share_p.to_numpy(), int(in_u.sum()), config.gate_null_draws, config.seed
    )
    return {
        "rank_auc": round(rank_auc(score, is_positive), 4),
        "total_variation": round(float(deviation.sum() / 2), 5),
        "null_total_variation": round(floor, 5),
        "total_variation_allowed": round(floor * config.gate_marginal_tv_multiple, 5),
        "rank_auc_tolerance": config.gate_rank_auc_tolerance,
        "largest_deviation": round(float(deviation.max()), 5),
        "substrates_absent_from_u": float(
            (in_u.reindex(share_p.index).fillna(0) == 0).sum()
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Draw U for one class and write its table, its rejects and its report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="class name, e.g. amide_coupling")
    parser.add_argument("root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)

    start = time.perf_counter()
    config = load_unlabeled_config(args.config)
    positives = pd.read_parquet(
        args.root / "positives" / args.name / "positives.parquet"
    )
    substrates = pd.read_parquet(
        args.root / "private" / "features" / args.name / "substrates.parquet",
        columns=["key_stereo", "n_heavy_atoms"],
    )
    unlabeled, rejected = unlabeled_pairs(positives, substrates, config)

    width = positives[positives["family"] == config.p_width]
    gate = {
        "acid": marginals(width, unlabeled, "key_a_stereo", config),
        "amine": marginals(width, unlabeled, "key_b_stereo", config),
    }
    # Raise before writing. A U whose marginals do not match P's reopens the shortcut the sampler
    # exists to close, and it would be invisible in every table downstream.
    for role, measured in gate.items():
        drift = abs(measured["rank_auc"] - 0.5)
        if drift > config.gate_rank_auc_tolerance:
            raise ValueError(
                f"{role} popularity rank-AUC {measured['rank_auc']} is {drift:.4f} from 0.5, "
                f"past the configured {config.gate_rank_auc_tolerance}; nothing was written"
            )
        if measured["total_variation"] > measured["total_variation_allowed"]:
            raise ValueError(
                f"{role} marginal total variation {measured['total_variation']} is past "
                f"{measured['total_variation_allowed']}, which is "
                f"{config.gate_marginal_tv_multiple} times the simulated null "
                f"{measured['null_total_variation']}; nothing was written"
            )

    out = args.root / "unlabeled" / args.name
    stem = args.config.stem
    write_table(
        pa.Table.from_pandas(unlabeled, preserve_index=False),
        out / f"{stem}.parquet",
        root=args.root,
    )
    write_table(
        pa.Table.from_pandas(rejected, preserve_index=False),
        out / f"{stem}.rejected.parquet",
        root=args.root,
    )
    report: dict[str, object] = {
        "stage": "unlabeled_literature",
        "class": args.name,
        "positives": {
            "pairs": len(width),
            "acids": int(width["key_a_stereo"].nunique()),
            "amines": int(width["key_b_stereo"].nunique()),
            # The share this stage exists to neutralise: a substrate reported once sits in P once
            # and would sit in a uniform U as often as one reported hundreds of times.
            "single_partner_acids": int(
                (width["key_a_stereo"].value_counts() == 1).sum()
            ),
            "single_partner_amines": int(
                (width["key_b_stereo"].value_counts() == 1).sum()
            ),
        },
        "draws": {
            "k": config.k,
            "seed": config.seed,
            "requested": config.k * len(width),
            "out": int(unlabeled["n_draws"].sum()),
            "rejected": {"pair_in_p_flat": len(rejected)},
            "pairs_out": len(unlabeled),
        },
        "gate": gate,
        "seconds": round(time.perf_counter() - start, 1),
    }
    write_report(out / f"{stem}.report.json", report, configs=[args.config])
    print(
        f"{args.name} {stem}: {len(unlabeled):,} pairs from "
        f"{int(unlabeled['n_draws'].sum()):,} draws -> {out}"
    )


if __name__ == "__main__":
    main()
