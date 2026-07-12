"""Stage 2-3: merge the per-source parsed parquets, dedup, and build the group catalogue.

Reads every ``<source>_parsed.parquet`` from stage 1, reports cross-source overlap, then groups the
pooled reactions into *"same reaction, commonish conditions"* candidate plates and writes a ranked
catalogue. A group is one (reaction class) x (condition cluster); the ``other`` bucket is sub-split
by its reaction-center signature (the unsupervised layer). Within each group the plate-relevant
metrics are computed on **unique substrate pairs** (so a reaction reported many times across sources
does not inflate the matrix size), while raw report frequency is kept as a popularity signal.

Condition cluster = (metal, ligand[only when metal-catalysed], base family, solvent class, temp bin)
-- the family binning is the "small deviations OK" knob.

Usage:
    uv run python -m scripts_JSP.rxngroup_merge_group --yield-threshold 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pubench.rxngroup.sources import PUBLIC_SOURCES

IN_DIR = Path("datasets/processed/private/rxn_groups")


def _class_labels(df: pd.DataFrame) -> pd.Series:
    """Tier-1 label per row (vectorised): the named class, or ``other:<center-sig>`` for ``other``.

    Args:
        df: Parsed reactions with ``reaction_class`` and ``center_sig`` columns.

    Returns:
        The tier-1 class label for every row.
    """
    labels = df["reaction_class"].astype("object")
    is_other = df["reaction_class"] == "other"
    sig = df["center_sig"].fillna("").astype("object")
    other_label = np.where(sig.to_numpy() != "", "other:" + sig, "other:unmapped")
    labels = labels.mask(is_other, pd.Series(other_label, index=df.index))
    return labels


def _load(in_dir: Path) -> pd.DataFrame:
    """Load and concatenate every ``*_parsed.parquet`` in ``in_dir``."""
    files = sorted(p for p in in_dir.glob("*_parsed.parquet"))
    if not files:
        raise FileNotFoundError(f"No *_parsed.parquet in {in_dir}; run rxngroup_parse first.")
    frames = [pd.read_parquet(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    logging.info("loaded %s rows from %d source files", f"{len(df):,}", len(files))
    return df


def _overlap_report(df: pd.DataFrame) -> dict[str, int]:
    """Cross-source dedup statistics on the canonical reaction key."""
    per_key_sources = df.groupby("rxn_key")["source"].nunique()
    return {
        "total_rows": len(df),
        "unique_reactions": int(df["rxn_key"].nunique()),
        "reactions_in_multiple_sources": int((per_key_sources > 1).sum()),
    }


def build_groups(df: pd.DataFrame, *, yield_threshold: float) -> pd.DataFrame:
    """Group pooled reactions into (class x condition-cluster) rows with plate-selection metrics.

    Args:
        df: The pooled per-source parsed reactions.
        yield_threshold: Percent yield above which a reaction counts as a "hit" (headroom proxy).

    Returns:
        One row per group, ranked by the number of unique substrate pairs (matrix size).
    """
    df = df.copy()
    df["class_label"] = _class_labels(df)
    # Ligand only distinguishes clusters for metal-catalysed reactions.
    df["cluster_ligand"] = np.where(
        (df["metal"] != "none") & df["ligand_key"].notna(), df["ligand_key"], "na"
    )
    df["pair"] = df["partner_a"].fillna("") + ">>" + df["partner_b"].fillna("")

    keys = ["class_label", "metal", "cluster_ligand", "base_family", "solvent_family", "temp_bin"]
    records: list[dict[str, object]] = []
    for key, g in df.groupby(keys, dropna=False):
        has_pair = g["partner_a"].notna()
        pub = g["source"].isin(PUBLIC_SOURCES)
        pairs = g.loc[has_pair, "pair"]
        yields = g["yield_value"].dropna()
        src_counts = g["source"].value_counts()
        rep = g.loc[has_pair].head(1)
        records.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n_reports": len(g),
                "n_unique_reactions": int(g["rxn_key"].nunique()),
                "n_unique_pairs": int(pairs.nunique()),
                # Same matrix from redistributable (open) sources only -> the reproducible core.
                "n_pairs_public": int(g.loc[has_pair & pub, "pair"].nunique()),
                "n_electrophiles": int(g.loc[has_pair, "partner_a"].nunique()),
                "n_nucleophiles": int(g.loc[has_pair, "partner_b"].nunique()),
                "sources": ",".join(sorted(src_counts.index)),
                "n_public": int(pub.sum()),
                "n_licensed": int((~pub).sum()),
                "yield_n": len(yields),
                "yield_median": float(yields.median()) if len(yields) else float("nan"),
                "yield_frac_hit": float((yields >= yield_threshold).mean())
                if len(yields)
                else float("nan"),
                "example_a": (rep["partner_a"].iloc[0] if len(rep) else None),
                "example_b": (rep["partner_b"].iloc[0] if len(rep) else None),
            }
        )
    groups = pd.DataFrame.from_records(records)
    return groups.sort_values(["n_unique_pairs", "n_reports"], ascending=False).reset_index(
        drop=True
    )


def main() -> None:
    """Merge, dedup-report, group, and persist the ranked catalogue."""
    parser = argparse.ArgumentParser(
        description="Merge + group parsed reactions into plate groups."
    )
    parser.add_argument("--in-dir", type=Path, default=IN_DIR)
    parser.add_argument(
        "--yield-threshold",
        type=float,
        default=20.0,
        help="Percent-yield hit threshold (matches Chemlex Conversion>=20).",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=0,
        help="Drop NAMED groups with fewer than this many unique substrate pairs (0 = keep all, "
        "so unimolecular FGIs with no substrate matrix are retained).",
    )
    parser.add_argument(
        "--min-other-reports",
        type=int,
        default=25,
        help="Keep an 'other:<center-sig>' group only if reported at least this many times "
        "(tames the long tail of one-off unnamed transformations).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s"
    )
    df = _load(args.in_dir)
    overlap = _overlap_report(df)
    logging.info(
        "dedup: %s rows -> %s unique reactions (%s reported in >1 source)",
        f"{overlap['total_rows']:,}",
        f"{overlap['unique_reactions']:,}",
        f"{overlap['reactions_in_multiple_sources']:,}",
    )

    # Full catalogue (all sources, incl. licensed Pistachio) + a public-only reproducible catalogue.
    _catalogue(df, args, tag="all")
    _catalogue(df[df["source"].isin(PUBLIC_SOURCES)], args, tag="public")


def _catalogue(df: pd.DataFrame, args: argparse.Namespace, *, tag: str) -> None:
    """Build, threshold, persist and print one catalogue (``all`` or ``public``)."""
    groups = build_groups(df, yield_threshold=args.yield_threshold)
    is_other = groups["class_label"].str.startswith("other")
    # Keep every named group with >= min_pairs (0 -> all, incl. no-matrix FGIs); threshold 'other'.
    keep = (~is_other & (groups["n_unique_pairs"] >= args.min_pairs)) | (
        is_other & (groups["n_reports"] >= args.min_other_reports)
    )
    groups = groups[keep].reset_index(drop=True)
    logging.info(
        "[%s] %s named + %s other groups kept (from %s rows)",
        tag,
        f"{int((~is_other).sum()):,}",
        f"{int(is_other.sum()):,}",
        f"{len(df):,}",
    )

    out_parquet = args.in_dir / f"reaction_condition_groups_{tag}.parquet"
    out_csv = args.in_dir / f"groups_summary_{tag}.csv"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    groups.to_parquet(out_parquet, index=False)
    groups.to_csv(out_csv, index=False)
    logging.info("[%s] wrote %s groups -> %s (+ .csv)", tag, f"{len(groups):,}", out_parquet)

    named = groups[~groups["class_label"].str.startswith("other")]
    cols = [
        "class_label",
        "metal",
        "base_family",
        "solvent_family",
        "temp_bin",
        "n_unique_pairs",
        "n_pairs_public",
        "n_electrophiles",
        "n_nucleophiles",
        "n_reports",
        "sources",
    ]
    with pd.option_context("display.max_rows", 40, "display.width", 220):
        print(f"\n=== [{tag}] Top named-reaction condition groups (by unique substrate pairs) ===")
        print(named[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
