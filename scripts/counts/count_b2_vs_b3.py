"""Is the B2-over-B3 gap real, and why do two near-identical steric scores give different AUCs?

Two questions behind the first stage S1 run:
1. B5 is the rank sum of B4 and a steric term. Two steric terms that agree at Spearman 0.98 give
   0.791 and 0.775. This prints how many distinct values each takes, because B4 has only four, so
   almost all of B5's power is in how the steric term breaks B4's ties.
2. B2 sees which acid went with which amine; B3 cannot. B2 wins by about 0.012. This runs Owen's
   pigeonhole bootstrap (resampling acids and amines, not reactions) for the paired difference, so
   we know whether that gap is more than sampling noise before anyone writes it down as a finding.

Validation only; the test side is never read (asserted).

Usage:
    uv run python scripts/counts/count_b2_vs_b3.py ~/University/Phd/Dataset/Datasets
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

if len(sys.argv) != 2:
    raise SystemExit("usage: scripts/counts/count_b2_vs_b3.py <Datasets root>")
ROOT = Path(sys.argv[1]).expanduser()
FAMILY = "HATU_DIPEA_DMF"
RESAMPLES = 2000
SEED = 20261022
# Read from the model config rather than re-typed here, so this script cannot drift from the B4
# the stage actually scores.
CLASS_SCORE = yaml.safe_load(
    Path("configs/models/baselines.yml").read_text(encoding="utf-8")
)["amine_subclass_scores"]


def weighted_auc(scores: np.ndarray, positive: np.ndarray, weight: np.ndarray) -> float:
    """ROC-AUC with a weight per reaction; tied scores count half, as Mann-Whitney does."""
    _, group = np.unique(scores, return_inverse=True)
    negative_weight = np.bincount(
        group, weights=weight * ~positive, minlength=group.max() + 1
    )
    below = np.cumsum(negative_weight) - negative_weight
    positive_weight = weight * positive
    wins = positive_weight * (below[group] + 0.5 * negative_weight[group])
    return float(wins.sum() / (positive_weight.sum() * (weight * ~positive).sum()))


def tie_structure(features: pd.DataFrame, validation: pd.DataFrame) -> None:
    """Why two steric terms that rank the same give different AUCs: they break B4's ties differently."""
    amine = validation["key_b_stereo"]
    subclass = amine.map(features["subclass"]).map(CLASS_SCORE)
    one = -amine.map(features["n_heavy_at_2_bonds"]).astype(float)
    two = -(
        amine.map(features["alpha_sum_heavy_degree"])
        + amine.map(features["n_heavy_at_2_bonds"])
    ).astype(float)
    # A key missing from the features would map to NaN, rank as NaN, and numpy's unique sorts NaN
    # last, so a lost substrate would silently become the best-scoring row and the printed AUC would
    # be wrong with no sign of it.
    assert subclass.notna().all() and one.notna().all() and two.notna().all(), (
        "a validation amine is missing from the feature table"
    )
    positive = validation["label"].to_numpy(dtype=bool)
    ones = np.ones(len(validation))
    print(
        f"B4 alone: {subclass.nunique()} distinct values, AUC {weighted_auc(subclass.to_numpy(), positive, ones):.3f}"
    )
    for name, steric in (("one descriptor", one), ("two descriptors", two)):
        combined = subclass.rank().to_numpy() + steric.rank().to_numpy()
        print(
            f"  {name:16s}: steric {steric.nunique():>2} values, B5 {len(set(combined)):>2} values, "
            f"AUC {weighted_auc(combined, positive, ones):.3f}"
        )
    print(
        f"  Spearman between the two steric terms: {one.corr(two, method='spearman'):.3f}"
    )


def pair_bootstrap(
    validation: pd.DataFrame, left: str, right: str, factors: str = "both"
) -> None:
    """Pigeonhole bootstrap of AUC(left) - AUC(right): resample whole acids and whole amines.

    ``factors`` is ``both``, ``acids`` or ``amines``; the one-factor runs show which of the two
    small substrate sets carries the uncertainty.

    The resampled units are the **flat** keys. Our fingerprints are chirality-blind, so two
    stereoisomers are one molecule to every model here, and resampling them separately would treat
    one molecule as two independent draws. Validation holds 94 acid and 112 amine flat keys against
    95 and 113 stereo keys (one collapsing pair on each side, 2026-10-22).
    """
    positive = validation["label"].to_numpy(dtype=bool)
    acids, acid_index = np.unique(validation["key_a_flat"], return_inverse=True)
    amines, amine_index = np.unique(validation["key_b_flat"], return_inverse=True)
    scores = {name: validation[name].to_numpy() for name in (left, right)}
    plain = {
        name: weighted_auc(value, positive, np.ones(len(validation)))
        for name, value in scores.items()
    }
    generator = np.random.default_rng(SEED)
    differences = np.empty(RESAMPLES)
    for draw in range(RESAMPLES):
        acid_count = (
            np.bincount(
                generator.integers(0, len(acids), len(acids)), minlength=len(acids)
            )
            if factors in ("both", "acids")
            else np.ones(len(acids), dtype=int)
        )
        amine_count = (
            np.bincount(
                generator.integers(0, len(amines), len(amines)), minlength=len(amines)
            )
            if factors in ("both", "amines")
            else np.ones(len(amines), dtype=int)
        )
        weight = (acid_count[acid_index] * amine_count[amine_index]).astype(float)
        kept = weight > 0
        if (
            not (positive[kept] * weight[kept]).any()
            or not (~positive[kept] * weight[kept]).any()
        ):
            differences[draw] = np.nan
            continue
        differences[draw] = weighted_auc(scores[left], positive, weight) - weighted_auc(
            scores[right], positive, weight
        )
    usable = differences[~np.isnan(differences)]
    low, high = np.percentile(usable, [2.5, 97.5])
    print(
        f"{left} {plain[left]:.3f} vs {right} {plain[right]:.3f} [{factors:6s}]: "
        f"difference {plain[left] - plain[right]:+.3f}, 95 % interval [{low:+.3f}, {high:+.3f}], "
        f"half-width {(high - low) / 2:.3f}, {left} wins {(usable > 0).mean():.1%} of {len(usable)}"
    )


def width_coincidence(scored: pd.DataFrame) -> None:
    """Are B3's two P widths the same score vector, or two different ones with the same AUC?

    B3 lands on 0.621 at both widths, which looks like a bug in which the class width silently
    replaces the family width. It is not: the two vectors disagree badly (printed below), so the
    equal AUC is a coincidence of the ranking and nothing more.
    """
    both = scored[scored["model"] == "B3"].pivot_table(
        index=["key_a_stereo", "key_b_stereo"], columns="p_width", values="score"
    )
    family, whole = both.columns[0], both.columns[1]
    print(
        f"B3 score vectors identical across P widths: {both[family].equals(both[whole])}; "
        f"Spearman {both.corr(method='spearman').iloc[0, 1]:.3f}; "
        f"means {both[family].mean():.3f} ({family}) and {both[whole].mean():.3f} ({whole})"
    )


def main() -> None:
    """Print the tie structure of B5 and the bootstrap interval of B2 minus B3."""
    features = pd.read_parquet(
        ROOT
        / "derived"
        / "private"
        / "features"
        / "amide_coupling"
        / "substrates.parquet"
    ).set_index("key_stereo")
    validation = pd.read_parquet(
        ROOT
        / "derived_samples"
        / "private"
        / "benchmarks"
        / "chemlex"
        / FAMILY
        / "validation.parquet"
    )
    assert "test" not in set(validation["both_unseen"]), (
        "the validation file must hold no test rows"
    )
    validation = validation.assign(label=validation["label"].astype(bool))
    scored = pd.read_parquet(
        ROOT
        / "derived"
        / "private"
        / "scores"
        / "amide_coupling"
        / FAMILY
        / "baselines.parquet"
    )
    scored = scored[scored["score_set"] == "validation"]

    print("--- why two near-identical steric terms give different AUCs")
    tie_structure(features, validation)

    wide = validation[
        ["key_a_stereo", "key_b_stereo", "key_a_flat", "key_b_flat", "label"]
    ].copy()
    for (model, width), group in scored.groupby(["model", "p_width"], dropna=False):
        name = (
            model
            if isinstance(width, str) is False
            else f"{model}_{'hatu' if width == FAMILY else 'class'}"
        )
        merged = wide.merge(
            group[["key_a_stereo", "key_b_stereo", "score"]],
            on=["key_a_stereo", "key_b_stereo"],
        )
        wide[name] = merged["score"].to_numpy()

    print("\n--- is B3's equal AUC at both P widths one vector or two?")
    width_coincidence(scored)

    print("\n--- does knowing the pair beat knowing the two substrates?")
    for width in ("class", "hatu"):
        pair_bootstrap(wide, f"B2_{width}", f"B3_{width}")
    print("\n--- for scale, the same test against the strongest baseline")
    pair_bootstrap(wide, "B6", "B2_class")
    print("\n--- how big an edge over a pre-registered baseline is detectable here")
    for left, right in (("B6", "B4"), ("B4", "B1"), ("B6", "B1")):
        pair_bootstrap(wide, left, right)
    print("\n--- which of the two small substrate sets carries the uncertainty")
    for factors in ("acids", "amines"):
        pair_bootstrap(wide, "B6", "B2_class", factors)
        pair_bootstrap(wide, "B2_class", "B3_class", factors)


if __name__ == "__main__":
    main()
