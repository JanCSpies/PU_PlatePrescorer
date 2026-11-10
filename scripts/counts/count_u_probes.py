"""The probe numbers behind stage U's sampler (ADR 015 and ADR 017).

Every figure that justifies how U is drawn is produced here, so the design can be checked rather
than taken on trust. Published under ADR 011 because these numbers go into the paper's methods.

What it answers, in the order the design argues it:

1. **The structure the sampler must preserve.** The literature is disassortative in size — a complex
   fragment is coupled to a simple building block — so P has a negative correlation between the two
   sides' heavy-atom counts. A sampler that destroys it leaves a shortcut.
2. **What an independent draw costs.** Replacing *both* sides from the marginals keeps popularity
   matched and breaks that structure, and two numbers then separate P from U.
3. **What the implemented sampler gives.** One side replaced at a time, from the same heavy-atom
   bin.
4. **The gate and the diagnostics.** The gate is the **rank**-AUC of one side's raw literature
   frequency: a single monotone feature can be ranked directly, and fitting a classifier to it
   measures the classifier instead (on the real draw it returns 0.377 where the rank-AUC is 0.504).
   The joint probes need two features, so they use a classifier and are read against a uniformly
   drawn control.

Usage:
    uv run python scripts/counts/count_u_probes.py ~/University/Phd/Dataset/Datasets
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

if len(sys.argv) != 2:
    raise SystemExit("usage: scripts/counts/count_u_probes.py <Datasets root>")
ROOT = Path(sys.argv[1]).expanduser()
CLASS = "amide_coupling"
U_CONFIG = "unlabeled_literature_k10"
SEED = 20261028


def rank_auc(score: np.ndarray, positive: np.ndarray) -> float:
    """ROC-AUC of a one-dimensional score, ties sharing their average rank.

    Used for the gate. A classifier on a single monotone feature with these class sizes has nothing
    to fit and reports its own noise, so the feature is ranked directly instead.
    """
    ranks = pd.Series(score).rank().to_numpy()
    n_positive = int(positive.sum())
    n_negative = len(score) - n_positive
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def model_auc(positive: np.ndarray, unlabeled: np.ndarray) -> float:
    """Cross-validated ROC-AUC of a classifier separating P from U on a multi-column block."""
    features = np.vstack([positive, unlabeled])
    label = np.r_[np.ones(len(positive)), np.zeros(len(unlabeled))]
    learner = HistGradientBoostingClassifier(
        random_state=7, max_iter=120, early_stopping=False
    )
    return float(
        cross_val_score(learner, features, label, cv=5, scoring="roc_auc").mean()
    )


def draw_independent(
    acids: np.ndarray, amines: np.ndarray, generator: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """The rejected sampler: both sides drawn from P's marginals, independently of each other."""
    return generator.choice(acids, len(acids)), generator.choice(amines, len(amines))


def main() -> None:
    """Print the structure, the two samplers, and stage U's gate and diagnostics."""
    positives = pd.read_parquet(
        ROOT / "derived" / "positives" / CLASS / "positives.parquet"
    )
    positives = positives[positives["family"] == "all"]
    size = pd.read_parquet(
        ROOT / "derived" / "private" / "features" / CLASS / "substrates.parquet",
        columns=["key_stereo", "n_heavy_atoms"],
    ).set_index("key_stereo")["n_heavy_atoms"]
    unlabeled = pd.read_parquet(
        ROOT / "derived" / "unlabeled" / CLASS / f"{U_CONFIG}.parquet"
    )
    drawn = unlabeled.loc[unlabeled.index.repeat(unlabeled["n_draws"])]

    acids = positives["key_a_stereo"].to_numpy()
    amines = positives["key_b_stereo"].to_numpy()
    per_acid = positives["key_a_stereo"].value_counts()
    per_amine = positives["key_b_stereo"].value_counts()

    def sizes(frame_acids: pd.Series, frame_amines: pd.Series) -> np.ndarray:
        return np.column_stack(
            [
                size.reindex(frame_acids).to_numpy(),
                size.reindex(frame_amines).to_numpy(),
            ]
        ).astype(float)

    def popularity(frame_acids: pd.Series, frame_amines: pd.Series) -> np.ndarray:
        return np.log1p(
            np.column_stack(
                [
                    frame_acids.map(per_acid).fillna(0).to_numpy(),
                    frame_amines.map(per_amine).fillna(0).to_numpy(),
                ]
            )
        )

    p_sizes = sizes(positives["key_a_stereo"], positives["key_b_stereo"])
    print("--- 1. the structure the sampler must preserve")
    print(f"P: corr(size_acid, size_amine) = {np.corrcoef(p_sizes.T)[0, 1]:+.3f}")

    generator = np.random.default_rng(SEED)
    free_a, free_b = draw_independent(acids, amines, generator)
    free_sizes = sizes(pd.Series(free_a), pd.Series(free_b))
    print("\n--- 2. what an independent draw costs (the rejected sampler)")
    print(f"U: corr(size_acid, size_amine) = {np.corrcoef(free_sizes.T)[0, 1]:+.3f}")
    print(f"joint size probe               = {model_auc(p_sizes, free_sizes):.3f}")

    u_sizes = sizes(drawn["key_a_stereo"], drawn["key_b_stereo"])
    print("\n--- 3. what the implemented sampler gives")
    print(f"U: corr(size_acid, size_amine) = {np.corrcoef(u_sizes.T)[0, 1]:+.3f}")

    print("\n--- 4. the gate (a single feature, ranked directly)")
    gate_score = np.r_[
        positives["key_a_stereo"].map(per_acid).to_numpy(float),
        drawn["key_a_stereo"].map(per_acid).to_numpy(float),
    ]
    is_positive = np.r_[
        np.ones(len(positives), dtype=bool), np.zeros(len(drawn), dtype=bool)
    ]
    print(f"one-side popularity, rank-AUC  = {rank_auc(gate_score, is_positive):.4f}")
    print(
        "the same feature through a classifier = "
        f"{model_auc(gate_score[is_positive].reshape(-1, 1), gate_score[~is_positive].reshape(-1, 1)):.4f}"
        "   (the instrument, not the data)"
    )

    print(
        "\n--- 5. the diagnostics (two features, so a classifier, read against a control)"
    )
    uniform_a = generator.choice(np.unique(acids), len(drawn))
    uniform_b = generator.choice(np.unique(amines), len(drawn))
    p_popularity = popularity(positives["key_a_stereo"], positives["key_b_stereo"])
    print(
        f"joint popularity, matched U    = "
        f"{model_auc(p_popularity, popularity(drawn['key_a_stereo'], drawn['key_b_stereo'])):.3f}"
    )
    print(
        f"joint popularity, uniform U    = "
        f"{model_auc(p_popularity, popularity(pd.Series(uniform_a), pd.Series(uniform_b))):.3f}"
        "   (the control: without it 0.5 means nothing)"
    )
    print(f"joint size, matched U          = {model_auc(p_sizes, u_sizes):.3f}")


if __name__ == "__main__":
    main()
