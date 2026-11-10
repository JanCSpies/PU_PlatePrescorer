"""What can separate P from U, feature set by feature set, on both unlabeled spaces.

The estimator follows the column-count rule the project already uses in count_adversarial:
logistic regression above 100 columns, boosting below. A boosted tree with 120 iterations on
4,096 sparse columns and 6,000 rows underfits, and reported morgan BELOW morgan_centre -- an
instrument result, not a data one. The folds are SHUFFLED: the pool is built label by label and
key-sorted within each, so unshuffled stratified folds hand each split a contiguous block of the
alphabet and the fingerprint rows came out BELOW chance.

One instrument throughout -- the same gradient-boosted classifier under 5-fold CV -- so the rows are
comparable. Reading across a column answers "what does the fingerprint add over frequency and
size"; reading across the two U's answers the anchor question with a different instrument from the
K-NN probe. Label-free: no Chemlex, no yield.

Usage:
    uv run python scripts/counts/count_block_ladder.py ~/University/Phd/Dataset/Datasets
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from platepu.features.blocks import block_matrix
from platepu.score.axes import make_model
from platepu.score.pair_knn import balanced_sample

if len(sys.argv) != 2:
    raise SystemExit("usage: scripts/counts/count_block_ladder.py <Datasets root>")
ROOT = Path(sys.argv[1]).expanduser() / "derived"
PER_CLASS, SEED = 3000, 20261029
STEMS = ("unlabeled_literature_k10", "unlabeled_both_sides_k10")

positives = pd.read_parquet(f"{ROOT}/positives/amide_coupling/positives.parquet")
positives = positives[positives["family"] == "all"]
substrates = pd.read_parquet(
    f"{ROOT}/private/features/amide_coupling/substrates.parquet"
)
by_key = substrates.set_index("key_stereo")
per_acid = positives["key_a_stereo"].value_counts()
per_amine = positives["key_b_stereo"].value_counts()


def features(pool: pd.DataFrame, name: str) -> np.ndarray:
    """One feature set for every pair in the pool, both sides concatenated."""
    a, b = pool["key_a_stereo"], pool["key_b_stereo"]
    if name == "frequency":
        return np.log1p(
            np.column_stack(
                [
                    a.map(per_acid).fillna(0).to_numpy(),
                    b.map(per_amine).fillna(0).to_numpy(),
                ]
            )
        )
    rows = [by_key.loc[a].reset_index(), by_key.loc[b].reset_index()]
    return np.hstack([block_matrix(side, name) for side in rows])


print(f"P {len(positives):,}  balanced pool {PER_CLASS:,} per class, seed {SEED}\n")
sets = ["frequency", "size alone", "descriptors", "morgan_centre", "morgan"]
print(f"{'feature set':16} " + "  ".join(f"{s.split('_')[-1][:9]:>9}" for s in STEMS))
for name in sets:
    line = []
    for stem in STEMS:
        unlabeled = pd.read_parquet(f"{ROOT}/unlabeled/amide_coupling/{stem}.parquet")
        labelled = pd.concat(
            [
                positives[["key_a_stereo", "key_b_stereo"]].assign(label=1),
                unlabeled[["key_a_stereo", "key_b_stereo"]].assign(label=0),
            ],
            ignore_index=True,
        )
        pool = balanced_sample(labelled, PER_CLASS, SEED)
        matrix = features(pool, name)
        label = pool["label"].to_numpy()
        scores = cross_val_score(
            make_model(matrix.shape[1]),
            matrix,
            label,
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
            scoring="roc_auc",
        )
        line.append(float(scores.mean()))
    print(f"{name:16} " + "  ".join(f"{v:9.4f}" for v in line))
