"""A control with a known answer, to find out whether my pipeline or my conclusion is wrong.

Three measurements through the SAME path -- block_matrix, make_model, cross-validated ROC-AUC:

1. literature-only against lab-only substrates on morgan. The published count script reports 0.859
   for acids and 0.879 for amines. If this path cannot reproduce that, the path is broken and the
   P-vs-U fingerprint numbers mean nothing.
2. the same, with folds shuffled, since the published script does not shuffle.
3. P against U on the pair-level fingerprint, raw counts and binarised, to test whether treating
   counts as magnitudes is what a linear model cannot use.

Usage:
    uv run python scripts/counts/count_pipeline_control.py ~/University/Phd/Dataset/Datasets
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
    raise SystemExit("usage: scripts/counts/count_pipeline_control.py <Datasets root>")
ROOT = Path(sys.argv[1]).expanduser() / "derived"
SEED, PER_CLASS = 20261029, 3000
substrates = pd.read_parquet(
    f"{ROOT}/private/features/amide_coupling/substrates.parquet"
)


def auc(matrix: np.ndarray, label: np.ndarray, shuffle: bool) -> float:
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED) if shuffle else 5
    model = make_model(matrix.shape[1])
    return float(
        cross_val_score(model, matrix, label, cv=folds, scoring="roc_auc").mean()
    )


print("=== 1+2. the known answer: literature-only vs lab-only substrates, morgan")
print(
    f"{'role':8} {'n_lit':>7} {'n_lab':>6} {'unshuffled':>11} {'shuffled':>9}   published"
)
for role, want in (("acid", 0.859), ("amine", 0.879)):
    rows = substrates[substrates["role"] == role]
    literature = rows["in_positives"] & ~rows["in_library"]
    lab = rows["in_library"] & ~rows["in_positives"]
    used = rows[literature | lab].reset_index(drop=True)
    source = used["in_library"].to_numpy(dtype=int)
    matrix = block_matrix(used, "morgan")
    print(
        f"{role:8} {int((1 - source).sum()):>7,} {int(source.sum()):>6,} "
        f"{auc(matrix, source, False):>11.3f} {auc(matrix, source, True):>9.3f}   {want}"
    )

print("\n=== 3. P against U on pair fingerprints, raw counts vs binarised")
positives = pd.read_parquet(f"{ROOT}/positives/amide_coupling/positives.parquet")
positives = positives[positives["family"] == "all"]
by_key = substrates.set_index("key_stereo")
unlabeled = pd.read_parquet(
    f"{ROOT}/unlabeled/amide_coupling/unlabeled_literature_k10.parquet"
)
labelled = pd.concat(
    [
        positives[["key_a_stereo", "key_b_stereo"]].assign(label=1),
        unlabeled[["key_a_stereo", "key_b_stereo"]].assign(label=0),
    ],
    ignore_index=True,
)
pool = balanced_sample(labelled, PER_CLASS, SEED)
label = pool["label"].to_numpy()
print(f"{'block':16} {'raw counts':>11} {'binarised':>10}")
for block in ("morgan_centre", "morgan"):
    sides = [
        block_matrix(by_key.loc[pool[key]].reset_index(), block)
        for key in ("key_a_stereo", "key_b_stereo")
    ]
    raw = np.hstack(sides)
    print(
        f"{block:16} {auc(raw, label, True):>11.4f} {auc((raw > 0).astype(float), label, True):>10.4f}"
    )
