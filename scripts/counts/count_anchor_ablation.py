"""The anchor ablation: Chen's K-NN probe on the anchored U and on the unanchored control.

Reads only P and the two unlabeled tables. No label, no Chemlex. The reading rule was fixed before
this ran: every number here is a COMPARISON between the two U's, never a level, because P can be
told from U by real chemistry and a high score alone proves nothing.

Usage:
    uv run python scripts/counts/count_anchor_ablation.py ~/University/Phd/Dataset/Datasets
"""

import sys
from pathlib import Path

import pandas as pd

from platepu.score.pair_knn import balanced_sample, pair_knn_auc

if len(sys.argv) != 2:
    raise SystemExit("usage: scripts/counts/count_anchor_ablation.py <Datasets root>")
ROOT = Path(sys.argv[1]).expanduser() / "derived"
PER_CLASS = 3000
SEED = 20261029
KS = (1, 3, 5, 11, 25)
STEMS = ("unlabeled_literature_k10", "unlabeled_both_sides_k10")

positives = pd.read_parquet(f"{ROOT}/positives/amide_coupling/positives.parquet")
positives = positives[positives["family"] == "all"]
substrates = pd.read_parquet(
    f"{ROOT}/private/features/amide_coupling/substrates.parquet"
)
print(f"P pairs {len(positives):,}   substrates {len(substrates):,}")

for block in ("morgan_centre", "morgan"):
    print(f"\n=== block {block}")
    print(f"{'U':28} " + " ".join(f"K={k:<6}" for k in KS) + " best")
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
        auc = pair_knn_auc(pool, substrates, block, KS)
        best = max(auc, key=lambda k: auc[k])
        print(f"{stem:28} " + " ".join(f"{auc[k]:<8.4f}" for k in KS) + f" K={best}")
