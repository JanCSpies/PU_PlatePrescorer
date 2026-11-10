"""Validation ROC-AUC of every baseline at the 20 % yield cut-off.

Reads the stage S1 score table and the Chemlex **validation** labels. The test side is never
opened; the file name is checked below. Numbers here replace the pre-cut-off figures in ADRs 008/012/
013/015, which were computed on P = 28,158.

Usage:
    uv run python scripts/counts/count_baseline_aucs.py ~/University/Phd/Dataset/Datasets \
        ~/University/Phd/Dataset/Datasets/derived_samples/private/benchmarks/chemlex/HATU_DIPEA_DMF/validation.parquet
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

if len(sys.argv) != 3:
    raise SystemExit(
        "usage: scripts/counts/count_baseline_aucs.py <Datasets root> <validation parquet>"
    )
ROOT = Path(sys.argv[1]).expanduser()
SCORES = (
    f"{ROOT}/derived/private/scores/amide_coupling/HATU_DIPEA_DMF/baselines.parquet"
)
VALID = Path(sys.argv[2]).expanduser()

# A file-name check, not a substring test on the whole path: the test side is never read,
# and `assert` would vanish under python -O.
if VALID.name != "validation.parquet":
    raise SystemExit(f"refusing {VALID.name}: only validation.parquet is ever read")

labels = pd.read_parquet(VALID, columns=["key_a_stereo", "key_b_stereo", "label"])
scores = pd.read_parquet(SCORES)
scores = scores[scores["score_set"] == "validation"]

print(
    f"validation pairs: {len(labels):,}   positives: {int(labels['label'].sum()):,} "
    f"({100 * labels['label'].mean():.1f} %)"
)
print()
print(f"{'model':6} {'block':14} {'width':16} {'distinct':>9} {'AUC':>7}")
for (model, block, width), rows in scores.groupby(
    ["model", "feature_block", "p_width"], dropna=False
):
    merged = labels.merge(
        rows, on=["key_a_stereo", "key_b_stereo"], validate="one_to_one"
    )
    if len(merged) != len(labels):
        sys.exit(f"{model}: {len(merged)} of {len(labels)} pairs matched")
    auc = roc_auc_score(merged["label"], merged["score"])
    print(
        f"{model:6} {block!s:14} {width!s:16} {merged['score'].nunique():9,} {auc:7.4f}"
    )
