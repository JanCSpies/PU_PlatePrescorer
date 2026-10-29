"""What re-folding a Morgan fingerprint to fewer bits leaves.

Descriptive only: no label, neither axis. RDKit folds an environment into ``hash % fpSize``, so when
``n`` divides 2048 the bucket at width ``n`` is the sum of the 2048-wide buckets at stride ``n``.
Re-folding is therefore exact and needs no stage F rebuild. This chooses the candidate grid.

Usage:
    uv run python scripts/counts/count_fold_width.py ~/University/Phd/Dataset/Datasets
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from platepu.features.blocks import block_matrix

if len(sys.argv) != 2:
    raise SystemExit("usage: scripts/counts/count_fold_width.py <Datasets root>")
ROOT = Path(sys.argv[1]).expanduser() / "derived"
rows = pd.read_parquet(
    f"{ROOT}/private/features/amide_coupling/substrates.parquet",
    columns=["key_stereo", "morgan", "morgan_centre"],
)
print(f"substrates: {len(rows):,}")

for block in ("morgan", "morgan_centre"):
    counts = block_matrix(rows, block).astype(np.int32)
    print(f"\n=== {block}")
    print(
        f"  {'width':>6} {'bits used':>10} {'mean set/mol':>13} {'distinct fps':>13} {'share unique':>13}"
    )
    for width in (2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4):
        folded = counts.reshape(len(counts), width, -1).sum(axis=2)
        present = folded > 0
        distinct = len({row.tobytes() for row in present})
        print(
            f"  {width:>6} {int((present.any(0)).sum()):>10} {present.sum(1).mean():>13.1f} "
            f"{distinct:>13,} {distinct / len(counts):>12.1%}"
        )
