"""How often each Morgan bit is set, and what a frequency floor leaves.

Descriptive only: it reads no label and computes neither axis. Its job is to choose a sensible
candidate grid before the rule is pre-registered, so the grid does not waste half its points on
floors that leave nothing.

Usage:
    uv run python scripts/counts/count_bit_frequency.py ~/University/Phd/Dataset/Datasets
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from platepu.features.blocks import block_matrix

if len(sys.argv) != 2:
    raise SystemExit("usage: scripts/counts/count_bit_frequency.py <Datasets root>")
ROOT = Path(sys.argv[1]).expanduser() / "derived"
FEATURES = f"{ROOT}/private/features/amide_coupling/substrates.parquet"

rows = pd.read_parquet(
    FEATURES, columns=["key_stereo", "role", "morgan", "morgan_centre"]
)
print(f"substrates: {len(rows):,}")

for block in ("morgan", "morgan_centre"):
    counts = block_matrix(rows, block)
    present = counts > 0
    freq = present.sum(axis=0)  # in how many substrates each bit occurs
    used = freq[freq > 0]
    print(f"\n=== {block}: {len(used):,} of {counts.shape[1]:,} bits ever set")
    print(
        "  bit document-frequency quantiles:",
        {q: int(np.quantile(used, q / 100)) for q in (10, 25, 50, 75, 90, 99)},
        "max",
        int(used.max()),
    )
    print(
        f"  {'floor m':>8} {'bits kept':>10} {'mean bits/mol':>14} {'distinct fps':>13}"
    )
    for m in (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500):
        keep = freq >= m
        if not keep.any():
            print(f"  {m:>8} {0:>10} {'-':>14} {'-':>13}")
            continue
        masked = present[:, keep]
        distinct = len({row.tobytes() for row in masked})
        print(
            f"  {m:>8} {int(keep.sum()):>10} {masked.sum(1).mean():>14.1f} {distinct:>13,}"
        )
