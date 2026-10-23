"""Adversarial validation: can a classifier tell a literature substrate from a lab one?

The diagnostic the dataset-shift literature asks for (Ben-David et al., Mach. Learn. 2010, the proxy
A-distance). If a feature block separates the two sources, it carries provenance and not only
chemistry, and a PU model built on it can learn "looks like a patent molecule" instead of "reacts".
Reported per role and per feature block, as 5-fold cross-validated ROC-AUC. 0.5 means the block
cannot tell the sources apart; 1.0 means it tells them apart perfectly.

Usage:
    uv run python scripts/counts/count_adversarial.py ~/University/Phd/Dataset/Datasets
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

ROOT = Path(
    sys.argv[1] if len(sys.argv) > 1 else "~/University/Phd/Dataset/Datasets"
).expanduser()
DESCRIPTORS = [
    "centre_in_ring", "n_alpha", "alpha_n_aromatic", "alpha_n_in_ring", "alpha_max_heavy_degree",
    "alpha_sum_heavy_degree", "n_heavy_at_2_bonds", "n_heavy_at_3_bonds", "n_ewg_on_ring",
]  # fmt: skip
SEED = 20261022


def matrix(rows: pd.DataFrame, block: str) -> np.ndarray:
    """The feature block as a dense float matrix."""
    if block == "descriptors":
        return rows[DESCRIPTORS].to_numpy(dtype=float)
    if block == "descriptors + size":
        return rows[[*DESCRIPTORS, "n_heavy_atoms"]].to_numpy(dtype=float)
    if block == "size alone":
        return rows[["n_heavy_atoms"]].to_numpy(dtype=float)
    return np.stack([np.asarray(fp) for fp in rows[block]]).astype(float)


def main() -> None:
    """Print the separability of literature from lab substrates, per role and feature block."""
    table = pd.read_parquet(
        ROOT
        / "derived"
        / "private"
        / "features"
        / "amide_coupling"
        / "substrates.parquet"
    )
    for role in ("acid", "amine"):
        rows = table[table["role"] == role]
        # Substrates in both sources carry no provenance signal, so they are left out.
        only_literature = rows["in_positives"] & ~rows["in_library"]
        only_lab = rows["in_library"] & ~rows["in_positives"]
        used = rows[only_literature | only_lab].reset_index(drop=True)
        source = used["in_library"].to_numpy(dtype=int)
        print(
            f"\n{role}: {int((source == 0).sum()):,} literature-only against "
            f"{int(source.sum()):,} lab-only substrates"
        )
        for block in (
            "size alone",
            "descriptors",
            "descriptors + size",
            "morgan_centre",
            "morgan",
        ):
            features = matrix(used, block)
            model = (
                LogisticRegression(max_iter=2000, C=0.1)
                if features.shape[1] > 100
                else HistGradientBoostingClassifier(random_state=SEED, max_iter=200)
            )
            auc = cross_val_score(
                model, features, source, cv=5, scoring="roc_auc"
            ).mean()
            print(f"   {block:22s} {features.shape[1]:>5} columns   AUC {auc:.3f}")


if __name__ == "__main__":
    main()
