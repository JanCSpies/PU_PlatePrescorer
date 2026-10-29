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

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from platepu.features.blocks import block_matrix
from platepu.score.axes import make_model

if len(sys.argv) != 2:
    raise SystemExit("usage: scripts/counts/count_adversarial.py <Datasets root>")
ROOT = Path(sys.argv[1]).expanduser()
SEED = 20261022


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
            features = block_matrix(used, block)
            # The project's one estimator rule, so this published figure and the block ladder it is
            # read against cannot drift apart. `make_model` fixes `random_state` at 0; the override
            # keeps this script's own seed, and with it the number it has always printed.
            model = make_model(features.shape[1])
            if isinstance(model, HistGradientBoostingClassifier):
                model.set_params(random_state=SEED)
            auc = cross_val_score(
                model, features, source, cv=5, scoring="roc_auc"
            ).mean()
            print(f"   {block:22s} {features.shape[1]:>5} columns   AUC {auc:.3f}")


if __name__ == "__main__":
    main()
