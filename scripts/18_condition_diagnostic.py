"""In-domain diagnostic: are reaction conditions *learnable* for JnJ feasibility at all?

The condition-aware transfer eval (scripts/17) tests literature->JnJ. This companion isolates a
different question with a **supervised, within-JnJ** probe: train on the JnJ VAL rows and predict
the JnJ TEST rows (substrate-disjoint, but the two share the *same* condition sets), comparing
substrate-only vs substrate+condition features.

If substrate+condition clearly beats substrate-only here, then conditions carry real, learnable
signal and the transfer failure in scripts/17 is specifically a **condition-domain mismatch**
(literature uses different catalysts/bases than JnJ) — motivating condition *descriptors* (a shared
physical space) rather than raw fingerprints.

NOTE: this is a **diagnostic** that trains on JnJ VAL labels — it deliberately breaks the
"no JnJ labels in training" purity of the main result, and is reported separately as such.

Usage:
    uv run python scripts/18_condition_diagnostic.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bhpu.eval.metrics import feasibility_metrics
from bhpu.eval.protocol import FrozenEvalProtocol
from bhpu.features.condition_fp import condition_block, condition_ecfp_map
from bhpu.models.boosting import CatBoostFeasibilityModel

FEATURES_DIR = Path("datasets/processed/features")
ARTIFACTS_DIR = Path("artifacts")


def _load_ecfp(name: str) -> np.ndarray:
    """Load a cached substrate ECFP matrix (row-aligned to its frame)."""
    with np.load(FEATURES_DIR / f"{name}__ecfp2048.npz") as data:
        return np.asarray(data["X"], dtype=np.float32)


def main() -> None:
    """Entry point: supervised in-domain substrate vs substrate+condition on JnJ VAL->TEST."""
    jnj = pd.read_parquet(FEATURES_DIR / "test_jnj.parquet")
    protocol = FrozenEvalProtocol(jnj)
    sub = _load_ecfp("test_jnj")
    cmap = condition_ecfp_map([jnj])
    cond = condition_block(jnj, cmap)

    val_idx, test_idx = protocol.val_index, protocol.test_index
    y_val, y_test = protocol.labels("val"), protocol.labels("test")

    results: list[dict[str, object]] = []
    print(f"{'features':<20}{'TEST_AUC':>10}{'TEST_AP':>9}")
    print("-" * 40)
    for name, matrix in (
        ("substrate", sub),
        ("substrate+condition", np.hstack([sub, cond]).astype(np.float32)),
    ):
        model = CatBoostFeasibilityModel(
            iterations=400, depth=6, learning_rate=0.05, random_state=0
        )
        model.fit(matrix[val_idx], y_val)
        test = feasibility_metrics(y_test, model.predict_proba(matrix[test_idx]))
        row = {
            "features": name,
            "test_auc": round(test.roc_auc, 3),
            "test_auprc": round(test.auprc, 3),
        }
        results.append(row)
        print(f"{name:<20}{row['test_auc']:>10}{row['test_auprc']:>9}")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "condition_diagnostic.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {ARTIFACTS_DIR / 'condition_diagnostic.json'}")


if __name__ == "__main__":
    main()
