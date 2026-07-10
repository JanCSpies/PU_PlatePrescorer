r"""Run *our* simple HistGB+L2 (pubench ``sup_histgb_l2``) on *Florian's* data assembly.

Fills the missing cell of the reconciliation 2x2 (LAB_JOURNAL 2026-07-09 "Two tracks reconciled"):

    ============================  ==============================  ==============================
    model \\ data                 our final_datasets              Florian's bhpu (lit+ORD, his U)
    ============================  ==============================  ==============================
    HistGB+L2 (ours)              0.723 corr08 / 0.711 raw        <-- THIS SCRIPT
    IW-PU CatBoost (his)          0.686 corr08 / 0.674 raw        0.720 (his RESULTS.md)
    ============================  ==============================  ==============================

The point is a *clean* swap: the featurizer (``corr08`` / raw ``ecfp``) and the model
(``sup_histgb_l2``: HistGradientBoosting, ``max_iter=400``, ``learning_rate=0.05``,
``l2_regularization=1.0``) are the *identical* pubench code that scored 0.723/0.711 on our data.
Only the data changes to Florian's assembly:

* **P** = his literature positives (``literature_clean.parquet`` rows with ``labels == 1``)
  + his mined ORD C-N corpus (``cn_corpus_ord.parquet``) -- i.e. the "literature + ORD" P that
  produced his 0.720.
* **U** = his transductive JnJ substrate grid (``unlabeled_grid.parquet``).
* **VAL / TEST** = his JnJ plate split by the shared frozen protocol
  (:class:`bhpu.eval.protocol.FrozenEvalProtocol`), so val/test match his 0.720 exactly.

Substrate-only featurization (``use_conditions=False``) needs only the ``Aryl SMILES`` /
``Amine SMILES`` columns, which every one of his frames already carries, so no schema
materialisation is required -- we featurize his frames directly.

If HistGB+L2 also lands ~0.72 here, both data assemblies hit the same literature->JnJ ceiling
under either learner (method is not the lever -- Occam holds). If it is materially lower, his
IW-PU-on-his-data would be a genuine data+method combination worth keeping.

Usage:
    uv run python scripts_JSP/reproduce_histgb_on_fb_data.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bhpu.data import schema as bh_schema
from bhpu.eval.metrics import feasibility_metrics
from bhpu.eval.protocol import FrozenEvalProtocol
from pubench.features import build_featurizer
from pubench.models import build_model

# --- Florian's data assembly (his cached frames + mined ORD corpus) --------------------------
FEATURES_DIR = Path("datasets/processed/features")
LIT_CLEAN = FEATURES_DIR / "literature_clean.parquet"
UNLABELED_GRID = FEATURES_DIR / "unlabeled_grid.parquet"
TEST_JNJ = FEATURES_DIR / "test_jnj.parquet"
#: His mined ORD corpus lives in his home tree (not symlinked into our datasets/processed).
ORD_CORPUS = Path(
    "/home/employee/fboser/fboser_mydata/hackathon_claude/datasets/processed/cn_corpus_ord.parquet"
)

OUT_DIR = Path("artifacts/pubench/fb_data_histgb")

#: The exact pubench model that scored 0.723/0.711 on our data (E6 resolved_config).
MODEL_SPEC: dict[str, object] = {
    "base": "histgb",
    "iterations": 400,
    "learning_rate": 0.05,
    "l2_regularization": 1.0,
}
#: The two featurizers of record: cheap correlation-condensed ECFP (~982-d) and raw 4096-d ECFP.
FEATURIZER_SPECS: dict[str, tuple[str, dict[str, object]]] = {
    "corr08": (
        "reduced_ecfp",
        {
            "method": "correlation",
            "n_bits": 2048,
            "min_freq": 0.01,
            "corr_threshold": 0.8,
            "use_conditions": False,
        },
    ),
    "raw_ecfp": ("ecfp", {"n_bits": 2048, "use_conditions": False}),
}

_SUBSTRATE_COLS = [bh_schema.ARYL_SMILES, bh_schema.AMINE_SMILES]


@dataclass(frozen=True)
class Splits:
    """The four SMILES frames of Florian's assembly (+ val/test labels)."""

    positives: pd.DataFrame
    unlabeled: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    val_y: np.ndarray
    test_y: np.ndarray


def load_florian_splits() -> Splits:
    """Assemble P / U / VAL / TEST from Florian's cached frames and his mined ORD corpus."""
    lit = pd.read_parquet(LIT_CLEAN)
    lit_pos = lit.loc[lit[bh_schema.LABEL] == 1, _SUBSTRATE_COLS].reset_index(drop=True)

    ord_corpus = pd.read_parquet(ORD_CORPUS)
    ord_pos = ord_corpus.rename(
        columns={"aryl_smiles": bh_schema.ARYL_SMILES, "amine_smiles": bh_schema.AMINE_SMILES}
    )[_SUBSTRATE_COLS].reset_index(drop=True)

    positives = pd.concat([lit_pos, ord_pos], ignore_index=True)

    unlabeled = pd.read_parquet(UNLABELED_GRID)[_SUBSTRATE_COLS].reset_index(drop=True)

    jnj = pd.read_parquet(TEST_JNJ)
    protocol = FrozenEvalProtocol(jnj)
    val = jnj.iloc[protocol.val_index][_SUBSTRATE_COLS].reset_index(drop=True)
    test = jnj.iloc[protocol.test_index][_SUBSTRATE_COLS].reset_index(drop=True)
    return Splits(
        positives=positives,
        unlabeled=unlabeled,
        val=val,
        test=test,
        val_y=protocol.labels("val"),
        test_y=protocol.labels("test"),
    )


def run_one(featurizer_label: str, splits: Splits, *, seed: int = 0) -> dict[str, object]:
    """Featurize + fit sup_histgb_l2 (P-vs-U) for one featurizer; return VAL/TEST metrics."""
    name, params = FEATURIZER_SPECS[featurizer_label]
    featurizer = build_featurizer(name, params)

    # Fit the (label-free) featurizer on P + U, exactly like the pubench runner (_featurize).
    fit_records = pd.concat([splits.positives, splits.unlabeled], ignore_index=True, sort=False)
    featurizer.fit(fit_records)

    x_p = featurizer.transform(splits.positives)
    x_u = featurizer.transform(splits.unlabeled)
    x_val = featurizer.transform(splits.val)
    x_test = featurizer.transform(splits.test)

    model = build_model("supervised", dict(MODEL_SPEC), seed=seed)
    model.fit(x_p, x_u)

    val = feasibility_metrics(splits.val_y, model.predict_proba(x_val))
    test = feasibility_metrics(splits.test_y, model.predict_proba(x_test))
    return {
        "featurizer": featurizer_label,
        "n_features": int(x_p.shape[1]),
        "val_roc_auc": val.roc_auc,
        "val_auprc": val.auprc,
        "test_roc_auc": test.roc_auc,
        "test_auprc": test.auprc,
    }


def main() -> None:
    """Run both featurizers on Florian's data and print + persist the comparison."""
    splits = load_florian_splits()
    counts = {
        "n_positives": len(splits.positives),
        "n_unlabeled": len(splits.unlabeled),
        "n_val": len(splits.val),
        "n_test": len(splits.test),
        "val_pos_rate": float(np.mean(splits.val_y)),
        "test_pos_rate": float(np.mean(splits.test_y)),
    }
    print("=== Florian's data assembly (his lit+ORD P, his transductive U, his JnJ split) ===")
    print(json.dumps(counts, indent=2))
    print("  (sanity: his RESULTS.md reports val 2,624 / test 1,920)\n")

    results = [run_one(label, splits) for label in FEATURIZER_SPECS]

    cols = ("featurizer", "n_feat", "VAL_AUC", "VAL_AP", "TE_AUC", "TE_AP")
    widths = (10, 7, 8, 7, 8, 7)
    pairs = enumerate(zip(cols, widths, strict=True))
    print(" ".join(f"{c:>{w}}" if i else f"{c:<{w}}" for i, (c, w) in pairs))
    print("-" * 52)
    for r in results:
        print(
            f"{r['featurizer']:<10} {r['n_features']:>7} "
            f"{r['val_roc_auc']:>8.3f} {r['val_auprc']:>7.3f} "
            f"{r['test_roc_auc']:>8.3f} {r['test_auprc']:>7.3f}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"model": MODEL_SPEC, "counts": counts, "results": results}
    (OUT_DIR / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
