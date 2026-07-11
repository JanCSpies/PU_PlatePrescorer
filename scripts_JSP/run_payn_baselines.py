"""PAYN-style baseline comparison for the Chemlex amide feasibility model.

PAYN (Boser et al., *JACS* 2026) reports every fold as a three-model bar: the PU model, a
**positives-only** naive skyline, and a **true** (real-labels) upper bound. This script computes the
two reference bars for our Chemlex amide task so the report can place our literature-PU model
between them:

* **positives-only** -- a one-class model trained on ONLY the literature positives (no unlabeled
  pool): each test reaction is scored by its mean Tanimoto to the ``K_NEIGHBORS`` nearest of the
  258,645 mined amide positives. It answers "how far does *looks like a known reaction* get you?"
  with zero notion of what fails -- the naive skyline. Deterministic (no seed).
* **true (upper bound)** -- the *same* ``sup_histgb_l2`` learner, but trained on the Chemlex TRAIN
  split's REAL 0/1 labels (in-domain supervised), tested on the same both-unseen TEST. The ceiling
  our zero-target-label literature-PU model is chasing. Seeded (HistGB) -> mean/std over 5 seeds.

Our PU model (literature P + combinatorial U) is read from ``multiseed_metrics.json`` /
``eval_arrays.npz`` by the report -- not recomputed here. All three models are scored on the
identical 2,234-row both-unseen Chemlex test (feasible = Conversion >= 20 %) with the same raw-ECFP
(4096-d substrate) features, so the only thing that varies across the three bars is *what data the
model learned from*.

Saves ``payn_baselines.json`` (per-seed + mean/std) and ``payn_baseline_arrays.npz`` (test labels +
seed-averaged positives-only / true test scores) for the report's comparison page.

Run with:  uv run python -m scripts_JSP.run_payn_baselines
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scripts_JSP.run_dataset_switch import (
    AMIDE_P,
    CHEMLEX,
    FEATURIZERS,
    MODEL_SPEC,
    _frame,
)
from sklearn.metrics import average_precision_score, roc_auc_score

from pubench.features import build_featurizer
from pubench.models import build_model

SEEDS = [0, 1, 2, 3, 4]
K_NEIGHBORS = 5  # positives-only: mean Tanimoto to the K nearest literature positives
REF_CHUNK = 20_000  # positives streamed in chunks so the reference never fully materialises
OUT_JSON = Path("artifacts/dataset_switch/payn_baselines.json")
OUT_NPZ = Path("artifacts/dataset_switch/payn_baseline_arrays.npz")


def _tanimoto_topk_scores(
    x_test: np.ndarray, positives: pd.DataFrame, featurizer: object
) -> np.ndarray:
    """Positives-only score: mean Tanimoto of each test row to its K nearest literature positives.

    The full positive set is streamed in ``REF_CHUNK`` blocks; a running per-test-row top-K of
    Tanimoto similarities is kept, so the reference matrix is never fully materialised. Tanimoto on
    binary ECFP is ``|A & B| / (|A| + |B| - |A & B|)``, computed as a single dense matmul per block.

    Args:
        x_test: Binary (0/1) test feature matrix, shape ``(n_test, n_bits)``.
        positives: Frame with the featurizer's substrate columns for the literature positives.
        featurizer: A fitted featurizer exposing ``transform(frame) -> np.ndarray``.

    Returns:
        Per-test-row feasibility score (higher = more positive-like), shape ``(n_test,)``.
    """
    xt = (x_test > 0).astype(np.float32)
    a = xt.sum(axis=1)  # |A| per test row
    n_test = xt.shape[0]
    topk = np.full((n_test, K_NEIGHBORS), -1.0, dtype=np.float32)  # running K best similarities
    for start in range(0, len(positives), REF_CHUNK):
        chunk = positives.iloc[start : start + REF_CHUNK]
        xr = (featurizer.transform(chunk) > 0).astype(np.float32)  # type: ignore[attr-defined]
        b = xr.sum(axis=1)  # |B| per reference row
        inter = xt @ xr.T  # |A & B|
        denom = a[:, None] + b[None, :] - inter
        sims = np.where(denom > 0, inter / denom, 0.0).astype(np.float32)
        merged = np.concatenate([topk, sims], axis=1)
        topk = -np.sort(-merged, axis=1)[:, :K_NEIGHBORS]  # keep the K largest similarities
    return topk.mean(axis=1)


def _metrics(y: np.ndarray, scores: np.ndarray, seed: int | None = None) -> dict[str, float]:
    """ROC-AUC + AUPRC (with an optional seed tag) for a score vector."""
    row = {
        "roc_auc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
    }
    if seed is not None:
        row["seed"] = float(seed)
    return row


def _block(per_seed: list[dict[str, float]]) -> dict[str, object]:
    """Wrap per-seed metric rows with mean/std over the metric keys."""
    keys = [k for k in per_seed[0] if k != "seed"]
    mean = {k: float(np.mean([r[k] for r in per_seed])) for k in keys}
    std = {k: float(np.std([r[k] for r in per_seed])) for k in keys}
    return {"per_seed": per_seed, "mean": mean, "std": std}


def main() -> None:
    """Compute the positives-only and true-upper-bound bars and persist metrics + scores."""
    fname, params = FEATURIZERS["raw_ecfp"]
    featurizer = build_featurizer(fname, params)

    chem = pd.read_excel(CHEMLEX)
    split = chem["Stratified_Split_Both_Unseen"]
    label = (chem["Conversion"] >= 20).astype(np.int64)
    train_rows, test_rows = split == "train", split == "test"

    test = _frame(chem.loc[test_rows, "Acid"], chem.loc[test_rows, "Amine"])
    test_y = label[test_rows].to_numpy()
    featurizer.fit(test)  # raw ECFP: fixed 2048x2 bits, so the fit frame does not select features
    x_test = featurizer.transform(test)

    # --- true upper bound: same HistGB, trained on the Chemlex TRAIN split's REAL labels ----------
    train_pos = _frame(
        chem.loc[train_rows & (label == 1), "Acid"], chem.loc[train_rows & (label == 1), "Amine"]
    )
    train_neg = _frame(
        chem.loc[train_rows & (label == 0), "Acid"], chem.loc[train_rows & (label == 0), "Amine"]
    )
    x_pos, x_neg = featurizer.transform(train_pos), featurizer.transform(train_neg)
    true_rows, true_scores = [], []
    for seed in SEEDS:
        model = build_model("supervised", dict(MODEL_SPEC), seed=seed)
        model.fit(x_pos, x_neg)  # P=real positives, U=real negatives -> a genuine supervised fit
        s = model.predict_proba(x_test)
        true_scores.append(s)
        true_rows.append(_metrics(test_y, s, seed))
        print(f"  [true]  seed {seed}: AUC {true_rows[-1]['roc_auc']:.3f} "
              f"AP {true_rows[-1]['auprc']:.3f}")
    true_block = _block(true_rows)

    # --- positives-only: one-class kNN density over ALL 258k literature positives (deterministic) -
    pos = pd.read_parquet(AMIDE_P)
    positives = _frame(pos["acid_smiles"], pos["amine_smiles"])
    print(f"  [pos-only] kNN Tanimoto to {len(positives):,} positives (k={K_NEIGHBORS})...")
    pos_scores = _tanimoto_topk_scores(x_test, positives, featurizer)
    pos_block = _block([_metrics(test_y, pos_scores, 0)])
    print(f"  [pos-only] AUC {pos_block['mean']['roc_auc']:.3f} "  # type: ignore[index]
          f"AP {pos_block['mean']['auprc']:.3f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "model": MODEL_SPEC,
                "k_neighbors": K_NEIGHBORS,
                "n_positives": len(positives),
                "test": {"n": len(test_y), "base_rate": float(test_y.mean())},
                "baselines": {"positives_only": pos_block, "true": true_block},
            },
            indent=2,
        )
    )
    np.savez(
        OUT_NPZ,
        test_y=test_y,
        positives_only_scores=pos_scores,
        true_scores=np.mean(true_scores, axis=0),
    )
    print(f"\nWrote {OUT_JSON} and {OUT_NPZ}")
    print(f"positives-only AUC {pos_block['mean']['roc_auc']:.3f}   "  # type: ignore[index]
          f"true(upper) AUC {true_block['mean']['roc_auc']:.3f}"  # type: ignore[index]
          f"+/-{true_block['std']['roc_auc']:.3f}")  # type: ignore[index]


if __name__ == "__main__":
    main()
