"""Second-reaction validation: run the Chemlex pipeline on the HiTEA Buchwald-Hartwig C-N set.

Mirrors the Chemlex amide evaluation (multi-seed PU + PAYN-style baselines) on a *different reaction
class* to show the literature-PU feasibility method generalises. HiTEA is a small, natural BH C-N
screen, so several Chemlex experiments degenerate and are handled honestly:

* **Transductive only.** The unlabeled grid U is the 33x28 halide x amine matrix; every one of the
  36 test substrates appears in it, so an *inductive* variant (drop test substrates from U) would
  empty U. HiTEA cannot support the inductive regime -- reported as a caveat, not run.
* **No validation split.** The only labels are the 36 test pairs, so operating points are chosen by
  **plate budget** (run the top-K), exactly as in the Chemlex report -- no val threshold needed.
* **True upper bound collapses.** With 36 non-repeating substrate pairs, an in-domain supervised
  model is not viable; we quantify this with leave-one-out CV (expected AUC ~ random or worse) and
  surface it as the argument *for* literature transfer rather than a competing bar.

Models (all scored on the identical 36-pair substrate-level test, feasible = max yield over the
screened conditions >= ``THRESHOLD_PCT`` %, raw-ECFP substrate features):

* **PU - ours**  : ``sup_histgb_l2`` on literature P (merged BH positives) vs the candidate grid U.
* **positives-only** : one-class kNN, mean Tanimoto to the k nearest of the merged BH positives.
* **true (LOO)** : the same HistGB, leave-one-out CV on the 36 real labels (in-domain proxy).

Saves ``hitea_metrics.json`` (per-seed + mean/std + bootstrap CI) and ``hitea_arrays.npz`` (test
labels + seed-averaged PU / positives-only / true-LOO test scores) for ``make_report_hitea.py``.

Run with:  uv run python -m scripts_JSP.run_hitea_eval
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scripts_JSP.run_dataset_switch import FEATURIZERS, MODEL_SPEC, build_hitea
from scripts_JSP.run_payn_baselines import _block, _metrics, _tanimoto_topk_scores
from sklearn.metrics import roc_auc_score

from pubench.features import build_featurizer
from pubench.models import build_model

SEEDS = [0, 1, 2, 3, 4]
K_NEIGHBORS = 5
THRESHOLD_PCT = 10  # HiTEA substrate-level feasibility cutoff (matches build_hitea)
OUT_JSON = Path("artifacts/dataset_switch/hitea_metrics.json")
OUT_NPZ = Path("artifacts/dataset_switch/hitea_arrays.npz")


def _bootstrap_auc(y: np.ndarray, scores: np.ndarray, n: int = 2000) -> tuple[float, float]:
    """Percentile 95% CI for ROC-AUC by resampling the (small) test set; seeded for reproducibility.

    On n=36 this interval is wide by construction -- reporting it honestly is the point.
    """
    rng = np.random.default_rng(0)
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        aucs.append(roc_auc_score(y[s], scores[s]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _pu_multiseed(
    x_p: np.ndarray, x_u: np.ndarray, x_test: np.ndarray, y: np.ndarray
) -> tuple[dict[str, object], np.ndarray]:
    """Fit the PU model over all seeds; return the metric block + seed-averaged scores."""
    rows, scores = [], []
    for seed in SEEDS:
        model = build_model("supervised", dict(MODEL_SPEC), seed=seed)
        model.fit(x_p, x_u)
        s = model.predict_proba(x_test)
        scores.append(s)
        rows.append(_metrics(y, s, seed))
    return _block(rows), np.mean(scores, axis=0)


def _true_loo(x_test: np.ndarray, y: np.ndarray) -> tuple[dict[str, object], np.ndarray]:
    """In-domain leave-one-out CV on the 36 real labels (the true-upper-bound proxy for HiTEA).

    For each held-out pair, the same HistGB is trained on the other 35 (real positives vs real
    negatives) and used to score the held-out pair. With non-repeating substrates this is expected
    to collapse -- which is exactly why an in-domain model is not viable here.
    """
    loo = np.zeros(len(y), dtype=float)
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        x_pos, x_neg = x_test[keep][y[keep] == 1], x_test[keep][y[keep] == 0]
        model = build_model("supervised", dict(MODEL_SPEC), seed=0)
        model.fit(x_pos, x_neg)
        loo[i] = float(model.predict_proba(x_test[i : i + 1])[0])
    return _block([_metrics(y, loo, 0)]), loo


def main() -> None:
    """Run the HiTEA BH transfer + baselines and persist metrics + scores for the second report."""
    splits = build_hitea()
    y = splits.test_y
    print(f"### HiTEA BH: P={len(splits.positives)} U={len(splits.unlabeled)} "
          f"test={len(y)} (pos {y.sum()}/{len(y)}, base {y.mean():.3f})")

    pu: dict[str, object] = {}
    pu_scores_raw = np.zeros(len(y))
    for flabel in FEATURIZERS:
        fname, params = FEATURIZERS[flabel]
        featurizer = build_featurizer(fname, params)
        featurizer.fit(
            pd.concat([splits.positives, splits.unlabeled], ignore_index=True, sort=False)
        )
        x_p = featurizer.transform(splits.positives)
        x_u = featurizer.transform(splits.unlabeled)
        x_test = featurizer.transform(splits.test)
        block, scores = _pu_multiseed(x_p, x_u, x_test, y)
        pu[flabel] = block
        m = block["mean"]  # type: ignore[index]
        s = block["std"]  # type: ignore[index]
        print(f"  PU {flabel:<9} AUC {m['roc_auc']:.3f}+/-{s['roc_auc']:.3f} AP {m['auprc']:.3f}")
        if flabel == "raw_ecfp":
            pu_scores_raw = scores

    # --- PAYN-style baselines on raw ECFP (positives-only + in-domain LOO) -----------------------
    fname, params = FEATURIZERS["raw_ecfp"]
    featurizer_raw = build_featurizer(fname, params)
    featurizer_raw.fit(splits.test)  # raw ECFP: fixed bits, fit frame does not select features
    x_test_raw = featurizer_raw.transform(splits.test)
    pos_scores = _tanimoto_topk_scores(x_test_raw, splits.positives, featurizer_raw)
    pos_block = _block([_metrics(y, pos_scores, 0)])
    true_block, true_scores = _true_loo(x_test_raw, y)
    print(f"  positives-only AUC {pos_block['mean']['roc_auc']:.3f}   "  # type: ignore[index]
          f"true(LOO) AUC {true_block['mean']['roc_auc']:.3f}")  # type: ignore[index]

    ci = _bootstrap_auc(y, pu_scores_raw)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "reaction": "Buchwald-Hartwig C-N (HiTEA)",
                "model": MODEL_SPEC,
                "seeds": SEEDS,
                "k_neighbors": K_NEIGHBORS,
                "threshold_pct": THRESHOLD_PCT,
                "regime": "transductive_only",
                "inductive_note": "inductive impossible: 36 test substrates span the whole grid U",
                "n_positives": len(splits.positives),
                "test": {"n": len(y), "base_rate": float(y.mean())},
                "pu": pu,
                "pu_bootstrap_ci": ci,
                "baselines": {"positives_only": pos_block, "true_loo": true_block},
            },
            indent=2,
        )
    )
    np.savez(
        OUT_NPZ,
        test_y=y,
        pu_scores=pu_scores_raw,
        positives_only_scores=pos_scores,
        true_loo_scores=true_scores,
    )
    print(f"\nWrote {OUT_JSON} and {OUT_NPZ}")


if __name__ == "__main__":
    main()
