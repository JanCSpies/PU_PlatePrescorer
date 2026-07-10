"""Multi-seed evaluation of the Chemlex amide feasibility model (transductive + inductive).

Featurizes each (regime, featurizer) once, then fits ``sup_histgb_l2`` under 5 seeds (the HistGB
bin-subsample on our >200k P+U pool is seeded, giving genuine model-training variance). For every
seed we record TEST ROC-AUC / AUPRC and, at the **val-selected Youden threshold**, the full test
confusion matrix and the operating-point metrics that matter for plate pre-scoring:

* **specificity / TN** -- failed reactions correctly skipped (experiments saved);
* **precision (hit rate)** -- of the wells we would run, the fraction that actually work;
* **recall** -- of all feasible reactions, the fraction we would still catch;
* **NPV** -- of the wells we skip, the fraction that would truly have failed.

Saves ``multiseed_metrics.json`` (per-seed + mean/std, all four regime x featurizer cells) and
``eval_arrays.npz`` (seed-averaged raw-ECFP test/val scores for the report's curves + bootstrap CI).

Run with:  uv run python -m scripts_JSP.run_multiseed_eval
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scripts_JSP.run_chemlex_inductive import build_chemlex_inductive
from scripts_JSP.run_dataset_switch import FEATURIZERS, MODEL_SPEC, Splits, build_chemlex
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from pubench.features import build_featurizer
from pubench.models import build_model

SEEDS = [0, 1, 2, 3, 4]
OUT_JSON = Path("artifacts/dataset_switch/multiseed_metrics.json")
OUT_NPZ = Path("artifacts/dataset_switch/eval_arrays.npz")


def _youden_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Return the score threshold maximising Youden's J (tpr - fpr) on a labelled split."""
    fpr, tpr, thr = roc_curve(y, scores)
    return float(thr[int(np.argmax(tpr - fpr))])


def _confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Confusion counts + plate-relevant operating-point metrics at a fixed threshold."""
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    n = tp + fp + tn + fn
    p = tp + fn
    denom_prec = tp + fp
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": tp / denom_prec if denom_prec else float("nan"),  # plate hit rate
        "recall": tp / p if p else float("nan"),  # feasible caught
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),  # failures skipped
        "npv": tn / (tn + fn) if (tn + fn) else float("nan"),  # skipped-and-truly-failed
        "accuracy": (tp + tn) / n if n else float("nan"),
        "frac_run": denom_prec / n if n else float("nan"),  # fraction of plate we would run
    }


def _mean_std(rows: list[dict[str, float]], key: str) -> tuple[float, float]:
    """Mean and (population) std of ``key`` across per-seed metric dicts."""
    vals = np.array([r[key] for r in rows], dtype=float)
    return float(vals.mean()), float(vals.std())


def _evaluate(
    regime: str, splits: Splits, flabel: str
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Featurize once, fit over all seeds, return the metric block + stacked test/val scores."""
    fname, params = FEATURIZERS[flabel]
    featurizer = build_featurizer(fname, params)
    featurizer.fit(pd.concat([splits.positives, splits.unlabeled], ignore_index=True, sort=False))
    x_p = featurizer.transform(splits.positives)
    x_u = featurizer.transform(splits.unlabeled)
    x_val = featurizer.transform(splits.val) if splits.val is not None else None
    x_test = featurizer.transform(splits.test)

    per_seed: list[dict[str, float]] = []
    test_scores_all, val_scores_all = [], []
    for seed in SEEDS:
        model = build_model("supervised", dict(MODEL_SPEC), seed=seed)
        model.fit(x_p, x_u)
        test_scores = model.predict_proba(x_test)
        test_scores_all.append(test_scores)
        row: dict[str, float] = {
            "seed": seed,
            "roc_auc": float(roc_auc_score(splits.test_y, test_scores)),
            "auprc": float(average_precision_score(splits.test_y, test_scores)),
        }
        if x_val is not None and splits.val_y is not None:
            val_scores = model.predict_proba(x_val)
            val_scores_all.append(val_scores)
            thr = _youden_threshold(splits.val_y, val_scores)
            row["threshold"] = thr
            row.update(_confusion(splits.test_y, (test_scores >= thr).astype(int)))
        per_seed.append(row)
        print(f"  [{regime}/{flabel}] seed {seed}: AUC {row['roc_auc']:.3f} AP {row['auprc']:.3f}")

    block: dict[str, object] = {"per_seed": per_seed, "mean": {}, "std": {}}
    metric_keys = [k for k in per_seed[0] if k != "seed"]
    for k in metric_keys:
        m, s = _mean_std(per_seed, k)
        block["mean"][k] = m  # type: ignore[index]
        block["std"][k] = s  # type: ignore[index]
    return block, np.array(test_scores_all), np.array(val_scores_all)


def main() -> None:
    """Run the full multi-seed grid and persist metrics + seed-averaged raw-ECFP scores."""
    regimes = {"transductive": build_chemlex(), "inductive": build_chemlex_inductive()}
    metrics: dict[str, dict[str, object]] = {}
    arrays: dict[str, np.ndarray] = {}
    for regime, splits in regimes.items():
        metrics[regime] = {}
        arrays[f"{regime}_test_y"] = splits.test_y
        if splits.val_y is not None:
            arrays[f"{regime}_val_y"] = splits.val_y
        for flabel in FEATURIZERS:
            block, test_scores, val_scores = _evaluate(regime, splits, flabel)
            metrics[regime][flabel] = block
            if flabel == "raw_ecfp":  # keep the production model's scores for the report
                arrays[f"{regime}_test_scores"] = test_scores.mean(axis=0)
                if val_scores.size:
                    arrays[f"{regime}_val_scores"] = val_scores.mean(axis=0)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seeds": SEEDS, "model": MODEL_SPEC, "metrics": metrics}
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    np.savez(OUT_NPZ, **arrays)
    print(f"\nWrote {OUT_JSON} and {OUT_NPZ}")
    for regime in metrics:
        for flabel in FEATURIZERS:
            m = metrics[regime][flabel]["mean"]  # type: ignore[index]
            s = metrics[regime][flabel]["std"]  # type: ignore[index]
            print(f"{regime:<12} {flabel:<9} AUC {m['roc_auc']:.3f}±{s['roc_auc']:.3f}  "
                  f"AP {m['auprc']:.3f}±{s['auprc']:.3f}")


if __name__ == "__main__":
    main()
