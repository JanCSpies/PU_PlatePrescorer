"""Run one PU method across the full Chemlex evaluation grid (the champion's grid).

Companion to :mod:`scripts_JSP.run_multiseed_eval`, which fixed the model to ``sup_histgb_l2``.
Here the model is a CLI argument so every registered PU learner can be scored on the *same*
Chemlex amide grid and compared head-to-head against the HistGB supervised bar. The scientific
question: on an **in-scope** dataset (Chemlex amide C-N, unlike out-of-domain JnJ 2024), do any of
the shift-aware / risk-based PU learners actually beat the naive P-vs-U HistGB baseline? On JnJ they
did not; this checks whether that ordering survives a more in-scope target.

For the chosen model we reuse the exact assembly + metrics of the champion run:

* **regimes:** transductive (full acid x amine U) + inductive (U from non-test substrates only);
* **featurizers:** ``raw_ecfp`` (2048) + ``corr08`` (correlation-reduced 2048);
* **seeds:** 5 (genuine model-training variance);
* **metrics:** TEST ROC-AUC / AUPRC per seed, plus the val-Youden operating point (precision /
  recall / specificity / NPV) that matters for plate pre-scoring.

Each model writes ``artifacts/dataset_switch/allmodels/<label>.json`` with the same block layout as
``multiseed_metrics.json`` so :mod:`scripts_JSP.summarize_chemlex_models` can pool them with the
existing supervised numbers.

The ``prior`` for nnPU / IW-PU is set to the Chemlex validation base rate (0.237), mirroring the JnJ
runs which used their own val base rate (0.32) -- not JnJ's stale prior, and not a test-peeking one.

Usage:
    uv run python -m scripts_JSP.run_chemlex_all_models --model bagging
    uv run python -m scripts_JSP.run_chemlex_all_models --model nnpu --smoke   # fast harness check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scripts_JSP.run_chemlex_inductive import build_chemlex_inductive
from scripts_JSP.run_dataset_switch import FEATURIZERS, Splits, build_chemlex
from scripts_JSP.run_multiseed_eval import (
    SEEDS,
    _confusion,
    _mean_std,
    _youden_threshold,
)
from sklearn.metrics import average_precision_score, roc_auc_score

from pubench.features import build_featurizer
from pubench.models import build_model

#: Chemlex validation (train-split) base rate; used as the fixed class prior for prior-based PU
#: methods, mirroring the JnJ protocol (val base rate) without touching test labels.
CHEMLEX_VAL_PRIOR = 0.237

#: label -> (registry key, constructor params) for every PU learner we benchmarked on JnJ. Params
#: are exactly the JnJ specs (configs/pu_bench.*.yaml); only the prior is retargeted to Chemlex.
MODEL_ZOO: dict[str, tuple[str, dict[str, object]]] = {
    "sup_histgb_l2": (
        "supervised",
        {"base": "histgb", "iterations": 400, "learning_rate": 0.05, "l2_regularization": 1.0},
    ),
    "bagging": ("bagging_pu", {"n_estimators": 15, "max_iter": 200}),
    "elkan": ("elkan_noto", {"max_iter": 300, "learning_rate": 0.1}),
    "spy": ("spy", {"iterations": 200, "depth": 6, "spy_fraction": 0.15}),
    "nnpu": ("nnpu", {"prior": CHEMLEX_VAL_PRIOR, "epochs": 200, "learning_rate": 0.05}),
    "iw_pu": (
        "iw_pu",
        {
            "prior": CHEMLEX_VAL_PRIOR,
            "epochs": 200,
            "density_ratio": "rulsif",
            "project_dim": 64,
        },
    ),
}

OUT_DIR = Path("artifacts/dataset_switch/allmodels")


def _evaluate(
    model_key: str,
    params: dict[str, object],
    regime: str,
    splits: Splits,
    flabel: str,
    seeds: list[int],
) -> dict[str, object]:
    """Featurize once, fit ``model_key`` over all ``seeds``, return the per-seed + mean/std block.

    Args:
        model_key: Registry key of the PU model to build.
        params: Constructor parameters forwarded to :func:`pubench.models.build_model`.
        regime: ``"transductive"`` or ``"inductive"`` (for logging only).
        splits: The assembled P / U / VAL / TEST frames for this regime.
        flabel: Featurizer label (key into :data:`FEATURIZERS`).
        seeds: Seeds to average over.

    Returns:
        A block ``{"per_seed": [...], "mean": {...}, "std": {...}}`` matching the champion run.
    """
    fname, fparams = FEATURIZERS[flabel]
    featurizer = build_featurizer(fname, fparams)
    featurizer.fit(pd.concat([splits.positives, splits.unlabeled], ignore_index=True, sort=False))
    x_p = featurizer.transform(splits.positives)
    x_u = featurizer.transform(splits.unlabeled)
    x_val = featurizer.transform(splits.val) if splits.val is not None else None
    x_test = featurizer.transform(splits.test)

    per_seed: list[dict[str, float]] = []
    for seed in seeds:
        model = build_model(model_key, dict(params), seed=seed)
        model.fit(x_p, x_u)
        test_scores = model.predict_proba(x_test)
        row: dict[str, float] = {
            "seed": seed,
            "roc_auc": float(roc_auc_score(splits.test_y, test_scores)),
            "auprc": float(average_precision_score(splits.test_y, test_scores)),
        }
        if x_val is not None and splits.val_y is not None:
            val_scores = model.predict_proba(x_val)
            thr = _youden_threshold(splits.val_y, val_scores)
            row["threshold"] = thr
            row.update(_confusion(splits.test_y, (test_scores >= thr).astype(int)))
        per_seed.append(row)
        print(f"  [{regime}/{flabel}] seed {seed}: AUC {row['roc_auc']:.3f} AP {row['auprc']:.3f}")

    block: dict[str, object] = {"per_seed": per_seed, "mean": {}, "std": {}}
    for k in (key for key in per_seed[0] if key != "seed"):
        m, s = _mean_std(per_seed, k)
        block["mean"][k] = m  # type: ignore[index]
        block["std"][k] = s  # type: ignore[index]
    return block


def main() -> None:
    """Parse ``--model``, run the (regime x featurizer x seed) grid, and persist the results."""
    parser = argparse.ArgumentParser(description="Score one PU method on the Chemlex amide grid.")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_ZOO), help="Model label.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast harness check: transductive + raw_ecfp + seed 0 only (no file written).",
    )
    args = parser.parse_args()

    model_key, params = MODEL_ZOO[args.model]
    print(f"### chemlex_amide | model={args.model} ({model_key}) params={params}")

    if args.smoke:
        block = _evaluate(model_key, params, "transductive", build_chemlex(), "raw_ecfp", [0])
        print(f"SMOKE OK: {args.model} transductive/raw_ecfp seed0 -> {block['per_seed'][0]}")
        return

    regimes = {"transductive": build_chemlex(), "inductive": build_chemlex_inductive()}
    metrics: dict[str, dict[str, object]] = {}
    for regime, splits in regimes.items():
        metrics[regime] = {}
        for flabel in FEATURIZERS:
            metrics[regime][flabel] = _evaluate(
                model_key, params, regime, splits, flabel, list(SEEDS)
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.model}.json"
    payload = {
        "model_label": args.model,
        "model_key": model_key,
        "params": params,
        "seeds": list(SEEDS),
        "metrics": metrics,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")
    for regime in metrics:
        for flabel in FEATURIZERS:
            m = metrics[regime][flabel]["mean"]  # type: ignore[index]
            s = metrics[regime][flabel]["std"]  # type: ignore[index]
            print(
                f"{regime:<12} {flabel:<9} AUC {m['roc_auc']:.3f}±{s['roc_auc']:.3f}  "
                f"AP {m['auprc']:.3f}±{s['auprc']:.3f}"
            )


if __name__ == "__main__":
    main()
