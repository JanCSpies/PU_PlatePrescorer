"""Benchmark the beyond-SCAR / SAR PU learners on the Chemlex amide grid.

Sibling of :mod:`scripts_JSP.run_chemlex_all_models` for the models that need a **shadow-feature
matrix** (reactivity-orthogonal reporting-propensity signals). It reuses the exact champion grid
(transductive + inductive × raw_ecfp + corr08 × 5 seeds) and metric block layout, additionally
building shadow features (:mod:`pubench.features.propensity`) for P (from the mined positives'
precomputed InChIKey-14 keys — fast) and for U / VAL / TEST (from SMILES), and injecting them into
each SAR model via ``set_propensity`` before fit.

The scientific question the SCAR bake-off could not answer: on Chemlex, does correcting the
*feature-dependent reporting bias* (which the shadow features capture — they separate P from U at
AUC ≈ 0.93, unlike the ECFP-space collapse of IW-PU) beat the naive P-vs-U HistGB champion (0.771)?

Each model writes ``artifacts/dataset_switch/allmodels/<label>.json`` in the same block layout as
``multiseed_metrics.json`` so :mod:`scripts_JSP.summarize_chemlex_models` pools it into the
leaderboard automatically.

Usage:
    uv run python -m scripts_JSP.run_chemlex_sar --model prop_ips
    uv run python -m scripts_JSP.run_chemlex_sar --model sar_em --smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scripts_JSP.run_chemlex_inductive import build_chemlex_inductive
from scripts_JSP.run_dataset_switch import AMIDE_P, FEATURIZERS, Splits, build_chemlex
from scripts_JSP.run_multiseed_eval import SEEDS, _confusion, _mean_std, _youden_threshold
from sklearn.metrics import average_precision_score, roc_auc_score

from pubench.features import build_featurizer
from pubench.features.propensity import ShadowFeatureBuilder
from pubench.models import build_model
from pubench.models.sar import _ShadowMixin

#: label -> (registry key, constructor params) for the SAR / beyond-SCAR arms.
MODEL_ZOO: dict[str, tuple[str, dict[str, object]]] = {
    # IPS reweighting of positives in shadow space, then the champion's HistGB on outcome features.
    "prop_ips": (
        "prop_ips",
        {"iterations": 400, "learning_rate": 0.05, "l2_regularization": 1.0, "clip_quantile": 0.05},
    ),
    # Bekker SAR-EM: joint EM of feasibility f(x) and reporting propensity e(x_e).
    "sar_em": (
        "sar_em",
        {"iterations": 200, "learning_rate": 0.05, "l2_regularization": 1.0, "em_iters": 5},
    ),
}

OUT_DIR = Path("artifacts/dataset_switch/allmodels")


def _shadow_for_positives(builder: ShadowFeatureBuilder) -> pd.DataFrame:
    """Precompute the P shadow features from the mined-positive InChIKey-14 keys (row-aligned to P).

    ``build_chemlex`` builds ``positives`` from ``AMIDE_P`` in file order, so the parquet's
    ``acid_key`` / ``amine_key`` columns align row-for-row with the featurised P matrix — no RDKit
    re-canonicalisation of the 258k pairs needed.
    """
    p = pd.read_parquet(AMIDE_P, columns=["acid_key", "amine_key"])
    return p


def _evaluate(
    model_key: str,
    params: dict[str, object],
    regime: str,
    splits: Splits,
    flabel: str,
    seeds: list[int],
    e_p: np.ndarray,
    builder: ShadowFeatureBuilder,
) -> dict[str, object]:
    """Featurize P/U/VAL/TEST + shadow features once, fit all seeds, return the metric block."""
    fname, fparams = FEATURIZERS[flabel]
    featurizer = build_featurizer(fname, fparams)
    featurizer.fit(pd.concat([splits.positives, splits.unlabeled], ignore_index=True, sort=False))
    x_p = featurizer.transform(splits.positives)
    x_u = featurizer.transform(splits.unlabeled)
    x_val = featurizer.transform(splits.val) if splits.val is not None else None
    x_test = featurizer.transform(splits.test)

    # Shadow features for U (constant across featurizers/seeds; U differs by regime).
    e_u = builder.from_smiles(splits.unlabeled.iloc[:, 0], splits.unlabeled.iloc[:, 1])

    per_seed: list[dict[str, float]] = []
    for seed in seeds:
        model = build_model(model_key, dict(params), seed=seed)
        assert isinstance(model, _ShadowMixin)
        model.set_propensity(e_p, e_u)
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
    """Parse ``--model``, build shadow features, run the grid, and persist the results."""
    parser = argparse.ArgumentParser(description="Score one SAR PU method on the Chemlex grid.")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_ZOO), help="Model label.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast harness check: transductive + raw_ecfp + seed 0 only (no file written).",
    )
    args = parser.parse_args()

    model_key, params = MODEL_ZOO[args.model]
    print(f"### chemlex_amide SAR | model={args.model} ({model_key}) params={params}")

    builder = ShadowFeatureBuilder()
    p_keys = _shadow_for_positives(builder)
    e_p = builder.from_keys(p_keys["acid_key"], p_keys["amine_key"])
    print(f"shadow features: P {e_p.shape} (from precomputed keys)")

    if args.smoke:
        block = _evaluate(
            model_key, params, "transductive", build_chemlex(), "raw_ecfp", [0], e_p, builder
        )
        print(f"SMOKE OK: {args.model} transductive/raw_ecfp seed0 -> {block['per_seed'][0]}")
        return

    regimes = {"transductive": build_chemlex(), "inductive": build_chemlex_inductive()}
    metrics: dict[str, dict[str, object]] = {}
    for regime, splits in regimes.items():
        metrics[regime] = {}
        for flabel in FEATURIZERS:
            metrics[regime][flabel] = _evaluate(
                model_key, params, regime, splits, flabel, list(SEEDS), e_p, builder
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
