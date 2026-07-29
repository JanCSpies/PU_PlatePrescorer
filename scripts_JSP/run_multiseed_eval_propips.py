"""Multi-seed plate-prescoring eval of the ``prop_ips`` beyond-SCAR model (for the PDF report).

Sibling of :mod:`scripts_JSP.run_multiseed_eval` (which fits the ``sup_histgb_l2`` champion). Fits
``prop_ips`` -- inverse-propensity reweighting of the literature positives in reactivity-orthogonal
shadow space, then the champion's HistGB on outcome features -- across the identical grid
(transductive + inductive x raw_ecfp + corr08 x 5 seeds) and writes the **same artifact layout**
:mod:`scripts_JSP.make_report` consumes, suffixed ``_propips`` so the champion report stays intact:

* ``multiseed_metrics_propips.json`` -- per-seed + mean/std for all four cells, with the
  val-selected Youden-threshold confusion matrix (identical block layout to the champion file).
* ``eval_arrays_propips.npz`` -- seed-averaged raw-ECFP test/val scores per regime (the report's
  curves, confusion matrices, plate-operating curve and bootstrap CI read from here).

``prop_ips`` scores VAL/TEST from the outcome-feature HistGB alone -- the shadow features enter only
at fit time, to weight the positives -- so only P and U need shadow features here (built exactly as
in :mod:`scripts_JSP.run_chemlex_sar`). Render the report from the output with:

    uv run python -m scripts_JSP.make_report --model propips

Run the eval on SLURM (matches the ~50 min cx_sar prop_ips job):
    sbatch scripts_JSP/run_multiseed_eval_propips.sbatch
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scripts_JSP.run_chemlex_inductive import build_chemlex_inductive
from scripts_JSP.run_chemlex_sar import MODEL_ZOO, _shadow_for_positives
from scripts_JSP.run_dataset_switch import FEATURIZERS, Splits, build_chemlex
from scripts_JSP.run_multiseed_eval import SEEDS, _confusion, _mean_std, _youden_threshold
from sklearn.metrics import average_precision_score, roc_auc_score

from pubench.features import build_featurizer
from pubench.features.propensity import ShadowFeatureBuilder
from pubench.models import build_model
from pubench.models.sar import _ShadowMixin

MODEL_LABEL = "prop_ips"
OUT_JSON = Path("artifacts/dataset_switch/multiseed_metrics_propips.json")
OUT_NPZ = Path("artifacts/dataset_switch/eval_arrays_propips.npz")


def _evaluate(
    regime: str, splits: Splits, flabel: str, e_p: np.ndarray, builder: ShadowFeatureBuilder
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Featurize + build U shadow once, fit prop_ips per seed; return block + test/val scores."""
    fname, params = FEATURIZERS[flabel]
    featurizer = build_featurizer(fname, params)
    featurizer.fit(pd.concat([splits.positives, splits.unlabeled], ignore_index=True, sort=False))
    x_p = featurizer.transform(splits.positives)
    x_u = featurizer.transform(splits.unlabeled)
    x_val = featurizer.transform(splits.val) if splits.val is not None else None
    x_test = featurizer.transform(splits.test)
    e_u = builder.from_smiles(splits.unlabeled.iloc[:, 0], splits.unlabeled.iloc[:, 1])

    model_key, mparams = MODEL_ZOO[MODEL_LABEL]
    per_seed: list[dict[str, float]] = []
    test_scores_all, val_scores_all = [], []
    for seed in SEEDS:
        model = build_model(model_key, dict(mparams), seed=seed)
        assert isinstance(model, _ShadowMixin)
        model.set_propensity(e_p, e_u)
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
    for k in (key for key in per_seed[0] if key != "seed"):
        m, s = _mean_std(per_seed, k)
        block["mean"][k] = m  # type: ignore[index]
        block["std"][k] = s  # type: ignore[index]
    return block, np.array(test_scores_all), np.array(val_scores_all)


def main() -> None:
    """Run the full multi-seed prop_ips grid and persist metrics + seed-averaged raw-ECFP scores."""
    model_key, mparams = MODEL_ZOO[MODEL_LABEL]
    print(f"### chemlex_amide multiseed | model={MODEL_LABEL} ({model_key}) params={mparams}")

    builder = ShadowFeatureBuilder()
    p_keys = _shadow_for_positives(builder)
    e_p = builder.from_keys(p_keys["acid_key"], p_keys["amine_key"])
    print(f"shadow features: P {e_p.shape} (from precomputed keys)")

    regimes = {"transductive": build_chemlex(), "inductive": build_chemlex_inductive()}
    metrics: dict[str, dict[str, object]] = {}
    arrays: dict[str, np.ndarray] = {}
    for regime, splits in regimes.items():
        metrics[regime] = {}
        arrays[f"{regime}_test_y"] = splits.test_y
        if splits.val_y is not None:
            arrays[f"{regime}_val_y"] = splits.val_y
        for flabel in FEATURIZERS:
            block, test_scores, val_scores = _evaluate(regime, splits, flabel, e_p, builder)
            metrics[regime][flabel] = block
            if flabel == "raw_ecfp":  # keep the production model's scores for the report
                arrays[f"{regime}_test_scores"] = test_scores.mean(axis=0)
                if val_scores.size:
                    arrays[f"{regime}_val_scores"] = val_scores.mean(axis=0)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seeds": list(SEEDS),
        "model": {"label": MODEL_LABEL, "key": model_key, "params": mparams},
        "metrics": metrics,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    np.savez(OUT_NPZ, **arrays)
    print(f"\nWrote {OUT_JSON} and {OUT_NPZ}")
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
