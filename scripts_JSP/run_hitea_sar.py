"""Second-reaction check for the beyond-SCAR arm: prop_ips / sar_em vs the champion on HiTEA BH.

The Chemlex amide grid showed the SAR reporting-bias correction ``prop_ips`` beating the naive
P-vs-U HistGB champion by +0.024 ROC-AUC. This script asks whether that is a real, transferable
effect or a lucky one-shot by rerunning the *identical* head-to-head on a **different reaction
class** -- the HiTEA Buchwald-Hartwig C-N screen -- under one harness so featurization, seeds and
test set are shared across models.

HiTEA is small and transductive-only (see :mod:`scripts_JSP.run_hitea_eval`): P = merged BH
positives (64k pairs), U = the 33x28 aryl x amine matrix, TEST = 36 substrate pairs collapsed to
substrate-level feasibility (max yield over conditions >= 10 %). There is no validation split, so we
report the threshold-free ROC-AUC / AUPRC (mean +/- std over 5 seeds) plus a percentile bootstrap CI
on the seed-averaged scores -- the n=36 interval is wide by construction and reported honestly.

Because the SAR models need reactivity-orthogonal shadow features, we also log the **shadow P-vs-U
separability AUC** (how well the popularity/availability features alone tell reported positives from
the candidate grid). On amide this is ~0.93; a much lower value here means the reporting bias is
less identifiable for BH substrates and the correction has less to work with -- context for a delta.

Models (all on the identical 36-pair test, raw-ECFP + corr08 substrate features):
    * ``sup_histgb_l2`` -- the naive P-vs-U HistGB champion (SCAR baseline).
    * ``prop_ips``      -- inverse-propensity reweighting of positives in shadow space (beyond-SCAR)
    * ``sar_em``        -- Bekker 2019 joint EM of feasibility ``f`` and propensity ``e``.

Saves ``hitea_sar_metrics.json`` (per-model per-seed + mean/std + bootstrap CI, both featurizers,
plus the champion delta) for the write-up / journal.

Run with:  uv run python -m scripts_JSP.run_hitea_sar          (full: 3 models x 2 feats x 5 seeds)
           uv run python -m scripts_JSP.run_hitea_sar --smoke   (raw_ecfp + seed 0, no file written)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scripts_JSP.run_chemlex_sar import MODEL_ZOO
from scripts_JSP.run_dataset_switch import FEATURIZERS, MERGED_BH, MODEL_SPEC, Splits, build_hitea
from scripts_JSP.run_multiseed_eval import _mean_std
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from pubench.features import build_featurizer
from pubench.features.propensity import ShadowFeatureBuilder
from pubench.models import build_model
from pubench.models.sar import _ShadowMixin

SEEDS = [0, 1, 2, 3, 4]
OUT_JSON = Path("artifacts/dataset_switch/hitea_sar_metrics.json")

#: label -> (registry key, params). The champion is the SCAR baseline the SAR arm must beat.
MODELS: dict[str, tuple[str, dict[str, object]]] = {
    "sup_histgb_l2": ("supervised", dict(MODEL_SPEC)),
    "prop_ips": MODEL_ZOO["prop_ips"],
    "sar_em": MODEL_ZOO["sar_em"],
}


def _bootstrap_auc(y: np.ndarray, scores: np.ndarray, n: int = 2000) -> tuple[float, float]:
    """Percentile 95% CI for ROC-AUC by resampling the (n=36) test set; seeded for reproducibility.

    On this small a test the interval is wide by construction -- reporting it honestly is the point.
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


def _shadow_separability(e_p: np.ndarray, e_u: np.ndarray) -> float:
    """In-sample ROC-AUC of a balanced logistic telling reported P from the candidate grid U.

    A proxy for how *identifiable* the reporting propensity is on these substrates: if the
    popularity/availability shadow features cannot separate P from U the SAR correction degenerates
    to the naive P-vs-U classifier.
    """
    x = np.vstack([e_p, e_u])
    s = np.r_[np.ones(len(e_p)), np.zeros(len(e_u))]
    scaler = StandardScaler().fit(x)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(scaler.transform(x), s)
    return float(roc_auc_score(s, lr.predict_proba(scaler.transform(x))[:, 1]))


def _fit_predict(
    label: str,
    key: str,
    params: dict[str, object],
    seed: int,
    x_p: np.ndarray,
    x_u: np.ndarray,
    x_test: np.ndarray,
    e_p: np.ndarray,
    e_u: np.ndarray,
) -> np.ndarray:
    """Fit one model (injecting shadow features for the SAR arm) and score the test frame."""
    model = build_model(key, dict(params), seed=seed)
    if isinstance(model, _ShadowMixin):
        model.set_propensity(e_p, e_u)
    model.fit(x_p, x_u)
    return np.asarray(model.predict_proba(x_test), dtype=float)


def _evaluate_featurizer(
    flabel: str, splits: Splits, e_p: np.ndarray, builder: ShadowFeatureBuilder
) -> tuple[dict[str, dict[str, object]], float]:
    """Featurize P/U/TEST + U shadow once, then score every model over all seeds for one feat."""
    fname, params = FEATURIZERS[flabel]
    featurizer = build_featurizer(fname, params)
    featurizer.fit(pd.concat([splits.positives, splits.unlabeled], ignore_index=True, sort=False))
    x_p = featurizer.transform(splits.positives)
    x_u = featurizer.transform(splits.unlabeled)
    x_test = featurizer.transform(splits.test)
    e_u = builder.from_smiles(splits.unlabeled.iloc[:, 0], splits.unlabeled.iloc[:, 1])
    sep = _shadow_separability(e_p, e_u)

    out: dict[str, dict[str, object]] = {}
    for label, (key, mparams) in MODELS.items():
        per_seed: list[dict[str, float]] = []
        scores_stack: list[np.ndarray] = []
        for seed in SEEDS:
            scores = _fit_predict(label, key, mparams, seed, x_p, x_u, x_test, e_p, e_u)
            scores_stack.append(scores)
            per_seed.append(
                {
                    "seed": seed,
                    "roc_auc": float(roc_auc_score(splits.test_y, scores)),
                    "auprc": float(average_precision_score(splits.test_y, scores)),
                }
            )
        mean_scores = np.mean(scores_stack, axis=0)
        mean: dict[str, float] = {}
        std: dict[str, float] = {}
        for k in ("roc_auc", "auprc"):
            mean[k], std[k] = _mean_std(per_seed, k)
        ci = list(_bootstrap_auc(splits.test_y, mean_scores))
        out[label] = {"per_seed": per_seed, "mean": mean, "std": std, "bootstrap_ci": ci}
        print(
            f"  {flabel:<9} {label:<14} AUC {mean['roc_auc']:.3f}±{std['roc_auc']:.3f} "
            f"AP {mean['auprc']:.3f}  CI[{ci[0]:.3f},{ci[1]:.3f}]"
        )
    return out, sep


def _auc(block: dict[str, object]) -> float:
    """Read the mean ROC-AUC out of a per-model metric block."""
    mean = block["mean"]
    assert isinstance(mean, dict)
    return float(mean["roc_auc"])


def main() -> None:
    """Run the champion + SAR arm on HiTEA BH and persist the head-to-head metrics."""
    parser = argparse.ArgumentParser(description="SAR arm vs champion on the HiTEA BH set.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast harness check: raw_ecfp + seed 0 for every model, no file written.",
    )
    args = parser.parse_args()

    splits = build_hitea()
    y = splits.test_y
    print(
        f"### HiTEA BH SAR: P={len(splits.positives)} U={len(splits.unlabeled)} "
        f"test={len(y)} (pos {int(y.sum())}/{len(y)}, base {y.mean():.3f})"
    )

    builder = ShadowFeatureBuilder()
    # P shadow features from the precomputed BH keys (aryl_key/amine_key align row-for-row with P).
    p_keys = pd.read_parquet(MERGED_BH, columns=["aryl_key", "amine_key"])
    e_p = builder.from_keys(p_keys["aryl_key"], p_keys["amine_key"])
    print(f"shadow features: P {e_p.shape} (from precomputed BH keys)")

    if args.smoke:
        fname, params = FEATURIZERS["raw_ecfp"]
        featurizer = build_featurizer(fname, params)
        featurizer.fit(
            pd.concat([splits.positives, splits.unlabeled], ignore_index=True, sort=False)
        )
        x_p = featurizer.transform(splits.positives)
        x_u = featurizer.transform(splits.unlabeled)
        x_test = featurizer.transform(splits.test)
        e_u = builder.from_smiles(splits.unlabeled.iloc[:, 0], splits.unlabeled.iloc[:, 1])
        print(f"shadow P-vs-U separability AUC: {_shadow_separability(e_p, e_u):.3f}")
        for label, (key, mparams) in MODELS.items():
            s = _fit_predict(label, key, mparams, 0, x_p, x_u, x_test, e_p, e_u)
            print(f"SMOKE {label:<14} raw_ecfp seed0 AUC {roc_auc_score(y, s):.3f}")
        return

    metrics: dict[str, dict[str, dict[str, object]]] = {}
    separability: dict[str, float] = {}
    for flabel in FEATURIZERS:
        cell, sep = _evaluate_featurizer(flabel, splits, e_p, builder)
        metrics[flabel] = cell
        separability[flabel] = sep
        print(f"  {flabel}: shadow P-vs-U separability AUC {sep:.3f}")

    # Champion deltas per featurizer (positive = SAR arm beats the naive HistGB baseline).
    champ = {f: _auc(metrics[f]["sup_histgb_l2"]) for f in FEATURIZERS}
    deltas = {
        f: {label: _auc(metrics[f][label]) - champ[f] for label in MODELS} for f in FEATURIZERS
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reaction": "Buchwald-Hartwig C-N (HiTEA)",
        "regime": "transductive_only",
        "seeds": SEEDS,
        "n_positives": len(splits.positives),
        "n_unlabeled": len(splits.unlabeled),
        "test": {"n": len(y), "base_rate": float(y.mean())},
        "models": {
            label: {"key": key, "params": params} for label, (key, params) in MODELS.items()
        },
        "shadow_pu_separability": separability,
        "metrics": metrics,
        "delta_vs_champion": deltas,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT_JSON}")
    for f in FEATURIZERS:
        for label in MODELS:
            print(f"{f:<9} {label:<14} Δ vs champ {deltas[f][label]:+.3f}")


if __name__ == "__main__":
    main()
