"""Do GFN2-xTB descriptors help, and does the transfer survive an inductive U?

Two focused questions on the winning config (importance-weighted PU-bagging, literature+ORD),
run directly with :class:`DomainAdaptedPULearner` (far lighter than the full 7-arm harness):

1. **Feature ablation** — ECFP vs ECFP⊕xTB vs xTB-only. Does adding per-substrate quantum
   electronics (HOMO/LUMO/gap/dipole/charges) lift transfer beyond substrate fingerprints?
2. **Inductive robustness** — ECFP with the transductive grid U vs an inductive U (TEST substrates
   removed), to confirm the headline gain is not leaning on seeing test substrate structures.

Writes ``artifacts/qm_and_inductive.json``. Usage:
    uv run python scripts/11_qm_and_inductive.py [--n-estimators 60]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bhpu.data import schema
from bhpu.eval import feasibility_metrics
from bhpu.eval.protocol import FrozenEvalProtocol, _substrate_key
from bhpu.eval.transfer import load_corpus_features
from bhpu.features.xtb import XTBDescriptorFeaturizer
from bhpu.models.boosting import CatBoostFeasibilityModel
from bhpu.pu.adapted import DomainAdaptedPULearner

FEATURES_DIR = Path("datasets/processed/features")
CORPUS = Path("datasets/processed/cn_corpus_ord.parquet")
XTB_CACHE = FEATURES_DIR / "xtb_substrate_cache.parquet"
ARTIFACTS_DIR = Path("artifacts")


def _load_npz(name: str) -> np.ndarray:
    with np.load(FEATURES_DIR / name) as data:
        matrix: np.ndarray = data["X"].astype(np.float32)
    return matrix


def _fit_eval(
    x_pos: np.ndarray,
    x_grid: np.ndarray,
    x_source: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_estimators: int,
) -> dict[str, float]:
    """Fit importance-weighted PU-bagging and return VAL/TEST ROC-AUC + AUPRC."""

    def factory() -> CatBoostFeasibilityModel:
        return CatBoostFeasibilityModel(iterations=400, depth=6, random_state=0)

    features = np.vstack([x_pos, x_grid])
    selection = np.concatenate([np.ones(len(x_pos)), np.zeros(len(x_grid))])
    learner = DomainAdaptedPULearner(
        factory, use_coral=False, use_importance_weights=True, n_estimators=n_estimators
    )
    learner.fit(features, selection, source_marginal=x_source)
    val = feasibility_metrics(y_val, learner.predict_proba(x_val))
    test = feasibility_metrics(y_test, learner.predict_proba(x_test))
    return {
        "val_auc": val.roc_auc,
        "val_ap": val.auprc,
        "test_auc": test.roc_auc,
        "test_ap": test.auprc,
        "domain_auc": learner.domain_auc_,
        "ess": learner.effective_sample_size_,
    }


def main() -> None:
    """Entry point: run the feature ablation and the inductive-U robustness check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=60)
    args = parser.parse_args()

    lit_frame = pd.read_parquet(FEATURES_DIR / "literature_clean.parquet")
    jnj_frame = pd.read_parquet(FEATURES_DIR / "test_jnj.parquet")
    grid_frame = pd.read_parquet(FEATURES_DIR / "unlabeled_grid.parquet")
    ord_frame = pd.read_parquet(CORPUS).rename(
        columns={"aryl_smiles": schema.ARYL_SMILES, "amine_smiles": schema.AMINE_SMILES}
    )

    protocol = FrozenEvalProtocol(jnj_frame)
    protocol.assert_training_clean(lit_frame)
    pos_mask = lit_frame[schema.LABEL].to_numpy() == 1
    val_idx, test_idx = protocol.val_index, protocol.test_index
    y_val, y_test = protocol.labels("val"), protocol.labels("test")

    # ECFP feature blocks (cached; ORD featurised on the fly in the same bit space).
    ecfp = {
        "lit": _load_npz("literature_clean__ecfp2048.npz"),
        "grid": _load_npz("unlabeled_grid__ecfp2048.npz"),
        "jnj": _load_npz("test_jnj__ecfp2048.npz"),
        "ord": load_corpus_features(CORPUS),
    }
    # xTB descriptor blocks (per-substrate, from the cache).
    cache = XTBDescriptorFeaturizer.load_cache(str(XTB_CACHE))
    featurizer = XTBDescriptorFeaturizer(cache)
    xtb = {
        "lit": featurizer.transform(lit_frame),
        "grid": featurizer.transform(grid_frame),
        "jnj": featurizer.transform(jnj_frame),
        "ord": featurizer.transform(ord_frame),
    }
    print(f"xTB cache: {len(cache)} substrates; xtb dim {xtb['lit'].shape[1]}")

    def blocks(kind: str) -> dict[str, np.ndarray]:
        """Assemble feature blocks for a feature kind (ecfp / ecfp_xtb / xtb)."""
        if kind == "ecfp":
            return ecfp
        if kind == "xtb":
            return xtb
        return {k: np.hstack([ecfp[k], xtb[k]]).astype(np.float32) for k in ecfp}

    # Inductive grid mask: drop grid rows reusing a TEST substrate.
    test_rows = jnj_frame.loc[test_idx]
    test_keys = {
        k
        for col in (schema.ARYL_SMILES, schema.AMINE_SMILES)
        for s in test_rows[col]
        if (k := _substrate_key(s)) is not None
    }
    grid_aryl = [_substrate_key(s) for s in grid_frame[schema.ARYL_SMILES]]
    grid_amine = [_substrate_key(s) for s in grid_frame[schema.AMINE_SMILES]]
    inductive_mask = np.array(
        [
            a not in test_keys and n not in test_keys
            for a, n in zip(grid_aryl, grid_amine, strict=True)
        ]
    )
    print(f"Inductive grid keeps {int(inductive_mask.sum())}/{len(inductive_mask)} rows")

    results: dict[str, dict[str, float]] = {}

    # 1) Feature ablation (transductive grid).
    for kind in ("ecfp", "ecfp_xtb", "xtb"):
        b = blocks(kind)
        x_pos = np.vstack([b["lit"][pos_mask], b["ord"]]).astype(np.float32)
        x_source = np.vstack([b["lit"], b["ord"]]).astype(np.float32)
        results[f"{kind}__transductive"] = _fit_eval(
            x_pos, b["grid"], x_source, b["jnj"][val_idx], y_val, b["jnj"][test_idx], y_test,
            n_estimators=args.n_estimators,
        )
        print(f"{kind:<9} transductive  -> {json.dumps(results[f'{kind}__transductive'])}")

    # 2) Inductive robustness (ECFP).
    b = blocks("ecfp")
    x_pos = np.vstack([b["lit"][pos_mask], b["ord"]]).astype(np.float32)
    x_source = np.vstack([b["lit"], b["ord"]]).astype(np.float32)
    results["ecfp__inductive"] = _fit_eval(
        x_pos, b["grid"][inductive_mask], x_source, b["jnj"][val_idx], y_val,
        b["jnj"][test_idx], y_test, n_estimators=args.n_estimators,
    )
    print(f"ecfp      inductive     -> {json.dumps(results['ecfp__inductive'])}")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "qm_and_inductive.json").write_text(json.dumps(results, indent=2))
    print("\nWrote artifacts/qm_and_inductive.json")


if __name__ == "__main__":
    main()
