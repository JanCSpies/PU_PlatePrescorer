"""Run the literature->Test transfer (``sup_histgb_l2``) on the two switched datasets.

For each candidate we build the transductive PU setup and score the held-out test with the *same*
model + featurizers that gave 0.72/0.76 on JnJ (pubench ``sup_histgb_l2``; ``corr08`` + raw ECFP):

* **Chemlex (amide):** P = mined amide corpus (258k pairs); U = the 272x231 acid x amine substrate
  matrix (transductive candidate space); VAL/TEST = the shipped ``Stratified_Split_Both_Unseen``
  (substrate-disjoint), label = Conversion >= 20 %.
* **HiTEA (BH):** P = reused ``positives_merged_bh``; U = the 33x28 aryl x amine matrix; TEST = the
  36 substrate pairs collapsed to substrate-level feasibility (max yield over conditions >= 10 %).
  No VAL (only 36 pairs) -- ROC-AUC is threshold-free; it is a cheap BH sanity check.

Substrate featurization is chemistry-agnostic (two Morgan blocks), so the amide acid/amine are
mapped onto the ``Aryl SMILES`` / ``Amine SMILES`` slots the featurizer expects.

Usage:
    uv run python scripts_JSP/run_dataset_switch.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from bhpu.data import schema as bh_schema
from bhpu.eval.metrics import feasibility_metrics
from pubench.features import build_featurizer
from pubench.models import build_model

ARYL, AMINE = bh_schema.ARYL_SMILES, bh_schema.AMINE_SMILES

AMIDE_P = Path("final_datasets/private/positives_amide_merged.parquet")
MERGED_BH = Path("final_datasets/private/positives_merged_bh.parquet")
CHEMLEX = Path("datasets/raw/Chemlex/Chemlex_Acidamine_Wetlab_Data.xlsx")
HITEA = Path("datasets/raw/HiTEA/buchwald.csv")
OUT = Path("artifacts/dataset_switch/transfer_results.json")

MODEL_SPEC: dict[str, object] = {
    "base": "histgb",
    "iterations": 400,
    "learning_rate": 0.05,
    "l2_regularization": 1.0,
}
FEATURIZERS: dict[str, tuple[str, dict[str, object]]] = {
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


@dataclass(frozen=True)
class Splits:
    """P / U / (optional) VAL / TEST substrate frames (+ labels) for one dataset."""

    positives: pd.DataFrame
    unlabeled: pd.DataFrame
    test: pd.DataFrame
    test_y: np.ndarray
    val: pd.DataFrame | None = None
    val_y: np.ndarray | None = None


def _frame(aryl: pd.Series, amine: pd.Series) -> pd.DataFrame:
    """Assemble a two-column substrate frame with the featurizer's expected column names."""
    return pd.DataFrame({ARYL: aryl.to_numpy(), AMINE: amine.to_numpy()})


def _matrix(aryl: pd.Series, amine: pd.Series) -> pd.DataFrame:
    """Full cartesian aryl x amine substrate matrix (the transductive unlabeled grid U)."""
    pairs = list(product(aryl.dropna().unique(), amine.dropna().unique()))
    a, n = zip(*pairs, strict=True)
    return pd.DataFrame({ARYL: a, AMINE: n})


def build_chemlex() -> Splits:
    """Assemble the Chemlex amide pipeline (mined P, matrix U, both-unseen VAL/TEST)."""
    c = pd.read_excel(CHEMLEX)
    p = pd.read_parquet(AMIDE_P)
    positives = _frame(p["acid_smiles"], p["amine_smiles"])
    unlabeled = _matrix(c["Acid"], c["Amine"])
    split = c["Stratified_Split_Both_Unseen"]
    label = (c["Conversion"] >= 20).astype(np.int64)
    val_rows, test_rows = split == "train", split == "test"
    return Splits(
        positives=positives,
        unlabeled=unlabeled,
        test=_frame(c.loc[test_rows, "Acid"], c.loc[test_rows, "Amine"]),
        test_y=label[test_rows].to_numpy(),
        val=_frame(c.loc[val_rows, "Acid"], c.loc[val_rows, "Amine"]),
        val_y=label[val_rows].to_numpy(),
    )


def build_hitea() -> Splits:
    """Assemble the HiTEA BH pipeline (merged_bh P, matrix U, 36-pair substrate-level TEST)."""
    h = pd.read_csv(HITEA)
    p = pd.read_parquet(MERGED_BH)
    positives = _frame(p[ARYL], p[AMINE])
    unlabeled = _matrix(h["halide"], h["nuc"])
    # Collapse to substrate-level feasibility: a pair reacts if any condition gives yield >= 10 %.
    pair_yield = h.groupby(["halide", "nuc"])["Product_Yield_PCT_Area_UV"].max().reset_index()
    test = _frame(pair_yield["halide"], pair_yield["nuc"])
    test_y = (pair_yield["Product_Yield_PCT_Area_UV"] >= 10).astype(np.int64).to_numpy()
    return Splits(positives=positives, unlabeled=unlabeled, test=test, test_y=test_y)


def run_one(
    name: str, splits: Splits, featurizer_label: str, *, seed: int = 0
) -> dict[str, object]:
    """Featurize + fit sup_histgb_l2 (P-vs-U) for one featurizer; return TEST (+ VAL) metrics."""
    fname, params = FEATURIZERS[featurizer_label]
    featurizer = build_featurizer(fname, params)
    featurizer.fit(pd.concat([splits.positives, splits.unlabeled], ignore_index=True, sort=False))

    x_p = featurizer.transform(splits.positives)
    x_u = featurizer.transform(splits.unlabeled)
    model = build_model("supervised", dict(MODEL_SPEC), seed=seed)
    model.fit(x_p, x_u)

    test_scores = model.predict_proba(featurizer.transform(splits.test))
    test = feasibility_metrics(splits.test_y, test_scores)
    row: dict[str, object] = {
        "dataset": name,
        "featurizer": featurizer_label,
        "n_features": int(x_p.shape[1]),
        "n_P": len(splits.positives),
        "n_U": len(splits.unlabeled),
        "n_test": len(splits.test),
        "test_pos_rate": float(np.mean(splits.test_y)),
        "test_roc_auc": test.roc_auc,
        "test_auprc": test.auprc,
    }
    if splits.val is not None and splits.val_y is not None:
        val_scores = model.predict_proba(featurizer.transform(splits.val))
        val = feasibility_metrics(splits.val_y, val_scores)
        row["val_roc_auc"] = val.roc_auc
        row["val_auprc"] = val.auprc
    return row


def main() -> None:
    """Run both datasets x both featurizers and print + persist the comparison."""
    datasets = {"chemlex_amide": build_chemlex(), "hitea_bh": build_hitea()}
    results: list[dict[str, object]] = []
    for name, splits in datasets.items():
        print(f"\n### {name}: P={len(splits.positives)} U={len(splits.unlabeled)} "
              f"test={len(splits.test)} (pos {np.mean(splits.test_y):.1%})")
        for flabel in FEATURIZERS:
            r = run_one(name, splits, flabel)
            results.append(r)
            val = f" | VAL {r['val_roc_auc']:.3f}" if "val_roc_auc" in r else ""
            print(f"  {flabel:<9} n_feat={r['n_features']:<5} "
                  f"TEST ROC-AUC {r['test_roc_auc']:.3f}  AP {r['test_auprc']:.3f}{val}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"model": MODEL_SPEC, "results": results}, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
