"""Inductive check for Chemlex: rebuild U from *non-test* substrates only, score the same test.

The transductive run (`run_dataset_switch.py`) built U from the full 272x231 acid x amine matrix,
which *contains* the held-out test substrates -- so its 0.775 is a transductive (plate-pre-scorer)
number. Chemlex's ``Stratified_Split_Both_Unseen`` makes train-row and test-row substrates disjoint
(0 shared acids, 0 shared amines), so we can drop every test-row substrate from U and still have a
non-empty pool: the 100 non-test acids x 155 non-test amines = 15,500 combos. Every one of the 2,234
test pairs then has *both* substrates absent from U -> scoring them is fully **inductive**.

Compare to the transductive numbers (raw ECFP 0.775 / corr08 0.729):

* AUC stays ~0.77  -> a genuine novel-substrate predictor (JnJ collapsed to 0.40 under this drop).
* AUC collapses    -> the gain was transductive, like JnJ.

Note this isolates U-transductivity only; P (literature amide) still legitimately contains some test
substrates in *other* pairs (the realistic 69%-acid / 26%-amine coverage), exactly as the JnJ
inductive check kept its literature P intact. U size differs from the transductive run (15,500 vs
62,832) -- an unavoidable consequence of the drop, same as the JnJ grid check.

Usage:
    uv run python scripts_JSP/run_chemlex_inductive.py
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scripts_JSP.run_dataset_switch import (
    AMIDE_P,
    AMINE,
    ARYL,
    CHEMLEX,
    FEATURIZERS,
    Splits,
    _frame,
    run_one,
)

OUT = Path("artifacts/dataset_switch/inductive_results.json")


def build_chemlex_inductive() -> Splits:
    """Assemble the inductive Chemlex pipeline: U spans only non-test substrates."""
    c = pd.read_excel(CHEMLEX)
    split = c["Stratified_Split_Both_Unseen"]
    test_rows, val_rows = split == "test", split == "train"

    test_acids = set(c.loc[test_rows, "Acid"].dropna().astype(str))
    test_amines = set(c.loc[test_rows, "Amine"].dropna().astype(str))
    keep_acids = sorted(set(c["Acid"].dropna().astype(str)) - test_acids)
    keep_amines = sorted(set(c["Amine"].dropna().astype(str)) - test_amines)
    grid = list(product(keep_acids, keep_amines))
    u_aryl, u_amine = zip(*grid, strict=True)
    unlabeled = pd.DataFrame({ARYL: u_aryl, AMINE: u_amine})

    p = pd.read_parquet(AMIDE_P)
    label = (c["Conversion"] >= 20).astype(np.int64)
    return Splits(
        positives=_frame(p["acid_smiles"], p["amine_smiles"]),
        unlabeled=unlabeled,
        test=_frame(c.loc[test_rows, "Acid"], c.loc[test_rows, "Amine"]),
        test_y=label[test_rows].to_numpy(),
        val=_frame(c.loc[val_rows, "Acid"], c.loc[val_rows, "Amine"]),
        val_y=label[val_rows].to_numpy(),
    )


def main() -> None:
    """Run the inductive Chemlex variant for both featurizers and persist results."""
    splits = build_chemlex_inductive()
    print(
        f"### chemlex_amide INDUCTIVE: P={len(splits.positives)} "
        f"U={len(splits.unlabeled)} (non-test substrates only) "
        f"test={len(splits.test)} (pos {np.mean(splits.test_y):.1%})"
    )
    results = []
    for flabel in FEATURIZERS:
        r = run_one("chemlex_amide_inductive", splits, flabel)
        results.append(r)
        val = f" | VAL {r['val_roc_auc']:.3f}" if "val_roc_auc" in r else ""
        print(
            f"  {flabel:<9} n_feat={r['n_features']:<5} "
            f"TEST ROC-AUC {r['test_roc_auc']:.3f}  AP {r['test_auprc']:.3f}{val}"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"results": results}, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
