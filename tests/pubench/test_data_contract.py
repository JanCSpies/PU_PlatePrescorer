"""Tests for the final_datasets adapter and the PUDataset contract (synthetic parquets)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pubench.data import FinalDatasetsAdapter, Split


def _write_final_datasets(root: Path) -> None:
    """Write a tiny synthetic final_datasets tree (P merged, U, val/test plate)."""
    (root / "private").mkdir(parents=True, exist_ok=True)
    cond = {
        "metal": "Pd",
        "ligand_smiles": "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
        "ligand_key": "L",
        "cond_base_smiles": "O=C([O-])[O-]",
        "cond_base_key": "B",
        "cond_solvent_smiles": "Cc1ccccc1",
        "cond_solvent_key": "S",
    }
    positives = pd.DataFrame(
        [
            {
                "Aryl SMILES": "Brc1ccccc1",
                "Amine SMILES": "C1COCCN1",
                "Product SMILES": "c1ccc(N2CCOCC2)cc1",
                "label": 1,
                **cond,
            }
        ]
    )
    positives.to_parquet(root / "private" / "positives_merged_bh.parquet", index=False)

    # Unlabeled grid: substrate-only (Product SMILES must be derived by the adapter).
    unlabeled = pd.DataFrame([{"Aryl SMILES": "Brc1ccncc1", "Amine SMILES": "C1CCNCC1", **cond}])
    unlabeled.to_parquet(root / "unlabeled_grid.parquet", index=False)

    plate = pd.DataFrame(
        [
            {
                "Aryl SMILES": "Brc1ccc(C)cc1",
                "Amine SMILES": "C1CCNC1",
                "labels": 1,
                "eval_split": "val",
                **cond,
            },
            {
                "Aryl SMILES": "Brc1ccc(F)cc1",
                "Amine SMILES": "C1CCOCC1",
                "labels": 0,
                "eval_split": "val",
                **cond,
            },
            {
                "Aryl SMILES": "Brc1cccnc1",
                "Amine SMILES": "C1CCNCC1",
                "labels": 1,
                "eval_split": "test",
                **cond,
            },
            {
                "Aryl SMILES": "Brc1ccccn1",
                "Amine SMILES": "C1COCCN1",
                "labels": 0,
                "eval_split": "test",
                **cond,
            },
        ]
    )
    plate["Product SMILES"] = "c1ccccc1"
    plate.to_parquet(root / "test.parquet", index=False)


def test_adapter_loads_all_splits_with_contract(tmp_path: Path) -> None:
    """The adapter yields P/U/val/test with the right labels and a derived U product."""
    _write_final_datasets(tmp_path)
    dataset = FinalDatasetsAdapter(root=tmp_path, positives="merged").load()

    assert set(dataset.available_splits) == {Split.P, Split.U, Split.VAL, Split.TEST}
    # P and U carry no ground truth; val/test do.
    assert dataset.labels(Split.P) is None
    assert dataset.labels(Split.U) is None
    np.testing.assert_array_equal(dataset.labels(Split.VAL), np.array([1, 0]))
    np.testing.assert_array_equal(dataset.labels(Split.TEST), np.array([1, 0]))
    # Product SMILES was derived for the substrate-only unlabeled grid.
    u_products = dataset.records(Split.U)["Product SMILES"]
    assert u_products.notna().all()
    # Every split exposes the featurizer's required condition columns.
    for split in dataset.available_splits:
        cols = dataset.records(split).columns
        assert {"Aryl SMILES", "Amine SMILES", "metal", "ligand_smiles"}.issubset(cols)
