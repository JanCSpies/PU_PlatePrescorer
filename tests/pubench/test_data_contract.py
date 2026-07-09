"""Tests for the final_datasets adapter and the PUDataset contract (synthetic parquets)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pubench.data import BaseKeyFilter, FinalDatasetsAdapter, Split


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
    # Every split exposes the featurizer's required condition columns, including the identity keys
    # the physics-descriptor featurizer looks up by.
    for split in dataset.available_splits:
        cols = dataset.records(split).columns
        assert {
            "Aryl SMILES",
            "Amine SMILES",
            "metal",
            "ligand_smiles",
            "ligand_key",
            "cond_base_key",
            "cond_solvent_key",
        }.issubset(cols)


def _write_mixed_base_datasets(root: Path) -> None:
    """Write a synthetic tree where every split carries two base keys (``CARB`` and ``OTHER``)."""
    (root / "private").mkdir(parents=True, exist_ok=True)

    def cond(base_key: str) -> dict[str, str]:
        return {
            "metal": "Pd",
            "ligand_smiles": "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
            "ligand_key": "L",
            "cond_base_smiles": "O=C([O-])[O-]",
            "cond_base_key": base_key,
            "cond_solvent_smiles": "Cc1ccccc1",
            "cond_solvent_key": "S",
        }

    def rows(specs: list[tuple[str, dict[str, object]]]) -> pd.DataFrame:
        return pd.DataFrame(
            [{**base, **extra} for base_key, extra in specs for base in [cond(base_key)]]
        )

    rows(
        [
            ("CARB", {"Aryl SMILES": "Brc1ccccc1", "Amine SMILES": "C1COCCN1", "label": 1}),
            ("OTHER", {"Aryl SMILES": "Brc1ccncc1", "Amine SMILES": "C1CCNCC1", "label": 1}),
        ]
    ).to_parquet(root / "private" / "positives_merged_bh.parquet", index=False)

    rows(
        [
            ("CARB", {"Aryl SMILES": "Brc1ccc(C)cc1", "Amine SMILES": "C1CCNC1"}),
            ("OTHER", {"Aryl SMILES": "Brc1ccc(F)cc1", "Amine SMILES": "C1CCOCC1"}),
        ]
    ).to_parquet(root / "unlabeled_grid.parquet", index=False)

    plate = rows(
        [
            (
                "CARB",
                {
                    "Aryl SMILES": "Brc1cccnc1",
                    "Amine SMILES": "C1CCNCC1",
                    "labels": 1,
                    "eval_split": "val",
                },
            ),
            (
                "OTHER",
                {
                    "Aryl SMILES": "Brc1ccccn1",
                    "Amine SMILES": "C1COCCN1",
                    "labels": 0,
                    "eval_split": "val",
                },
            ),
            (
                "CARB",
                {
                    "Aryl SMILES": "Brc1ccc(Cl)cc1",
                    "Amine SMILES": "C1CCCNC1",
                    "labels": 0,
                    "eval_split": "test",
                },
            ),
            (
                "OTHER",
                {
                    "Aryl SMILES": "Brc1ccc(Br)cc1",
                    "Amine SMILES": "CNC",
                    "labels": 1,
                    "eval_split": "test",
                },
            ),
        ]
    )
    plate["Product SMILES"] = "c1ccccc1"
    plate.to_parquet(root / "test.parquet", index=False)


def test_base_filter_restricts_only_targeted_splits(tmp_path: Path) -> None:
    """A carbonate-only filter keeps ``CARB`` rows in targeted splits and passes others through."""
    _write_mixed_base_datasets(tmp_path)
    base_filter = BaseKeyFilter(
        keys=frozenset({"CARB"}), apply_to=frozenset({Split.P, Split.U, Split.VAL, Split.TEST})
    )
    dataset = FinalDatasetsAdapter(
        root=tmp_path, positives="merged", base_filter=base_filter
    ).load()

    for split in (Split.P, Split.U, Split.VAL, Split.TEST):
        keys = dataset.records(split)["cond_base_key"].tolist()
        assert keys == ["CARB"], f"{split} not filtered to carbonate: {keys}"
    # The surviving val/test rows keep their labels aligned after filtering.
    np.testing.assert_array_equal(dataset.labels(Split.VAL), np.array([1]))
    np.testing.assert_array_equal(dataset.labels(Split.TEST), np.array([0]))


def test_base_filter_leaves_unlisted_splits_whole(tmp_path: Path) -> None:
    """``apply_to`` scoping: filtering only P/U leaves val and test at full size."""
    _write_mixed_base_datasets(tmp_path)
    base_filter = BaseKeyFilter(keys=frozenset({"CARB"}), apply_to=frozenset({Split.P, Split.U}))
    dataset = FinalDatasetsAdapter(
        root=tmp_path, positives="merged", base_filter=base_filter
    ).load()

    assert dataset.frame(Split.P).n == 1
    assert dataset.frame(Split.U).n == 1
    assert dataset.frame(Split.VAL).n == 2  # untouched
    assert dataset.frame(Split.TEST).n == 2  # untouched
