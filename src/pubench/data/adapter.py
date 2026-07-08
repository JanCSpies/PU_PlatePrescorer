"""Adapter that loads the pre-sampled ``final_datasets`` splits into a :class:`PUDataset`.

This is a *thin* adapter: it does not resample or re-split anything. It reads the parquet files
written by ``scripts_JSP/assemble_datasets.py``, projects each onto the canonical schema
(:mod:`pubench.data.schema`), derives the ``Product SMILES`` for the unlabeled pool (which is
stored substrate-only), and wraps the four splits in a :class:`PUDataset`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bhpu.chem.reaction import build_cn_coupling_product
from pubench.data import schema
from pubench.data.dataset import PUDataset, Split, SplitFrame

#: Columns kept for every split (the featurizer's input contract).
_CANONICAL_COLUMNS: tuple[str, ...] = (
    schema.ARYL_SMILES,
    schema.AMINE_SMILES,
    schema.PRODUCT_SMILES,
    schema.METAL,
    schema.LIGAND_SMILES,
    schema.BASE_SMILES,
    schema.SOLVENT_SMILES,
)


@dataclass(frozen=True, slots=True)
class FinalDatasetsAdapter:
    """Load P / U / val / test from a ``final_datasets`` directory.

    Parameters
    ----------
    root:
        Directory holding the assembled parquet files (e.g. ``final_datasets``).
    positives:
        Which positives corpus to use as P — a key of :data:`schema.DEFAULT_POSITIVES`
        (``"merged"`` or ``"ord"``).
    """

    root: Path
    positives: str = "merged"

    def load(self) -> PUDataset:
        """Read the four splits and assemble a :class:`PUDataset`.

        Returns:
        -------
        PUDataset
            Container with the P, U, val and test :class:`SplitFrame` objects.

        Raises:
        ------
        FileNotFoundError
            If a required parquet file is missing.
        KeyError
            If ``positives`` is not a known positives-corpus key.
        """
        p_frame = self._load_positives()
        u_frame = self._load_unlabeled()
        val_frame, test_frame = self._load_eval()
        return PUDataset(
            frames={
                Split.P: p_frame,
                Split.U: u_frame,
                Split.VAL: val_frame,
                Split.TEST: test_frame,
            }
        )

    def _load_positives(self) -> SplitFrame:
        """Load the positives corpus (all rows are PU-positive, ``y`` is undefined)."""
        if self.positives not in schema.DEFAULT_POSITIVES:
            raise KeyError(
                f"Unknown positives key {self.positives!r}; "
                f"choices: {sorted(schema.DEFAULT_POSITIVES)}."
            )
        records = self._read(schema.DEFAULT_POSITIVES[self.positives])
        return SplitFrame(role=Split.P, records=_project(records), y=None)

    def _load_unlabeled(self) -> SplitFrame:
        """Load the unlabeled grid, deriving the missing ``Product SMILES`` from substrates."""
        records = self._read(schema.DEFAULT_UNLABELED)
        records = _project(records, derive_product=True)
        return SplitFrame(role=Split.U, records=records, y=None)

    def _load_eval(self) -> tuple[SplitFrame, SplitFrame]:
        """Load the JnJ plate and split it into labelled val / test by ``eval_split``."""
        plate = self._read(schema.DEFAULT_EVAL)
        frames: list[SplitFrame] = []
        for role, tag in ((Split.VAL, "val"), (Split.TEST, "test")):
            subset = plate.loc[plate[schema.EVAL_SPLIT] == tag].reset_index(drop=True)
            y = subset[schema.LABEL].to_numpy().astype(np.int64)
            frames.append(SplitFrame(role=role, records=_project(subset), y=y))
        return frames[0], frames[1]

    def _read(self, relative: str) -> pd.DataFrame:
        """Read a parquet file under :attr:`root`, raising a clear error if absent."""
        path = self.root / relative
        if not path.exists():
            raise FileNotFoundError(f"Expected dataset file not found: {path}")
        return pd.read_parquet(path)


def _project(records: pd.DataFrame, *, derive_product: bool = False) -> pd.DataFrame:
    """Project ``records`` onto the canonical schema, optionally deriving the product.

    Parameters
    ----------
    records:
        A raw split frame read from parquet.
    derive_product:
        If ``True`` and ``Product SMILES`` is absent, build it from the substrates via
        :func:`bhpu.chem.reaction.build_cn_coupling_product` (deterministic BH product).

    Returns:
    -------
    pandas.DataFrame
        A frame containing exactly the canonical columns.
    """
    frame = records.copy()
    if derive_product and schema.PRODUCT_SMILES not in frame.columns:
        frame[schema.PRODUCT_SMILES] = [
            build_cn_coupling_product(aryl, amine)
            for aryl, amine in zip(
                frame[schema.ARYL_SMILES], frame[schema.AMINE_SMILES], strict=True
            )
        ]
    missing = [c for c in _CANONICAL_COLUMNS if c not in frame.columns]
    for column in missing:
        frame[column] = None
    return frame.loc[:, list(_CANONICAL_COLUMNS)].reset_index(drop=True)
