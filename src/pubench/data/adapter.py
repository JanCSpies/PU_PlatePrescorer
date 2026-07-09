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


@dataclass(frozen=True, slots=True)
class BaseKeyFilter:
    """Restrict named splits to rows whose reaction base is in an allowlist.

    Used to carve the *in-distribution-condition* subset of the JnJ plate — e.g. carbonate-only
    (``cond_base_key == "BVKZGUZCCUSVTD"``), the one base that overlaps the literature corpus
    (49.7 % of P vs 18.8 % of test), whereas P2-Et phosphazene / HMDS / silanolate are absent from
    P by design (Neves' *robust OOD* screen; see ``docs/decisions/007-...``). Filtering P/U/test to
    a shared base regime lets us test whether conditions crash transfer *only* because of the
    distributional mismatch, not the representation.

    Attributes:
    ----------
    keys:
        Allowlist of canonical ``cond_base_key`` values (InChIKey-14). A row survives iff its base
        key is in this set (``NaN`` base keys never match, so unknown-base rows are dropped).
    apply_to:
        The splits the filter is applied to; other splits pass through unchanged.
    """

    keys: frozenset[str]
    apply_to: frozenset[Split]

    def mask(self, records: pd.DataFrame, split: Split) -> pd.DataFrame:
        """Return ``records`` filtered to :attr:`keys` if ``split`` is in :attr:`apply_to`.

        Parameters
        ----------
        records:
            A raw split frame carrying the ``cond_base_key`` column.
        split:
            Which split ``records`` belongs to.

        Returns:
        -------
        pandas.DataFrame
            The row-filtered frame (index reset) when the filter applies to ``split``, otherwise
            ``records`` unchanged.
        """
        if split not in self.apply_to:
            return records
        keep = records[schema.BASE_KEY].isin(self.keys)
        return records.loc[keep].reset_index(drop=True)


#: Columns kept for every split (the featurizer's input contract). Includes both the condition
#: SMILES (for fingerprint featurizers) and the canonical identity keys (for the physics-descriptor
#: featurizer, which looks up ligand/base/solvent descriptors by key).
_CANONICAL_COLUMNS: tuple[str, ...] = (
    schema.ARYL_SMILES,
    schema.AMINE_SMILES,
    schema.PRODUCT_SMILES,
    schema.METAL,
    schema.LIGAND_SMILES,
    schema.BASE_SMILES,
    schema.SOLVENT_SMILES,
    schema.LIGAND_KEY,
    schema.BASE_KEY,
    schema.SOLVENT_KEY,
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
    base_filter:
        Optional :class:`BaseKeyFilter` restricting one or more splits to an in-distribution base
        regime (e.g. carbonate-only). ``None`` (default) loads every split whole.
    """

    root: Path
    positives: str = "merged"
    base_filter: BaseKeyFilter | None = None

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
        records = self._filter(records, Split.P)
        return SplitFrame(role=Split.P, records=_project(records), y=None)

    def _load_unlabeled(self) -> SplitFrame:
        """Load the unlabeled grid, deriving the missing ``Product SMILES`` from substrates."""
        records = self._read(schema.DEFAULT_UNLABELED)
        records = self._filter(records, Split.U)
        records = _project(records, derive_product=True)
        return SplitFrame(role=Split.U, records=records, y=None)

    def _load_eval(self) -> tuple[SplitFrame, SplitFrame]:
        """Load the JnJ plate and split it into labelled val / test by ``eval_split``."""
        plate = self._read(schema.DEFAULT_EVAL)
        frames: list[SplitFrame] = []
        for role, tag in ((Split.VAL, "val"), (Split.TEST, "test")):
            subset = plate.loc[plate[schema.EVAL_SPLIT] == tag].reset_index(drop=True)
            subset = self._filter(subset, role)
            y = subset[schema.LABEL].to_numpy().astype(np.int64)
            frames.append(SplitFrame(role=role, records=_project(subset), y=y))
        return frames[0], frames[1]

    def _filter(self, records: pd.DataFrame, split: Split) -> pd.DataFrame:
        """Apply :attr:`base_filter` to ``records`` for ``split`` (no-op if unset)."""
        if self.base_filter is None:
            return records
        return self.base_filter.mask(records, split)

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
