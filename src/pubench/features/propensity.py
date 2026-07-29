"""Reactivity-orthogonal *shadow* features for SAR / beyond-SCAR PU learning.

The SCAR PU methods assume a positive's probability of being *reported* (i.e. landing in the
literature set ``P`` rather than staying hidden in the unlabeled pool ``U``) is a constant ``c``.
Real literature positives violate this: chemists preferentially report *popular / available*
building blocks. Correcting that bias (SAR-PU; Bekker et al. 2019) needs a **shadow variable** — a
feature that drives *labeling* (reporting) but not the *outcome* (reaction feasibility). If the
propensity is modelled on the same substrate fingerprints that carry feasibility it is not
separately identifiable and the estimator collapses back to the naive P-vs-U classifier.

This module builds such shadow features from two precomputed, deliberately **non-structural**
signals keyed by connectivity ``InChIKey-14``:

* ``report_frequency.parquet`` — how often a building block appears as a reactant across the whole
  literature/patent corpus (popularity / reportedness), and
* ``availability_zinc_instock.parquet`` — whether the block is commercially in-stock (ZINC).

Both are popularity/availability proxies, orthogonal to the reaction chemistry, so a propensity
model fit on them is identifiable. Missing report counts are treated as ``0`` (a substrate absent
from an ~832k-reaction corpus is genuinely obscure — the *missingness itself* is the low-popularity
signal), with an explicit missing-indicator column so a learner can separate "obscure" from "absent
metadata".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bhpu.chem.canonical import inchikey, strip_and_canonicalise

#: Precomputed shadow-signal tables (see ``scripts_JSP/compute_report_frequency.py`` /
#: ``compute_availability.py``), keyed by connectivity InChIKey-14 in ``substrate_key``.
REPORT_FREQ_PATH = Path("datasets/processed/features/report_frequency.parquet")
AVAILABILITY_PATH = Path("datasets/processed/features/availability_zinc_instock.parquet")

#: Ordered names of the per-pair shadow features produced by :class:`ShadowFeatureBuilder`.
SHADOW_FEATURE_NAMES: tuple[str, ...] = (
    "acid_log_report",
    "amine_log_report",
    "acid_in_stock",
    "amine_in_stock",
    "acid_report_missing",
    "amine_report_missing",
)


class ShadowFeatureBuilder:
    """Map substrate pairs to a small matrix of reactivity-orthogonal propensity features.

    The builder loads the report-frequency and availability tables once, then produces a fixed
    6-column feature matrix (see :data:`SHADOW_FEATURE_NAMES`) for either substrate *keys*
    (``InChIKey-14``, already present on the mined positives) or substrate *SMILES* (canonicalised
    to keys on demand, with a cache — the unlabeled/test matrices have only a few hundred unique
    substrates). Column order and semantics are identical across both entry points so ``P``, ``U``,
    ``VAL`` and ``TEST`` share a feature space.
    """

    def __init__(
        self,
        report_freq_path: Path = REPORT_FREQ_PATH,
        availability_path: Path = AVAILABILITY_PATH,
    ) -> None:
        """Load and index the shadow-signal tables.

        Args:
            report_freq_path: Parquet with ``substrate_key`` and ``report_count`` columns.
            availability_path: Parquet with a ``substrate_key`` column listing in-stock substrates.
        """
        rf = pd.read_parquet(report_freq_path)
        self._report: dict[str, int] = dict(
            zip(rf["substrate_key"], rf["report_count"].astype(int), strict=True)
        )
        self._in_stock: set[str] = set(pd.read_parquet(availability_path)["substrate_key"])
        self._key_cache: dict[str, str | None] = {}

    def _key(self, smiles: str) -> str | None:
        """Return the connectivity InChIKey-14 for a SMILES (cached), or ``None`` if unparseable."""
        if smiles not in self._key_cache:
            try:
                self._key_cache[smiles] = inchikey(strip_and_canonicalise(smiles) or smiles)
            except Exception:  # any RDKit failure ⇒ treat as unkeyable
                self._key_cache[smiles] = None
        return self._key_cache[smiles]

    def _row(self, acid_key: str | None, amine_key: str | None) -> list[float]:
        """Build the 6-feature shadow row for one substrate pair from its two keys."""
        acid_rc = self._report.get(acid_key or "", 0)
        amine_rc = self._report.get(amine_key or "", 0)
        return [
            float(np.log1p(acid_rc)),
            float(np.log1p(amine_rc)),
            float((acid_key in self._in_stock) if acid_key else 0.0),
            float((amine_key in self._in_stock) if amine_key else 0.0),
            float(acid_rc == 0),
            float(amine_rc == 0),
        ]

    def from_keys(self, acid_keys: pd.Series, amine_keys: pd.Series) -> np.ndarray:
        """Shadow matrix for pairs whose ``InChIKey-14`` keys are already known (the mined P).

        Args:
            acid_keys: Per-row acid connectivity keys.
            amine_keys: Per-row amine connectivity keys.

        Returns:
            ``(n_pairs, 6)`` float array in :data:`SHADOW_FEATURE_NAMES` order.
        """
        rows = [
            self._row(a if isinstance(a, str) else None, n if isinstance(n, str) else None)
            for a, n in zip(acid_keys, amine_keys, strict=True)
        ]
        return np.asarray(rows, dtype=np.float64)

    def from_smiles(self, acid_smiles: pd.Series, amine_smiles: pd.Series) -> np.ndarray:
        """Shadow matrix for pairs given as SMILES (U / VAL / TEST substrate frames).

        Args:
            acid_smiles: Per-row acid SMILES.
            amine_smiles: Per-row amine SMILES.

        Returns:
            ``(n_pairs, 6)`` float array in :data:`SHADOW_FEATURE_NAMES` order.
        """
        rows = [
            self._row(self._key(str(a)), self._key(str(n)))
            for a, n in zip(acid_smiles, amine_smiles, strict=True)
        ]
        return np.asarray(rows, dtype=np.float64)
