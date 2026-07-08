"""Structural fingerprints of reaction *conditions* (catalyst, base, solvent).

The substrate featurisers ignore the reaction conditions, but conditions drive a large share of
Buchwald-Hartwig feasibility (which is why HTE screens them). This module ECFP-fingerprints the
catalyst/base/solvent SMILES so that similar conditions map close in feature space. The JnJ catalyst
SMILES are full Pd precatalyst complexes, so a single fingerprint captures the metal *and* the
ligand; base and solvent are fingerprinted likewise.

Conditions are fingerprinted per unique (catalyst, base, solvent) SMILES triple and looked up per
row, so the cost is set by the (small) number of distinct condition sets, not the row count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bhpu.data import schema
from bhpu.features.fingerprints import ECFPFeaturizer

#: SMILES columns that define a reaction condition set.
CONDITION_SMILES_COLUMNS: tuple[str, str, str] = (
    schema.CATALYST_SMILES,
    schema.BASE_SMILES,
    schema.SOLVENT_SMILES,
)

#: Default bits per condition component (catalyst / base / solvent).
DEFAULT_CONDITION_BITS: int = 1024


def condition_ecfp_map(
    frames: list[pd.DataFrame], *, n_bits: int = DEFAULT_CONDITION_BITS
) -> dict[tuple[str, str, str], np.ndarray]:
    """ECFP every unique (catalyst, base, solvent) SMILES triple across the given frames.

    Args:
        frames: Frames carrying the condition SMILES columns.
        n_bits: Bits per condition component.

    Returns:
        A ``(catalyst, base, solvent) -> concatenated ECFP`` lookup.
    """
    triples = pd.concat([f[list(CONDITION_SMILES_COLUMNS)] for f in frames], ignore_index=True)
    triples = triples.dropna().drop_duplicates().reset_index(drop=True)
    featurizer = ECFPFeaturizer(substrate_columns=CONDITION_SMILES_COLUMNS, n_bits=n_bits)
    matrix = featurizer.fit_transform(triples).astype(np.float32)
    keys = triples[list(CONDITION_SMILES_COLUMNS)].to_numpy()
    return {(str(k[0]), str(k[1]), str(k[2])): matrix[i] for i, k in enumerate(keys)}


def condition_block(
    frame: pd.DataFrame,
    cmap: dict[tuple[str, str, str], np.ndarray],
    *,
    n_bits: int = DEFAULT_CONDITION_BITS,
) -> np.ndarray:
    """Assemble the per-row condition ECFP block from a triple map (zeros for unseen triples).

    Args:
        frame: Reactions with the condition SMILES columns.
        cmap: Lookup from :func:`condition_ecfp_map`.
        n_bits: Bits per condition component (must match the map).

    Returns:
        An ``(n_rows, 3 * n_bits)`` float32 condition-fingerprint matrix.
    """
    width = n_bits * len(CONDITION_SMILES_COLUMNS)
    rows = [
        cmap.get((str(cat), str(base), str(solv)), np.zeros(width, dtype=np.float32))
        for cat, base, solv in frame[list(CONDITION_SMILES_COLUMNS)].to_numpy()
    ]
    return np.vstack(rows).astype(np.float32)
