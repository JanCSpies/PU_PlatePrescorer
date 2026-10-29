"""The one place a feature block becomes a matrix (ADR 018).

A "block" is what a model or a probe is allowed to see: a fingerprint column, the hand-written
descriptors, or size on its own. The same four lines were written out in ``score/baselines.py``,
``scripts/counts/count_adversarial.py``, ``scripts/features.py`` and a scratch script, each with a
different dtype, and ADR 018 is about to measure a family of candidate blocks -- which would have
added a fifth copy.

Rewired onto this module: ``score.baselines._fingerprints``, ``count_adversarial``,
``count_bit_frequency`` and ``count_fold_width``. **Not** rewired: ``features.morgan_summary``,
which summarises a block for the stage F report rather than feeding a model, and whose numbers are
part of a committed report -- it is the one surviving copy of the idiom.

This module converts and nothing else. It does **not** binarise: a caller that wants bits writes
``matrix > 0`` itself, so the one place that decides what a count of 3 means is the caller that
cares.
"""

import numpy as np
import pandas as pd

# The nine descriptors a model may see. ``n_heavy_atoms`` is deliberately **not** among them: size
# alone separates literature substrates from lab ones at about 0.67 (ADR 012 item 2), so it is
# reported on its own axis, and folding it into a block would hide that contribution inside it.
# ``features.substrate.DESCRIPTORS`` is a different set with a different job -- the ten columns
# stage F writes -- and the two are kept apart on purpose.
MODEL_DESCRIPTORS = (
    "centre_in_ring",
    "n_alpha",
    "alpha_n_aromatic",
    "alpha_n_in_ring",
    "alpha_max_heavy_degree",
    "alpha_sum_heavy_degree",
    "n_heavy_at_2_bonds",
    "n_heavy_at_3_bonds",
    "n_ewg_on_ring",
)
SIZE = "n_heavy_atoms"
# Columns holding one fixed-size count vector per substrate (stage F writes uint8, 2,048 wide).
FINGERPRINT_BLOCKS = ("morgan", "morgan_centre")
# Blocks assembled from scalar columns, in the order the matrix must carry them.
COLUMN_BLOCKS: dict[str, tuple[str, ...]] = {
    "descriptors": MODEL_DESCRIPTORS,
    "descriptors + size": (*MODEL_DESCRIPTORS, SIZE),
    "size alone": (SIZE,),
}


def block_matrix(substrates: pd.DataFrame, block: str) -> np.ndarray:
    """``block`` as a dense float64 matrix, one row per substrate in ``substrates``' own order.

    Args:
        substrates: Any frame carrying the block's columns; the stage F table, or a subset of it.
        block: A fingerprint column (``FINGERPRINT_BLOCKS``) or an assembled one
            (``COLUMN_BLOCKS``).

    Returns:
        ``(len(substrates), width)`` of float64, values unchanged -- counts stay counts.

    Raises:
        ValueError: ``block`` is not a block this project defines, or it is one whose columns this
            frame does not carry. The two are different mistakes -- a misspelt name against a table
            built without that column -- so they raise different messages.
    """
    if block in COLUMN_BLOCKS:
        columns = COLUMN_BLOCKS[block]
        _require(substrates, columns, block)
        return substrates[list(columns)].to_numpy(dtype=np.float64)
    if block in FINGERPRINT_BLOCKS:
        _require(substrates, (block,), block)
        return np.stack(
            [np.asarray(fingerprint) for fingerprint in substrates[block]]
        ).astype(np.float64)
    known = sorted([*COLUMN_BLOCKS, *FINGERPRINT_BLOCKS])
    raise ValueError(f"unknown feature block {block!r}; known blocks: {known}")


def _require(substrates: pd.DataFrame, columns: tuple[str, ...], block: str) -> None:
    """Refuse a known block whose columns this frame lacks, naming every one of them."""
    missing = [name for name in columns if name not in substrates.columns]
    if missing:
        raise ValueError(
            f"the feature table lacks column(s) {missing} of block {block!r}"
        )
