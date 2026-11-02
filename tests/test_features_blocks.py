"""Tests for platepu.features.blocks, written before the module on 2026-10-28.

The module is the one place where a feature block of the stage F table becomes a dense matrix. It
replaces the copies of the idiom in ``score.similarity.fingerprints`` (which was
``score.baselines._fingerprints`` until 2026-10-30), ``count_adversarial``,
``count_bit_frequency`` and ``count_fold_width``, and two sets of descriptors that shared one name.
``scripts/features.py`` ``morgan_summary`` is the one surviving copy: it summarises a block for the
stage F report rather than feeding a model, and its numbers are in a committed report.

The dtype rule pinned here: **every matrix is float64, with the values unchanged**. Fingerprint
counts (uint8 in the table) are widened, never binarised, never scaled; int16 descriptors are
widened too. Callers that want bits write ``(matrix > 0)`` themselves.

Error shapes pinned here (the substrings the tests match):
    unknown name:   ``unknown feature block 'morgan_ecfp'; known blocks: [...]``
    missing column: ``the feature table lacks column(s) ['n_alpha'] of block 'descriptors'``

The fixtures are made up: three substrates, a 6-bit fingerprint, descriptor values that differ per
column so a swapped column cannot go unseen.
"""

import numpy as np
import pandas as pd
import pytest

from platepu.features.blocks import FINGERPRINT_BLOCKS, MODEL_DESCRIPTORS, block_matrix
from platepu.features.substrate import DESCRIPTORS

NINE = (
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
KEYS = ["k_first", "k_second", "k_third"]


def _fingerprint(*counts: int) -> np.ndarray:
    return np.array(counts, dtype=np.uint8)


def _table() -> pd.DataFrame:
    """Three substrates. Descriptor column number c (0 to 9, in ``DESCRIPTORS`` order) holds
    ``10 * c + row``; the frame lists them reversed, so frame order is not model order."""
    values = {
        name: np.array([10 * column + row for row in range(3)], dtype=np.int16)
        for column, name in enumerate(DESCRIPTORS)
    }
    descriptors = pd.DataFrame({name: values[name] for name in reversed(DESCRIPTORS)})
    fingerprints = pd.DataFrame(
        {
            "key_stereo": KEYS,
            "morgan": [
                _fingerprint(0, 1, 0, 0, 0, 3),
                _fingerprint(1, 0, 0, 0, 0, 0),
                _fingerprint(0, 0, 2, 0, 1, 0),
            ],
            "morgan_centre": [
                _fingerprint(1, 0, 0),
                _fingerprint(0, 1, 0),
                _fingerprint(0, 0, 1),
            ],
        }
    )
    return pd.concat([fingerprints, descriptors], axis=1)


def test_block_matrix_of_a_fingerprint_column() -> None:
    matrix = block_matrix(_table(), "morgan")
    assert matrix.shape == (3, 6)
    assert matrix.dtype == np.float64
    assert matrix[0].tolist() == [0, 1, 0, 0, 0, 3]  # row order is the frame's
    assert matrix[1].tolist() == [1, 0, 0, 0, 0, 0]
    assert matrix[2].tolist() == [0, 0, 2, 0, 1, 0]


def test_block_matrix_follows_the_frame_row_order_not_the_key() -> None:
    reversed_table = _table().iloc[::-1].reset_index(drop=True)
    matrix = block_matrix(reversed_table, "morgan_centre")
    assert matrix.tolist() == [[0, 0, 1], [0, 1, 0], [1, 0, 0]]


def test_block_matrix_of_the_nine_descriptors() -> None:
    matrix = block_matrix(_table(), "descriptors")
    assert matrix.shape == (3, 9)
    assert matrix.dtype == np.float64
    # Column c of DESCRIPTORS holds 10 * c + row; the model's nine skip column 0 (n_heavy_atoms),
    # so output column j is DESCRIPTORS column j + 1, in MODEL_DESCRIPTORS order.
    for row in range(3):
        assert matrix[row].tolist() == [10 * (j + 1) + row for j in range(9)]


def test_block_matrix_of_descriptors_plus_size_appends_the_heavy_atoms_last() -> None:
    matrix = block_matrix(_table(), "descriptors + size")
    assert matrix.shape == (3, 10)
    assert matrix[:, :9].tolist() == block_matrix(_table(), "descriptors").tolist()
    assert matrix[:, 9].tolist() == [0, 1, 2]  # n_heavy_atoms is DESCRIPTORS column 0


def test_block_matrix_of_size_alone_is_the_heavy_atoms() -> None:
    matrix = block_matrix(_table(), "size alone")
    assert matrix.shape == (3, 1)
    assert matrix[:, 0].tolist() == [0, 1, 2]


def test_block_matrix_refuses_an_unknown_block() -> None:
    with pytest.raises(ValueError, match="unknown feature block 'morgan_ecfp'"):
        block_matrix(_table(), "morgan_ecfp")


def test_block_matrix_refuses_a_block_the_table_lacks() -> None:
    """A known composite whose column is gone says so; it is not a misspelling."""
    without = _table().drop(columns=["n_alpha"])
    with pytest.raises(
        ValueError, match=r"lacks column\(s\) \['n_alpha'\] of block 'descriptors'"
    ):
        block_matrix(without, "descriptors")


def test_block_matrix_names_a_missing_size_column() -> None:
    without = _table().drop(columns=["n_heavy_atoms"])
    with pytest.raises(
        ValueError, match=r"lacks column\(s\) \['n_heavy_atoms'\] of block 'size alone'"
    ):
        block_matrix(without, "size alone")


def test_model_descriptors_exclude_heavy_atoms() -> None:
    """Stops the model's nine and the stage F ten from being quietly merged: size has its own axis
    (it alone separates the two sources at about 0.67, ADR 012 item 2)."""
    assert "n_heavy_atoms" not in MODEL_DESCRIPTORS
    assert tuple(MODEL_DESCRIPTORS) == NINE
    assert len(DESCRIPTORS) == 10
    assert "n_heavy_atoms" in DESCRIPTORS
    assert set(MODEL_DESCRIPTORS) == set(DESCRIPTORS) - {"n_heavy_atoms"}


def test_block_matrix_does_not_binarise() -> None:
    matrix = block_matrix(_table(), "morgan")
    assert matrix.max() == 3.0
    assert matrix[0, 5] == 3.0
    assert matrix[2, 2] == 2.0


def _near_table() -> pd.DataFrame:
    """Two substrates with a full-width (2,048) morgan_near column, as stage F writes it."""
    first, second = np.zeros(2048, dtype=np.uint8), np.zeros(2048, dtype=np.uint8)
    first[5], first[2047] = 1, 3
    second[0] = 2
    return pd.DataFrame({"key_stereo": ["k1", "k2"], "morgan_near": [first, second]})


def test_block_matrix_of_morgan_near_is_float64_and_full_width() -> None:
    matrix = block_matrix(_near_table(), "morgan_near")
    assert matrix.shape == (2, 2048)
    assert matrix.dtype == np.float64
    assert (matrix[0, 5], matrix[0, 2047], matrix[1, 0]) == (
        1.0,
        3.0,
        2.0,
    )  # counts, not bits


def test_unknown_block_error_lists_morgan_near_among_the_known() -> None:
    with pytest.raises(ValueError, match="known blocks:.*'morgan_near'"):
        block_matrix(_near_table(), "morgan_ecfp")


def test_morgan_near_is_a_fingerprint_block_and_its_name_carries_no_distance() -> None:
    """The distance is chosen per class and lives in configs/features.yml (ADR 030); a column named
    for one value (morgan_d6) would lie for a class that picks another."""
    assert "morgan_near" in FINGERPRINT_BLOCKS
    assert not any(any(ch.isdigit() for ch in name) for name in FINGERPRINT_BLOCKS)
