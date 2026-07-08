"""Tests for the ECFP featurizer and the featurizer registry."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pubench.features import FEATURIZERS, ECFPFeaturizer, build_featurizer


def _records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Aryl SMILES": ["Brc1ccccc1", "Brc1ccncc1"],
            "Amine SMILES": ["C1COCCN1", None],  # a missing SMILES -> zero block
            "metal": ["Pd", "Cu"],
            "ligand_smiles": ["c1ccc(P(c2ccccc2)c2ccccc2)cc1", None],
            "cond_base_smiles": ["O=C([O-])[O-]", "O"],
            "cond_solvent_smiles": ["Cc1ccccc1", "C1CCOC1"],
        }
    )


def test_ecfp_is_registered() -> None:
    """ECFPFeaturizer self-registers under 'ecfp'."""
    assert "ecfp" in FEATURIZERS.names()
    assert isinstance(build_featurizer("ecfp"), ECFPFeaturizer)


def test_ecfp_shape_and_zero_block() -> None:
    """Feature width matches n_features and missing SMILES yield a zero block."""
    feat = ECFPFeaturizer(radius=2, n_bits=64, use_conditions=True)
    x = feat.fit_transform(_records())
    assert x.shape == (2, feat.n_features)
    assert x.dtype == np.float32
    # row 1 has no amine and no ligand -> those two 64-bit blocks are all zero.
    amine_block = x[1, 64:128]
    assert amine_block.sum() == 0.0


def test_ecfp_substrate_only_is_narrower() -> None:
    """Disabling conditions drops the condition + metal blocks."""
    with_cond = ECFPFeaturizer(n_bits=64, use_conditions=True)
    subs_only = ECFPFeaturizer(n_bits=64, use_conditions=False)
    assert subs_only.n_features < with_cond.n_features
    assert subs_only.fit_transform(_records()).shape == (2, subs_only.n_features)


def test_metal_onehot() -> None:
    """The metal one-hot marks the right column."""
    feat = ECFPFeaturizer(n_bits=8, use_conditions=True, metal_vocab=("Pd", "Cu", "Ni"))
    x = feat.fit_transform(_records())
    metal_block = x[:, -3:]
    np.testing.assert_array_equal(metal_block[0], [1, 0, 0])  # Pd
    np.testing.assert_array_equal(metal_block[1], [0, 1, 0])  # Cu
