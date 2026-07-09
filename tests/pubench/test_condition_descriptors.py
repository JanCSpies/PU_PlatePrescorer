"""Tests for the physical condition-descriptor featurizer and the composite featurizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pubench.features import (
    FEATURIZERS,
    CompositeFeaturizer,
    ConditionDescriptorFeaturizer,
    build_featurizer,
)
from pubench.features.condition_descriptors import (
    BASE_DESCRIPTOR_NAMES,
    BASE_TABLE,
    LIGAND_DESCRIPTOR_NAMES,
    SOLVENT_DESCRIPTOR_NAMES,
    SOLVENT_TABLE,
)

_LIG_WIDTH = len(LIGAND_DESCRIPTOR_NAMES)


def _records() -> pd.DataFrame:
    """Two rows using real base/solvent keys; one known and one unknown ligand key."""
    return pd.DataFrame(
        {
            "Aryl SMILES": ["Brc1ccccc1", "Brc1ccncc1"],
            "Amine SMILES": ["C1COCCN1", "C1CCNCC1"],
            "ligand_key": ["LIG_KNOWN", "LIG_UNKNOWN"],
            "cond_base_key": ["BVKZGUZCCUSVTD", "KRHYYFGTRYWZRS"],  # carbonate, fluoride
            "cond_solvent_key": ["WYURNTSHIVDZCO", "IAZDPXIOMUYVGZ"],  # THF, DMSO
            "metal": ["Pd", "Pd"],
        }
    )


def _fake_cache() -> dict[str, np.ndarray]:
    return {"LIG_KNOWN": np.arange(_LIG_WIDTH, dtype=np.float64)}


def test_tables_cover_expected_counts() -> None:
    """The curated tables carry the 13 bases and 18 solvents seen in the corpus."""
    assert len(BASE_TABLE) == 13
    assert len(SOLVENT_TABLE) == 18
    assert all(len(v) == len(BASE_DESCRIPTOR_NAMES) for v in BASE_TABLE.values())
    assert all(len(v) == len(SOLVENT_DESCRIPTOR_NAMES) for v in SOLVENT_TABLE.values())


def test_condition_featurizer_registered() -> None:
    """The condition-descriptor featurizer self-registers."""
    assert "condition_descriptors" in FEATURIZERS.names()
    assert isinstance(build_featurizer("condition_descriptors"), ConditionDescriptorFeaturizer)


def test_condition_shape_and_no_nan() -> None:
    """Output width = ligand ⊕ base ⊕ solvent ⊕ metal, and missing values are imputed away."""
    feat = ConditionDescriptorFeaturizer(ligand_cache=_fake_cache(), standardize=True)
    x = feat.fit_transform(_records())
    expected = _LIG_WIDTH + len(BASE_DESCRIPTOR_NAMES) + len(SOLVENT_DESCRIPTOR_NAMES) + 3
    assert x.shape == (2, expected)
    assert feat.n_features == expected
    assert len(feat.feature_names) == expected
    assert np.isfinite(x).all()  # unknown ligand / any nan is median-imputed


def test_metal_onehot_marks_pd() -> None:
    """Without standardisation the Pd one-hot column is 1 for both rows."""
    feat = ConditionDescriptorFeaturizer(ligand_cache=_fake_cache(), standardize=False)
    x = feat.fit_transform(_records())
    pd_col = _LIG_WIDTH + len(BASE_DESCRIPTOR_NAMES) + len(SOLVENT_DESCRIPTOR_NAMES)  # first metal
    assert np.allclose(x[:, pd_col], 1.0)


def test_condition_transform_before_fit_raises() -> None:
    """Transforming before fitting is a hard error."""
    with pytest.raises(RuntimeError):
        ConditionDescriptorFeaturizer(ligand_cache=_fake_cache()).transform(_records())


def test_composite_concatenates_blocks() -> None:
    """CompositeFeaturizer stacks an ECFP block and a condition-descriptor block."""
    feat = CompositeFeaturizer(
        components=[
            {"name": "ecfp", "params": {"n_bits": 32, "use_conditions": False}},
            {"name": "condition_descriptors", "params": {"ligand_cache": _fake_cache()}},
        ]
    )
    x = feat.fit_transform(_records())
    ecfp_width = 2 * 32
    cond_width = _LIG_WIDTH + len(BASE_DESCRIPTOR_NAMES) + len(SOLVENT_DESCRIPTOR_NAMES) + 3
    assert x.shape == (2, ecfp_width + cond_width)
    assert x.dtype == np.float32


def test_composite_rejects_empty() -> None:
    """A composite with no components is rejected."""
    with pytest.raises(ValueError):
        CompositeFeaturizer(components=[])
