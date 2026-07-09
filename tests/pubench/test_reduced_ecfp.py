"""Tests for the dimensionality-reduced ECFP featurizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pubench.features import FEATURIZERS, ReducedECFPFeaturizer, build_featurizer


def _records() -> pd.DataFrame:
    """A small, structurally diverse substrate set (enough columns to survive reduction)."""
    aryls = [
        "Brc1ccccc1",
        "Brc1ccncc1",
        "Clc1ccc(C)cc1",
        "Ic1ccc(OC)cc1",
        "Brc1cccc(C(F)(F)F)c1",
        "Clc1ccc(N(C)C)cc1",
        "Brc1ccc2ccccc2c1",
        "Clc1ncccc1",
    ]
    amines = [
        "C1COCCN1",
        "C1CCNCC1",
        "CN(C)C",
        "NCc1ccccc1",
        "c1ccc(N)cc1",
        "CCNCC",
        "C1CCC(N)CC1",
        "Nc1ccncc1",
    ]
    return pd.DataFrame({"Aryl SMILES": aryls, "Amine SMILES": amines})


def test_registered() -> None:
    """ReducedECFPFeaturizer self-registers under 'reduced_ecfp'."""
    assert "reduced_ecfp" in FEATURIZERS.names()
    assert isinstance(build_featurizer("reduced_ecfp"), ReducedECFPFeaturizer)


def test_svd_gives_dense_low_dim_block() -> None:
    """SVD reduction yields a dense (n_rows, n_components) float32 matrix."""
    feat = ReducedECFPFeaturizer(n_bits=256, method="svd", n_components=4, use_conditions=False)
    x = feat.fit_transform(_records())
    assert x.shape == (8, 4)
    assert x.dtype == np.float32
    assert feat.n_features == 4
    # Dense: the SVD projection is (almost surely) not mostly zeros like the raw fingerprint.
    assert (x != 0).mean() > 0.5


def test_svd_caps_components_at_available_bits() -> None:
    """Requesting more components than surviving bits is capped, not an error."""
    feat = ReducedECFPFeaturizer(n_bits=256, method="svd", n_components=10_000)
    x = feat.fit_transform(_records())
    assert 1 <= x.shape[1] < 10_000  # capped to the available rank, not the request
    assert x.shape[1] == feat.n_features


def test_none_passthrough_matches_raw_width() -> None:
    """method='none' returns the raw fingerprint (only constant columns removed)."""
    feat = ReducedECFPFeaturizer(n_bits=128, method="none", use_conditions=False)
    x = feat.fit_transform(_records())
    # substrate-only, 2 blocks of 128 bits = 256, minus all-zero columns.
    assert x.shape[0] == 8
    assert x.shape[1] <= 256


def test_min_freq_drops_rare_bits() -> None:
    """A higher min_freq keeps strictly fewer bits than the constant-only filter."""
    base = ReducedECFPFeaturizer(n_bits=256, method="variance", min_freq=0.0)
    strict = ReducedECFPFeaturizer(n_bits=256, method="variance", min_freq=0.3)
    assert strict.fit_transform(_records()).shape[1] < base.fit_transform(_records()).shape[1]


def test_correlation_prune_does_not_widen() -> None:
    """Correlation pruning keeps a subset of the frequency-filtered bits."""
    variance = ReducedECFPFeaturizer(n_bits=256, method="variance")
    corr = ReducedECFPFeaturizer(n_bits=256, method="correlation", corr_threshold=0.8)
    assert corr.fit_transform(_records()).shape[1] <= variance.fit_transform(_records()).shape[1]


def test_transform_before_fit_raises() -> None:
    """Transforming before fitting is a hard error."""
    feat = ReducedECFPFeaturizer(n_bits=64)
    with pytest.raises(RuntimeError):
        feat.transform(_records())


def test_rejects_bad_params() -> None:
    """Unknown method and out-of-range min_freq are rejected at construction."""
    with pytest.raises(ValueError):
        ReducedECFPFeaturizer(method="pca")
    with pytest.raises(ValueError):
        ReducedECFPFeaturizer(min_freq=0.7)
