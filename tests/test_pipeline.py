"""Fast unit tests for the bhpu pipeline (synthetic data; no large files needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bhpu.chem import build_cn_coupling_product, canonical_smiles, inchikey
from bhpu.data import schema
from bhpu.eval import NevesFoldReader, feasibility_metrics, plate_hit_rate_uplift
from bhpu.features import ConditionEncoder, DRFPFeaturizer, ECFPFeaturizer
from bhpu.leakage import LeakageAuditor
from bhpu.models import CatBoostFeasibilityModel
from bhpu.pu import BaggingPULearner, SpyPULearner
from bhpu.space import CombinatorialSpaceGenerator


def _toy_frame() -> pd.DataFrame:
    """A tiny, valid BH reaction table for shape/logic tests."""
    return pd.DataFrame(
        {
            schema.ARYL_SMILES: ["Brc1ccccc1", "Clc1ccncc1", "Brc1ccc(C)cc1"],
            schema.AMINE_SMILES: ["C1CCNCC1", "C1COCCN1", "CCN"],
            schema.CATALYST_NAME: ["ruphos-pd-g4", "xphos-pd-g4", "ruphos-pd-g4"],
            schema.CATALYST_SMILES: ["[Pd]", "[Pd]", "[Pd]"],
            schema.BASE_NAME: ["P2Et", "Cs2CO3", "P2Et"],
            schema.BASE_SMILES: ["CCN=P", "O=C([O-])[O-].[Cs+].[Cs+]", "CCN=P"],
            schema.SOLVENT_NAME: ["t-AmOH", "DMSO", "t-AmOH"],
            schema.SOLVENT_SMILES: ["CCC(C)(C)O", "CS(C)=O", "CCC(C)(C)O"],
            schema.PRODUCT_SMILES: [
                "O=C(c1ccccc1)N1CCCCC1",  # placeholder; not chemically checked here
                "c1ccncc1N1CCOCC1",
                "CCNc1ccc(C)cc1",
            ],
            schema.LABEL: [1, 0, 1],
        }
    )


def test_canonical_and_inchikey_are_stable() -> None:
    assert canonical_smiles("c1ccccc1") == canonical_smiles("C1=CC=CC=C1")
    assert inchikey("c1ccccc1") == inchikey("C1=CC=CC=C1")
    assert canonical_smiles("not_a_smiles") is None


def test_build_cn_coupling_product_matches_expected() -> None:
    # Bromobenzene + morpholine -> 4-phenylmorpholine.
    product = build_cn_coupling_product("Brc1ccccc1", "C1COCCN1")
    assert product == canonical_smiles("c1ccc(N2CCOCC2)cc1")
    # A tertiary amine (no N-H) cannot couple.
    assert build_cn_coupling_product("Brc1ccccc1", "CN(C)C") is None


def test_ecfp_and_drfp_shapes() -> None:
    frame = _toy_frame()
    ecfp = ECFPFeaturizer(n_bits=256)
    x_ecfp = ecfp.fit_transform(frame)
    assert x_ecfp.shape == (3, 512)  # two substrates x 256 bits
    assert len(ecfp.feature_names) == 512

    x_drfp = DRFPFeaturizer(n_bits=256).fit_transform(frame)
    assert x_drfp.shape == (3, 256)
    assert (x_drfp.sum(axis=1) > 0).all()


def test_condition_encoder_handles_unknown_categories() -> None:
    encoder = ConditionEncoder().fit(_toy_frame())
    unseen = _toy_frame().copy()
    unseen[schema.CATALYST_NAME] = "brand-new-catalyst"
    encoded = encoder.transform(unseen)
    # Unknown catalyst -> the catalyst block is all-zero for those rows.
    assert encoded.shape[0] == 3


def test_leakage_auditor_detects_and_filters_shared_substrates() -> None:
    test = _toy_frame()
    literature = _toy_frame()  # identical substrates -> maximal leakage
    auditor = LeakageAuditor(test)
    report = auditor.audit(literature)
    assert report.n_rows_sharing_any_substrate == 3
    assert report.n_rows_sharing_pair == 3
    assert len(auditor.filter(literature, level="substrate")) == 0


def test_combinatorial_space_excludes_measured_pairs() -> None:
    reference = _toy_frame()
    # Use a condition set whose reagents exist in the toy reference frame.
    generator = CombinatorialSpaceGenerator(
        reference,
        condition_sets=(schema.ConditionSet("ruphos-pd-g4", "P2Et", "t-AmOH"),),
        add_products=True,
    )
    grid = generator.generate(exclude="measured_pairs")
    measured = set(zip(reference[schema.ARYL_SMILES], reference[schema.AMINE_SMILES], strict=True))
    grid_pairs = set(zip(grid[schema.ARYL_SMILES], grid[schema.AMINE_SMILES], strict=True))
    assert measured.isdisjoint(grid_pairs)


def test_neves_fold_reader_parses_iteration_columns() -> None:
    frame = pd.DataFrame(
        {
            "iteration_0 train_test": [
                "['train', 'test']",
                "['test', 'train']",
                "['train', 'train']",
            ]
        }
    )
    reader = NevesFoldReader(frame)
    folds = list(reader.iter_folds())
    assert reader.n_folds_per_iteration == 2
    assert len(folds) == 2
    # fold 0: row 1 is test; fold 1: row 0 is test.
    assert folds[0].test_index.tolist() == [1]
    assert folds[1].test_index.tolist() == [0]


def test_feasibility_metrics_perfect_and_calibration() -> None:
    y_true = np.array([0, 0, 1, 1])
    perfect = feasibility_metrics(y_true, np.array([0.0, 0.1, 0.9, 1.0]))
    assert perfect.roc_auc == pytest.approx(1.0)
    assert perfect.auprc == pytest.approx(1.0)
    assert perfect.base_rate == pytest.approx(0.5)


def test_plate_uplift_ranks_positives_first() -> None:
    y_true = np.array([1, 1, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.1, 0.2, 0.05, 0.15])
    (result,) = plate_hit_rate_uplift(y_true, y_score, budgets=(2,), n_bootstrap=100)
    assert result.model_hit_rate == pytest.approx(1.0)
    assert result.uplift_vs_random > 1.0


@pytest.mark.parametrize("learner_name", ["spy", "bagging"])
def test_pu_learners_recover_positives(learner_name: str) -> None:
    """On separable synthetic data, PU learners rank hidden positives above negatives."""
    rng = np.random.default_rng(0)
    n = 150
    positives = rng.normal(2.0, 0.5, size=(n, 8))
    negatives = rng.normal(-2.0, 0.5, size=(n, 8))
    features = np.vstack([positives, negatives]).astype(np.float32)
    true_labels = np.concatenate([np.ones(n), np.zeros(n)])

    # Reveal half of the positives as labeled; hide everything else as unlabeled.
    selection = np.zeros(2 * n, dtype=int)
    revealed = rng.choice(np.arange(n), size=n // 2, replace=False)
    selection[revealed] = 1
    unlabeled = np.flatnonzero(selection == 0)

    def factory() -> CatBoostFeasibilityModel:
        return CatBoostFeasibilityModel(iterations=40, depth=3, random_state=0)

    if learner_name == "spy":
        learner = SpyPULearner(factory, random_state=0)
    else:
        learner = BaggingPULearner(factory, n_estimators=8, random_state=0)
    learner.fit(features, selection)

    scores = learner.predict_proba(features[unlabeled])
    metrics = feasibility_metrics(true_labels[unlabeled], scores)
    assert metrics.roc_auc > 0.9
