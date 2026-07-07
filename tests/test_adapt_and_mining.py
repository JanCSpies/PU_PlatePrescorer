"""Unit tests for domain adaptation and the C–N coupling miner (synthetic + tiny real SMILES)."""

from __future__ import annotations

import numpy as np

from bhpu.adapt import CoralAligner, DomainDiscriminator
from bhpu.chem import canonical_smiles
from bhpu.data.mining import BHCouplingDetector, CNCouplingMiner, RawReaction
from bhpu.eval import feasibility_metrics
from bhpu.models import CatBoostFeasibilityModel
from bhpu.pu import DomainAdaptedPULearner


def _bromobenzene_morpholine_reaction(product: str) -> RawReaction:
    """A raw reaction of bromobenzene + morpholine + Pd catalyst with a given product."""
    return RawReaction(
        molecules=(
            "Brc1ccccc1",  # aryl bromide
            "C1COCCN1",  # morpholine (N–H amine)
            "[Pd]",  # palladium
            "O=C([O-])[O-].[Cs+].[Cs+]",  # base
        ),
        products=(product,),
        source="unit-test",
    )


def test_detector_confirms_known_coupling_and_flags_pd() -> None:
    detector = BHCouplingDetector()
    true_product = canonical_smiles("c1ccc(N2CCOCC2)cc1")
    assert true_product is not None
    match = detector.detect(_bromobenzene_morpholine_reaction(true_product))
    assert match is not None
    assert canonical_smiles(match.aryl_smiles) == canonical_smiles("Brc1ccccc1")
    assert canonical_smiles(match.amine_smiles) == canonical_smiles("C1COCCN1")
    assert match.has_pd is True


def test_detector_rejects_when_product_does_not_match() -> None:
    detector = BHCouplingDetector()
    # Aryl halide + amine are present, but the recorded product is unrelated -> not a coupling.
    assert detector.detect(_bromobenzene_morpholine_reaction("c1ccccc1")) is None


def test_detector_strips_atom_maps_from_stored_smiles() -> None:
    detector = BHCouplingDetector()
    true_product = canonical_smiles("c1ccc(N2CCOCC2)cc1")
    assert true_product is not None
    # Atom-mapped reactants (as some ORD rows carry) must yield clean, unmapped stored SMILES.
    mapped = RawReaction(
        molecules=(
            "Br[c:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1",
            "[NH:7]1[CH2:8][CH2:9][O:10][CH2:11][CH2:12]1",
            "[Pd]",
        ),
        products=(true_product,),
        source="unit-test",
    )
    match = detector.detect(mapped)
    assert match is not None
    assert ":" not in match.aryl_smiles and ":" not in match.amine_smiles
    assert canonical_smiles(match.aryl_smiles) == canonical_smiles("Brc1ccccc1")


def test_detector_rejects_reaction_without_amine() -> None:
    detector = BHCouplingDetector()
    reaction = RawReaction(
        molecules=("Brc1ccccc1", "OB(O)c1ccccc1"),  # Suzuki-type partners, no N–H amine
        products=("c1ccc(-c2ccccc2)cc1",),
        source="unit-test",
    )
    assert detector.detect(reaction) is None


def test_miner_dedups_pairs_and_audits_leakage() -> None:
    product = canonical_smiles("c1ccc(N2CCOCC2)cc1")
    assert product is not None
    # JnJ reference: morpholine is a JnJ amine -> any pair reusing it must be dropped.
    from bhpu.chem.canonical import inchikey, strip_and_canonicalise

    morpholine_key = inchikey(strip_and_canonicalise("C1COCCN1") or "C1COCCN1")
    assert morpholine_key is not None
    miner = CNCouplingMiner(frozenset({morpholine_key}), frozenset())
    # Two identical detections (dedup to one pair) + it reuses a JnJ amine (leak -> dropped).
    miner.mine(iter([_bromobenzene_morpholine_reaction(product)] * 2))
    frame = miner.to_frame()
    assert miner.report.n_detected == 2
    assert miner.report.n_pairs_dropped_substrate_leak == 2
    assert len(frame) == 0  # both dropped for reusing the JnJ amine

    # Now with no leakage reference: the two identical detections collapse to a single pair.
    clean = CNCouplingMiner(frozenset(), frozenset())
    clean.mine(iter([_bromobenzene_morpholine_reaction(product)] * 2))
    clean_frame = clean.to_frame()
    assert len(clean_frame) == 1
    assert int(clean_frame.iloc[0]["n_records"]) == 2


def test_domain_discriminator_separates_and_weights() -> None:
    rng = np.random.default_rng(0)
    source = rng.normal(-2.0, 0.5, size=(200, 6)).astype(np.float32)
    target = rng.normal(2.0, 0.5, size=(200, 6)).astype(np.float32)

    def factory() -> CatBoostFeasibilityModel:
        return CatBoostFeasibilityModel(iterations=60, depth=3, random_state=0)

    disc = DomainDiscriminator(factory, n_splits=3, random_state=0).fit(source, target)
    assert disc.domain_auc_ > 0.95  # trivially separable -> strong "shortcut"
    weights = disc.importance_weights(source)
    assert np.all(weights >= 0)
    assert weights.mean() == np.float32(weights.mean())  # finite
    assert DomainDiscriminator.effective_sample_size(weights) <= len(weights)


def test_coral_aligns_source_covariance_to_target() -> None:
    rng = np.random.default_rng(1)
    source = rng.normal(0.0, 1.0, size=(500, 4))
    # Target has a different scale/covariance and mean.
    target = rng.normal(3.0, 1.0, size=(500, 4)) @ np.array(
        [[2.0, 0.5, 0, 0], [0.5, 1.5, 0, 0], [0, 0, 0.7, 0], [0, 0, 0, 1.2]]
    )
    aligned = CoralAligner(shrinkage=1e-3).fit_transform(source, target)
    # After alignment the covariance should be much closer to the target's than the source's was.
    before = np.linalg.norm(np.cov(source, rowvar=False) - np.cov(target, rowvar=False))
    after = np.linalg.norm(np.cov(aligned, rowvar=False) - np.cov(target, rowvar=False))
    assert after < 0.25 * before
    assert np.allclose(aligned.mean(axis=0), target.mean(axis=0), atol=0.3)


def _domain_shift_scenario() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthesise a literature→target domain shift with a separable feasibility signal.

    Four *class* dims carry feasibility (positive ``+1.5`` / negative ``−1.5``); four *domain* dims
    carry a spurious literature offset (``+4`` in literature, ``0`` in the target). A naive P-vs-U
    classifier keys on the domain dims (the shortcut) and cannot rank the target; domain adaptation
    should remove the offset and recover the class signal.

    Returns:
        ``(features, selection, target_x, target_y, source_marginal)``.
    """
    rng = np.random.default_rng(2)
    n = 150

    def block(label: float, domain_offset: float, count: int) -> np.ndarray:
        cls = rng.normal(1.5 if label else -1.5, 0.5, size=(count, 4))
        dom = rng.normal(domain_offset, 0.5, size=(count, 4))
        return np.hstack([cls, dom]).astype(np.float32)

    positives = block(1.0, 4.0, n)  # observed literature positives (domain-shifted)
    source_marginal = np.vstack([block(1.0, 4.0, n), block(0.0, 4.0, n)])  # all literature (P+N)
    target_pos, target_neg = block(1.0, 0.0, n), block(0.0, 0.0, n)  # target domain (no offset)
    unlabeled = np.vstack([target_pos, target_neg])
    target_y = np.concatenate([np.ones(n), np.zeros(n)])

    features = np.vstack([positives, unlabeled])
    selection = np.concatenate([np.ones(n), np.zeros(len(unlabeled))])
    return features, selection, unlabeled, target_y, source_marginal


def _factory() -> CatBoostFeasibilityModel:
    return CatBoostFeasibilityModel(iterations=80, depth=3, random_state=0)


def test_domain_adaptation_beats_naive_under_shift() -> None:
    features, selection, target_x, target_y, source_marginal = _domain_shift_scenario()

    naive = DomainAdaptedPULearner(
        _factory, use_coral=False, use_importance_weights=False, n_estimators=12, random_state=0
    ).fit(features, selection)
    adapted = DomainAdaptedPULearner(
        _factory, use_coral=True, use_importance_weights=True, n_estimators=12, random_state=0
    ).fit(features, selection, source_marginal=source_marginal)

    naive_auc = feasibility_metrics(target_y, naive.predict_proba(target_x)).roc_auc
    adapted_auc = feasibility_metrics(target_y, adapted.predict_proba(target_x)).roc_auc

    # The domain shortcut sinks the naive learner; adaptation recovers the feasibility signal.
    assert adapted_auc > 0.85
    assert adapted_auc > naive_auc
    assert 0.0 <= float(np.min(adapted.predict_proba(target_x))) <= 1.0
