"""Unit tests for the amide-coupling miner (synthetic + tiny real SMILES)."""

from __future__ import annotations

from bhpu.chem.canonical import inchikey
from bhpu.data.mining import RawReaction
from pubench.mining import (
    AmideCouplingDetector,
    AmideMiner,
    MinedAmide,
    enumerate_amide_products,
)

_BENZOIC_ACID = "OC(=O)c1ccccc1"
_ANILINE = "Nc1ccccc1"
#: Benzanilide -- the amide of benzoic acid + aniline.
_BENZANILIDE = "O=C(Nc1ccccc1)c1ccccc1"


def _reaction(molecules: tuple[str, ...], product: str) -> RawReaction:
    """A raw reaction with the given candidate molecules and a single recorded product."""
    return RawReaction(molecules=molecules, products=(product,), source="unit-test")


def test_template_builds_expected_amide() -> None:
    """The condensation template reproduces the known benzanilide product."""
    products = enumerate_amide_products(_BENZOIC_ACID, _ANILINE)
    assert inchikey(_BENZANILIDE) in {inchikey(p) for p in products}


def test_detects_confirmed_amide() -> None:
    """A reaction whose recorded product matches the enumerated amide is detected."""
    detector = AmideCouplingDetector()
    match = detector.detect(_reaction((_BENZOIC_ACID, _ANILINE, "CN(C)C=O"), _BENZANILIDE))
    assert isinstance(match, MinedAmide)
    assert inchikey(match.acid_smiles) == inchikey(_BENZOIC_ACID)
    assert inchikey(match.amine_smiles) == inchikey(_ANILINE)


def test_rejects_unconfirmed_product() -> None:
    """Acid + amine present but the recorded product is unrelated -> not an amide coupling."""
    detector = AmideCouplingDetector()
    assert detector.detect(_reaction((_BENZOIC_ACID, _ANILINE), "CCO")) is None


def test_rejects_tertiary_amine() -> None:
    """A tertiary amine has no N-H, so no amine nucleophile is found."""
    detector = AmideCouplingDetector()
    assert detector.detect(_reaction((_BENZOIC_ACID, "CN(C)C"), _BENZANILIDE)) is None


def test_miner_dedups_repeated_pair() -> None:
    """Two records of the same substrate pair collapse to one row with n_records == 2."""
    miner = AmideMiner(frozenset())
    for _ in range(2):
        miner.add_match(
            MinedAmide(_BENZOIC_ACID, _ANILINE, _BENZANILIDE, "s", None, 50.0)
        )
    frame = miner.to_frame()
    assert len(frame) == 1
    assert int(frame.iloc[0]["n_records"]) == 2


def test_miner_drops_exact_test_pair() -> None:
    """A mined pair equal to a Test pair is dropped and counted as exact leakage."""
    test_pair = (inchikey(_BENZOIC_ACID), inchikey(_ANILINE))
    miner = AmideMiner(frozenset({test_pair}))
    miner.add_match(MinedAmide(_BENZOIC_ACID, _ANILINE, _BENZANILIDE, "s", None, None))
    frame = miner.to_frame()
    assert frame.empty
    assert miner.report.n_exact_test_pair_hits == 1
    assert miner.report.n_pairs_dropped_test_leak == 1
