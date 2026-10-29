"""Tests for platepu.score.axes, written before the module on 2026-10-28.

The module makes the provenance axis of ADR 018 reproducible: can a feature block tell a
literature-only substrate (source 0) from a lab-only one (source 1)? 0.5 means no. The behaviour is
that of scratch/count_block_two_axes.py (seed 20261022 there); ADR 012 quotes repeats, size
adjustment and a bootstrap that scripts/counts/count_adversarial.py does not have.

Every fixture is made up, small and seeded. Exact numbers are worked out by hand in the test that
uses them. Size levels 1, 2, 3, 4 with 50 rows each give quartile edges 1.75, 2.5, 3.25, so level k
lands in quartile k - 1 and the quartile of every row is known without running anything.

Names this file fixes for the module (the tests fail until it has them):
    ProvenanceSettings(seed, repeats, resamples)           frozen
    ProvenanceResult(auc, adjusted_auc, with_size_auc, ci_low, ci_high)   frozen
    provenance_auc(features, source, size, settings) -> ProvenanceResult
Decision pinned in test_bootstrap_interval_covers_all_repeats: repeat r uses seed ``settings.seed + r``;
``auc`` is the mean over repeats; the interval bootstraps all repeats, not repeat 0 only.
``provenance_auc`` must call ``out_of_fold`` through the module namespace (the test replaces it).
"""

import dataclasses

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from platepu.score import axes
from platepu.score.axes import (
    ProvenanceSettings,
    out_of_fold,
    provenance_auc,
    size_adjusted_auc,
    with_size,
)

FAST = ProvenanceSettings(seed=7, repeats=2, resamples=50)


def _four_quartiles() -> np.ndarray:
    """Size column of four 50-row quartiles: levels 1, 2, 3, 4."""
    return np.repeat([1.0, 2.0, 3.0, 4.0], 50)


def _source_by_quartile(lab_counts: list[int], lit_counts: list[int]) -> np.ndarray:
    """Source column (1 = lab) with the given class counts in each quartile."""
    parts = [
        np.array([1] * lab + [0] * lit)
        for lab, lit in zip(lab_counts, lit_counts, strict=True)
    ]
    return np.concatenate(parts)


def _separating_scores(
    source: np.ndarray, quartiles_reversed: tuple[int, ...]
) -> np.ndarray:
    """Score = source (AUC 1) except in the listed quartiles, where it is 1 - source (AUC 0)."""
    scores = source.astype(float)
    for quartile in quartiles_reversed:
        inside = slice(50 * quartile, 50 * quartile + 50)
        scores[inside] = 1.0 - source[inside]
    return scores


# ------------------------------------------------------------------ size-adjusted AUC
def _size_confounded_fixture() -> tuple[np.ndarray, np.ndarray]:
    """1000 rows: size level 10/20/30/40 (+-0.5 jitter) and lab share 0.2/0.4/0.6/0.8 per level.

    The source depends on size only through the level, so inside one quartile size carries nothing.
    """
    generator = np.random.default_rng(1)
    level = np.repeat([10.0, 20.0, 30.0, 40.0], 250)
    size = level + generator.uniform(-0.5, 0.5, level.size)
    share = np.repeat([0.2, 0.4, 0.6, 0.8], 250)
    source = (generator.uniform(size=level.size) < share).astype(int)
    return size, source


def test_size_adjusted_auc_of_size_itself_is_about_half():
    """ADR 012's sanity check: scoring rows by their size must leave nothing inside a quartile."""
    size, source = _size_confounded_fixture()
    assert size_adjusted_auc(size, source, size) == pytest.approx(0.5, abs=0.06)


def test_size_adjusted_auc_differs_from_the_plain_auc():
    """The control for the test above: unadjusted, size alone separates the sources well."""
    size, source = _size_confounded_fixture()
    plain = roc_auc_score(source, size)
    assert plain > 0.7
    assert abs(plain - size_adjusted_auc(size, source, size)) > 0.2


@pytest.mark.parametrize(
    ("lab_in_last", "lit_in_last"),
    [(4, 46), (46, 4)],
    ids=["four_lab_rows", "four_literature_rows"],
)
def test_a_quartile_with_too_few_of_either_class_is_skipped(lab_in_last, lit_in_last):
    """Quartiles 0-2 hold 10 lab and 40 literature rows with AUC 1; quartile 3 has AUC 0.

    With four rows of one class, quartile 3 is dropped and only the AUC-1 quartiles remain.
    """
    source = _source_by_quartile([10, 10, 10, lab_in_last], [40, 40, 40, lit_in_last])
    size = _four_quartiles()
    scores = _separating_scores(source, quartiles_reversed=(3,))
    assert size_adjusted_auc(scores, source, size) == pytest.approx(1.0)


def test_a_quartile_with_exactly_five_of_each_class_counts():
    """Same fixture with five lab rows in quartile 3: weights are 400, 400, 400 and 5 * 45 = 225.

    (3 * 400 * 1 + 225 * 0) / (3 * 400 + 225) = 1200 / 1425.
    """
    source = _source_by_quartile([10, 10, 10, 5], [40, 40, 40, 45])
    size = _four_quartiles()
    scores = _separating_scores(source, quartiles_reversed=(3,))
    assert size_adjusted_auc(scores, source, size) == pytest.approx(1200 / 1425)


def test_eight_lab_acids_among_two_hundred_literature_acids_have_no_adjusted_auc():
    """The shape of the real problem in miniature (189 lab-only against 11,203 literature-only).

    Sizes 0..207 with a lab row at every 26th position: each size quartile holds about 52 rows and
    two lab ones, never the five that MIN_PER_CLASS asks for. All four quartiles are skipped, so
    the function must say so instead of dividing zero by zero.
    """
    size = np.arange(208, dtype=float)
    source = np.zeros(208, dtype=int)
    source[::26] = 1
    assert source.sum() == 8
    with pytest.raises(
        ValueError, match="no size quartile holds 5 of both classes"
    ) as raised:
        size_adjusted_auc(size, source, size)
    assert "8 lab and 200 literature rows" in str(raised.value)


def test_quartiles_are_weighted_by_the_pairs_they_hold():
    """Quartile 0 (10 lab, 40 lit, AUC 1, 400 pairs); quartile 1 (25, 25, AUC 0, 625 pairs);
    quartiles 2 and 3 hold four lab rows and are skipped.

    n_lab * n_literature weighting gives 400 / 1025. Equal weighting would give 0.5, weighting by
    n_lab alone 10 / 35, by rows held 0.5.
    """
    source = _source_by_quartile([10, 25, 4, 4], [40, 25, 46, 46])
    size = _four_quartiles()
    scores = _separating_scores(source, quartiles_reversed=(1,))
    assert size_adjusted_auc(scores, source, size) == pytest.approx(400 / 1025)


# ------------------------------------------------------------------ out-of-fold scores
def _informative_features(n_rows: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Five columns, the first shifted by the source, so a model finds something."""
    generator = np.random.default_rng(3)
    source = np.tile([0, 1], n_rows // 2)
    features = generator.normal(size=(n_rows, 5))
    features[:, 0] += 1.5 * source
    return features, source


def test_out_of_fold_scores_every_row_exactly_once():
    """One finite probability per row. A row left out of every fold would stay unscored (NaN or 0).

    Not proven here: that no row is scored twice; the next test covers who scored it.
    """
    features, source = _informative_features()
    scores = out_of_fold(features, source, seed=1)
    assert scores.shape == (len(source),)
    assert np.isfinite(scores).all()
    assert ((scores > 0) & (scores < 1)).all()


def test_out_of_fold_never_scores_a_row_with_a_model_that_saw_it():
    """A memorisation trap. Each row has its own one-hot column (120 columns, so logistic
    regression) and the labels are random: a model that has seen a row recognises its column and
    would score it perfectly (AUC about 1). Out of fold the row's column was never trained, so all
    scores in a fold are ties and the AUC must be about 0.5.

    Proves: no row is scored by a model fitted on it, for this estimator. Does not prove: that the
    folds partition the rows, or that nothing leaks through preprocessing outside the folds.
    """
    n_rows = 120
    source = np.tile([0, 1], n_rows // 2)
    np.random.default_rng(5).shuffle(source)
    one_hot = np.eye(n_rows)
    scores = out_of_fold(one_hot, source, seed=1)
    assert roc_auc_score(source, scores) == pytest.approx(0.5, abs=0.15)


def test_out_of_fold_is_reproducible_from_the_seed():
    features, source = _informative_features()
    first = out_of_fold(features, source, seed=1)
    assert np.array_equal(first, out_of_fold(features, source, seed=1))


def test_out_of_fold_with_a_second_seed_gives_something_different():
    features, source = _informative_features()
    assert not np.allclose(
        out_of_fold(features, source, seed=1), out_of_fold(features, source, seed=2)
    )


# ------------------------------------------------------------------ with_size
def test_with_size_appends_one_column_and_changes_nothing_else():
    """The last column is size / 30. The 30 is a crude scaling that puts the column on the order of
    a fingerprint bit (a molecule has about 30 heavy atoms); it is not chemistry and not fitted.
    """
    features = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 5.0], [1.0, 1.0, 0.0]])
    before = features.copy()
    size = np.array([15.0, 30.0, 45.0])
    result = with_size(features, size)
    assert result.shape == (3, 4)
    assert np.array_equal(result[:, :3], features)
    assert np.allclose(result[:, 3], [0.5, 1.0, 1.5])
    assert np.array_equal(features, before)


# ------------------------------------------------------------------ provenance_auc
def test_settings_and_result_are_frozen():
    result = provenance_auc(*_with_size_column(*_informative_features()), FAST)
    with pytest.raises(dataclasses.FrozenInstanceError):
        FAST.seed = 1  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.auc = 1.0  # type: ignore[misc]


def _with_size_column(features: np.ndarray, source: np.ndarray):
    """(features, source, size) with a size column unrelated to the source."""
    size = np.random.default_rng(9).integers(10, 40, len(source)).astype(float)
    return features, source, size


def test_bootstrap_interval_contains_the_point_estimate_and_is_reproducible():
    args = _with_size_column(*_informative_features())
    result = provenance_auc(*args, FAST)
    assert result.ci_low < result.auc < result.ci_high
    assert result == provenance_auc(*args, FAST)


def test_provenance_auc_separates_an_informative_block():
    """The first column is shifted by 1.5 standard deviations: the plain AUC is clearly above 0.5."""
    result = provenance_auc(*_with_size_column(*_informative_features()), FAST)
    assert result.auc > 0.75


def test_bootstrap_interval_covers_all_repeats(monkeypatch):
    """The point estimate is the mean over repeats, so the interval must bootstrap all of them.

    Out-of-fold scores are replaced: repeat 0 separates perfectly (AUC 1), repeats 1 and 2 reverse
    the classes (AUC 0). The point estimate is 1/3. An interval from repeat 0 alone would be [1, 1]
    and exclude it; resampling rows and averaging the three repeats on the same rows gives 1/3 in
    every resample.
    """
    settings = ProvenanceSettings(seed=100, repeats=3, resamples=40)

    def fake_out_of_fold(features, source, seed):
        return source.astype(float) if seed == settings.seed else 1.0 - source

    monkeypatch.setattr(axes, "out_of_fold", fake_out_of_fold)
    features, source, size = _with_size_column(*_informative_features())
    result = provenance_auc(features, source, size, settings)
    assert result.auc == pytest.approx(1 / 3)
    assert result.ci_low == pytest.approx(1 / 3)
    assert result.ci_high == pytest.approx(1 / 3)


def test_provenance_of_two_identical_classes_is_about_half():
    """Literature and lab rows drawn from one distribution: every figure sits near 0.5.

    1500 rows, not 400: out-of-fold AUC of pure noise is biased low (0.40 at 400 rows with one seed)
    and only settles near 0.5 with more rows.
    """
    generator = np.random.default_rng(11)
    features = generator.normal(size=(1500, 4))
    source = generator.integers(0, 2, 1500)
    size = generator.integers(10, 40, 1500).astype(float)
    result = provenance_auc(features, source, size, FAST)
    for figure in (
        result.auc,
        result.adjusted_auc,
        result.with_size_auc,
        result.ci_low,
        result.ci_high,
    ):
        assert figure == pytest.approx(0.5, abs=0.1)
