"""The provenance axis on which a feature block is judged (ADR 018 item 2).

Provenance asks: can a classifier tell a substrate that only the literature reports from one that
only the lab stocks, using this block alone? Low is good -- a block that answers it well is partly
describing where a molecule came from rather than how it reacts.

Three things make the number mean what it says, and ADR 012 quotes all three although the published
``scripts/counts/count_adversarial.py`` computes none of them. They are ported here; wiring that
script onto this module is the outstanding work ADR 018's consequences record, so until it is done
the published provenance figure is still one unrepeated, unadjusted AUC with no interval:

- **Repeats.** One 5-fold split is one draw; the figure is the mean over ``repeats`` of them, each
  with its own seed.
- **A size adjustment.** Size alone separates the two sources at about 0.67, because literature
  molecules really are bigger, so 0.5 is not the floor. The adjusted figure scores inside size
  quartiles, where size barely varies, and a block that only inherited size reads about 0.5 there.
- **A bootstrap interval.** The adversarial problem is 11,203 literature-only acids against 189
  lab-only ones, so the estimate is fragile and a gap under about 0.04 is not a gap.

The interval brackets the **mean over repeats**, which is the figure reported beside it: each
resample draws its rows once and averages every repeat's AUC on those rows, so it covers row
sampling and split-to-split noise together. The scratch version this was ported from bootstrapped
repeat 0 only, which brackets a single split's AUC instead -- a different quantity from the one it
was printed next to.

``provenance_auc`` reaches ``out_of_fold`` through the module namespace, so a test can replace it.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Above this many columns the boosting model is slow and no better, so the linear one is used.
WIDE = 100
FOLDS = 5
# Fewest of either class a size quartile must hold before its AUC is worth computing.
MIN_PER_CLASS = 5
# A crude scaling that puts the size column on the order of a fingerprint bit. It is arithmetic,
# not chemistry: nothing about amide coupling happens at 30 heavy atoms.
SIZE_SCALE = 30.0


@dataclass(frozen=True)
class ProvenanceSettings:
    """The knobs of one provenance measurement, together so the call stays inside five arguments.

    Attributes:
        seed: Repeat ``r`` uses ``seed + r``, and the bootstrap draws from ``seed``.
        repeats: How many 5-fold splits are averaged.
        resamples: Bootstrap resamples behind the interval.
    """

    seed: int
    repeats: int
    resamples: int


@dataclass(frozen=True)
class ProvenanceResult:
    """What one block scores on the provenance axis for one role. Lower is better throughout."""

    auc: float
    adjusted_auc: float
    with_size_auc: float
    ci_low: float
    ci_high: float


def make_model(n_columns: int) -> LogisticRegression | HistGradientBoostingClassifier:
    """The estimator for a block of this width, chosen by column count and nothing else."""
    if n_columns > WIDE:
        return LogisticRegression(max_iter=2000, C=0.1)
    return HistGradientBoostingClassifier(random_state=0, max_iter=200)


def out_of_fold(features: np.ndarray, source: np.ndarray, seed: int) -> np.ndarray:
    """Out-of-fold probability of "lab-only" for every row, from one 5-fold split."""
    folds = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
    model = make_model(features.shape[1])
    if hasattr(model, "random_state"):
        model.set_params(random_state=seed)
    return cross_val_predict(
        model, features, source, cv=folds, method="predict_proba", n_jobs=FOLDS
    )[:, 1]


def size_adjusted_auc(
    scores: np.ndarray, source: np.ndarray, size: np.ndarray
) -> float:
    """ROC-AUC inside size quartiles, averaged with weight ``n_lab * n_literature``.

    Inside a quartile size barely varies, so whatever a block inherits from size cannot separate the
    two sources there. Scoring the rows **by** their size is the sanity check: it reads about 0.5
    here and far from it unadjusted.

    A quartile holding fewer than ``MIN_PER_CLASS`` of either class is skipped rather than scored,
    because an AUC on four rows is noise with a number attached.

    Raises:
        ValueError: every quartile was skipped, so there is no adjusted figure to return. The
            adversarial problem is about 189 lab-only acids against 11,203 literature-only ones,
            and lab molecules are the small ones, so this is reachable input rather than a
            theoretical case -- it is named here instead of surfacing as a ZeroDivisionError.
    """
    edges = np.quantile(size, [0.25, 0.5, 0.75])
    quartile = np.searchsorted(edges, size, side="right")
    total, weight_sum = 0.0, 0.0
    for index in range(4):
        inside = quartile == index
        n_lab = int(source[inside].sum())
        n_literature = int((1 - source[inside]).sum())
        if n_lab < MIN_PER_CLASS or n_literature < MIN_PER_CLASS:
            continue
        weight = float(n_lab * n_literature)
        total += weight * roc_auc_score(source[inside], scores[inside])
        weight_sum += weight
    if weight_sum == 0.0:
        raise ValueError(
            f"no size quartile holds {MIN_PER_CLASS} of both classes, so there is no adjusted "
            f"AUC: {int(source.sum())} lab and {int(len(source) - source.sum())} literature rows"
        )
    return total / weight_sum


def with_size(features: np.ndarray, size: np.ndarray) -> np.ndarray:
    """``features`` with the scaled size as one more column.

    It always appends, and it cannot check: the block's name does not reach this far. So
    ``with_size_auc`` is meaningless for a block that already carries size -- ``"descriptors +
    size"`` and ``"size alone"`` of ``features.blocks`` are exactly those two -- because it would
    score size twice. Choosing a block is the caller's job; this function does arithmetic.
    """
    return np.hstack([features, (size / SIZE_SCALE)[:, None]])


def provenance_auc(
    features: np.ndarray,
    source: np.ndarray,
    size: np.ndarray,
    settings: ProvenanceSettings,
) -> ProvenanceResult:
    """How much this block says about where a substrate came from.

    Args:
        features: One row per substrate, as ``features.blocks.block_matrix`` returns it.
        source: 1 for a lab-only substrate, 0 for a literature-only one.
        size: Heavy-atom count per row, for the adjustment and the extra column.
        settings: Seed, repeats and bootstrap resamples.

    Returns:
        The mean AUC over repeats, its size-adjusted counterpart, the AUC with size appended, and a
        95 % bootstrap interval around the first of those.
    """
    plain, adjusted, plus, every_repeat = [], [], [], []
    for repeat in range(settings.repeats):
        seed = settings.seed + repeat
        scores = out_of_fold(features, source, seed)
        every_repeat.append(scores)
        plain.append(roc_auc_score(source, scores))
        adjusted.append(size_adjusted_auc(scores, source, size))
        plus.append(
            roc_auc_score(source, out_of_fold(with_size(features, size), source, seed))
        )
    low, high = _interval(every_repeat, source, settings)
    return ProvenanceResult(
        auc=float(np.mean(plain)),
        adjusted_auc=float(np.mean(adjusted)),
        with_size_auc=float(np.mean(plus)),
        ci_low=low,
        ci_high=high,
    )


def _interval(
    every_repeat: list[np.ndarray], source: np.ndarray, settings: ProvenanceSettings
) -> tuple[float, float]:
    """The 95 % bootstrap interval of the **mean over repeats**, the figure it sits beside.

    Each resample draws its rows once and averages every repeat's AUC on those same rows, so the
    interval covers row sampling and split-to-split noise together. A resample holding one class
    only has no AUC and is drawn again.
    """
    generator = np.random.default_rng(settings.seed)
    means: list[float] = []
    while len(means) < settings.resamples:
        pick = generator.integers(0, len(source), len(source))
        if not 0 < source[pick].sum() < len(source):
            continue
        means.append(
            float(np.mean([roc_auc_score(source[pick], s[pick]) for s in every_repeat]))
        )
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)
