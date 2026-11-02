"""The one place stage S2 fits a learner (docs/design/modeling.md "Stage S2"; ADR 016 item 3).

Two sub-steps fit the same learner on different rows: S2a fits P against U to score the spies
(``score/pu.py``), S2b fits P against the reliable negatives. So the learner is called from here and
nowhere else, and the ``Learner`` record is the only thing that varies between the two -- there is no
second place where a default could differ.

**Every fit is weighted, and that is not an option.** Stage U writes 102,200 distinct pairs carrying
110,585 draws: ``n_draws`` is the row's sampling weight, 5,735 rows carry more than one and the
largest carries 58. Five places in this repository already treat the draw as the unit --
``positives/unlabeled.py``'s own docstring ("one row per distinct pair with ``n_draws`` as its
weight"), the stage U **gate**, which expands the table by ``n_draws`` before measuring, so its
certified rank-AUCs of 0.5049 and 0.5095 are properties of the draw distribution and not of the rows
(``scripts/unlabeled_literature.py``, which also reports ``out`` as 110,585),
``scripts/counts/count_u_probes.py``, ``scratch/count_spy_threshold.py``, which passes
``sample_weight = n_draws`` to every fit, and the P:U ratio quoted throughout as 1:9.7, which is
110,585/11,396 and not the 8.97 the rows give.

This module was written on 2026-10-30 without the weight, which meant the stage would have fitted a U
that the gate never certified, dropped 7.6 % of the drawn weight, counted a pair drawn 58 times once,
and failed to reproduce the very measurement it was built from (ADR 026). ``weights`` is therefore a
required argument with no default, and a positives row passes 1.0 -- an optional weight is a weight
somebody forgets.

Nothing here reads a config record: the parameters arrive as values, so this module cannot grow a
dependency on stage S2a's configuration and S2b can call it with its own.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class Learner:
    """The learner's own settings, which travel together because they always travel together.

    One record rather than two arguments, because ``fit_predict`` had reached §5's five-parameter
    limit and the sample weight had to go in. ``params`` is the config's ``learner_params``, which
    ``PuConfig`` has already refused unless it fixes ``max_iter`` and writes ``early_stopping`` out
    as false (ADR 016 item 3); ``seed`` is the config's, and it seeds the folds as well as the
    learner.
    """

    params: Mapping[str, Any]
    seed: int


def fit_predict(
    train_features: np.ndarray,
    train_target: np.ndarray,
    scored_features: np.ndarray,
    weights: np.ndarray,
    learner: Learner,
) -> np.ndarray:
    """One weighted fit of ``HistGradientBoostingClassifier``, and its positive-class probability.

    ``weights`` runs with ``train_features``, one per training row: ``n_draws`` for an unlabeled pair
    and 1.0 for a positive. The module docstring carries why it is required rather than optional.

    **float32 on the way out**, which is where the stage stores it: a threshold must be a number the
    stored scores can be compared against exactly, or a row could be flagged against a value that is
    not in the table (``tests/test_score_pu.py`` pins this).

    ``early_stopping`` and ``max_iter`` are not defaulted here. ``PuConfig`` refuses a config that
    leaves them out, so they always arrive in ``learner.params``; a default in this module would be a
    second opinion about the thing ADR 016 item 3 wrote down.
    """
    model = HistGradientBoostingClassifier(random_state=learner.seed, **learner.params)
    model.fit(train_features, train_target, sample_weight=weights)
    probability: np.ndarray = model.predict_proba(scored_features)[:, 1]
    return probability.astype(np.float32)


def cross_fitted_scores(
    features: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    n_folds: int,
    learner: Learner,
) -> tuple[np.ndarray, np.ndarray]:
    """``(fold, score)``: the fold each row sat out, and its score from the model that missed it.

    Out of fold is stage S2a's primary scoring (ADR 016 item 1). A spy sits in its own training set
    labelled unlabeled, so a flexible learner can partly memorise it; because every U row carries
    that same label, the direction of the contamination is neither known nor bounded. Scoring each
    row by a model that never saw it removes the question instead of answering it.

    The folds are stratified on ``target`` and shuffled from ``learner.seed``, which keeps the **P
    share** even across folds. It does **not** even out the spies, and an earlier version of this
    docstring claimed it did (pr-reviewer, 2026-10-30): a spy carries ``target`` 0, so it is
    stratified together with U and a fold can hold more spies than another. Whether to stratify on
    the three groups instead -- P, spy, U -- is a change of behaviour and therefore a measurement,
    not a docstring fix, and it is open. What the threshold rests on is the spies' scores pooled over
    all folds, which this does deliver. The weights are not stratified on either: a fold can hold
    more drawn weight than another, and the pooled scores are again what the threshold uses.
    """
    fold = np.empty(len(target), dtype=int)
    score = np.empty(len(target), dtype=np.float32)
    splitter = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=learner.seed
    )
    for index, (train, test) in enumerate(splitter.split(features, target)):
        fold[test] = index
        score[test] = fit_predict(
            features[train], target[train], features[test], weights[train], learner
        )
    return fold, score
