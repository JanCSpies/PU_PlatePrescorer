"""Stage S2a, the spy step (docs/design/modeling.md "Stage S2"; ADR 013 item 3, 016, 024, 026).

Hide a fraction of the known positives inside U, fit a classifier on what is left of P against U plus
those hidden rows (the **spies**), put the threshold where about 95 % of the spies score above it,
and call every U row below it a **reliable negative**. ``score/pu_audit.py`` measures what the
threshold selected; stage S2b fits the final model on P plus these rows.

**This is a heuristic with a measured weakness, and the code must not read as more (ADR 026).** On
the real amide data the reliable-negative set is *depleted* of the 1,121 literature negatives we can
check it against: at tolerance 0.05 it calls 14.3 % of U reliably negative while catching 3.9 % of
the known failures, a lift of 0.27 -- three to four times worse than drawing at random
(``scratch/count_spy_threshold.py``, 2026-10-30). Two consequences for anyone reading this module:

- The **realised spy recall** below is recall on *reported* positives. The positives the method must
  find are the unreported ones (ADR 014 item 1, ADR 019 item 3), so a realised recall of 0.95 says
  the threshold did what it was told and says nothing about whether it cut the right part of the
  space. Bekker et al. 2019 (p. 11, via ADR 026) describes exactly that failure: a SCAR rule can
  take the wrong subspace while looking calibrated.
- The naive control of ADR 016 item 2 -- the same classifier with no spies and no threshold -- is
  therefore a serious candidate and not a courtesy, and stage S2b runs it beside this.

**No label and no metric.** Stage S2 scores pairs; every comparison, stratum and interval is stage E
(ADR 008 item 6, a pre-registration). Nothing here reads a benchmark label, and no function is even
handed one. Reading and writing files is the script's job (CLAUDE.md §5), so ``score`` imports only
``features.blocks`` and its own modules.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from platepu.io import strictest_licence
from platepu.score.pairs import pair_matrix
from platepu.score.pu_audit import score_deciles, selection_shift
from platepu.score.pu_config import SCORINGS, PuConfig, config_sha256
from platepu.score.pu_fit import Learner, cross_fitted_scores, fit_predict

# The four keys that identify a pair, in the order stages P and U write them.
PAIR_KEYS = ("key_a_stereo", "key_b_stereo", "key_a_flat", "key_b_flat")
# How far from a whole number the product `spy_recall * n_spies` may sit and still be read as that
# number. A configured recall has two or three decimals, so a product within 1e-9 of an integer is
# the integer meant; without this a `ceil` can turn 19.000000000000004 into 20 and recall the lot.
TIE_SLACK = 1e-9


def spy_threshold(spy_scores: np.ndarray, spy_recall: float) -> float:
    """The score above which about ``spy_recall`` of the spies sit, by the written tie rule.

    ``k = ceil(spy_recall * n)`` spies must be kept, so the threshold is the **k-th largest** spy
    score; a spy counts as recalled when its score is ``>=`` the threshold and a U row is a reliable
    negative only when its score is strictly ``<`` it. Two properties follow, and both are
    deliberate: the realised recall is never **below** the configured one, and a tie never moves the
    threshold -- every spy level with the k-th is recalled too, so the realised recall can come out
    above 0.95 and the report carries both numbers. Fingerprints collide and ties are common here,
    and splitting one would mean choosing between identical rows at random.

    Raises:
        ValueError: ``spy_recall`` is outside (0, 1); or there are no spies, in which case guessing
            a threshold would put "reliable" on arbitrary rows.
    """
    if not 0.0 < spy_recall < 1.0:
        raise ValueError(
            f"spy_recall must be in the open interval (0, 1), not {spy_recall}"
        )
    count = spy_scores.size
    if count == 0:
        raise ValueError(
            "no spies: a threshold cannot be cut from an empty score array"
        )
    recalled = min(count, max(1, math.ceil(spy_recall * count - TIE_SLACK)))
    return float(np.sort(spy_scores)[count - recalled])


def spy_labels(
    positives: pd.DataFrame, unlabeled: pd.DataFrame, config: PuConfig
) -> pd.DataFrame:
    """One row per positive and per unlabeled pair, with the label the learner is actually given.

    The whole method rests on the spy looking unlabeled: a spy is a known positive (``in_p`` true)
    carrying ``label`` 0, exactly like a U row, so the fit cannot tell it from one. The remaining
    positives carry 1 and every U row carries 0 and ``in_p`` false.

    Every row also carries the **weight** the fit must give it: ``n_draws`` for an unlabeled pair,
    1.0 for a positive, spies included -- a spy is a reported positive and not a draw.
    ``score/pu_fit.py`` carries the evidence that the draw is the unit and what ignoring it cost.

    ``spy_fraction`` of P is rounded **down**, so the configured fraction is never exceeded, and the
    draw is ``config.seed``'s. ``std_version`` travels with the row from the table it came from; the
    licence does not, because it is a property of the fitted model's whole training set and
    ``reliable_negatives`` takes it from ``io.strictest_licence``.

    Raises:
        ValueError: the fraction rounds down to no spies at all -- checked here, before the six fits,
            because the alternative is finding out from ``spy_threshold`` about seven minutes into a
            real run (``scratch/profile_pair_fit.py``); or, from ``_draw_weights``, an unlabeled
            table without ``n_draws`` or with a non-positive one.
    """
    weights = _draw_weights(unlabeled)
    columns = [*PAIR_KEYS, "std_version"]
    n_spies = int(np.floor(config.spy_fraction * len(positives)))
    if n_spies < 1:
        raise ValueError(
            f"spy_fraction {config.spy_fraction} of {len(positives)} positives rounds down to no "
            f"spies; the threshold is cut from their scores and cannot be cut from none"
        )
    label = np.ones(len(positives), dtype=int)
    label[
        np.random.default_rng(config.seed).choice(
            len(positives), size=n_spies, replace=False
        )
    ] = 0
    return pd.concat(
        [
            positives[columns].assign(in_p=True, weight=1.0, label=label),
            unlabeled[columns].assign(in_p=False, weight=weights, label=0),
        ],
        ignore_index=True,
    )


def _draw_weights(unlabeled: pd.DataFrame) -> np.ndarray:
    """``n_draws`` as a float weight, refusing a table that cannot supply one.

    Required, not defaulted: silently taking a missing or zero ``n_draws`` for 1.0 is exactly the
    defect of 2026-10-30, where the stage would have fitted the unweighted U that stage U's gate
    never certified (``score/pu_fit.py`` has the five pieces of evidence).

    Raises:
        ValueError: ``n_draws`` is absent, or a row carries zero or less. A zero-weight row is a row
            that sits in the table and not in the fit, which nothing in the design describes.
    """
    if "n_draws" not in unlabeled.columns:
        raise ValueError(
            "the unlabeled table lacks n_draws, the draw weight every fit needs; stage U writes "
            "one row per distinct pair and n_draws is its multiplicity"
        )
    weights = unlabeled["n_draws"].to_numpy(dtype=np.float64)
    if weights.size and weights.min() <= 0:
        raise ValueError(
            f"n_draws must be positive: {int((weights <= 0).sum())} row(s) are not, "
            f"the smallest being {weights.min():g}"
        )
    return weights


@dataclass(frozen=True)
class _Scored:
    """One spy fit read both ways: the fold each row sat out, the two scorings, their thresholds.

    Both scorings are computed in one call because ADR 016's consequence asks the report for both
    thresholds and the size of both reliable-negative sets. The configured one goes in the table;
    the other is there so the cost of PAYN's in-sample choice is a measurement.
    """

    fold: np.ndarray
    spy: np.ndarray
    unlabeled_row: np.ndarray
    scores: Mapping[str, np.ndarray]
    thresholds: Mapping[str, float]


def reliable_negatives(
    positives: pd.DataFrame,
    unlabeled: pd.DataFrame,
    substrates: pd.DataFrame,
    config: PuConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """``(negatives, report)``: one row per U pair and per spy, with its score and its flag.

    Args:
        positives: The positives of ``config.p_width``, with the four pair keys, ``std_version`` and
            ``licence``. The spies are drawn from here.
        unlabeled: The stage U table named by ``config.unlabeled``, same columns.
        substrates: The stage F table; only ``config.feature_block`` and, for the report,
            ``n_heavy_atoms`` are read from it.
        config: Hashed into every row. No label is passed, and none is read (ADR 008 item 6).

    Returns:
        The table of ``docs/design/modeling.md`` "S2a output", and the audit report that goes beside
        it. ``reliable_negative`` is never true on a spy: a spy is a reported positive.

    Raises:
        ValueError: from ``pair_matrix``, a pair key missing from the feature table or a
            ``std_version`` that disagrees across the three inputs; from ``spy_threshold``, no spies.
    """
    labels = spy_labels(positives, unlabeled, config)
    features = pair_matrix(labels, substrates, config.feature_block)
    scored = _score_rows(features, labels, config)
    licence = strictest_licence([*positives["licence"], *unlabeled["licence"]])
    table = _negatives_table(labels, scored, licence, config)
    return table, _audit_report(table, scored, unlabeled, substrates, config)


def _score_rows(
    features: np.ndarray, labels: pd.DataFrame, config: PuConfig
) -> _Scored:
    """Score every row both ways and cut a threshold from each, which is one call's work.

    Both scorings come out of one call because ADR 016's consequence asks the report for both
    thresholds and the size of both reliable-negative sets; ``score/pu_fit.py`` carries the reason
    out of fold is the primary one. Six fits at the real sizes, about seven minutes
    (``scratch/profile_pair_fit.py``, 2026-10-30, 67 s a fit).
    """
    target = labels["label"].to_numpy()
    weights = labels["weight"].to_numpy()
    learner = Learner(params=config.learner_params, seed=config.seed)
    fold, out_of_fold = cross_fitted_scores(
        features, target, weights, config.n_folds, learner
    )
    scores = {
        "out_of_fold": out_of_fold,
        "in_sample": fit_predict(features, target, features, weights, learner),
    }
    spy = labels["in_p"].to_numpy() & (target == 0)
    return _Scored(
        fold=fold,
        spy=spy,
        unlabeled_row=~labels["in_p"].to_numpy(),
        scores=scores,
        thresholds={
            name: spy_threshold(value[spy], config.spy_recall)
            for name, value in scores.items()
        },
    )


def _negatives_table(
    labels: pd.DataFrame, scored: _Scored, licence: str, config: PuConfig
) -> pd.DataFrame:
    """The S2a layout: the rows the learner saw as unlabeled, which is U plus the spies.

    The positives that stayed positive are not here. They were never candidates to be called
    negative, and stage S2b reads them from the positives table.

    One ``licence`` for the whole table, the strictest of P's and U's, as
    ``scripts/score_baselines.py`` already does for stage S1 ("a scored set's licence is the
    strictest its own pairs carry"). It carried each row's own licence until 2026-10-30, which was
    wrong in the only way that matters here: every score comes from a model fitted on **all** of P
    and U together, so an open U row's score is derived from every licensed row in P as well, and
    a per-row stamp would let ``io.write_table`` put it outside ``private/``. Today it changes
    nothing -- P and U are 100 %% ``open`` (counted 2026-10-30) -- and that is exactly why it is
    worth fixing before Pistachio enters P. The remaining gap, which this does **not** close and
    stage S1 shares: the feature table's own licences, where 535 ``internal`` and 16
    ``noncommercial`` rows are read by this fit.
    """
    kept = (labels["label"] == 0).to_numpy()
    rows = labels.loc[kept]
    in_p = rows["in_p"].to_numpy()
    score = scored.scores[config.spy_scoring][kept]
    threshold = scored.thresholds[config.spy_scoring]
    return pd.DataFrame(
        {
            **{key: rows[key].to_numpy() for key in PAIR_KEYS},
            "in_p": in_p,
            "weight": rows["weight"].to_numpy(),
            "fold": scored.fold[kept],
            "spy_score": score,
            "reliable_negative": ~in_p & (score < threshold),
            "threshold": threshold,
            "model": config.model,
            "feature_block": config.feature_block,
            "seed": config.seed,
            "config_sha256": config_sha256(config),
            "std_version": rows["std_version"].to_numpy(),
            "licence": licence,
        }
    )


def _audit_report(
    table: pd.DataFrame,
    scored: _Scored,
    unlabeled: pd.DataFrame,
    substrates: pd.DataFrame,
    config: PuConfig,
) -> dict[str, Any]:
    """The audit record of ``modeling.md`` "S2a output", plain JSON throughout.

    ``spy_recall_realised`` is counted from the table rather than from the rule, so the two can
    disagree and be seen to. Read it with the module docstring: it is recall on reported positives.
    """
    in_p = table["in_p"].to_numpy()
    spy_scores = table.loc[in_p, "spy_score"].to_numpy()
    kept = int(table["reliable_negative"].sum())
    report: dict[str, Any] = {
        "n_spies": int(in_p.sum()),
        "spy_recall_configured": float(config.spy_recall),
        "spy_recall_realised": float(
            (spy_scores >= scored.thresholds[config.spy_scoring]).mean()
        ),
        "n_reliable_negatives": kept,
        "reliable_negative_share": kept / len(unlabeled),
        # Both shares, because the fit is draw-weighted and the two answer different questions:
        # what share of the distinct pairs was kept, and what share of the drawn mass. The spy
        # sweep reports the same pair (`rn_share_pairs`, `rn_share_weighted`).
        "reliable_negative_share_weighted": float(
            table.loc[table["reliable_negative"], "weight"].sum()
            / table.loc[~in_p, "weight"].sum()
        ),
        "spy_score_deciles": score_deciles(spy_scores),
        "unlabeled_score_deciles": score_deciles(
            table.loc[~in_p, "spy_score"].to_numpy()
        ),
        "selection_shift": selection_shift(
            table[table["reliable_negative"]], unlabeled, substrates
        ),
    }
    for name in SCORINGS:
        report[f"threshold_{name}"] = scored.thresholds[name]
        report[f"n_reliable_negatives_{name}"] = int(
            (scored.scores[name][scored.unlabeled_row] < scored.thresholds[name]).sum()
        )
    return report
