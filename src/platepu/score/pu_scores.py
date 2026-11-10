"""Stage S2b, the final fit and the two arms (docs/design/modeling.md "S2b"; ADR 028, ADR 029).

S2a cut the reliable negatives and audited them. This fits the final model **once** and scores the
pairs the plate and the evaluation are about. It has two arms, and ``config.negatives`` is the whole
of the difference between them:

- the **spy arm**, primary (ADR 014 item 1): P against the rows of an S2a table flagged
  ``reliable_negative``, each carrying the draw weight it was given there (ADR 029);
- the **naive arm**, secondary (ADR 016 item 2, better motivated by ADR 026 item 3): the same
  learner with no spies and no threshold, P against **every** row of U weighted by ``n_draws``, the
  raw probability used as the ranking. It answers what the spy step bought, and the first real S2a
  run gave that question more weight rather than less: the reliable negatives came out larger than U
  (33.78 heavy atoms against 32.36, 36.61 on ``morgan``) and thinner (36 % of their substrates
  appear in exactly one pair, against 0 % anywhere in U), while the grid the plate is picked from
  averages 23.73. The naive arm touches none of that.

**No label and no metric.** Nothing here reads a benchmark label and nothing computes a ROC-AUC: the
validation read is a separate step, as ``scripts/counts/count_baseline_aucs.py`` is for stage S1, and
every stratum, comparison and interval is stage E (ADR 008 item 6). Reading and writing files is the
script's job, so both functions below take values and return values.
"""

from typing import Any

import numpy as np
import pandas as pd

from platepu.features.blocks import SIZE
from platepu.score.pairs import pair_matrix
from platepu.score.pu_audit import score_deciles
from platepu.score.pu_config import PuConfig, PuScoreConfig, config_sha256
from platepu.score.pu_fit import Learner, fit_predict

PAIR_KEYS = ["key_a_stereo", "key_b_stereo", "key_a_flat", "key_b_flat"]
STEREO_KEYS = ("key_a_stereo", "key_b_stereo")
FLAT_KEYS = ("key_a_flat", "key_b_flat")
# The keys an S2a run fixed that an S2b run must not contradict. The spy settings are deliberately
# not among them: they live in the S2a record and are traceable through its hash (ADR 028 item 3).
SHARED_KEYS = ("feature_block", "p_width", "unlabeled", "learner", "learner_params")


def negative_source(config: PuScoreConfig) -> str:
    """Which table the negative class comes from, which **is** the arm (ADR 028 item 2).

    The one place the arm is decided, so ``training_rows`` and the report cannot disagree about
    which run they are describing.
    """
    return "unlabeled" if config.negatives is None else "reliable_negatives"


def check_negatives_pointer(
    config: PuScoreConfig, spy_config: PuConfig, negatives: pd.DataFrame
) -> None:
    """Refuse a spy arm whose negatives were cut under a different configuration.

    The pointer is checked and not trusted (ADR 028 item 3). A negatives table cut under ``morgan``
    must never be fitted under ``morgan_centre``: the reliable negatives depend on the block, so the
    two would be a model trained on a selection made for another model, and nothing in the output
    would say so.

    Raises:
        ValueError: one of ``SHARED_KEYS`` disagrees between the two records, naming the key; or the
            table's ``config_sha256`` is not the one distinct value the S2a record hashes to -- which
            also catches a table holding two configurations' rows.
    """
    for key in SHARED_KEYS:
        mine, theirs = getattr(config, key), getattr(spy_config, key)
        if mine != theirs:
            raise ValueError(
                f"{key} disagrees with the S2a config {config.negatives!r}: this run has {mine!r}, "
                f"the negatives were cut under {theirs!r}"
            )
    expected = config_sha256(spy_config)
    found = sorted(set(negatives["config_sha256"]))
    if found != [expected]:
        raise ValueError(
            f"the negatives table's config_sha256 is {found} where the S2a config "
            f"{config.negatives!r} hashes to {expected}; it was cut under another configuration"
        )


def training_rows(
    positives: pd.DataFrame, negatives: pd.DataFrame, config: PuScoreConfig
) -> pd.DataFrame:
    """The rows the final model is fitted on: ``PAIR_KEYS``, ``target`` and ``weight``.

    **The arm's rule lives here and not in the caller.** Which rows are the negative class *is* the
    definition of the arm, so the whole table arrives and this function selects: the spy arm keeps
    the rows flagged ``reliable_negative`` and reads their weights from the ``weight`` column the S2a
    table carries (ADR 029), the naive arm keeps every row and weighs it by ``n_draws``. A caller
    that filtered first could filter differently, and nothing would catch it (§5).

    A positive weighs 1.0, which is what makes the weighted P-to-negative ratio read as the project
    quotes it (about 1:9.7 for the naive arm; ADR 024, ADR 026).

    Raises:
        ValueError: a pair is in both classes, naming it. The real data cannot produce one -- stage U
            subtracts every reported pair on flat keys, and ``reliable_negative`` is never true on a
            spy -- so this is a broken input and not a case to resolve.
    """
    if negative_source(config) == "unlabeled":
        chosen, weights = negatives, negatives["n_draws"]
    else:
        chosen = negatives[negatives["reliable_negative"].to_numpy()]
        weights = chosen["weight"]
    train = pd.concat(
        [
            positives[PAIR_KEYS].assign(target=1, weight=1.0),
            chosen[PAIR_KEYS].assign(
                target=0, weight=weights.to_numpy(dtype=np.float64)
            ),
        ],
        ignore_index=True,
    )
    _refuse_both_classes(train)
    return train


def pair_scores(
    train: pd.DataFrame,
    scored: dict[str, pd.DataFrame],
    substrates: pd.DataFrame,
    config: PuScoreConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """``(scores, report)``: one float32 array per scored set, from **one** fit.

    The scored sets are stacked into a single matrix and predicted in one call, so the learner is
    fitted once however many sets are passed. They are not cross-fitted: no validation pair is in P,
    and the pairs that *are* in the training rows are counted in the report and kept rather than
    dropped, because dropping them would filter a scoring set (ADR 028 item 7).
    """
    features = pair_matrix(train, substrates, config.feature_block)
    blocks = {
        name: pair_matrix(pairs, substrates, config.feature_block)
        for name, pairs in scored.items()
    }
    predicted = fit_predict(
        features,
        train["target"].to_numpy(),
        np.vstack(list(blocks.values())),
        train["weight"].to_numpy(),
        Learner(params=config.learner_params, seed=config.seed),
    )
    scores, start = {}, 0
    for name, block in blocks.items():
        scores[name] = predicted[start : start + len(block)]
        start += len(block)
    return scores, _report(train, scored, scores, substrates, negative_source(config))


def _refuse_both_classes(train: pd.DataFrame) -> None:
    """Stop on a pair that is both a positive and a negative, which no weighting could resolve."""
    pairs = list(zip(train["key_a_stereo"], train["key_b_stereo"], strict=True))
    target = train["target"].to_numpy()
    positive = {pair for pair, is_p in zip(pairs, target, strict=True) if is_p == 1}
    both = sorted(
        {
            pair
            for pair, is_p in zip(pairs, target, strict=True)
            if is_p == 0 and pair in positive
        }
    )
    if both:
        raise ValueError(
            f"{len(both)} pair(s) are in both classes, e.g. {both[:5]}; a positive cannot also be "
            f"a negative, so one of the two inputs is wrong"
        )


def _report(
    train: pd.DataFrame,
    scored: dict[str, pd.DataFrame],
    scores: dict[str, np.ndarray],
    substrates: pd.DataFrame,
    source: str,
) -> dict[str, Any]:
    """Sizes, the ratio both ways, and the label-free diagnostics of each scored set."""
    negative = (train["target"] == 0).to_numpy()
    weight = train["weight"].to_numpy()
    return {
        "n_positives": int((~negative).sum()),
        "n_negatives": int(negative.sum()),
        "negative_source": source,
        # Negatives per positive, the direction ADR 024 and ADR 026 quote as "1:9.7", so the report
        # and the records read the same way round.
        "negatives_per_positive_rows": float(negative.sum() / (~negative).sum()),
        "negatives_per_positive_weight": float(
            weight[negative].sum() / weight[~negative].sum()
        ),
        "negative_weight_sum": float(weight[negative].sum()),
        "scored": {
            name: _set_report(pairs, scores[name], train, substrates)
            for name, pairs in scored.items()
        },
    }


def _set_report(
    pairs: pd.DataFrame,
    values: np.ndarray,
    train: pd.DataFrame,
    substrates: pd.DataFrame,
) -> dict[str, Any]:
    """One scored set's diagnostics, none of which touches a label.

    ``score_deciles`` is stage S2a's own function, so the word means one thing in both reports. The
    tie structure is here because the plate takes the top 96 and a reader must know whether that was
    chemistry or a tie. The Spearman correlation against pair size is the size check the stage S
    design demands beside every model, and the two overlap counts say how much of the scored set the
    fit had already seen (ADR 028 item 7).
    """
    sizes = substrates.set_index("key_stereo")[SIZE]
    heavy = pairs["key_a_stereo"].map(sizes).to_numpy(dtype=np.float64) + pairs[
        "key_b_stereo"
    ].map(sizes).to_numpy(dtype=np.float64)
    counts = pd.Series(values).value_counts()
    return {
        "score_deciles": score_deciles(values),
        "n_distinct_scores": len(counts),
        "largest_tie": int(counts.iloc[0]),
        "spearman_heavy_atoms": float(
            pd.Series(values).corr(pd.Series(heavy), method="spearman")
        ),
        "n_in_training_stereo": _in_training(pairs, train, STEREO_KEYS),
        "n_in_training_flat": _in_training(pairs, train, FLAT_KEYS),
    }


def _in_training(
    pairs: pd.DataFrame, train: pd.DataFrame, keys: tuple[str, str]
) -> int:
    """How many scored pairs the fit already saw, on one key type.

    Both counts are reported because they answer different questions: stereo says the identical pair
    was in training, flat says a stereoisomer of it was, and the fingerprints are chirality-blind,
    so to the model the flat match is the one that matters.
    """
    known = {tuple(row) for row in train[list(keys)].to_numpy()}
    return int(sum(tuple(row) in known for row in pairs[list(keys)].to_numpy()))
