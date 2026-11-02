"""Pair rows for stage S: the features of both substrates on one row (docs/design/modeling.md).

The feature table of stage F holds one row per substrate; every model and baseline scores a pair, so
each side is joined on once and prefixed ``a_`` (role A, the acid) and ``b_``. Nothing is dropped and
nothing is reordered, so a caller can put its own columns back beside the result.

Two shapes of the same join live here, because the design puts the pair rows **and** the feature
matrix in this module. ``pair_features`` returns a frame a baseline reads by column name;
``pair_matrix`` returns the dense numeric matrix a learner is fitted on, which stage S2 needs for
both its sub-steps (``score/pu.py``) and which must contain the fingerprint and nothing else.
"""

import numpy as np
import pandas as pd

from platepu.features.blocks import FINGERPRINT_BLOCKS, block_matrix

# (prefix, the column of ``pairs`` that holds that side's key)
SIDES = (("a", "key_a_stereo"), ("b", "key_b_stereo"))


def substrate_index(substrates: pd.DataFrame) -> dict[str, int]:
    """Row position of every ``key_stereo``, so a pair row can pick its two feature rows.

    One function because two callers index the same table by key: ``pair_matrix`` below and
    ``score/similarity.fingerprints``, which carried this code until 2026-10-30 (CLAUDE.md §5, keys
    live in exactly one function). The messages are that module's, unchanged, and its tests pin them.

    Raises:
        ValueError: the table lacks ``key_stereo``; or a ``key_stereo`` occurs on more than one row.

            The duplicate refusal is the one worth explaining. A dictionary keeps the **last** row of
            a repeated key, so every feature of that substrate would be read off whichever row
            happened to come last -- a wrong number with no exception and no rejected row. Stage F
            writes one row per ``key_stereo`` (``features/substrate.py`` groups by it), so this does
            not arise; it is refused rather than resolved because a rule for which duplicate wins
            would be a branch for a case the data does not have (§8). Identical duplicates are
            refused too, for the same reason: telling them apart is that branch.
    """
    if "key_stereo" not in substrates.columns:
        raise ValueError(
            "the feature table lacks column(s) ['key_stereo'], the row index"
        )
    keys = substrates["key_stereo"]
    repeated = keys.duplicated(keep=False)
    if repeated.any():
        # Sorted by `str` because a null key is reachable and `sorted` refuses to order it against
        # a string. `drop_duplicates` counts two nulls as one duplicated value, where `nunique`
        # would drop them and report none.
        shown = sorted(keys[repeated].drop_duplicates(), key=str)
        raise ValueError(
            f"duplicated key_stereo values: {shown[:5]} ({len(shown)} on "
            f"{int(repeated.sum())} rows); one row per substrate is required"
        )
    return {key: row for row, key in enumerate(keys)}


def pair_matrix(
    pairs: pd.DataFrame, substrates: pd.DataFrame, block: str
) -> np.ndarray:
    """``block`` for the acid then for the amine, one row per pair in ``pairs``' own order.

    The matrix a pair model is fitted on, and the whole of what it may see: ADR 008 item 1 keeps
    hand-written chemistry in the baselines and in the reporting, so the ten descriptors, the amine
    subclasses and ``n_heavy_atoms`` are out even when the table carries them. Only a fingerprint
    block is accepted, and any other name raises rather than being quietly dropped.

    Returns:
        ``(len(pairs), 2 * width)`` of **float64**, ``block_matrix``'s own dtype, counts unchanged
        (ADR 018 -- a count of 3 stays 3).

        This returned float32 for a few hours on 2026-10-30, on the reasoning that a Morgan count
        is at most 255 and the learner bins its input to float32 anyway, so the matrix could be
        halved. The second half of that is **false**, and pr-reviewer caught it:
        ``sklearn.ensemble._hist_gradient_boosting.common.X_DTYPE`` is float64 in scikit-learn
        1.9.1, so a float32 matrix is copied to float64 inside ``fit`` and the peak is the sum of
        both. Measured, one fit of 40,000 x 4,096 in a process of its own
        (``scratchpad/dtype_peak.py``, 2026-10-30): float32 input 0.66 GB with a peak of 2.81 GB,
        float64 input 1.31 GB with a peak of **2.16 GB**. The narrow matrix cost 0.65 GB more, and
        float64 also restores the configuration ``scratch/profile_pair_fit.py`` profiled (11.54 GB
        peak, 22 % of this machine's RAM).

    Raises:
        ValueError: ``block`` is not a fingerprint block, naming it; or, from ``_check`` and
            ``substrate_index``, a pair key is missing from the feature table, the table mixes
            ``std_version`` or disagrees with the pairs', or a ``key_stereo`` is repeated.
    """
    if block not in FINGERPRINT_BLOCKS:
        raise ValueError(
            f"feature block {block!r} is not a fingerprint, and a pair model may see nothing "
            f"else (ADR 008 item 1); it must be one of {list(FINGERPRINT_BLOCKS)}"
        )
    _check(pairs, substrates.set_index("key_stereo"))
    index = substrate_index(substrates)
    bits = block_matrix(substrates, block)
    return np.hstack(
        [bits[[index[key] for key in pairs[column]]] for _, column in SIDES]
    )


def pair_features(pairs: pd.DataFrame, substrates: pd.DataFrame) -> pd.DataFrame:
    """``pairs`` with every substrate column joined on twice, prefixed by side.

    Args:
        pairs: One row per scored pair, with ``key_a_stereo`` and ``key_b_stereo``; its own columns
            are kept untouched.
        substrates: The stage F table, with ``key_stereo`` and the feature columns.

    Returns:
        ``pairs`` plus ``a_<column>`` and ``b_<column>`` for every column of ``substrates`` except
        ``key_stereo``, in the same row order and with the same index.

    Raises:
        ValueError: a pair's key is missing from ``substrates`` (counted 2026-10-21: coverage is
            100 %, so a miss is input we have not seen), ``substrates`` mixes ``std_version``, which
            would put two standardisations in one table, or ``pairs`` carries a ``std_version``
            column that disagrees with it.
    """
    known = substrates.set_index("key_stereo")
    _check(pairs, known)
    paired = pairs.copy()
    for side, key in SIDES:
        for column in known.columns:
            paired[f"{side}_{column}"] = pairs[key].map(known[column])
    return paired


def _check(pairs: pd.DataFrame, known: pd.DataFrame) -> None:
    """Stop on a key the feature table does not have, or on two standardisations meeting.

    The keys themselves carry the standardisation: a key written under one ``std_version`` can be
    absent from, or mean something else in, a table built under another. The key check above catches
    that only when the strings differ; two standardisations that agree on these molecules and differ
    elsewhere would pass it silently. Where the pairs record their own ``std_version`` (the positives
    and the benchmark pairs do; the inventory pairs from ``library/unlabeled.py`` do not), it must be the
    one the features were built under. CLAUDE.md §6: one standardisation for P, U and TEST.
    """
    for _, key in SIDES:
        missing = sorted(set(pairs[key]) - set(known.index))
        if missing:
            raise ValueError(f"{key} not in the feature table: {missing[:5]}")
    if "std_version" not in known.columns:
        return
    versions = sorted(set(known["std_version"]))
    if len(versions) > 1:
        raise ValueError(f"the feature table mixes std_version values {versions}")
    if "std_version" in pairs.columns:
        theirs = sorted(set(pairs["std_version"]))
        if theirs != versions:
            raise ValueError(
                f"the pairs carry std_version {theirs}, the feature table {versions}"
            )
