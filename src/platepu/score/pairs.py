"""Pair rows for stage S: the features of both substrates on one row (docs/design/modeling.md).

The feature table of stage F holds one row per substrate; every model and baseline scores a pair, so
each side is joined on once and prefixed ``a_`` (role A, the acid) and ``b_``. Nothing is dropped and
nothing is reordered, so a caller can put its own columns back beside the result.
"""

import pandas as pd

# (prefix, the column of ``pairs`` that holds that side's key)
SIDES = (("a", "key_a_stereo"), ("b", "key_b_stereo"))


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
