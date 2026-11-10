"""Validation and test sets of one benchmark for one condition family (docs/design/modeling.md).

The benchmark's ``test`` side is ours untouched until the end. Everything else that is labelled, in
the family, and shares no substrate with that test side is our validation set. Family membership is
the positives rule: the condition contains every member, extras allowed.

Until 2026-10-22 the validation set was the benchmark's own ``train`` side alone. That threw away the
rows Chemlex discarded to keep *its* train and test clean, a reason that does not bind us because our
model never trains on Chemlex. Taking them back, and at the same time closing a leak the benchmark's
own split had, raised the validation set from 1,725 reactions over 53 distinct amines to 2,615 over
113, which is what the uncertainty of every comparison depends on. The test side is unchanged. The
old validation set is **not** a subset: 150 of its rows carried a stereoisomer of a test substrate
and are now excluded.
"""

from collections.abc import Set as AbstractSet

import pandas as pd

from platepu.standardise.molecule import condition_from_text, in_family


def family_split(
    labels: pd.DataFrame, members: AbstractSet[str], split_column: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(validation, test) labelled rows of ``labels`` in the family.

    Args:
        labels: Benchmark reactions with ``condition``, ``label``, the four substrate key columns
            and ``split_column``.
        members: The condition family's standardised reagents.
        split_column: The benchmark's own split; only its ``test`` value is read.

    Returns:
        The validation rows, which are every labelled row of the family whose acid and whose amine
        appear nowhere in the test side, and the test rows. The two share no substrate, so the test
        side stays unseen whatever the benchmark's own split did with the rest.

    The exclusion is on the **flat** keys, not the stereo keys. Our fingerprints are computed without
    chirality, so two stereoisomers carry the identical feature vector and a stereoisomer of a test
    substrate sitting in validation is, to every model we build, the same molecule. The benchmark's
    own train side had this leak (5 acids, 1 amine on 2026-10-22); excluding on flat keys removes it.
    """
    member = labels["condition"].map(
        lambda condition: in_family(condition_from_text(condition), members)
    )
    usable = labels[member & labels["label"].notna()]
    test = usable[usable[split_column] == "test"]
    # Usable rows but no test row is never a legitimate state: it means the split column was spelled
    # or typed differently ("Test", "test ", a null), and then every labelled row becomes validation
    # while the seal looks intact, because an empty test side shares no substrate with anything. A
    # family with no usable rows at all is a different, allowed case -- both frames come back empty,
    # which tests/test_benchmark_split.py pins.
    if not usable.empty and test.empty:
        raise ValueError(
            f"no row of {split_column!r} equals 'test', so the whole family would become "
            f"validation; values present: {sorted(set(usable[split_column].dropna()))[:5]}"
        )
    unseen = ~usable["key_a_flat"].isin(set(test["key_a_flat"])) & ~usable[
        "key_b_flat"
    ].isin(set(test["key_b_flat"]))
    return usable[unseen], test
