"""The project's one similarity between substrates of the same side (CLAUDE.md §5).

Three things need it and that is why it is here rather than inside ``score/baselines.py``, where it
began: the nearest-pair baselines B2, B3 and B8; the pair K-NN probe of the anchor ablation
(``score/pair_knn.py``), which until now reached past the privacy of two underscored names in a
module called after baselines; and the per-side base kernel of ADR 023, which is the third consumer
and the reason §5's threshold for extracting a shared helper is met.

**Binarised, on purpose.** ``features.blocks.block_matrix`` never binarises — a count of 3 stays 3
(ADR 018) — and this module does, because Tanimoto asks which environments occur and not how often.
ADR 023 fixes the binarised form as the base kernel's, so the choice is now a decision and not a
convenience.

**float32, also on purpose.** Every step before the single division is exact in float32: the bits
are 0 or 1 and the inner dimension is 2,048, so each dot product is an integer far below 2^24. The
division is the only rounding in the path, and `tests/test_similarity.py` pins all of it. The
consequence worth knowing, because it is a trap: the division rounds by **up to** about 6e-8
relative per entry. That is the unit roundoff, 2^-24 = 5.96e-8, and it is a **bound**, not a typical
error: dividing random ratios of integers in 1..2048 in float32 and comparing against float64 gives
a worst case of 5.95e-8 against a mean of 2.1e-8, reproduced three times on 2026-10-30 at 200,000
and 400,000 draws and at three seeds. That is a one-line check and no script holds it, which is why
the method is written out here rather than cited. ADR 025's positive-semidefiniteness check built its Gram matrices in float64 from the
raw bits rather than through this function, reasoning that rounding of that size can surface as a
negative eigenvalue; that reasoning is recorded there and is **not** re-measured here. Anyone
widening this to float64 is changing a documented property and has a test to change first.
"""

import numpy as np
import pandas as pd

from platepu.features.blocks import block_matrix


def fingerprints(
    substrates: pd.DataFrame, block: str
) -> tuple[dict[str, int], np.ndarray]:
    """Row index of every key, and ``block``'s binarised counts as a float matrix for the dot product.

    Raises:
        ValueError: the table lacks ``key_stereo``; a ``key_stereo`` occurs on more than one row;
            or, from ``block_matrix``, the block is unknown or the table lacks its columns. The
            column is checked before the block, so a frame missing both reports the column.

            The duplicate refusal is the one worth explaining. The index below keeps the **last**
            row of a repeated key, so every similarity for that substrate would be read off
            whichever fingerprint happened to come last -- a wrong number with no exception and no
            rejected row. Stage F writes one row per ``key_stereo`` (``features/substrate.py``
            groups by it), so this does not arise; it is refused rather than resolved because a rule
            for which duplicate wins would be a branch for a case the data does not have (§8).
            Identical duplicates are refused too, for the same reason: telling them apart is that
            branch.
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
    index = {key: row for row, key in enumerate(keys)}
    # The conversion lives in one place (ADR 018); binarising is this module's own choice, because
    # Tanimoto is about which environments occur and not how often.
    return index, (block_matrix(substrates, block) > 0).astype(np.float32)


def side_similarity(
    scored: pd.Series, positive: pd.Series, index: dict[str, int], bits: np.ndarray
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    """Tanimoto between the distinct scored and the distinct positive substrates of one side.

    Only the distinct substrates are compared: the lab holds 529 acids and 538 amines against about
    11,000 and 10,000 in the positives, so the matrix stays small and the pair reduction is left to
    the caller.

    The result is **float32**, and deliberately not widened; the module docstring carries the
    argument and the measured bound, and this is the consequence for the callers. Every step above
    is exact in float32: the bits are 0 or 1 and the inner dimension is 2,048, so each dot product
    is an integer well below 2^24, and ``union`` is an integer too. The single division is the only
    rounding in the path. B2 then applies ``minimum`` and ``max``, which select rather than compute,
    so its answer is unchanged; B3 adds, so it must accumulate into a float64 array, and it does.

    Two all-zero fingerprints give 0.0 rather than NaN, which the ``where`` guard below does. It is
    reachable: at ``d = 0`` a substrate whose reacting centre contributes no environment has an
    empty fingerprint.

    Raises:
        ValueError: a distinct key of either side is absent from the substrate table ``index`` was
            built from, so its fingerprint is unknown. The scored side is checked first, so the
            message is deterministic when both are short. It used to be a bare ``KeyError`` naming
            one key and neither the side nor how many.

            Up to five keys are named beside the counts, which is what ``positives/unlabeled.py``
            and ``features/substrate.py`` already do for this same check. I first reported counts
            alone, on the reasoning that a ``key_stereo`` can be an inventory or a Chemlex identity;
            the repository's own precedent says otherwise, and there is a reachable case that
            settles it. A **null** ``key_stereo`` in a pair table counts as absent -- that is the
            defect ``positives/unlabeled.py`` records as its own ``KeyError: nan`` -- and counts
            alone would send the reader to audit the feature table when the fault is nulls in the
            pairs. The names tell the two causes apart at a glance.
    """
    scored_keys, positive_keys = pd.unique(scored), pd.unique(positive)
    for side, side_keys in (("scored", scored_keys), ("positive", positive_keys)):
        absent = sorted((key for key in side_keys if key not in index), key=str)
        if absent:
            raise ValueError(
                f"{side} keys missing from the substrate table: {absent[:5]} "
                f"({len(absent)} of {len(side_keys)} distinct)"
            )
    left = bits[[index[key] for key in scored_keys]]
    right = bits[[index[key] for key in positive_keys]]
    shared = left @ right.T
    union = left.sum(1)[:, None] + right.sum(1)[None, :] - shared
    similarity = np.divide(shared, union, out=np.zeros_like(shared), where=union > 0)
    return (
        similarity,
        {key: row for row, key in enumerate(scored_keys)},
        {key: column for column, key in enumerate(positive_keys)},
    )
