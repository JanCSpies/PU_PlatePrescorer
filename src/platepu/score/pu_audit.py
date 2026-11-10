"""What the spy threshold selected, measured (docs/design/modeling.md "Stage S2a"; ADR 016, 026).

Separate from ``score/pu.py`` because choosing the reliable negatives and measuring what the choice
did are two jobs, and the second one is the reason stage S2a writes a table of its own: the reliable
negatives are the part of this method most in need of auditing.

**The number this module exists for.** Stage U matched U's per-side marginals to P's on purpose, and
the gate confirms it worked (rank-AUC 0.5049 for acids, 0.5095 for amines, ADR 026). The threshold
then keeps a subset of U, and a subset of a matched set is not matched. So every reliable-negative
set is shifted against U by construction, and the only honest thing to do is say how far:
``selection_shift`` is that statement. It is a description, not a gate -- ADR 017 is about not
turning a diagnostic into one.

**The shift is counted over rows, while the fit is weighted by ``n_draws``** (``score/pu_fit.py``).
That is deliberate for now and it is a known mismatch: U's certified marginal is the draw
distribution, so a row-based distance is measured against a U the gate did not certify. Making the
three measures draw-weighted would change every value in them and is a separate piece of work with
its own tests; the row counts are what ``tests/test_score_pu.py`` pins by hand today. The report
carries the reliable-negative share both ways, so the size of the mismatch is at least visible.
"""

import numpy as np
import pandas as pd

from platepu.features.blocks import SIZE

# (the report's suffix, the pair column that holds that side's substrate key)
SIDES = (("a", "key_a_stereo"), ("b", "key_b_stereo"))


def score_deciles(scores: np.ndarray) -> list[float]:
    """The nine decile cut points, ascending, as plain floats for the report JSON.

    Deciles and not a mean and spread: the two distributions this is asked of -- the spies' scores
    and U's -- are the ones whose overlap the threshold cuts, and a reader who wants to know whether
    the cut fell in a gap or in the middle of a mass needs the shape.
    """
    return [float(value) for value in np.quantile(scores, np.arange(1, 10) / 10)]


def selection_shift(
    reliable: pd.DataFrame, unlabeled: pd.DataFrame, substrates: pd.DataFrame
) -> dict[str, float | None]:
    """How far the reliable negatives sit from U as a whole, on three axes.

    Args:
        reliable: The rows of U the threshold kept, with the two stereo keys.
        unlabeled: Every U row, the set the shift is measured against.
        substrates: The stage F table, for ``n_heavy_atoms``.

    Returns:
        ``marginal_tv_a`` and ``marginal_tv_b``, the total variation distance between the
        reliable negatives' distribution over that side's substrates and U's own, in [0, 1] -- the
        direct reading of how much of stage U's matching the threshold undid;
        ``mean_heavy_atoms_*``, the mean heavy-atom count of a **pair** (the two sides summed, as
        ``modeling.md`` counts pair size), since size is the axis a model can read without chemistry;
        and ``single_partner_share_*``, the share of the **distinct substrates** of a set that occur
        in exactly one of its pairs, pooled over both roles -- a set carried by a few popular
        substrates is narrower than its row count suggests. Every value is ``None`` when the set it
        describes is empty, which an empty reliable-negative set makes reachable; a distance to an
        empty distribution is undefined, and ``None`` writes as ``null`` where a NaN would write as
        invalid JSON.
    """
    sizes = substrates.set_index("key_stereo")[SIZE]
    shift: dict[str, float | None] = {}
    for side, key in SIDES:
        shift[f"marginal_tv_{side}"] = _total_variation(reliable[key], unlabeled[key])
    for name, pairs in (("reliable", reliable), ("unlabeled", unlabeled)):
        shift[f"mean_heavy_atoms_{name}"] = _mean_pair_size(pairs, sizes)
        shift[f"single_partner_share_{name}"] = _single_partner_share(pairs)
    return shift


def _total_variation(selected: pd.Series, whole: pd.Series) -> float | None:
    """Half the summed absolute difference of the two key distributions; 0 means no shift."""
    if selected.empty:
        return None
    kept = selected.value_counts(normalize=True)
    everything = whole.value_counts(normalize=True)
    keys = kept.index.union(everything.index)
    difference = kept.reindex(keys, fill_value=0.0) - everything.reindex(
        keys, fill_value=0.0
    )
    return float(0.5 * difference.abs().sum())


def _mean_pair_size(pairs: pd.DataFrame, sizes: pd.Series) -> float | None:
    """Mean heavy atoms of a pair, the two sides summed."""
    if pairs.empty:
        return None
    acid, amine = (pairs[key].map(sizes) for _, key in SIDES)
    return float((acid + amine).mean())


def _single_partner_share(pairs: pd.DataFrame) -> float | None:
    """Share of the set's distinct substrates that appear in exactly one of its pairs.

    Both roles are pooled into one number, which stage F's refusal of a key under two roles makes
    safe: a ``key_stereo`` is an acid or an amine, never both.
    """
    if pairs.empty:
        return None
    appearances = pd.concat([pairs[key] for _, key in SIDES]).value_counts()
    return float((appearances == 1).mean())
