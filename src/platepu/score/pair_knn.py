"""Chen's cheap instrument: are P and U two clusters, or a gradient? (the anchor ablation).

Chen et al. settled the DUD-E decoy question partly with a plain fingerprint K-NN: if ``K = 1`` wins
and the AUC is high, the two classes sit in separate clusters and a model is reading the way the
negatives were built rather than the chemistry. On DUD-E the 1-NN reached AUC > 0.82 on all 102
targets and ``K = 1`` was best on 79 of them.

**The trap this module exists to avoid.** A positive pair compared against the positives finds
*itself*, at similarity 1.0 on both sides, and no unlabeled pair can -- reported pairs are
subtracted out of U on flat keys. Without leave-one-out the probe separates the classes perfectly
with no chemistry at all, and reports a number that looks like a result.

What is **not** a bug: similarity 1.0 on one *side*. An unlabeled pair's acid really is a published
acid; it was drawn from them. So self is excluded by **row position**, never by similarity value.
Excluding every exact match would close the probe's eyes along with the leak.

"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# The one similarity in the project (CLAUDE.md §5). It lived privately inside `baselines` until
# 2026-10-30, and this probe reached past the underscore to get at it; it is now public in its own
# module, which is what §5 asks for once a third caller needs it.
from platepu.score.similarity import fingerprints, side_similarity

SIDES = (("a", "key_a_stereo"), ("b", "key_b_stereo"))


def pair_similarity(
    labelled: pd.DataFrame, substrates: pd.DataFrame, block: str
) -> np.ndarray:
    """Every pair against every other, as the **minimum** of the two sides' Tanimoto.

    The minimum, not the mean: a pair sharing its acid with another but carrying an unrelated amine
    is not its analogue. The diagonal is left at 1.0 — the caller drops it by position.
    """
    index, bits = fingerprints(substrates, block)
    sides = []
    for _, key in SIDES:
        keys = labelled[key]
        similarity, rows, columns = side_similarity(keys, keys, index, bits)
        sides.append(
            similarity[keys.map(rows).to_numpy()][:, keys.map(columns).to_numpy()]
        )
    return np.minimum(sides[0], sides[1])


def neighbour_indices(
    labelled: pd.DataFrame, substrates: pd.DataFrame, block: str, k: int
) -> np.ndarray:
    """Row positions of each pair's ``k`` nearest neighbours, itself never among them.

    Ties go to the lower row position, so two runs agree: fingerprints collide and ties are common.

    Raises:
        ValueError: ``k`` is not below the pool size, which would ask for more neighbours than the
            pool can offer once each pair is excluded from its own row.
    """
    similarity = pair_similarity(labelled, substrates, block)
    size = len(similarity)
    if not 0 < k < size:
        raise ValueError(
            f"k must be between 1 and {size - 1} for a pool of {size}, not {k}"
        )
    # By position, not by value: a one-sided exact match is real and has to survive this step.
    np.fill_diagonal(similarity, -1.0)
    return np.argsort(-similarity, axis=1, kind="stable")[:, :k]


def pair_knn_auc(
    labelled: pd.DataFrame, substrates: pd.DataFrame, block: str, ks: tuple[int, ...]
) -> dict[int, float]:
    """ROC-AUC of the neighbour vote at each ``k``, in the order ``ks`` gives them.

    A pair's score is the share of its ``k`` neighbours that are positives. Which ``k`` wins is the
    reading: ``k = 1`` winning is Chen's signature of two clusters rather than a gradient.

    The order comes from ``neighbour_indices`` at the largest ``k``, and every smaller ``k`` is a
    prefix of it. One implementation of the exclusion and the tie rule, so a change to either cannot
    move this table without moving what ``neighbour_indices`` reports.
    """
    order = neighbour_indices(labelled, substrates, block, max(ks))
    label = labelled["label"].to_numpy()
    return {
        k: float(roc_auc_score(label, label[order[:, :k]].mean(axis=1))) for k in ks
    }


def balanced_sample(labelled: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """``per_class`` rows of each label, drawn without replacement and reproducible from ``seed``.

    All against all over the full data is 113,596 squared, which is not affordable; Chen sampled per
    target for the same reason. A balanced pool also keeps the neighbour vote meaningful — scoring
    against the positives alone would give every unlabeled pair none but positive neighbours.

    Raises:
        ValueError: a label has fewer rows than ``per_class``. Drawing with replacement instead
            would put the same pair in the pool twice and let it be its own neighbour under another
            row position, which is the one thing this module exists to prevent.
    """
    generator = np.random.default_rng(seed)
    parts = []
    for label in (1, 0):
        rows = labelled[labelled["label"] == label]
        if len(rows) < per_class:
            raise ValueError(
                f"label {label} has {len(rows)} rows, fewer than the {per_class} asked for"
            )
        parts.append(rows.iloc[generator.choice(len(rows), per_class, replace=False)])
    return pd.concat(parts, ignore_index=True)
