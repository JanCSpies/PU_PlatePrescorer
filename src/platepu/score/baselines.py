"""The baselines B0 to B8 of stage S1 (docs/design/modeling.md; ADR 006 item 5, ADR 007 item 5).

Every function takes the output of ``pair_features`` and returns one float per row, in row order.
None of them fits a parameter: B2, B3 and B8 only look up the positives. Higher is always better,
and a score is comparable only inside one scored set, so the caller scores one set at a time.

B8 is B2 read from the centred fingerprint, so both come out of ``nearest_pair_scores`` with the
block as an argument. The two bars that survive a change of reaction class are these, because B4 to
B6 are hand-written amide chemistry (ADR 014 item 2).
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from platepu.features.blocks import block_matrix

# Scored pairs per block of the B2 reduction. Read at call time so a test can shrink it; the answer
# does not depend on it. The cost is set by the number of positive *pairs*, which at the class width
# is 28,158 (over 11,543 distinct acids and 9,959 amines, counted 2026-10-22). With the columns
# selected first, one block holds three float32 arrays of CHUNK x 28,158, so 256 rows cost about
# 86 MB against the 1.7 GB that 5,000 would, and nothing larger is built on the way.
CHUNK = 256
# The two fingerprint blocks stage F writes. ``morgan_centre`` is rooted at the reacting atom and is
# not a subset of ``morgan`` (configs/features.yml).
MORGAN = "morgan"
MORGAN_CENTRE = "morgan_centre"


@dataclass(frozen=True)
class BaselinesConfig:
    """The two choices the baselines read from ``configs/models/baselines.yml``.

    Attributes:
        seed: The stream B0 draws from. It is stamped on every score row, so a run is reproducible
            from the table alone.
        amine_subclass_scores: One value per amine subclass for B4, with ``other`` named explicitly
            so a new subclass cannot slip in with a silent default.
    """

    seed: int
    amine_subclass_scores: Mapping[str, float]


def load_baselines_config(path: Path) -> BaselinesConfig:
    """Read the model YAML into a record, so no raw dictionary crosses into the scoring layer.

    Raises:
        ValueError: a key is missing, or its value is not of the shape the baselines need. The
            message names the file, because the caller sees only a path.
    """
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        seed = int(config["seed"])
        scores = {
            str(name): float(value)
            for name, value in config["amine_subclass_scores"].items()
        }
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{path} must give an integer seed and a mapping of amine subclass scores: {error}"
        ) from error
    return BaselinesConfig(seed=seed, amine_subclass_scores=scores)


def random_scores(paired: pd.DataFrame, seed: int, score_set: str) -> np.ndarray:
    """B0: uniform noise, the floor every other score is read against.

    The draw is made on the pairs sorted by their two keys and handed back in the caller's row
    order, so a pair's score does not depend on where its row sits. The stream is seeded from
    ``seed`` and ``score_set`` together: one stream over both sets would shift the validation
    scores whenever the inventory changed size.
    """
    order = np.lexsort(
        (paired["key_b_stereo"].to_numpy(), paired["key_a_stereo"].to_numpy())
    )
    generator = np.random.default_rng([seed, _entropy(score_set)])
    scores = np.empty(len(paired), dtype=np.float64)
    scores[order] = generator.random(len(paired))
    return scores


def popularity_scores(paired: pd.DataFrame, positives: pd.DataFrame) -> np.ndarray:
    """B1: how often each substrate is reported in the class, ``log1p(n_acid) + log1p(n_amine)``.

    Counts are stereo-agnostic (``key_flat``) and come from the class-wide positives, because
    popularity is a property of the substrate and not of one condition family. A substrate in no
    positive contributes ``log1p(0) = 0``; on our validation set that is most of them.
    """
    total = np.zeros(len(paired), dtype=np.float64)
    for side, key in (("a", "key_a_flat"), ("b", "key_b_flat")):
        reports = positives.groupby(key)["n_reports"].sum()
        total += np.log1p(
            paired[f"{side}_key_flat"].map(reports).fillna(0).to_numpy(float)
        )
    return total


def nearest_pair_scores(
    paired: pd.DataFrame,
    positives: pd.DataFrame,
    substrates: pd.DataFrame,
    block: str,
) -> np.ndarray:
    """B2 on ``block = "morgan"``, B8 on ``block = "morgan_centre"``: how close the pair is to the
    nearest reported pair.

    Tanimoto on the binarised counts, per side; the similarity of a scored pair to a positive pair
    is the **minimum** of the two sides, and the score is the maximum over the positives. The
    minimum, not the mean: a pair that shares its acid with a reported pair but carries an unrelated
    amine is not an analogue of it.

    B8 is this baseline rooted at the reacting atom, so it is this function with another argument
    and not a second copy: one definition, one similarity, and the two differ only in what the
    fingerprint looked at (ADR 013 item 5, ADR 014 item 2).

    Raises:
        ValueError: ``positives`` is empty (see ``_require_positives``), or ``block`` is not a
            column of ``substrates``.
    """
    _require_positives(positives, "B2")
    index, bits = _fingerprints(substrates, block)
    rows, by_positive = {}, {}
    for side, key in (("a", "key_a_stereo"), ("b", "key_b_stereo")):
        similarity, scored, positive = _side_similarity(
            paired[key], positives[key], index, bits
        )
        # Columns first, rows second. Selecting the positive columns on the distinct-substrate
        # matrix costs one small array per side (529 x 28,158 float32 at the class width, 60 MB);
        # doing it the other way round, as this did until 2026-10-22, built a CHUNK x 11,543
        # intermediate per block on the way, which is what made the peak 4.3 GB instead of 0.2.
        by_positive[side] = similarity[:, positives[key].map(positive).to_numpy()]
        rows[side] = paired[key].map(scored).to_numpy()
    scores = np.empty(len(paired), dtype=np.float64)
    for start in range(0, len(paired), CHUNK):
        # Named for the rows it slices, not "block": that word now means the feature block.
        chunk = slice(start, start + CHUNK)
        acid = by_positive["a"][rows["a"][chunk]]
        amine = by_positive["b"][rows["b"][chunk]]
        scores[chunk] = np.minimum(acid, amine).max(axis=1)
    return scores


def additive_scores(
    paired: pd.DataFrame, positives: pd.DataFrame, substrates: pd.DataFrame
) -> np.ndarray:
    """B3: the two substrates judged on their own, acid score + amine score.

    Per side, the Tanimoto to the single most similar positive substrate of that role. It reads the
    substrates of the positives and never their pairing, so it is the control for the question B2
    asks: does the pair matter, or are the two substrates on their own enough?

    Raises:
        ValueError: ``positives`` is empty (see ``_require_positives``).
    """
    _require_positives(positives, "B3")
    # B3 exists on the plain block only; there is no centred variant of it to choose between.
    index, bits = _fingerprints(substrates, MORGAN)
    total = np.zeros(len(paired), dtype=np.float64)
    for key in ("key_a_stereo", "key_b_stereo"):
        similarity, scored, _ = _side_similarity(
            paired[key], positives[key], index, bits
        )
        total += similarity.max(axis=1)[paired[key].map(scored).to_numpy()]
    return total


def subclass_scores(
    paired: pd.DataFrame, side: str, scores: Mapping[str, float]
) -> np.ndarray:
    """B4: the chemist's rule, one value per subclass of ``side`` (``"a"`` or ``"b"``).

    Raises:
        ValueError: a subclass the config does not list. Every subclass, ``other`` included, is
            named in the YAML, so a new class cannot slip in with a silent default.
    """
    values = paired[f"{side}_subclass"]
    unknown = sorted(set(values) - set(scores))
    if unknown:
        raise ValueError(f"subclasses missing from the model config: {unknown}")
    return values.map(scores).to_numpy(dtype=np.float64)


def rank_sum(*parts: np.ndarray) -> np.ndarray:
    """B5 and B6: the sum of the average ranks of each part.

    Ranks, so that a part's scale never reaches the result; average ranks, so that tied values share
    a place instead of being separated by the order they happen to sit in.
    """
    total = np.zeros(len(parts[0]), dtype=np.float64)
    for part in parts:
        total += pd.Series(part).rank().to_numpy(dtype=np.float64)
    return total


def size_scores(paired: pd.DataFrame) -> np.ndarray:
    """B7: minus the heavy atoms of the pair, so a smaller pair scores higher.

    A control, not a chemistry baseline. Its sign is the prior that a less hindered pair reacts
    faster. Validation pairs average 38.7 heavy atoms against 23.7 for inventory pairs, so this says
    how far that one difference alone carries a score.
    """
    heavy = paired["a_n_heavy_atoms"].to_numpy(float) + paired[
        "b_n_heavy_atoms"
    ].to_numpy(float)
    return -heavy


def _require_positives(positives: pd.DataFrame, model: str) -> None:
    """Stop before B2 or B3 reads an empty positive table.

    Without it both raise the same bare ``ValueError: zero-size array to reduction operation
    maximum`` from inside numpy, which names neither the baseline nor the cause. An empty table
    means the condition family or the class carries no positive pair at all, which is a question
    about the input and never a score of zero.
    """
    if positives.empty:
        raise ValueError(
            f"{model} has no positives to compare with: the table is empty"
        )


def _entropy(text: str) -> int:
    """A stable 64-bit seed from a string; ``hash()`` changes between runs."""
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")


def _fingerprints(
    substrates: pd.DataFrame, block: str
) -> tuple[dict[str, int], np.ndarray]:
    """Row index of every key, and ``block``'s binarised counts as a float matrix for the dot product.

    Raises:
        ValueError: from ``block_matrix``, when the block is unknown or the table lacks its columns.
    """
    index = {key: row for row, key in enumerate(substrates["key_stereo"])}
    # The conversion lives in one place (ADR 018); binarising is this caller's own choice, because
    # Tanimoto is about which environments occur and not how often.
    return index, (block_matrix(substrates, block) > 0).astype(np.float32)


def _side_similarity(
    scored: pd.Series, positive: pd.Series, index: dict[str, int], bits: np.ndarray
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    """Tanimoto between the distinct scored and the distinct positive substrates of one side.

    Only the distinct substrates are compared: the lab holds 529 acids and 538 amines against about
    11,000 and 10,000 in the positives, so the matrix stays small and the pair reduction is left to
    the caller.

    The result is **float32**, and deliberately not widened. Every step above is exact in float32:
    the bits are 0 or 1 and the inner dimension is 2,048, so each dot product is an integer well
    below 2^24, and ``union`` is an integer too. The single division is the only rounding in the
    path. B2 then applies ``minimum`` and ``max``, which select rather than compute, so its answer
    is unchanged; B3 adds, so it must accumulate into a float64 array, and it does.
    """
    scored_keys, positive_keys = pd.unique(scored), pd.unique(positive)
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
