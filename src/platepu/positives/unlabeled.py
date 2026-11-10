"""Stage U: the literature-built unlabeled space (docs/design/modeling.md, ADR 013 and ADR 015).

U is drawn from the substrates of the literature positives, never from the inventory (ADR 008
item 4). The class-wide cross-product is 114,956,737 pairs against 28,158 positives, so U is
sampled, and how it is sampled is the whole design: two shortcuts had to be closed, and both were
measured first.

**Popularity.** 67.4 % of acids and 67.3 % of amines appear in exactly one reported pair. Drawn
uniformly, a substrate reported 321 times sits in U about as often as one reported once, so a
classifier separates P from U by how often a molecule gets reported rather than by whether the pair
fits. Drawing replacements in proportion to P frequency removes it: a probe seeing one side's
frequency scores 0.497.

**The pair structure.** The literature is disassortative in size, ``corr(size_a, size_b) = -0.388``,
because a complex fragment is coupled to a simple building block. Replacing both sides **from the
marginals** destroys that and lets two numbers -- the heavy-atom counts -- separate P from U at
0.640, above B2 and B7. So the default replaces one side at a time, from the same heavy-atom bin.

Neither makes U bias-free and nothing can: a U matching P's joint structure in every respect would
*be* P. These are the shortcuts that could be named and measured; the residual is reported as a
diagnostic and never gated on (ADR 017).

**Two rules, and the difference between them is an experiment.** ``size_stratified_one_side`` keeps
one side of the template as an **anchor**; ``size_stratified_both_sides`` replaces both, each from
the decile of the side it replaces. The second is *not* the sampler the 0.640 above rejected: that
figure came from drawing both sides from the marginals with no size binning at all. Keeping each
replacement inside its own decile keeps the joint size structure; what it removes is the anchor.

The anchor is the open question. A model trained against the anchored U can learn "this acid goes
with amines like *that* one", which is chemistry, or publishing habit, and the two cannot be told
apart by assertion. Chen et al. settled the same question for the DUD-E decoy set by rebuilding the
negatives without the construction under suspicion and measuring what changed. The unanchored rule
exists for that comparison and for nothing else; it is not a candidate for production.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from platepu.io import strictest_licence

ONE_SIDE = "size_stratified_one_side"
BOTH_SIDES = "size_stratified_both_sides"
# Every rule implemented here. A config naming anything else is refused with this list, so a reader
# learns the options from the error rather than from the source.
MATCH = (ONE_SIDE, BOTH_SIDES)
SUBTRACT_ON = "key_flat"
# The pair columns U carries, in output order.
SIDES = (("a", "key_a_stereo", "key_a_flat"), ("b", "key_b_stereo", "key_b_flat"))
# ``n_draws`` is int16. A both-sides draw concentrates more mass on popular x popular pairs, so the
# ceiling is nearer than it is for the anchored rule, and a silent wrap would be invisible.
MAX_DRAWS = int(np.iinfo(np.int16).max)


@dataclass(frozen=True)
class UnlabeledConfig:
    """The choices stage U reads from ``configs/models/<stem>.yml``.

    Attributes:
        k: Counter-pairs drawn per positive pair. It *is* the P:U ratio, because the marginals are
            matched; there is no second knob.
        seed: The draw's stream, stamped on the report so a run is reproducible from it.
        size_bins: Quantile bins of ``n_heavy_atoms``, cut over the distinct P substrates of each
            role. A replacement comes from the bin of the substrate it replaces.
        p_width: The ``family`` of the positives to read; ``"all"`` is the class width (ADR 008
            item 8).
        subtract_p_on: ``"key_flat"``. The fingerprints are chirality-blind and isotope-blind, so
            two stereo variants are one input to the model and a drawn pair that is a positive
            under another drawing is a positive. Any other value raises: subtracting on stereo
            keys is not implemented, and a config that asked for it used to run and subtract on
            flat keys anyway.
        match: The sampling rule, one of ``MATCH``. ``size_stratified_one_side`` is the production
            rule; ``size_stratified_both_sides`` exists only for the anchor ablation and is not a
            candidate for production. Anything else raises, listing both, rather than falling back.
        gate_rank_auc_tolerance: How far the rank-AUC of one side's raw P frequency may sit from
            0.5 before the run is refused.
        gate_marginal_tv_multiple: How many times the simulated null a role's P-to-U total
            variation may reach. The null is the distance a **perfect** sampler shows from sampling
            noise alone, so the gate scales with the data instead of with one dataset; an absolute
            threshold reads 0.053 on this P and 0.105 on a 40-pair fixture drawn correctly.
        gate_null_draws: Multinomials used to estimate that null; its 99th percentile is the floor.
            Both are the gate of ADR 017, and the stage raises past them: a check that reports and
            continues is not a check.
    """

    k: int
    seed: int
    size_bins: int
    p_width: str
    subtract_p_on: str
    match: str
    gate_rank_auc_tolerance: float
    gate_marginal_tv_multiple: float
    gate_null_draws: int


def load_unlabeled_config(path: Path) -> UnlabeledConfig:
    """Read a stage U YAML; a missing or wrongly typed key raises ValueError naming the file."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return UnlabeledConfig(
            k=int(config["k"]),
            seed=int(config["seed"]),
            size_bins=int(config["size_bins"]),
            p_width=str(config["p_width"]),
            subtract_p_on=str(config["subtract_p_on"]),
            match=str(config["match"]),
            gate_rank_auc_tolerance=_positive(config, "gate_rank_auc_tolerance"),
            gate_marginal_tv_multiple=_positive(config, "gate_marginal_tv_multiple"),
            gate_null_draws=int(config["gate_null_draws"]),
        )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} is not a stage U config: {error}") from error


def _positive(config: dict[str, object], key: str) -> float:
    """A gate tolerance from the config; zero or negative would reject every run with a confusing
    error, so it is refused where it is read rather than where it bites."""
    value = float(config[key])  # type: ignore[arg-type]
    if value <= 0:
        raise ValueError(f"{key} must be positive, not {value}")
    return value


def unlabeled_pairs(
    positives: pd.DataFrame,
    reported: pd.DataFrame,
    substrates: pd.DataFrame,
    config: UnlabeledConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(unlabeled, rejected)`` drawn from ``positives`` under ``config``.

    Args:
        positives: The positives table; only rows of ``config.p_width`` are used. The draw pools
            and their frequency weights come from here and nowhere else, because U's marginals are
            matched to P's and a substrate outside P would break the match the gate protects.
        reported: Every pair the class reported, whatever its yield (``build_reported``). All of it
            is subtracted, not only the part that became P: since the yield cut-off, 15,641 pairs
            of this class are reported with no usable yield and 1,121 are known to fail, and a
            reported reaction drawn as unlabeled is the contamination this subtraction exists to
            prevent.
        substrates: The stage F table, read for ``key_stereo`` and ``n_heavy_atoms`` alone. It also
            holds inventory rows, which never reach U because the draw pools are built from the
            positives' own substrates.
        config: The sampling rule.

    Returns:
        The unlabeled pairs, one row per distinct pair with ``n_draws`` as its weight, and one
        rejected row per rejected draw.

    Raises:
        ValueError: the positives mix ``std_version``; ``config.match`` or
            ``config.subtract_p_on`` is not implemented; a positive's substrate is missing from
            ``substrates``; ``reported`` does not contain every positive; or a role's sizes cannot
            be binned.
    """
    if config.match not in MATCH:
        raise ValueError(
            f"match {config.match!r} is not implemented; expected one of {sorted(MATCH)}"
        )
    if config.subtract_p_on != SUBTRACT_ON:
        raise ValueError(
            f"subtract_p_on {config.subtract_p_on!r} is not implemented; "
            f"expected {SUBTRACT_ON!r}"
        )
    kept = positives[positives["family"] == config.p_width]
    versions = sorted(set(kept["std_version"]))
    if len(versions) > 1:
        raise ValueError(f"the positives mix std_version values {versions}")

    # A reported table that has lost positives would subtract less than it should, and the draw
    # would look healthy while seeding U with known reactions. Caught here, not downstream.
    stereo_pairs = set(zip(kept["key_a_stereo"], kept["key_b_stereo"], strict=True))
    absent = stereo_pairs - set(
        zip(reported["key_a_stereo"], reported["key_b_stereo"], strict=True)
    )
    if absent:
        raise ValueError(
            f"the reported table is missing {len(absent)} of the "
            f"{len(stereo_pairs)} positive pairs"
        )

    size = substrates.set_index("key_stereo")["n_heavy_atoms"]
    pools = {side: _pools(kept[key], size, config.size_bins) for side, key, _ in SIDES}
    known = set(zip(reported["key_a_flat"], reported["key_b_flat"], strict=True))
    flat = {
        side: dict(zip(kept[key], kept[flat_key], strict=True))
        for side, key, flat_key in SIDES
    }
    smiles = {
        side: dict(zip(kept[key], kept[f"substrate_{side}"], strict=True))
        for side, key, _ in SIDES
    }

    drawn, rejects = _draw_pairs(kept, pools, flat, known, config)
    return (
        _table(drawn, flat, smiles, kept),
        _rejected(rejects, flat),
    )


def _draw_pairs(
    kept: pd.DataFrame,
    pools: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    flat: dict[str, dict[str, str]],
    known: set[tuple[str, str]],
    config: UnlabeledConfig,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """``config.k`` draws against every positive pair, split into kept and already reported.

    One pass, so the generator is consumed in one order and the seed reproduces the run exactly.
    """
    generator = np.random.default_rng(config.seed)
    drawn: list[tuple[str, str]] = []
    rejects: list[tuple[str, str]] = []
    acids = kept["key_a_stereo"].to_numpy()
    amines = kept["key_b_stereo"].to_numpy()
    unanchored = config.match == BOTH_SIDES
    for row in range(len(kept)):
        for draw in range(config.k):
            pair = _counter_pair(
                acids[row], amines[row], pools, draw, unanchored, generator
            )
            target = (
                rejects if (flat["a"][pair[0]], flat["b"][pair[1]]) in known else drawn
            )
            target.append(pair)
    return drawn, rejects


def _counter_pair(
    acid: str,
    amine: str,
    pools: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    draw: int,
    unanchored: bool,
    generator: np.random.Generator,
) -> tuple[str, str]:
    """The pair drawn against one template pair on draw number ``draw``.

    Anchored: half the draws perturb the amine and half the acid, so each anchor keeps its own
    frequency exactly and the other side is replaced within its size bin. Unanchored: both sides are
    replaced on every draw, each from the decile of the side it replaces, so the template contributes
    its two deciles and nothing else. The acid is drawn before the amine in the unanchored branch,
    which is the order the seeded runs in the journal were produced in.
    """
    if unanchored:
        return (_draw(pools["a"], acid, generator), _draw(pools["b"], amine, generator))
    if draw % 2 == 0:
        return (acid, _draw(pools["b"], amine, generator))
    return (_draw(pools["a"], acid, generator), amine)


def _pools(
    keys: pd.Series, size: pd.Series, bins: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """For each substrate, the pool it may be replaced from and that pool's P-frequency weights.

    The bins are quantiles of ``n_heavy_atoms`` over the **distinct** substrates of this role that
    appear in P, so a replacement is always a molecule of comparable size. Weights are P counts, so
    the role's marginal distribution is preserved by the draw.
    """
    counts = keys.value_counts()
    missing = sorted(set(counts.index) - set(size.index))
    if missing:
        raise ValueError(
            f"positives substrate missing from the feature table: {missing[:5]}"
        )
    heavy = size.reindex(counts.index)
    labels = pd.qcut(heavy, bins, labels=False, duplicates="drop")
    # A label is NaN when every substrate of this role shares one heavy-atom count, so all the
    # quantile edges collapse, or when one of them carries no count at all -- which the `missing`
    # check above does not see, because the substrate is in the table and only its size is absent.
    # Either way the replacement pool is undefined; without this the draw died on `KeyError: nan`.
    if labels.isna().any():
        raise ValueError(
            f"the size distribution of {keys.name} could not be binned into {bins} quantiles: "
            f"{int(labels.isna().sum())} of {len(labels)} substrates fall in no bin, either "
            "because the role has a single heavy-atom count or because n_heavy_atoms is missing"
        )
    pools = {}
    for label in pd.unique(labels):
        members = counts.index[labels == label].to_numpy()
        weights = counts.reindex(members).to_numpy(dtype=float)
        pools[label] = (members, weights / weights.sum())
    return {key: pools[label] for key, label in zip(counts.index, labels, strict=True)}


def _draw(
    pools: dict[str, tuple[np.ndarray, np.ndarray]],
    key: str,
    generator: np.random.Generator,
) -> str:
    """One replacement for ``key``, from its own size bin, weighted by P frequency."""
    members, weights = pools[key]
    return str(generator.choice(members, p=weights))


def _table(
    drawn: list[tuple[str, str]],
    flat: dict[str, dict[str, str]],
    smiles: dict[str, dict[str, str]],
    kept: pd.DataFrame,
) -> pd.DataFrame:
    """The unlabeled table: one row per distinct pair, with its draws as the weight.

    Repeats are **not** dropped but counted. About 19 % of draws repeat a pair at k = 10, and
    repeats concentrate on popular substrates, so deduplicating them would pull exactly those below
    their P frequency and reopen the shortcut this stage exists to close (ADR 015 item 2).
    """
    pairs = pd.DataFrame(drawn, columns=["key_a_stereo", "key_b_stereo"])
    rows = (
        pairs.groupby(["key_a_stereo", "key_b_stereo"], sort=True)
        .size()
        .reset_index(name="n_draws")
    )
    # The cast below is to int16. Without this the count would wrap and the heaviest pair -- always
    # a popular one -- would come out small or negative, which is the kind of error that reads as a
    # chemistry result. ``rows`` is empty when every draw was rejected, which is a legitimate case.
    if not rows.empty:
        heaviest = rows.loc[rows["n_draws"].idxmax()]
        if heaviest["n_draws"] > MAX_DRAWS:
            raise ValueError(
                f"pair ({heaviest['key_a_stereo']}, {heaviest['key_b_stereo']}) was drawn "
                f"{heaviest['n_draws']} times, past the {MAX_DRAWS} an int16 holds"
            )
    return rows.assign(
        key_a_flat=rows["key_a_stereo"].map(flat["a"]),
        key_b_flat=rows["key_b_stereo"].map(flat["b"]),
        substrate_a=rows["key_a_stereo"].map(smiles["a"]),
        substrate_b=rows["key_b_stereo"].map(smiles["b"]),
        n_draws=rows["n_draws"].astype("int16"),
        licence=strictest_licence(kept["licence"]),
        std_version=kept["std_version"].iloc[0],
    )[
        [
            "key_a_stereo",
            "key_b_stereo",
            "key_a_flat",
            "key_b_flat",
            "substrate_a",
            "substrate_b",
            "n_draws",
            "licence",
            "std_version",
        ]
    ]


def _rejected(
    rejects: list[tuple[str, str]], flat: dict[str, dict[str, str]]
) -> pd.DataFrame:
    """One row per rejected draw, so the report's counts add up on draws and not on pairs."""
    pairs = pd.DataFrame(rejects, columns=["key_a_stereo", "key_b_stereo"])
    return pairs.assign(
        key_a_flat=pairs["key_a_stereo"].map(flat["a"]),
        key_b_flat=pairs["key_b_stereo"].map(flat["b"]),
        reason="pair_reported_flat",
    )
