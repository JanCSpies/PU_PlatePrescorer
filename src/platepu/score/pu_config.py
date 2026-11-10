"""The stage S2 config record, its file and its hash (docs/design/modeling.md; ADR 016).

Its own module because both sub-steps read the same config: S2a cuts the reliable negatives from it
(``score/pu.py``) and S2b fits the final model under it, and because a config that cannot mean what
it says should be refused once, where it is built, rather than in each step that uses it.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from platepu.features.blocks import FINGERPRINT_BLOCKS

# The one implemented learner (ADR 013 item 3, ADR 016 item 3); `learner` records it, it is not a
# branch, and any other value raises.
LEARNER = "HistGradientBoostingClassifier"
# Out of fold is primary, in sample is PAYN's choice kept as a measurement (ADR 016 item 1).
SCORINGS = ("out_of_fold", "in_sample")


@dataclass(frozen=True)
class PuConfig:
    """The choices stage S2 reads from ``configs/models/<stem>.yml``, hashed into every row.

    Attributes:
        model: The name the score rows carry, e.g. ``pair_hgb``.
        feature_block: ``morgan_centre`` (primary, ADR 014 item 2) or ``morgan``. Nothing else:
            ADR 008 item 1 keeps hand-written chemistry in the baselines, so the descriptors, the
            amine subclasses and ``n_heavy_atoms`` are out of the model. Each block gets its own
            full run, spies included, because the reliable negatives depend on the block.
        p_width: The ``family`` of positives to read; ``all`` is the class width (ADR 008 item 8).
        unlabeled: The stage U config stem whose table is used, recorded so a negatives table names
            the U it was cut from.
        spy_fraction: Share of P hidden in U. **Measured, never inherited** (ADR 026 item 5): PAYN's
            0.2 was tuned at P:U = 1:1 and ours is about 1:9.7.
        spy_recall: The share of spies the threshold must keep above it, 0.95. It is recall on
            reported positives; see the module docstring for what that does and does not mean.
        spy_scoring: ``out_of_fold`` or ``in_sample``. The second is a second config file and not a
            code path (ADR 016 item 1); the report carries both thresholds either way, so the cost
            of PAYN's choice is a number.
        n_folds: Folds of the out-of-fold scoring, 5.
        learner: Must be ``LEARNER``.
        learner_params: Every learner argument, so the config hash records it. ``max_iter`` must be
            given and ``early_stopping`` written out as ``false``: above 10,000 rows scikit-learn
            turns early stopping on by itself and carves an internal validation split with its own
            seed, which §5 calls hidden state (ADR 016 item 3).
        seed: One stream for the spy draw, the folds and the learner, stamped on every row.
    """

    model: str
    feature_block: str
    p_width: str
    unlabeled: str
    spy_fraction: float
    spy_recall: float
    spy_scoring: str
    n_folds: int
    learner: str
    learner_params: Mapping[str, Any]
    seed: int

    def __post_init__(self) -> None:
        """Refuse a config that cannot mean what it says, where it is built rather than where it
        bites: a degenerate fraction, an unimplemented choice, or a learner left to tune itself."""
        for name in ("spy_fraction", "spy_recall"):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(
                    f"{name} must be in the open interval (0, 1), not {value}; 0 hides no spy "
                    f"and 1 leaves no positive to learn from"
                )
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be at least 2, not {self.n_folds}")
        if self.spy_scoring not in SCORINGS:
            raise ValueError(
                f"spy_scoring must be one of {list(SCORINGS)}, not {self.spy_scoring!r}"
            )
        if self.feature_block not in FINGERPRINT_BLOCKS:
            raise ValueError(
                f"feature_block {self.feature_block!r} is not a fingerprint, and a PU model may "
                f"see nothing else (ADR 008 item 1); use one of {list(FINGERPRINT_BLOCKS)}"
            )
        if self.learner != LEARNER:
            raise ValueError(
                f"learner {self.learner!r} is not implemented; only {LEARNER!r} is, and a second "
                f"learner arrives with its own measurement and its own record (ADR 016 item 3)"
            )
        check_learner_params(self.learner_params)


def check_learner_params(params: Mapping[str, Any]) -> None:
    """The two parameters ADR 016 item 3 requires in writing, refused where they are read.

    A module function and not a method, because both config records need the identical two refusals
    and §5 wants one copy of a rule.
    """
    if params.get("max_iter") is None:
        raise ValueError(
            "learner_params must fix max_iter, so the number of boosting rounds is in the "
            "config and in its hash"
        )
    if params.get("early_stopping") is not False:
        raise ValueError(
            "learner_params must write early_stopping out as false (ADR 016 item 3); left to "
            "itself scikit-learn turns it on above 10,000 rows and splits with its own seed"
        )


def load_pu_config(path: Path) -> PuConfig:
    """Read a stage S2 YAML; a missing, unknown or wrongly typed key raises, naming the file.

    An **unknown** key raises rather than being ignored, which is the one refusal here that is not
    about types: every field below is required, so a typo cannot silently fall back to a default --
    but ``spy_recal: 0.90`` beside a correct ``spy_recall`` would have been read as the correct one
    and the typo dropped, and the run would be a different run than the file says.
    """
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        unknown = sorted(set(config) - set(PuConfig.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown key(s) {unknown}")
        return PuConfig(
            model=_text(config, "model"),
            feature_block=_text(config, "feature_block"),
            p_width=_text(config, "p_width"),
            unlabeled=_text(config, "unlabeled"),
            spy_fraction=float(config["spy_fraction"]),
            spy_recall=float(config["spy_recall"]),
            spy_scoring=_text(config, "spy_scoring"),
            n_folds=_whole(config, "n_folds"),
            learner=_text(config, "learner"),
            learner_params=dict(config["learner_params"]),
            seed=_whole(config, "seed"),
        )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} is not a stage S2 config: {error}") from error


def _text(config: dict[str, object], key: str) -> str:
    """A non-empty string. ``str`` alone would turn a blank YAML value into the word ``"None"``,
    and ``model: `` would then stamp ``"None"`` on every score row."""
    value = config[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string, not {value!r}")
    if not value.strip():
        raise ValueError(f"{key} must not be blank")
    return value


def _whole(config: dict[str, object], key: str) -> int:
    """A whole number. ``int`` alone truncates, so ``n_folds: 7.9`` would run 7 folds silently.

    ``bool`` is excluded explicitly because it is an ``int`` in Python, and ``seed: true`` would
    otherwise be read as the seed 1. ``TypeError`` rather than ``ValueError``: the loader turns
    both into one message naming the file.
    """
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be a whole number, not {value!r}")
    return value


# Quoted because `PuScoreConfig` is defined below: both records are hashed the same way, through
# `asdict`, so one function serves stage S2a and stage S2b.
def config_sha256(config: "PuConfig | PuScoreConfig") -> str:
    """The hash every score row carries, taken from the config's **values**.

    ``io.sha256`` hashes a config *file*, and stage S1 writes that. This one hashes the record, for
    one reason: the value that was used is what a row must be traceable to, and a comment or a
    reordered key changes the file's bytes without changing the run. The two readings differ and the
    column must mean one thing, so stage S2b takes its hash from here as well.
    """
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# The two arms of stage S2b (ADR 028 decision 4). They carry different `model` values because the
# stage S layout promises that (model, feature block, scored pair) identifies a row, so stage E
# needs no lookup of which arm a row came from.
ARMS = ("pair_hgb_spy", "pair_hgb_naive")


@dataclass(frozen=True)
class PuScoreConfig:
    """The choices stage S2b reads from ``configs/models/<stem>.yml`` (ADR 028).

    Its own record and not ``PuConfig``, because the naive arm needs none of ``spy_fraction``,
    ``spy_recall``, ``spy_scoring`` or ``n_folds``: those are properties of an S2a run, and a file
    carrying four keys that describe a run it is not would make a reader believe spies were drawn.
    No key here is inert in either arm.

    Attributes:
        model: ``pair_hgb_spy`` or ``pair_hgb_naive``, which is also the arm.
        feature_block: ``morgan_centre`` (primary, ADR 014 item 2) or ``morgan``, nothing else.
        p_width: The ``family`` of positives to train on; ``all`` is the class width.
        unlabeled: The stage U stem. In the naive arm it is the table read as the negative class; in
            the spy arm it is the assertion that the negatives were cut from that U, checked against
            the S2a config by ``pu_scores.check_negatives_pointer``.
        negatives: **The arm.** A stem names the S2a config whose ``<stem>.negatives.parquet``
            supplies the negative class; ``None`` means no such table -- train against all of U.
            Null for "this run read none" is what stage S1 already does with ``feature_block`` on a
            baseline that reads no fingerprint. An empty string is refused: it is not "no table".
        learner: Must be ``LEARNER``.
        learner_params: Every learner argument, under the same two refusals as ``PuConfig``.
        seed: One stream for the fit, stamped on every score row.
    """

    model: str
    feature_block: str
    p_width: str
    unlabeled: str
    negatives: str | None
    learner: str
    learner_params: Mapping[str, Any]
    seed: int

    def __post_init__(self) -> None:
        """Refuse a config that cannot mean what it says, where it is built."""
        if self.model not in ARMS:
            raise ValueError(f"model must be one of {list(ARMS)}, not {self.model!r}")
        if self.feature_block not in FINGERPRINT_BLOCKS:
            raise ValueError(
                f"feature_block {self.feature_block!r} is not a fingerprint, and a PU model may "
                f"see nothing else (ADR 008 item 1); use one of {list(FINGERPRINT_BLOCKS)}"
            )
        if self.learner != LEARNER:
            raise ValueError(
                f"learner {self.learner!r} is not implemented; only {LEARNER!r} is (ADR 016 item 3)"
            )
        if self.negatives is not None and not self.negatives.strip():
            raise ValueError(
                "negatives must be a config stem or null, never blank: null is the naive arm, and "
                "an empty string says nothing"
            )
        check_learner_params(self.learner_params)


def load_pu_score_config(path: Path) -> PuScoreConfig:
    """Read a stage S2b YAML; a missing, unknown or wrongly typed key raises, naming the file.

    ``negatives`` is the one key that may be null, and null is the naive arm. Every other key is
    required in both arms, so a typo cannot fall back to a default -- and a key from an S2a file,
    ``spy_fraction`` above all, is refused as unknown rather than ignored.
    """
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        unknown = sorted(set(config) - set(PuScoreConfig.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown key(s) {unknown}")
        negatives = config["negatives"]
        return PuScoreConfig(
            model=_text(config, "model"),
            feature_block=_text(config, "feature_block"),
            p_width=_text(config, "p_width"),
            unlabeled=_text(config, "unlabeled"),
            negatives=None if negatives is None else _text(config, "negatives"),
            learner=_text(config, "learner"),
            learner_params=dict(config["learner_params"]),
            seed=_whole(config, "seed"),
        )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} is not a stage S2b config: {error}") from error
