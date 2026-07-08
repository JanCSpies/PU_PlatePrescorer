"""PU-model extension point: the interchangeable learning algorithms and their assumptions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from pubench.registry import Registry

#: Registry of PU models, keyed by ``name``.
MODELS: Registry[PUModel] = Registry("model")


@dataclass(frozen=True, slots=True)
class Requires:
    """Declared side-information / assumptions of a PU algorithm.

    Surfaced in the comparison table so results are read alongside what each method needs and
    what it assumes about the data-generating process.

    Attributes:
    ----------
    needs_prior:
        The method needs the positive class prior ``pi`` (estimated or supplied).
    needs_propensity:
        The method fits/uses a reportability (labeling) propensity model.
    assumes_fixed_negatives:
        The method assumes the negative class-conditional is invariant across P and U/target.
    handles_dd_shift:
        The method is designed to remain valid under distribution shift of *both* the target
        positive and target negative populations (the setting of this study).
    """

    needs_prior: bool = False
    needs_propensity: bool = False
    assumes_fixed_negatives: bool = False
    handles_dd_shift: bool = False


@dataclass(frozen=True, slots=True)
class LabeledArrays:
    """A featurised, labelled slice (e.g. the validation split) passed to :meth:`PUModel.fit`."""

    X: np.ndarray
    y: np.ndarray


class PUModel(ABC):
    """Abstract base class for an interchangeable PU-learning algorithm.

    Subclasses declare a class-level ``name`` and ``requires`` and register with :data:`MODELS`.
    ``fit`` receives the featurised positive (P) and unlabeled (U) matrices and, optionally, a
    labelled validation slice for internal selection; ``predict_proba`` returns ``P(y = 1 | x)``.

    Attributes:
    ----------
    name:
        Unique registry key.
    requires:
        The method's :class:`Requires` metadata.
    """

    name: ClassVar[str]
    requires: ClassVar[Requires]

    def __init__(self, *, seed: int = 0) -> None:
        """Store the global seed (subclasses extend with their own hyperparameters)."""
        self.seed = seed

    @abstractmethod
    def fit(self, P: np.ndarray, U: np.ndarray, *, val: LabeledArrays | None = None) -> PUModel:
        """Fit the model on positives ``P`` and unlabeled ``U``.

        Parameters
        ----------
        P:
            Feature matrix of the positive set (all rows are PU-positive).
        U:
            Feature matrix of the unlabeled pool.
        val:
            Optional labelled validation slice for internal selection / early stopping.

        Returns:
        -------
        PUModel
            ``self``, fitted.
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``P(y = 1 | x)`` for each row of ``X`` as a 1-D array in ``[0, 1]``."""


def build_model(name: str, params: dict[str, Any] | None = None, *, seed: int = 0) -> PUModel:
    """Instantiate a registered PU model by name.

    Parameters
    ----------
    name:
        Registry key (e.g. ``"supervised"``, ``"nnpu"``).
    params:
        Keyword arguments forwarded to the model constructor.
    seed:
        Global seed, forwarded as ``seed=`` unless overridden in ``params``.

    Returns:
    -------
    PUModel
        The constructed model.
    """
    kwargs = {"seed": seed, **(params or {})}
    return MODELS.get(name)(**kwargs)
