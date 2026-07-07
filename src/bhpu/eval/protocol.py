"""Frozen, anti-cheat evaluation protocol: literature-only PU → JnJ (substrate-disjoint).

This module is the single source of truth for how we score the project. **Do not change
it to make a model look better** — that would be metric hacking. It enforces:

* **No JnJ labels in training.** Models train only on external data (literature/PU); JnJ
  is used solely for VAL (model selection) and TEST (final, touched once).
* **Substrate-disjoint holdout.** JnJ substrates are partitioned into VAL vs TEST groups;
  a reaction is VAL/TEST only if *both* its aryl and amine fall in that group (mixed
  reactions are discarded), so VAL and TEST share no aryl or amine.
* **Leakage guards.** :meth:`assert_training_clean` verifies external training data reuses
  no JnJ substrate (aryl or amine) and no JnJ (aryl, amine) pair.

The feasibility label is fixed at yield ≥ 10 % upstream (``schema.LABEL``).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from bhpu.chem.canonical import inchikey, strip_and_canonicalise
from bhpu.data import schema

#: Which substrate axis/axes must be disjoint between VAL and TEST.
HoldoutAxis = Literal["both", "amine", "aryl"]


def _substrate_key(smiles: object) -> str | None:
    """Salt-stripped connectivity InChIKey identity for a substrate SMILES."""
    if not isinstance(smiles, str) or not smiles:
        return None
    return inchikey(strip_and_canonicalise(smiles) or smiles)


class FrozenEvalProtocol:
    """Deterministic literature-only → JnJ evaluation with a substrate-disjoint split.

    Attributes:
        val_index: Positional indices of the JnJ VAL reactions (model selection).
        test_index: Positional indices of the JnJ TEST reactions (final; touch once).
        jnj_substrate_keys: All JnJ aryl+amine InChIKeys (for training leakage checks).
    """

    def __init__(
        self,
        jnj: pd.DataFrame,
        *,
        seed: int = 1234,
        val_fraction: float = 0.5,
        holdout_axis: HoldoutAxis = "both",
    ) -> None:
        """Build the frozen split from the JnJ partition.

        Args:
            jnj: The JnJ test partition (must expose the substrate + label columns).
            seed: RNG seed for the substrate partition (fixed for reproducibility).
            val_fraction: Fraction of substrates assigned to VAL (rest to TEST).
            holdout_axis: Disjointness requirement — ``"both"`` (aryl & amine),
                ``"amine"`` or ``"aryl"``.
        """
        self._jnj = jnj.reset_index(drop=True)
        self.seed = seed
        self.val_fraction = val_fraction
        self.holdout_axis = holdout_axis

        aryl_keys = np.array(
            [_substrate_key(s) for s in self._jnj[schema.ARYL_SMILES]], dtype=object
        )
        amine_keys = np.array(
            [_substrate_key(s) for s in self._jnj[schema.AMINE_SMILES]], dtype=object
        )
        self.jnj_substrate_keys: frozenset[str] = frozenset(
            k for k in [*aryl_keys, *amine_keys] if k is not None
        )
        self._jnj_pairs: frozenset[tuple[str, str]] = frozenset(
            (a, n)
            for a, n in zip(aryl_keys, amine_keys, strict=True)
            if a is not None and n is not None
        )

        aryl_val, aryl_test = self._partition(aryl_keys)
        amine_val, amine_test = self._partition(amine_keys)
        val_mask = self._membership(aryl_keys, amine_keys, aryl_val, amine_val)
        test_mask = self._membership(aryl_keys, amine_keys, aryl_test, amine_test)
        self.val_index: np.ndarray = np.flatnonzero(val_mask)
        self.test_index: np.ndarray = np.flatnonzero(test_mask)

    # -- Public API ---------------------------------------------------------

    def labels(self, subset: Literal["val", "test"]) -> np.ndarray:
        """Return the binary feasibility labels for VAL or TEST."""
        index = self.val_index if subset == "val" else self.test_index
        values: np.ndarray = self._jnj.loc[index, schema.LABEL].to_numpy().astype(int)
        return values

    def frame(self, subset: Literal["val", "test"]) -> pd.DataFrame:
        """Return the JnJ rows for VAL or TEST."""
        index = self.val_index if subset == "val" else self.test_index
        return self._jnj.loc[index].reset_index(drop=True)

    def assert_training_clean(self, training: pd.DataFrame) -> None:
        """Raise if external training data leaks a JnJ substrate or pair.

        Args:
            training: External (literature/PU) training reactions.

        Raises:
            AssertionError: If any training row reuses a JnJ aryl/amine or a JnJ pair.
        """
        aryl_keys = [_substrate_key(s) for s in training[schema.ARYL_SMILES]]
        amine_keys = [_substrate_key(s) for s in training[schema.AMINE_SMILES]]
        shared_substrate = sum(
            1
            for a, n in zip(aryl_keys, amine_keys, strict=True)
            if (a is not None and a in self.jnj_substrate_keys)
            or (n is not None and n in self.jnj_substrate_keys)
        )
        shared_pair = sum(
            1
            for a, n in zip(aryl_keys, amine_keys, strict=True)
            if a is not None and n is not None and (a, n) in self._jnj_pairs
        )
        if shared_substrate or shared_pair:
            raise AssertionError(
                f"Training data leaks JnJ: {shared_substrate} rows reuse a JnJ substrate, "
                f"{shared_pair} rows reuse a JnJ pair. Filter before training."
            )

    def summary(self) -> dict[str, object]:
        """Return a small dict describing the split (for logging/provenance)."""
        return {
            "seed": self.seed,
            "val_fraction": self.val_fraction,
            "holdout_axis": self.holdout_axis,
            "n_val": len(self.val_index),
            "n_test": len(self.test_index),
            "n_discarded_mixed": int(len(self._jnj) - len(self.val_index) - len(self.test_index)),
            "val_base_rate": float(self.labels("val").mean())
            if len(self.val_index)
            else float("nan"),
            "test_base_rate": float(self.labels("test").mean())
            if len(self.test_index)
            else float("nan"),
        }

    # -- Internal helpers ---------------------------------------------------

    def _partition(self, keys: np.ndarray) -> tuple[frozenset[str], frozenset[str]]:
        """Split the unique substrate keys deterministically into (val, test)."""
        unique = sorted({k for k in keys if k is not None})
        rng = np.random.default_rng(self.seed + len(unique))
        shuffled = list(rng.permutation(np.array(unique, dtype=object)))
        n_val = round(self.val_fraction * len(shuffled))
        return frozenset(shuffled[:n_val]), frozenset(shuffled[n_val:])

    def _membership(
        self,
        aryl_keys: np.ndarray,
        amine_keys: np.ndarray,
        aryl_set: frozenset[str],
        amine_set: frozenset[str],
    ) -> np.ndarray:
        """Boolean mask of reactions whose substrates fall in the given groups."""
        aryl_in: np.ndarray = np.array([k in aryl_set for k in aryl_keys])
        amine_in: np.ndarray = np.array([k in amine_set for k in amine_keys])
        if self.holdout_axis == "amine":
            return amine_in
        if self.holdout_axis == "aryl":
            return aryl_in
        combined: np.ndarray = aryl_in & amine_in
        return combined
