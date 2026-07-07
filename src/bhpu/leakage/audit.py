"""Audit and remove leakage between literature training data and the test set.

The JnJ test set is an **out-of-distribution substrate** benchmark: a literature
reaction leaks if it re-uses a test *substrate* (aryl halide or amine), even under
different conditions. This module computes substrate/pair/reaction overlaps on
canonical InChIKey identities and can filter offending rows before PU training.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel

from bhpu.chem.canonical import inchikey, strip_and_canonicalise
from bhpu.data import schema

#: Granularity at which a literature row is considered leaked.
FilterLevel = Literal["substrate", "pair"]


def _substrate_key(smiles: object) -> str | None:
    """Return a salt-stripped connectivity InChIKey identity for a substrate SMILES."""
    if not isinstance(smiles, str) or not smiles:
        return None
    cleaned = strip_and_canonicalise(smiles) or smiles
    return inchikey(cleaned)


class LeakageReport(BaseModel):
    """Overlap statistics between a literature frame and the test set.

    Attributes:
        n_literature_rows: Rows audited.
        n_test_aryls: Distinct test aryl-halide identities.
        n_test_amines: Distinct test amine identities.
        n_shared_aryls: Literature aryls also present in the test set.
        n_shared_amines: Literature amines also present in the test set.
        n_rows_sharing_any_substrate: Literature rows re-using a test aryl or amine.
        n_rows_sharing_pair: Literature rows re-using a full test (aryl, amine) pair.
        n_unparsable_rows: Rows with an aryl/amine SMILES that could not be keyed.
    """

    n_literature_rows: int
    n_test_aryls: int
    n_test_amines: int
    n_shared_aryls: int
    n_shared_amines: int
    n_rows_sharing_any_substrate: int
    n_rows_sharing_pair: int
    n_unparsable_rows: int


class LeakageAuditor:
    """Detect and remove literature/test leakage on canonical substrate identities.

    The test-set substrate and pair identities are precomputed once at construction;
    :meth:`audit` and :meth:`filter` can then be called for any number of candidate
    literature frames.
    """

    def __init__(self, test_frame: pd.DataFrame) -> None:
        """Precompute test-set substrate identities.

        Args:
            test_frame: The held-out test set (e.g. the JnJ partition).
        """
        aryl_keys = [_substrate_key(s) for s in test_frame[schema.ARYL_SMILES]]
        amine_keys = [_substrate_key(s) for s in test_frame[schema.AMINE_SMILES]]
        self._test_aryls: frozenset[str] = frozenset(k for k in aryl_keys if k is not None)
        self._test_amines: frozenset[str] = frozenset(k for k in amine_keys if k is not None)
        self._test_pairs: frozenset[tuple[str, str]] = frozenset(
            (a, n)
            for a, n in zip(aryl_keys, amine_keys, strict=True)
            if a is not None and n is not None
        )

    def _row_keys(self, frame: pd.DataFrame) -> tuple[list[str | None], list[str | None]]:
        """Return (aryl_keys, amine_keys) aligned to ``frame`` rows."""
        aryl_keys = [_substrate_key(s) for s in frame[schema.ARYL_SMILES]]
        amine_keys = [_substrate_key(s) for s in frame[schema.AMINE_SMILES]]
        return aryl_keys, amine_keys

    def audit(self, literature_frame: pd.DataFrame) -> LeakageReport:
        """Compute overlap statistics for ``literature_frame`` against the test set.

        Args:
            literature_frame: Candidate literature/HTE reactions.

        Returns:
            A populated :class:`LeakageReport`.
        """
        aryl_keys, amine_keys = self._row_keys(literature_frame)
        shared_aryls = {k for k in aryl_keys if k is not None and k in self._test_aryls}
        shared_amines = {k for k in amine_keys if k is not None and k in self._test_amines}

        rows_any_substrate = 0
        rows_pair = 0
        unparsable = 0
        for aryl, amine in zip(aryl_keys, amine_keys, strict=True):
            if aryl is None or amine is None:
                unparsable += 1
                continue
            if aryl in self._test_aryls or amine in self._test_amines:
                rows_any_substrate += 1
            if (aryl, amine) in self._test_pairs:
                rows_pair += 1

        return LeakageReport(
            n_literature_rows=len(literature_frame),
            n_test_aryls=len(self._test_aryls),
            n_test_amines=len(self._test_amines),
            n_shared_aryls=len(shared_aryls),
            n_shared_amines=len(shared_amines),
            n_rows_sharing_any_substrate=rows_any_substrate,
            n_rows_sharing_pair=rows_pair,
            n_unparsable_rows=unparsable,
        )

    def filter(
        self, literature_frame: pd.DataFrame, *, level: FilterLevel = "substrate"
    ) -> pd.DataFrame:
        """Return ``literature_frame`` with leaked rows removed.

        Args:
            literature_frame: Candidate literature reactions.
            level: ``"substrate"`` (strict; drop any row re-using a test aryl OR amine —
                correct for substrate-OOD evaluation) or ``"pair"`` (lenient; drop only
                rows re-using a full test (aryl, amine) pair).

        Returns:
            A filtered copy of ``literature_frame`` (unparsable rows are kept, since
            their identity vs the test set is unknown; they surface in the report).
        """
        aryl_keys, amine_keys = self._row_keys(literature_frame)
        keep_mask = []
        for aryl, amine in zip(aryl_keys, amine_keys, strict=True):
            if aryl is None or amine is None:
                keep_mask.append(True)
                continue
            if level == "substrate":
                leaked = aryl in self._test_aryls or amine in self._test_amines
            else:
                leaked = (aryl, amine) in self._test_pairs
            keep_mask.append(not leaked)
        return literature_frame.loc[keep_mask].copy()
