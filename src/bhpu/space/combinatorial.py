"""Enumerate the combinatorial aryl × amine × condition space (the unlabeled set).

The JnJ test set measured only 888 of the 110 × 68 = 7,480 possible substrate pairs.
For plate design we score the *whole* grid under the general conditions; the
unmeasured wells are the unlabeled set under the PU premise. This module enumerates
that grid, optionally excludes what the test set already measured (to avoid leakage),
and synthesises the expected C–N product so DRFP featurisation works on new wells.
"""

from __future__ import annotations

import itertools
from typing import Literal

import pandas as pd

from bhpu.chem.reaction import build_cn_coupling_product
from bhpu.data import schema
from bhpu.data.schema import ConditionSet

#: How to treat wells the test set already measured.
ExcludeMode = Literal["measured_pairs", "measured_rows", "none"]


class CombinatorialSpaceGenerator:
    """Enumerate the aryl × amine × condition grid from a reference (test) frame.

    Substrates and condition-reagent SMILES are read from the reference frame so the
    generated wells use identical chemistry to the benchmark.

    Attributes:
        condition_sets: Named (catalyst, base, solvent) sets to enumerate per pair.
        add_products: Whether to synthesise the BH product SMILES for each well.
    """

    def __init__(
        self,
        reference_frame: pd.DataFrame,
        *,
        condition_sets: tuple[ConditionSet, ...] = schema.GENERAL_CONDITIONS,
        add_products: bool = True,
    ) -> None:
        """Initialise the generator from a reference frame.

        Args:
            reference_frame: Frame supplying the substrate universe and the name→SMILES
                lookups for reagents (typically the JnJ test partition).
            condition_sets: Condition triples to pair with every substrate combination.
            add_products: If ``True``, populate ``Product SMILES`` via the BH template.
        """
        self._reference = reference_frame
        self.condition_sets = condition_sets
        self.add_products = add_products
        self._aryls: list[str] = sorted(reference_frame[schema.ARYL_SMILES].dropna().unique())
        self._amines: list[str] = sorted(reference_frame[schema.AMINE_SMILES].dropna().unique())
        self._name_to_smiles = self._build_reagent_lookup(reference_frame)
        self._measured_pairs, self._measured_rows = self._index_measured(reference_frame)

    def generate(self, *, exclude: ExcludeMode = "measured_rows") -> pd.DataFrame:
        """Enumerate the grid and return it as an unlabeled DataFrame.

        Args:
            exclude: ``"measured_rows"`` drops enumerated wells whose exact
                (aryl, amine, catalyst, base, solvent) was measured in the reference;
                ``"measured_pairs"`` drops every well of any measured (aryl, amine)
                pair (strict, substrate-disjoint from the measured design);
                ``"none"`` keeps the full grid.

        Returns:
            A DataFrame with the standard component columns, ``Product SMILES`` (if
            ``add_products``), and a boolean ``is_measured_pair`` column.

        Raises:
            KeyError: If a configured condition set names a reagent absent from the
                reference frame (so its SMILES cannot be resolved).
        """
        records: list[dict[str, object]] = []
        for aryl, amine in itertools.product(self._aryls, self._amines):
            pair_measured = (aryl, amine) in self._measured_pairs
            if exclude == "measured_pairs" and pair_measured:
                continue
            product = build_cn_coupling_product(aryl, amine) if self.add_products else None
            for condition in self.condition_sets:
                if (
                    exclude == "measured_rows"
                    and (aryl, amine, *condition.as_tuple()) in self._measured_rows
                ):
                    continue
                records.append(self._make_record(aryl, amine, condition, product, pair_measured))
        return pd.DataFrame.from_records(records)

    # -- Internal helpers ---------------------------------------------------

    def _make_record(
        self,
        aryl: str,
        amine: str,
        condition: ConditionSet,
        product: str | None,
        pair_measured: bool,
    ) -> dict[str, object]:
        """Assemble one enumerated-well record with resolved reagent SMILES."""
        catalyst, base, solvent = condition.as_tuple()
        return {
            schema.ARYL_SMILES: aryl,
            schema.AMINE_SMILES: amine,
            schema.CATALYST_NAME: catalyst,
            schema.CATALYST_SMILES: self._name_to_smiles[(schema.CATALYST_NAME, catalyst)],
            schema.BASE_NAME: base,
            schema.BASE_SMILES: self._name_to_smiles[(schema.BASE_NAME, base)],
            schema.SOLVENT_NAME: solvent,
            schema.SOLVENT_SMILES: self._name_to_smiles[(schema.SOLVENT_NAME, solvent)],
            schema.PRODUCT_SMILES: product,
            "is_measured_pair": pair_measured,
        }

    @staticmethod
    def _build_reagent_lookup(frame: pd.DataFrame) -> dict[tuple[str, str], str]:
        """Map each (name-column, name) to a representative reagent SMILES."""
        lookup: dict[tuple[str, str], str] = {}
        for name_col, smiles_col in (
            (schema.CATALYST_NAME, schema.CATALYST_SMILES),
            (schema.BASE_NAME, schema.BASE_SMILES),
            (schema.SOLVENT_NAME, schema.SOLVENT_SMILES),
        ):
            pairs = frame[[name_col, smiles_col]].dropna().drop_duplicates(subset=[name_col])
            for name, smiles in zip(pairs[name_col], pairs[smiles_col], strict=True):
                lookup[(name_col, str(name))] = str(smiles)
        return lookup

    @staticmethod
    def _index_measured(
        frame: pd.DataFrame,
    ) -> tuple[set[tuple[str, str]], set[tuple[str, str, str, str, str]]]:
        """Return the sets of measured (aryl, amine) pairs and full-condition rows."""
        pairs = set(zip(frame[schema.ARYL_SMILES], frame[schema.AMINE_SMILES], strict=True))
        rows = set(
            zip(
                frame[schema.ARYL_SMILES],
                frame[schema.AMINE_SMILES],
                frame[schema.CATALYST_NAME],
                frame[schema.BASE_NAME],
                frame[schema.SOLVENT_NAME],
                strict=True,
            )
        )
        return pairs, rows
