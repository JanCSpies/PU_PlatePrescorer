"""Assemble reaction SMILES and generate Buchwald–Hartwig C–N coupling products.

Two needs:

* **DRFP featurisation** wants a reaction SMILES ``reactants>>product``. The
  :class:`ReactionAssembler` builds one from the component columns, optionally
  including the reagents (catalyst/base/solvent) so the fingerprint captures the
  *conditions* as well as the substrates.
* The **enumerated unlabeled space** has no measured product column, so
  :func:`build_cn_coupling_product` synthesises the expected C–N coupling product
  from an aryl halide and an amine via an RDKit reaction template.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from bhpu.data import schema

# Buchwald–Hartwig C–N coupling template: aryl–halide + amine N–H -> aryl–N.
# The amine nitrogen must bear at least one H (primary or secondary amine); tertiary
# amines and imine/nitrile nitrogens are excluded so they correctly yield no product.
_CN_COUPLING_SMARTS: str = "[c:1][F,Cl,Br,I].[N;H1,H2;!$(N=*);!$(N#*);!$(N-[#7,#8]):2]>>[c:1][N:2]"
_CN_COUPLING_RXN = AllChem.ReactionFromSmarts(_CN_COUPLING_SMARTS)


@lru_cache(maxsize=200_000)
def build_cn_coupling_product(aryl_halide_smiles: str, amine_smiles: str) -> str | None:
    """Generate the expected BH C–N coupling product SMILES.

    Applies an aryl-halide + amine → aryl-amine template. When several products are
    possible (multiple halides or N–H sites) the first RDKit-enumerated product is
    returned — sufficient as a representative structure for reaction fingerprinting.

    Args:
        aryl_halide_smiles: SMILES of the aryl (pseudo)halide electrophile.
        amine_smiles: SMILES of the amine nucleophile.

    Returns:
        Canonical product SMILES, or ``None`` if the template does not apply
        (e.g. a tertiary amine) or either input is unparsable.
    """
    aryl = Chem.MolFromSmiles(aryl_halide_smiles)
    amine = Chem.MolFromSmiles(amine_smiles)
    if aryl is None or amine is None:
        return None
    products = _CN_COUPLING_RXN.RunReactants((aryl, amine))
    for outcome in products:
        product = outcome[0]
        try:
            Chem.SanitizeMol(product)
        except (Chem.AtomValenceException, Chem.KekulizeException, ValueError):
            continue
        product_smiles: str = Chem.MolToSmiles(product)
        return product_smiles
    return None


@lru_cache(maxsize=200_000)
def coupling_site_indices(
    aryl_halide_smiles: str, amine_smiles: str
) -> tuple[int, int | None, int] | None:
    """Locate the reacting atoms of a BH C–N coupling via the reaction template.

    Runs the same atom-mapped template used to build the product, then reads back which
    aryl carbon (SMARTS map ``1``) and amine nitrogen (map ``2``) formed the new C–N bond
    from RDKit's ``react_atom_idx`` provenance. Because it reuses the product template, the
    returned sites are exactly the atoms whose coupling the DRFP/product represent, and the
    polyhalide/poly-N–H ambiguity is resolved the same way as :func:`build_cn_coupling_product`
    (the first sanitisable outcome).

    Args:
        aryl_halide_smiles: SMILES of the aryl (pseudo)halide electrophile.
        amine_smiles: SMILES of the amine nucleophile.

    Returns:
        ``(aryl_carbon_idx, leaving_halide_idx, amine_nitrogen_idx)`` as atom indices on
        freshly parsed ``aryl``/``amine`` molecules (``leaving_halide_idx`` is ``None`` for
        a pseudohalide such as a triflate whose leaving atom is not a single halogen), or
        ``None`` if the template does not apply or an input is unparsable.
    """
    aryl = Chem.MolFromSmiles(aryl_halide_smiles)
    amine = Chem.MolFromSmiles(amine_smiles)
    if aryl is None or amine is None:
        return None
    for outcome in _CN_COUPLING_RXN.RunReactants((aryl, amine)):
        product = outcome[0]
        try:
            Chem.SanitizeMol(product)
        except (Chem.AtomValenceException, Chem.KekulizeException, ValueError):
            continue
        carbon_idx: int | None = None
        nitrogen_idx: int | None = None
        for atom in product.GetAtoms():
            mapno = atom.GetPropsAsDict().get("old_mapno")
            if not atom.HasProp("react_atom_idx"):
                continue
            if mapno == 1:
                carbon_idx = atom.GetIntProp("react_atom_idx")
            elif mapno == 2:
                nitrogen_idx = atom.GetIntProp("react_atom_idx")
        if carbon_idx is None or nitrogen_idx is None:
            continue
        leaving_idx = next(
            (
                neighbour.GetIdx()
                for neighbour in aryl.GetAtomWithIdx(carbon_idx).GetNeighbors()
                if neighbour.GetSymbol() in ("F", "Cl", "Br", "I")
            ),
            None,
        )
        return carbon_idx, leaving_idx, nitrogen_idx
    return None


@lru_cache(maxsize=200_000)
def enumerate_cn_coupling_products(aryl_halide_smiles: str, amine_smiles: str) -> frozenset[str]:
    """Enumerate *all* distinct BH C–N coupling products of an aryl halide and amine.

    Unlike :func:`build_cn_coupling_product` (which returns one representative product),
    this returns every sanitisable product across all halide positions and N–H sites. It
    is used by the reaction miner to *confirm* that a candidate (aryl halide, amine) pair
    truly produced one of a literature reaction's recorded products — a high-precision
    detector of aryl C–N bond formation.

    Args:
        aryl_halide_smiles: SMILES of the aryl (pseudo)halide electrophile.
        amine_smiles: SMILES of the amine nucleophile.

    Returns:
        A frozenset of canonical product SMILES (empty if the template does not apply or
        either input is unparsable).
    """
    aryl = Chem.MolFromSmiles(aryl_halide_smiles)
    amine = Chem.MolFromSmiles(amine_smiles)
    if aryl is None or amine is None:
        return frozenset()
    outcomes = _CN_COUPLING_RXN.RunReactants((aryl, amine))
    products: set[str] = set()
    for outcome in outcomes:
        product = outcome[0]
        try:
            Chem.SanitizeMol(product)
        except (Chem.AtomValenceException, Chem.KekulizeException, ValueError):
            continue
        products.add(Chem.MolToSmiles(product))
    return frozenset(products)


class ReactionAssembler:
    """Build reaction SMILES (``reactants>>product``) from component columns.

    Attributes:
        include_reagents: If ``True``, append catalyst/base/solvent to the reactant
            side so a downstream reaction fingerprint encodes the conditions too.
    """

    def __init__(self, *, include_reagents: bool = True) -> None:
        """Initialise the assembler.

        Args:
            include_reagents: Whether to include catalyst/base/solvent SMILES on the
                reactant side of the assembled reaction.
        """
        self.include_reagents = include_reagents

    def _reactant_columns(self) -> tuple[str, ...]:
        """Return the component columns used on the reactant side."""
        if self.include_reagents:
            return (schema.ARYL_SMILES, schema.AMINE_SMILES, *schema.CONDITION_SMILES_COLUMNS)
        return (schema.ARYL_SMILES, schema.AMINE_SMILES)

    def assemble_row(self, row: pd.Series) -> str | None:
        """Assemble a single reaction SMILES from a DataFrame row.

        Args:
            row: A row exposing the component SMILES columns and ``Product SMILES``.

        Returns:
            ``"reactant1.reactant2...>>product"``, or ``None`` if the product or any
            required reactant is missing/empty.
        """
        reactants = self._collect(row, self._reactant_columns())
        product = row.get(schema.PRODUCT_SMILES)
        if reactants is None or not isinstance(product, str) or not product:
            return None
        return f"{'.'.join(reactants)}>>{product}"

    def assemble(self, frame: pd.DataFrame) -> list[str | None]:
        """Assemble reaction SMILES for every row of ``frame``.

        Args:
            frame: DataFrame with the component and product SMILES columns.

        Returns:
            A list of reaction SMILES (or ``None`` per row where assembly fails),
            aligned to ``frame``'s row order.
        """
        return [self.assemble_row(row) for _, row in frame.iterrows()]

    @staticmethod
    def _collect(row: pd.Series, columns: Sequence[str]) -> list[str] | None:
        """Collect non-empty component SMILES for ``columns``; ``None`` if any missing."""
        values: list[str] = []
        for column in columns:
            value = row.get(column)
            if not isinstance(value, str) or not value:
                return None
            values.append(value)
        return values
