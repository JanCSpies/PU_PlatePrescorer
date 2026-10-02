"""Re-count amide-coupling condition bins on rxnutils-standardised ORDerly (exploration script).

Compares two counts on the same rows:
- raw: reactants and product from ORDerly's own columns (as in explore_condition_bins.py);
- rxnutils: reactants and product from the rxnutils reaction, after reactants2reagents.

Decisions (2026-10-02, agreed with Luka):
- The condition key always comes from ORDerly's agent_* / solvent_* columns, joined to the
  rxnutils output by row_id (= ORDerly row index). It is never rebuilt from the reaction SMILES.
- Same amide rule and same key rules as explore_condition_bins.py (Cl, [Cl-], PF6-, BF4- dropped).
- Rows rxnutils leaves without a reactant or without a product are rejected with a reason.
- A duplicate is the same reaction in the same bin: the bin key is already equal, so one bin
  counts each core_key (reactants + products) once. full_key (also reagents from the reaction
  SMILES, e.g. HCl) is reported next to it; it splits rows our key treats as equal.

Usage:
    uv run python scripts/recount_condition_bins.py ORDERLY_PARQUET RXNUTILS_OUT_TSV [--out FILE]
"""

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from explore_condition_bins import (
    AGENT_COLUMNS,
    SOLVENT_COLUMNS,
    ConditionKey,
    condition_part,
    is_amide_coupling,
)

ORDERLY_COLUMNS = [
    "reactant_000",
    "reactant_001",
    "product_000",
    *AGENT_COLUMNS,
    *SOLVENT_COLUMNS,
]
RXNUTILS_COLUMNS = [
    "row_id",
    "rsmi_nomap",
    "n_reactants",
    "n_products",
    "core_key",
    "full_key",
]
Triple = tuple[str, str, str]  # (reactant, reactant, product)


@dataclass
class Recount:
    """One counting run (raw or rxnutils).

    Attributes:
        steps: Rows left after each step, in order.
        rejected: Rows rejected before the steps, by reason.
        bins: For each condition key, the (core_key, full_key) of every reaction in it.
    """

    steps: dict[str, int] = field(default_factory=dict)
    rejected: Counter[str] = field(default_factory=Counter)
    bins: dict[ConditionKey, list[tuple[str, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def size(self, key: ConditionKey) -> int:
        """Reactions in a bin, duplicates included."""
        return len(self.bins.get(key, []))

    def unique_core(self, key: ConditionKey) -> int:
        """Reactions in a bin after removing duplicates by core_key."""
        return len({core for core, _ in self.bins.get(key, [])})

    def unique_full(self, key: ConditionKey) -> int:
        """Reactions in a bin after removing duplicates by full_key."""
        return len({full for _, full in self.bins.get(key, [])})


def load(orderly_path: Path, rxnutils_path: Path) -> pd.DataFrame:
    """Join ORDerly's columns and the rxnutils output on row_id (= ORDerly row index).

    Only rows present in both are kept (rxnutils receives every ORDerly row that has a
    mapped reaction, so normally all of them).
    """
    orderly = pd.read_parquet(orderly_path, columns=ORDERLY_COLUMNS)
    orderly.index = orderly.index.astype(str)
    rxnutils = pd.read_csv(
        rxnutils_path,
        sep="\t",
        usecols=RXNUTILS_COLUMNS,
        dtype=str,
        keep_default_na=False,
    ).set_index("row_id")
    return orderly.join(rxnutils, how="inner")


def raw_triple(row: pd.Series) -> Triple | None:
    """Two reactants and a product from ORDerly's own columns, or None if one is missing."""
    values = (row.reactant_000, row.reactant_001, row.product_000)
    if all(isinstance(value, str) for value in values):
        return values
    return None


def rxnutils_triple(rsmi_nomap: str) -> Triple | None:
    """Two reactants and one product from the rxnutils reaction (reactants>reagents>products).

    Returns None unless there are exactly two reactants and exactly one product, which is the
    same shape the raw count requires.
    """
    reactant_part, _, product_part = rsmi_nomap.split(">")
    reactants = [smiles for smiles in reactant_part.split(".") if smiles]
    products = [smiles for smiles in product_part.split(".") if smiles]
    if len(reactants) != 2 or len(products) != 1:
        return None
    return reactants[0], reactants[1], products[0]


def rxnutils_reject_reason(row: pd.Series) -> str | None:
    """Why rxnutils left a row unusable, or None. Counted, never dropped silently."""
    if row.n_reactants == "0":
        return "no reactant after rxnutils"
    if row.n_products == "0":
        return "no product after rxnutils"
    return None


def condition_key(row: pd.Series) -> ConditionKey | None:
    """(agents, solvents) from ORDerly's columns; None if empty or unparsable."""
    agents = condition_part([row[column] for column in AGENT_COLUMNS])
    solvents = condition_part([row[column] for column in SOLVENT_COLUMNS])
    if not agents or not solvents:
        return None
    return agents, solvents


def _amide_rows(
    table: pd.DataFrame, use_rxnutils: bool, result: Recount
) -> Iterator[pd.Series]:
    """Yield the amide-coupling rows, filling result.rejected and the first steps on the way."""
    n_shape = 0
    amide_rows = []
    for _, row in table.iterrows():
        if use_rxnutils and (reason := rxnutils_reject_reason(row)) is not None:
            result.rejected[reason] += 1
            continue
        triple = rxnutils_triple(row.rsmi_nomap) if use_rxnutils else raw_triple(row)
        if triple is None:
            continue
        n_shape += 1
        if is_amide_coupling(*triple):
            amide_rows.append(row)
    result.steps["two reactants + product"] = n_shape
    result.steps["amide coupling"] = len(amide_rows)
    yield from amide_rows


def recount(table: pd.DataFrame, use_rxnutils: bool) -> Recount:
    """Run the step funnel on the joined table and collect the bins with their hash keys.

    Args:
        table: Output of ``load``.
        use_rxnutils: Take reactants and product from the rxnutils reaction instead of
            ORDerly's columns. The condition key always comes from ORDerly's columns.

    Returns:
        Steps, reject reasons and bins (with core_key and full_key per reaction).
    """
    result = Recount()
    result.steps["all rows"] = len(table)
    for row in _amide_rows(table, use_rxnutils, result):
        key = condition_key(row)
        if key is not None:
            result.bins[key].append((row.core_key, row.full_key))
    result.steps["binned"] = sum(len(members) for members in result.bins.values())
    result.steps["binned, duplicates removed (core_key)"] = sum(
        result.unique_core(key) for key in result.bins
    )
    return result


def core_keys_in_several_bins(result: Recount) -> int:
    """How many reactions (core_key) appear under more than one condition key."""
    bins_per_core: dict[str, set[ConditionKey]] = defaultdict(set)
    for key, members in result.bins.items():
        for core, _ in members:
            bins_per_core[core].add(key)
    return sum(1 for keys in bins_per_core.values() if len(keys) > 1)


def comparison_table(raw: Recount, rxn: Recount, top: int = 20) -> pd.DataFrame:
    """Raw vs rxnutils vs deduplicated sizes for the largest raw bins."""
    largest = sorted(raw.bins, key=raw.size, reverse=True)[:top]
    rows = [
        {
            "agents": "; ".join(key[0]),
            "solvents": "; ".join(key[1]),
            "raw": raw.size(key),
            "rxnutils": rxn.size(key),
            "rxnutils_unique_core": rxn.unique_core(key),
            "rxnutils_unique_full": rxn.unique_full(key),
        }
        for key in largest
    ]
    return pd.DataFrame(rows)


def print_report(raw: Recount, rxn: Recount, table: pd.DataFrame) -> None:
    """Print both funnels, reject reasons, multi-condition reactions and the top-20 table."""
    print(f"{'step':40} {'raw':>9} {'rxnutils':>9}")
    for step in rxn.steps:
        print(f"{step:40} {raw.steps.get(step, 0):>9,} {rxn.steps[step]:>9,}")
    print(f"\nrejected by rxnutils flags: {dict(rxn.rejected) or 'none'}")
    print(
        f"reactions (core_key) under more than one condition key: {core_keys_in_several_bins(rxn):,}"
    )
    print("\nTop 20 raw bins: raw vs rxnutils vs deduplicated")
    print(table.to_string(index=False))


def main() -> None:
    """Join ORDerly and the rxnutils output, count both ways, print and save the comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("orderly", type=Path, help="ORDerly condition parquet file")
    parser.add_argument(
        "rxnutils_out", type=Path, help="rxnutils_out.tsv from run_rxnutils.py"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("scratch/rxnutils/recount_top20.csv")
    )
    args = parser.parse_args()

    table = load(args.orderly, args.rxnutils_out)
    raw = recount(table, use_rxnutils=False)
    rxn = recount(table, use_rxnutils=True)
    comparison = comparison_table(raw, rxn)
    print_report(raw, rxn, comparison)
    comparison.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
