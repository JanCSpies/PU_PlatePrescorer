"""Count amide-coupling condition bins in ORDerly (exploration script, not package code).

Answers one question before any reagent-role assignment is built: how many amide couplings
does ORDerly have under each exact condition combination (agents + solvents)?

Decisions (2026-10-02):
- ``Cl`` is dropped from every key. In ORDerly it is the counter-ion of EDC·HCl or of an amine
  hydrochloride (salts are always split into separate agents), never the coupling reagent.
  Ionic ``[Cl-]`` is dropped for the same reason (added after the first full run).
- PF6- and BF4- are dropped too: they are spectator counter-ions of the uronium/phosphonium
  activators, so HBTU (PF6-) and TBTU (BF4-), which share one cation, give one key. HATU,
  HBTU/TBTU, BOP and PyBOP stay distinct through their cations.
- Key = (sorted agents, sorted solvents) exactly as ORDerly assigns the roles. ORDerly lists
  liquid bases (DIPEA, Et3N, NMM) as solvents; every molecule always gets the same role, so
  counting is unaffected. Read the "solvents" column as "solvents and liquid bases".
- Bins whose "solvents" are only bases are kept but reported separately: the real solvent is
  usually in a referenced example ("prepared analogously to Example 5") that ORDerly never read.

Usage:
    uv run python scripts/explore_condition_bins.py PATH [--max-rows 20000] [--out FILE]
"""

import argparse
from collections import Counter
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog(
    "rdApp.*"
)  # RDKit warns on every odd SMILES; parse failures are counted

ACID = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
AMINE = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O);!$(NS=O)]")
AMIDE = Chem.MolFromSmarts("C(=O)[NX3]")

# Hand-made synonyms, raw SMILES -> raw SMILES; canonicalised before use. Fill from the
# "top agents" list of a run.
SYNONYMS = {
    "CCN=C=NCCCN(C)C.Cl": "CCN=C=NCCCN(C)C",  # EDC·HCl -> EDC
}
# Counter-ions, never the coupling reagent (decisions above): HCl, chloride, PF6-, BF4-.
DROPPED = frozenset({"Cl", "[Cl-]", "F[P-](F)(F)(F)(F)F", "F[B-](F)(F)F"})
BASES = frozenset({"CCN(C(C)C)C(C)C", "CCN(CC)CC", "CN1CCOCC1"})  # DIPEA, Et3N, NMM

AGENT_COLUMNS = ["agent_000", "agent_001", "agent_002"]
SOLVENT_COLUMNS = ["solvent_000", "solvent_001"]

ConditionKey = tuple[
    tuple[str, ...], tuple[str, ...]
]  # (sorted agents, sorted solvents)


@dataclass
class BinCount:
    """Result of one counting run.

    Attributes:
        steps: Rows left after each filtering step, in order.
        bins: Number of reactions per condition key.
        agents: Number of counted reactions each individual agent appears in.
        n_unparsable: Amide couplings excluded because a condition SMILES could not be parsed.
    """

    steps: dict[str, int] = field(default_factory=dict)
    bins: Counter[ConditionKey] = field(default_factory=Counter)
    agents: Counter[str] = field(default_factory=Counter)
    n_unparsable: int = 0


@cache
def canonical(smiles: str) -> str | None:
    """Return RDKit's canonical SMILES without atom-map numbers, or None if unparsable.

    Args:
        smiles: Any SMILES, possibly atom-mapped (``[CH3:1]N(C)C=O``).

    Returns:
        One unique string per molecule, so the same molecule written two ways gives one key.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return str(Chem.MolToSmiles(mol))


@cache
def matches(smiles: str, pattern: Chem.Mol) -> bool:
    """Return True if the molecule contains the SMARTS pattern; unparsable SMILES never match."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and mol.HasSubstructMatch(pattern)


def is_amide_coupling(reactant_a: str, reactant_b: str, product: str) -> bool:
    """Crude amide-coupling rule: carboxylic acid + amine -> amide, with no atom mapping.

    The acid can be either reactant, so both orders are tried. An acid chloride has no
    C(=O)OH and therefore never counts as the acid.

    Args:
        reactant_a: First reactant SMILES.
        reactant_b: Second reactant SMILES.
        product: Product SMILES.

    Returns:
        True if one reactant is an acid, the other an amine, and the product has an amide.
    """
    acid_then_amine = matches(reactant_a, ACID) and matches(reactant_b, AMINE)
    amine_then_acid = matches(reactant_b, ACID) and matches(reactant_a, AMINE)
    return (acid_then_amine or amine_then_acid) and matches(product, AMIDE)


@cache
def _canonical_synonyms() -> dict[str, str]:
    """SYNONYMS with keys and values canonicalised, so lookups match canonical SMILES."""
    table = {}
    for raw, target in SYNONYMS.items():
        key, value = canonical(raw), canonical(target)
        if key is None or value is None:
            raise ValueError(f"unparsable SMILES in SYNONYMS: {raw!r} -> {target!r}")
        table[key] = value
    return table


def condition_part(values: list[object]) -> tuple[str, ...] | None:
    """Turn one row's agents (or solvents) into a sorted tuple of canonical SMILES.

    Empty cells (NaN) are skipped, synonyms are applied and counter-ions are dropped. Sorting makes
    the order in the source irrelevant.

    Args:
        values: The cells of the agent (or solvent) columns of one row.

    Returns:
        The sorted tuple, possibly empty; None if any SMILES cannot be parsed.
    """
    parts = []
    for value in values:
        if not isinstance(value, str):
            continue
        smiles = canonical(value)
        if smiles is None:
            return None
        smiles = _canonical_synonyms().get(smiles, smiles)
        if smiles not in DROPPED:
            parts.append(smiles)
    return tuple(sorted(parts))


def count_bins(table: pd.DataFrame) -> BinCount:
    """Filter ORDerly rows step by step and count reactions per condition key.

    Steps: all rows -> two reactants and a product -> amide coupling -> has agents and
    solvents. Reactions without an agent or without a solvent are excluded, never put
    into an "unknown" bin.

    Args:
        table: ORDerly rows with reactant, product, agent and solvent columns.

    Returns:
        Row counts per step, the bins, and per-agent counts over the binned reactions.
    """
    result = BinCount()
    result.steps["all rows"] = len(table)

    present = (
        table.reactant_000.notna()
        & table.reactant_001.notna()
        & table.product_000.notna()
    )
    two = table[present]
    result.steps["two reactants + product"] = len(two)

    triples = zip(two.reactant_000, two.reactant_001, two.product_000, strict=True)
    amide = two[[is_amide_coupling(a, b, p) for a, b, p in triples]]
    result.steps["amide coupling"] = len(amide)

    for row in amide[AGENT_COLUMNS + SOLVENT_COLUMNS].itertuples(index=False):
        agents = condition_part(list(row[: len(AGENT_COLUMNS)]))
        solvents = condition_part(list(row[len(AGENT_COLUMNS) :]))
        if agents is None or solvents is None:
            result.n_unparsable += 1
        elif agents and solvents:
            result.bins[(agents, solvents)] += 1
            result.agents.update(agents)
    result.steps["has agents and solvents"] = result.bins.total()
    return result


def _join(smiles: tuple[str, ...]) -> str:
    """Join SMILES for display and CSV; '; ' because '+' and '.' occur inside SMILES."""
    return "; ".join(smiles)


def print_report(result: BinCount, top_bins: int = 20, top_agents: int = 30) -> None:
    """Print the step counts, the largest bins, base-only bins and the most common agents."""
    print("Rows left after each step:")
    for step, n in result.steps.items():
        print(f"  {step:28} {n:>9,}")
    print(
        f"  (amide couplings with an unparsable condition SMILES, excluded: {result.n_unparsable:,})"
    )

    print(f"\nTop {top_bins} condition bins (of {len(result.bins):,}):")
    for (agents, solvents), n in result.bins.most_common(top_bins):
        print(f"  {n:>6,}  {_join(agents)}  |  {_join(solvents)}")

    base_only = {key: n for key, n in result.bins.items() if set(key[1]) <= BASES}
    print(
        f"\nBins whose 'solvents' are only DIPEA/Et3N/NMM (solvent probably missing in source): "
        f"{len(base_only):,} bins, {sum(base_only.values()):,} reactions"
    )

    print(f"\nTop {top_agents} agents (reactions they appear in):")
    for agent, n in result.agents.most_common(top_agents):
        print(f"  {n:>6,}  {agent}")


def write_bins_csv(result: BinCount, path: Path) -> None:
    """Save every bin, largest first, with columns agents, solvents, n_reactions."""
    rows = [
        {"agents": _join(agents), "solvents": _join(solvents), "n_reactions": n}
        for (agents, solvents), n in result.bins.most_common()
    ]
    pd.DataFrame(rows, columns=["agents", "solvents", "n_reactions"]).to_csv(
        path, index=False
    )
    print(f"\nWrote {len(rows):,} bins to {path}")


def main() -> None:
    """Read ORDerly, count condition bins, print the report and save the CSV."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="ORDerly condition parquet file")
    parser.add_argument(
        "--max-rows", type=int, default=None, help="only the first N rows"
    )
    parser.add_argument("--out", type=Path, default=Path("condition_bins.csv"))
    args = parser.parse_args()

    columns = [
        "reactant_000",
        "reactant_001",
        "product_000",
        *AGENT_COLUMNS,
        *SOLVENT_COLUMNS,
    ]
    table = pd.read_parquet(args.path, columns=columns)
    if args.max_rows is not None:
        table = table.head(args.max_rows)
    result = count_bins(table)
    print_report(result)
    write_bins_csv(result, args.out)


if __name__ == "__main__":
    main()
