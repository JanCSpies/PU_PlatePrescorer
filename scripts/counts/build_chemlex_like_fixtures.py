"""Build tests/data/chemlex_like_6rows.parquet and the Chemlex rows of mapped_12rows.parquet.

The rows are hand-written in Chemlex's column layout with simple, common building blocks, so no
Chemlex data (CC BY-NC-ND) is in the repository (CLAUDE.md §9). They cover the cases the old fixture
covered: HATU drawn as the O-form (0) and as the N-oxide with no both-unseen label (1), EDCI as its
hydrochloride (2), PyBrOP with PF6 written as a cation (3), and a replicate pair that disagrees at
20 % (4: 31 %, 5: 0 %). Reactions 0, 1, 2, 4 and 5 are mapped for real with RXNMapper 0.4.3; row
``chemlex:6`` is mapped by hand, wrongly on purpose (``MIS_MAPPED``), so the stages meet a
mis-mapped reaction with 4 dynamic bonds, as RXNMapper produced for one Chemlex row on 2026-10-09.

Usage:
    uv run python scripts/counts/build_chemlex_like_fixtures.py ~/University/Phd/envs/rxnmapper/bin/python
"""

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from platepu.ingest.chemlex import records_from_table
from platepu.ingest.record import records_to_table
from platepu.reactions.mapping import join_mapped, map_reactions, rxnmapper
from platepu.reactions.smiles import rsmi_table

# parents[2], not [1]: this script lives in scripts/counts/, two levels below the repository
# root. It was moved here from scratch/ on 2026-10-22 and the depth was missed, which left the
# CLAUDE.md §9 promise that this code rebuilds the fixtures unkept until it was caught.
DATA = Path(__file__).parents[2] / "tests" / "data"
DIEA = "CCN(C(C)C)C(C)C"
PF6 = "F[P-](F)(F)(F)(F)F"
HATU_O_FORM = "CN(C)C(On1nnc2cccnc21)=[N+](C)C"
HATU_N_OXIDE = "CN(C)C(n1n[n+]([O-])c2ncccc21)=[N+](C)C"
DMF = "CN(C)C=O"
HATU = f"{DIEA}.{HATU_O_FORM}.{PF6}"

ROWS = [
    # acid, amine, reagents, product, conversion, random, one_unseen, both_unseen
    (
        "O=C(O)CCCc1ccccc1",
        "C[C@H](N)c1ccccc1",
        HATU,
        "C[C@H](NC(=O)CCCc1ccccc1)c1ccccc1",
        80.0,
        "test",
        "test",
        "test",
    ),
    (
        "O=C(O)c1ccco1",
        "Nc1ccc([N+](=O)[O-])cc1",
        f"{DIEA}.{HATU_N_OXIDE}.{PF6}",
        "O=C(Nc1ccc([N+](=O)[O-])cc1)c1ccco1",
        0.0,
        "train",
        "train",
        None,
    ),
    (
        "O=C(O)c1ccco1",
        "C1CCNCC1",
        f"CCN=C=NCCCN(C)C.Cl.{DIEA}",
        "O=C(c1ccco1)N1CCCCC1",
        12.5,
        "test",
        "train",
        "train",
    ),
    (
        "O=C(O)c1ccccc1",
        "Nc1ccc(N2CCOCC2)cc1",
        f"{DIEA}.Br[P+](N1CCCC1)(N1CCCC1)N1CCCC1.F[P+](F)(F)(F)(F)F",
        "O=C(Nc1ccc(N2CCOCC2)cc1)c1ccccc1",
        0.0,
        "train",
        "train",
        "test",
    ),
    (
        "O=C(O)c1ccc(Cl)cc1",
        "Nc1nccs1",
        HATU,
        "O=C(Nc1nccs1)c1ccc(Cl)cc1",
        31.0,
        "train",
        "test",
        "test",
    ),
    (
        "O=C(O)c1ccc(Cl)cc1",
        "Nc1nccs1",
        HATU,
        "O=C(Nc1nccs1)c1ccc(Cl)cc1",
        0.0,
        "test",
        "test",
        "test",
    ),
]
COLUMNS = [
    "Acid",
    "Amine",
    "Reagents",
    "Products",
    "Conversion",
    "Random_Split",
    "Stratified_Split_One_Unseen",
    "Stratified_Split_Both_Unseen",
]


def chemlex_like() -> pd.DataFrame:
    """The six rows in Chemlex's column order, index 0-5 as the sheet row index."""
    table = pd.DataFrame(ROWS, columns=COLUMNS)
    table.insert(3, "Solvent", DMF)
    return table


# 4-acetylbenzoic acid + methylamine, mapped by hand. The product is labelled as if the amide C were
# the ketone C (9) and the ring were mirrored, so every ring bond survives: the CGR sees C9-N2 and
# C3-C10 formed and C9-C10 and C3-OH broken (4 dynamic bonds), and C9 is no carboxyl C (no_class).
MIS_MAPPED = (
    "[CH3:10][C:9](=[O:11])[c:8]1[cH:7][cH:6][c:5]([C:3](=[O:4])O)[cH:13][cH:12]1.[CH3:1][NH2:2]."
    f"{DIEA}.CN(C)C(On1nnc2cccnc12)=[N+](C)C.{PF6}.{DMF}"
    ">>[CH3:1][NH:2][C:9](=[O:11])[c:8]1[cH:7][cH:6][c:5]([C:3]([CH3:10])=[O:4])[cH:13][cH:12]1"
)


def main() -> None:
    """Write both fixtures."""
    table = chemlex_like()
    table.to_parquet(DATA / "chemlex_like_6rows.parquet")
    stage2, _ = rsmi_table(records_to_table(records_from_table(table)))
    # The adapter stamps ``noncommercial`` because it assumes the real CC BY-NC-ND workbook
    # (ADR 009). These six rows are hand-written here, so they are genuinely ``open`` and the
    # fixture must say so -- it is the evidence that no Chemlex row is in the repository
    # (CLAUDE.md §9, ADR 011). Without this line the builder stopped reproducing the committed
    # fixture the moment the adapter changed, which is how it was caught on 2026-10-23.
    stage2 = stage2.set_column(
        stage2.schema.get_field_index("licence"),
        "licence",
        pa.array(["open"] * stage2.num_rows, pa.string()),
    )
    mapper = rxnmapper(Path(sys.argv[1]).expanduser(), batch_size=8)
    items = sorted(
        set(zip(stage2["rsmi_hash"].to_pylist(), stage2["rsmi"].to_pylist()))
    )
    cache = map_reactions(items, mapper)
    mapped, _ = join_mapped(stage2, cache)
    extra = pa.Table.from_pylist(
        [
            {
                "record_id": "chemlex:6",
                "licence": "open",
                "rsmi_hash": "hand-mis-mapped",
                "mapped_rsmi": MIS_MAPPED,
                "confidence": 0.5,
            }
        ],
        schema=mapped.schema,
    )
    old = pq.read_table(DATA / "mapped_12rows.parquet")
    orderly = old.filter(pc.starts_with(old["record_id"], "orderly:"))
    pq.write_table(
        pa.concat_tables([orderly, mapped, extra]), DATA / "mapped_12rows.parquet"
    )
    print(
        pa.concat_tables([mapped, extra])
        .select(["record_id", "mapped_rsmi"])
        .to_pandas()
        .to_string()
    )


if __name__ == "__main__":
    main()
