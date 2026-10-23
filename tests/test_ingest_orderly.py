"""Tests for the ORDerly adapter, on five hand-checked rows.

Fixture ``tests/data/orderly_5rows.parquet``: rows 638744, 669992, 321510, 635883 and 167959,
cut unchanged (all 18 columns, index kept) from ``samples_2k/orderly_condition.parquet`` on
2026-10-01 with pandas 3.0.6. ORDerly is open data.

The expected records follow the decisions in ``docs/design/rebuild.md`` ("ORDerly specifics"):
roles, salts and yields are kept exactly as ORDerly gives them; ``year`` is the patent grant year.
Each record was checked by Luka against the row's ``procedure_details``.
"""

from pathlib import Path

import pytest

from platepu.ingest.orderly import read_orderly
from platepu.ingest.record import ReactionRecord

FIXTURE = Path(__file__).parent / "data" / "orderly_5rows.parquet"

EXPECTED: dict[str, ReactionRecord] = {
    # amide coupling (EDC/HOBt)
    "638744": ReactionRecord(
        source="orderly",
        source_id="638744",
        licence="open",
        reactants=("CSc1cccc(Oc2ncc(F)cc2C(=O)N[C@H]2CC[C@@H](N)CC2)c1", "COCC(=O)O"),
        agents=("CCN=C=NCCCN(C)C", "Cl", "On1nnc2ccccc21"),
        solvents=("CCN(CC)CC", "CN(C)C=O"),
        products=("COCC(=O)N[C@H]1CC[C@@H](NC(=O)c2cc(F)cnc2Oc2cccc(SC)c2)CC1",),
        source_rxn="Cl.[NH2:2][C@@H:3]1[CH2:8][CH2:7][C@H:6]([NH:9][C:10](=[O:27])[C:11]2[CH:16]=[C:15]([F:17])[CH:14]=[N:13][C:12]=2[O:18][C:19]2[CH:24]=[CH:23][CH:22]=[C:21]([S:25][CH3:26])[CH:20]=2)[CH2:5][CH2:4]1.C(N(CC)CC)C.[CH3:35][O:36][CH2:37][C:38](O)=[O:39].Cl.CN(C)CCCN=C=NCC.ON1C2C=CC=CC=2N=N1>CN(C)C=O>[F:17][C:15]1[CH:14]=[N:13][C:12]([O:18][C:19]2[CH:24]=[CH:23][CH:22]=[C:21]([S:25][CH3:26])[CH:20]=2)=[C:11]([CH:16]=1)[C:10]([NH:9][C@H:6]1[CH2:7][CH2:8][C@@H:3]([NH:2][C:38](=[O:39])[CH2:37][O:36][CH3:35])[CH2:4][CH2:5]1)=[O:27]",
        temperature_c=25.0,
        reaction_time_h=18.0,
        yield_pct=68.8,
        yield_kind="isolated",
        year=2006,
        source_document="ord_dataset-437aa6654d5044ddaef3346dc4c6e08a",
    ),
    # Suzuki cross-coupling
    "669992": ReactionRecord(
        source="orderly",
        source_id="669992",
        licence="open",
        reactants=("OCc1cccc(Br)c1", "CC1(C)OB(c2ccc(N3CCOCC3)cc2)OC1(C)C"),
        agents=(
            "c1ccc([P](c2ccccc2)(c2ccccc2)[Pd]([P](c2ccccc2)(c2ccccc2)c2ccccc2)([P](c2ccccc2)(c2ccccc2)c2ccccc2)[P](c2ccccc2)(c2ccccc2)c2ccccc2)cc1",
            "O=C([O-])[O-]",
            "[Na+]",
        ),
        solvents=("O", "COCCOC"),
        products=("OCc1cccc(-c2ccc(N3CCOCC3)cc2)c1",),
        source_rxn="C(=O)([O-])[O-].[Na+].[Na+].CC1(C)C(C)(C)OB([C:15]2[CH:20]=[CH:19][C:18]([N:21]3[CH2:26][CH2:25][O:24][CH2:23][CH2:22]3)=[CH:17][CH:16]=2)O1.Br[C:29]1[CH:30]=[C:31]([CH2:35][OH:36])[CH:32]=[CH:33][CH:34]=1.COCCOC>C1C=CC([P]([Pd]([P](C2C=CC=CC=2)(C2C=CC=CC=2)C2C=CC=CC=2)([P](C2C=CC=CC=2)(C2C=CC=CC=2)C2C=CC=CC=2)[P](C2C=CC=CC=2)(C2C=CC=CC=2)C2C=CC=CC=2)(C2C=CC=CC=2)C2C=CC=CC=2)=CC=1.O>[N:21]1([C:18]2[CH:17]=[CH:16][C:15]([C:29]3[CH:34]=[CH:33][CH:32]=[C:31]([CH2:35][OH:36])[CH:30]=3)=[CH:20][CH:19]=2)[CH2:22][CH2:23][O:24][CH2:25][CH2:26]1",
        temperature_c=80.0,
        reaction_time_h=8.0,
        yield_pct=86.6,
        yield_kind="isolated",
        year=2014,
        source_document="ord_dataset-5e6956e6e8c24a168866a253f4a66c6c",
    ),
    # no temperature
    "321510": ReactionRecord(
        source="orderly",
        source_id="321510",
        licence="open",
        reactants=(
            "N#CCc1ccccc1-c1ccc(C(=O)N2Cc3ccc(C(=O)NCc4cccnc4)n3Cc3ccccc32)cc1",
            "NO",
        ),
        agents=("Cl", "O=C([O-])[O-]", "[K+]"),
        solvents=("O", "CCO"),
        products=(
            "NC(Cc1ccccc1-c1ccc(C(=O)N2Cc3ccc(C(=O)NCc4cccnc4)n3Cc3ccccc32)cc1)=NO",
        ),
        source_rxn="[C:1]([CH2:3][C:4]1[CH:9]=[CH:8][CH:7]=[CH:6][C:5]=1[C:10]1[CH:15]=[CH:14][C:13]([C:16]([N:18]2[C:24]3[CH:25]=[CH:26][CH:27]=[CH:28][C:23]=3[CH2:22][N:21]3[C:29]([C:32]([NH:34][CH2:35][C:36]4[CH:37]=[N:38][CH:39]=[CH:40][CH:41]=4)=[O:33])=[CH:30][CH:31]=[C:20]3[CH2:19]2)=[O:17])=[CH:12][CH:11]=1)#[N:2].Cl.[NH2:43][OH:44].C(=O)([O-])[O-].[K+].[K+]>C(O)C.O>[NH2:2][C:1](=[N:43][OH:44])[CH2:3][C:4]1[CH:9]=[CH:8][CH:7]=[CH:6][C:5]=1[C:10]1[CH:11]=[CH:12][C:13]([C:16]([N:18]2[C:24]3[CH:25]=[CH:26][CH:27]=[CH:28][C:23]=3[CH2:22][N:21]3[C:29]([C:32]([NH:34][CH2:35][C:36]4[CH:37]=[N:38][CH:39]=[CH:40][CH:41]=4)=[O:33])=[CH:30][CH:31]=[C:20]3[CH2:19]2)=[O:17])=[CH:14][CH:15]=1",
        temperature_c=None,
        reaction_time_h=None,
        yield_pct=22.3,
        yield_kind="isolated",
        year=2010,
        source_document="ord_dataset-ed680843f6d14f5c9901869b2a06b4a4",
    ),
    # no yield
    "635883": ReactionRecord(
        source="orderly",
        source_id="635883",
        licence="open",
        reactants=(
            "CC(C)(c1cc(-c2cccc(C(=O)O)c2)c2ncccc2c1)S(C)(=O)=O",
            "NC(=NO)c1ccc(F)cc1",
        ),
        agents=("O=C(n1ccnc1)n1ccnc1",),
        solvents=("CN(C)C=O", "O"),
        products=(
            "CC(C)(c1cc(-c2cccc(-c3nc(-c4ccc(F)cc4)no3)c2)c2ncccc2c1)S(C)(=O)=O",
        ),
        source_rxn="[CH3:1][C:2]([C:8]1[CH:9]=[C:10]2[C:15](=[C:16]([C:18]3[CH:19]=[C:20]([CH:24]=[CH:25][CH:26]=3)[C:21]([OH:23])=O)[CH:17]=1)[N:14]=[CH:13][CH:12]=[CH:11]2)([S:4]([CH3:7])(=[O:6])=[O:5])[CH3:3].C1N=CN(C(N2C=NC=C2)=O)C=1.[F:39][C:40]1[CH:45]=[CH:44][C:43]([C:46](=[N:48]O)[NH2:47])=[CH:42][CH:41]=1>CN(C=O)C.O>[F:39][C:40]1[CH:45]=[CH:44][C:43]([C:46]2[N:48]=[C:21]([C:20]3[CH:19]=[C:18]([C:16]4[CH:17]=[C:8]([C:2]([CH3:1])([S:4]([CH3:7])(=[O:5])=[O:6])[CH3:3])[CH:9]=[C:10]5[C:15]=4[N:14]=[CH:13][CH:12]=[CH:11]5)[CH:26]=[CH:25][CH:24]=3)[O:23][N:47]=2)=[CH:42][CH:41]=1",
        temperature_c=120.0,
        reaction_time_h=18.0,
        yield_pct=None,
        yield_kind=None,
        year=2009,
        source_document="ord_dataset-93908aaae836460ebd48d733eccad483",
    ),
    # salt split into ions
    "167959": ReactionRecord(
        source="orderly",
        source_id="167959",
        licence="open",
        reactants=(
            "COC(CNC(=O)C(CCC(C)(C)O)CC(OC(C)=O)C(Cc1cccc(F)c1)NC(=O)c1cnc2ccccc2n1)OC",
            "[NH4+]",
        ),
        agents=("CC(=O)[O-]", "O=C([O-])O", "[Na+]"),
        solvents=("CC(=O)O", "CCOC(C)=O"),
        products=(
            "CC(=O)OC(CC(CCC(C)(C)O)c1ncc[nH]1)C(Cc1cccc(F)c1)NC(=O)c1cnc2ccccc2n1",
        ),
        source_rxn="CO[CH:3](OC)[CH2:4][NH:5][C:6]([CH:8]([CH2:37][CH2:38][C:39]([OH:42])([CH3:41])[CH3:40])[CH2:9][CH:10]([O:33][C:34](=[O:36])[CH3:35])[CH:11]([NH:20][C:21]([C:23]1[CH:32]=[N:31][C:30]2[C:25](=[CH:26][CH:27]=[CH:28][CH:29]=2)[N:24]=1)=[O:22])[CH2:12][C:13]1[CH:18]=[CH:17][CH:16]=[C:15]([F:19])[CH:14]=1)=O.C([O-])(=O)C.[NH4+:49].C(=O)(O)[O-].[Na+]>C(O)(=O)C.C(OCC)(=O)C>[F:19][C:15]1[CH:14]=[C:13]([CH2:12][CH:11]([CH:10]([O:33][C:34](=[O:36])[CH3:35])[CH2:9][CH:8]([C:6]2[NH:5][CH:4]=[CH:3][N:49]=2)[CH2:37][CH2:38][C:39]([OH:42])([CH3:40])[CH3:41])[NH:20][C:21]([C:23]2[CH:32]=[N:31][C:30]3[C:25](=[CH:26][CH:27]=[CH:28][CH:29]=3)[N:24]=2)=[O:22])[CH:18]=[CH:17][CH:16]=1",
        temperature_c=115.0,
        reaction_time_h=None,
        yield_pct=16.8,
        yield_kind="isolated",
        year=2004,
        source_document="ord_dataset-e44331dc51de453ca14b7032593c1958",
    ),
}


@pytest.fixture(scope="module")
def records() -> dict[str, ReactionRecord]:
    """Read the fixture once and index the records by ``source_id``."""
    read, rejected = read_orderly(FIXTURE)
    assert rejected == []
    return {record.source_id: record for record in read}


def test_all_five_rows_become_records(records: dict[str, ReactionRecord]) -> None:
    """Every fixture row is readable, so none is rejected and none is lost."""
    assert sorted(records) == sorted(EXPECTED)


@pytest.mark.parametrize("source_id", sorted(EXPECTED))
def test_record_matches_hand_checked_expectation(
    records: dict[str, ReactionRecord], source_id: str
) -> None:
    """Each record equals the hand-checked expectation, field by field."""
    assert records[source_id] == EXPECTED[source_id]


def test_row_without_temperature_gives_none(records: dict[str, ReactionRecord]) -> None:
    """A missing temperature is None, never NaN (NaN != NaN would break comparisons)."""
    assert records["321510"].temperature_c is None


def test_row_without_yield_gives_none(records: dict[str, ReactionRecord]) -> None:
    """A missing yield is None, never NaN, and then there is no yield kind either."""
    assert records["635883"].yield_pct is None
    assert records["635883"].yield_kind is None


def test_literature_yield_is_labelled_isolated(
    records: dict[str, ReactionRecord],
) -> None:
    """A patent yield is taken to be an isolated yield, so it is never mixed up with an
    LC-MS conversion."""
    assert records["638744"].yield_kind == "isolated"


def test_overnight_is_eight_hours_as_given(records: dict[str, ReactionRecord]) -> None:
    """ORDerly turns "stirring ... overnight" into 8 h; the adapter keeps that value."""
    assert records["669992"].reaction_time_h == 8.0


def test_et3n_stays_in_solvents_as_given(records: dict[str, ReactionRecord]) -> None:
    """Wrong roles are kept: fixing them is the job of annotate (decision A6)."""
    assert "CCN(CC)CC" in records["638744"].solvents


def test_sodium_carbonate_keeps_one_sodium_as_given(
    records: dict[str, ReactionRecord],
) -> None:
    """ORDerly lists each ion once; the adapter does not rebuild Na2CO3 (decision A3)."""
    assert records["669992"].agents.count("[Na+]") == 1
    assert "[Na+].[Na+]" in (records["669992"].source_rxn or "")


def test_ammonium_acetate_stays_split_across_roles(
    records: dict[str, ReactionRecord],
) -> None:
    """NH4+ arrives as a reactant and acetate as an agent; both stay where they are."""
    assert "[NH4+]" in records["167959"].reactants
    assert "CC(=O)[O-]" in records["167959"].agents


def test_source_document_is_the_ord_dataset_file() -> None:
    """The record keeps ORDerly's extracted_from_file, so splits can follow the corpus' time blocks."""
    records, _ = read_orderly(FIXTURE)
    assert {record.source_id: record.source_document for record in records}[
        "638744"
    ] == ("ord_dataset-437aa6654d5044ddaef3346dc4c6e08a")
