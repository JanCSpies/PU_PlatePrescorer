"""Tests for stage 5, classify (configs/classes/amide_coupling.yml).

Hand-written mapped reactions cover the cases counted in the data; the fixture rows mapped for real
check the engine on RXNMapper output.
"""

from pathlib import Path

import classify as classify_script
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rdkit import Chem

from platepu.classify import groups
from platepu.classify.engine import (
    ClassRule,
    classify_table,
    load_class,
    match_reaction,
)
from platepu.classify.groups import group_counts
from platepu.reactions.clean import clean_table, load_pipeline

ROOT = Path(__file__).parents[1]
AMIDE = load_class(ROOT / "configs" / "classes" / "amide_coupling.yml")

ACID_AMINE = (
    "[CH3:1][C:2](=[O:3])O.[NH2:4][CH3:5]>CN(C)C=O>[CH3:1][C:2](=[O:3])[NH:4][CH3:5]"
)
CARBOXYLATE = (
    "[CH3:1][C:2](=[O:3])[O-].[NH2:4][CH3:5]>>[CH3:1][C:2](=[O:3])[NH:4][CH3:5]"
)
AMMONIUM = "[CH3:1][C:2](=[O:3])O.[NH3+:4][CH3:5]>>[CH3:1][C:2](=[O:3])[NH:4][CH3:5]"
SULFONAMIDE = (
    "[CH3:1][C:2](=[O:3])O.[NH2:4][S:5](=[O:6])(=[O:7])[CH3:8]>>"
    "[CH3:1][C:2](=[O:3])[NH:4][S:5](=[O:6])(=[O:7])[CH3:8]"
)
WEINREB = "[CH3:1][C:2](=[O:3])O.[CH3:4][NH:5][O:6][CH3:7]>>[CH3:1][C:2](=[O:3])[N:5]([CH3:4])[O:6][CH3:7]"
ENAMINONE = (
    "[CH3:1][C:2](=[O:3])O.[NH2:4][C:5]([CH3:6])=[CH:7][C:8](=[O:9])[CH3:10]>>"
    "[CH3:1][C:2](=[O:3])[NH:4][C:5]([CH3:6])=[CH:7][C:8](=[O:9])[CH3:10]"
)
HYDRAZIDE = "[CH3:1][C:2](=[O:3])O.[NH2:4][NH2:5]>>[CH3:1][C:2](=[O:3])[NH:4][NH2:5]"
LACTAM = "[NH2:1][CH2:2][CH2:3][CH2:4][C:5](=[O:6])O>>[NH:1]1[CH2:2][CH2:3][CH2:4][C:5]1=[O:6]"


def test_acid_and_amine_get_their_roles() -> None:
    """Role A is the molecule with the carboxyl C, role B the one with the N, both unmapped."""
    assert match_reaction(ACID_AMINE, [AMIDE]) == [("amide_coupling", "CC(=O)O", "CN")]


@pytest.mark.parametrize("rsmi", [CARBOXYLATE, AMMONIUM])
def test_salts_match_before_standardisation(rsmi: str) -> None:
    """Carboxylates and ammonium salts (ORDerly: 1,181 and 196) are the same class."""
    assert [match[0] for match in match_reaction(rsmi, [AMIDE])] == ["amide_coupling"]


@pytest.mark.parametrize("rsmi", [SULFONAMIDE, WEINREB, HYDRAZIDE])
def test_other_n_nucleophiles_are_not_amines(rsmi: str) -> None:
    """Decided 2026-10-09: sulfonamides, hydroxylamines and hydrazines are not amines here."""
    assert match_reaction(rsmi, [AMIDE]) == []


def test_intramolecular_bond_has_no_two_substrates() -> None:
    """A lactam forms its C-N bond inside one molecule: no pair of substrates."""
    assert match_reaction(LACTAM, [AMIDE]) == []


def test_enaminone_is_an_amine() -> None:
    """Decided 2026-10-09: vinylogous amides stay in the class (Chemlex: 502 rows, mostly negative)."""
    assert [match[0] for match in match_reaction(ENAMINONE, [AMIDE])] == [
        "amide_coupling"
    ]


def test_class_file_is_read_with_reacting_atoms() -> None:
    """The :1 in each SMARTS marks the reacting atom; the formed bond names two roles."""
    assert isinstance(AMIDE, ClassRule)
    assert (AMIDE.role_a, AMIDE.role_b) == ("acid", "amine")
    assert AMIDE.allowed_dynamic_bonds == (2,)


def test_table_rows_and_reject_reasons() -> None:
    """RXNMapper output: Chemlex-layout amide couplings match, the Suzuki coupling does not."""
    pipeline = load_pipeline(ROOT / "configs" / "clean.yml")
    cleaned, _ = clean_table(
        pq.read_table(ROOT / "tests" / "data" / "mapped_12rows.parquet"), pipeline
    )
    table, rejected = classify_table(cleaned, [AMIDE])
    rows = {row["record_id"]: row for row in table.to_pylist()}
    assert rows["chemlex:0"]["substrate_a"] == "O=C(O)CCCc1ccccc1"
    assert rows["chemlex:0"]["class"] == "amide_coupling"
    reasons = dict(
        zip(
            rejected["record_id"].to_pylist(),
            rejected["reason"].to_pylist(),
            strict=True,
        )
    )
    assert reasons["orderly:669992"] == "no_class"  # Suzuki
    # Mis-mapped (N bonded to the ketone, not the carboxyl C): not this class from the mapping.
    # Benchmarks take their roles from their own columns instead (decided 2026-10-09).
    assert reasons["chemlex:6"] == "no_class"
    assert isinstance(table, pa.Table)


def _cleaned(*rows: tuple[str, str]) -> pa.Table:
    """A stage-4 table of ``(record_id, clean_rsmi)`` rows with the other columns filled in."""
    return pa.table(
        {
            "record_id": [record_id for record_id, _ in rows],
            "licence": ["open"] * len(rows),
            "clean_rsmi": [rsmi for _, rsmi in rows],
            "n_reactants": [2] * len(rows),
            "n_products": [1] * len(rows),
            "n_dynamic_bonds": [2] * len(rows),
        }
    )


def test_lactam_is_rejected_as_no_formed_bond() -> None:
    """A bond formed inside one molecule is no formed bond between two substrates."""
    _, rejected = classify_table(_cleaned(("orderly:1", LACTAM)), [AMIDE])
    assert rejected.to_pylist() == [
        {"record_id": "orderly:1", "reason": "no_formed_bond"}
    ]


def test_unparsable_molecule_is_rejected_with_reason() -> None:
    """CLAUDE.md §6: a molecule RDKit cannot read is a rejected row with a reason, not a crash."""
    broken = "[CH3:1][C:2](=[O:3])O.[NH2:4]c1ccc>>[CH3:1][C:2](=[O:3])[NH:4]c1ccc"
    table, rejected = classify_table(
        _cleaned(("orderly:1", broken), ("orderly:2", ACID_AMINE)), [AMIDE]
    )
    assert rejected.to_pylist() == [{"record_id": "orderly:1", "reason": "unparsable"}]
    assert table["record_id"].to_pylist() == ["orderly:2"]


@pytest.mark.parametrize(
    ("acid_smarts", "message"),
    [("[CX3:1](=O", "unparsable SMARTS"), ("[CX3](=O)[OX2H1]", "exactly one atom")],
)
def test_broken_role_smarts_stops_loading(
    tmp_path: Path, acid_smarts: str, message: str
) -> None:
    """A typo or a missing :1 in a class file stops loading instead of matching nothing."""
    path = tmp_path / "broken.yml"
    path.write_text(
        "name: broken\n"
        f"roles:\n  acid:\n    smarts: '{acid_smarts}'\n  amine:\n    smarts: '[N;!H0:1]'\n"
        "formed_bond: [acid, amine]\nallowed_dynamic_bonds: [2]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        load_class(path)


@pytest.mark.parametrize(
    ("smiles", "counts"),
    [
        ("CC(=O)O", (1, 0)),  # acetic acid: one acid, no amine
        ("NCc1ccccc1", (0, 1)),  # benzylamine
        ("OC(=O)CCC(=O)O", (2, 0)),  # succinic acid: two acid groups
        ("NCCC(=O)O", (1, 1)),  # beta-alanine: both roles
    ],
)
def test_group_counts_per_role(smiles: str, counts: tuple[int, int]) -> None:
    """(groups of role A, groups of role B): the one count positives and the library both use."""
    assert group_counts(smiles, AMIDE) == counts


def test_script_without_class_files_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run outside the repository, the default class folder is empty: stop, do not reject everything."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        classify_script.main(["orderly", str(tmp_path)])


# Amine and acid subclasses (docs/design/modeling.md, Stage F). Hand-written SMILES, checked with RDKit
# on 2026-10-21: each has exactly one amine N, and the subclass is read at that N.
def test_amide_class_loads_four_amine_subclasses() -> None:
    """The amine role lists its four subclasses in the order of the YAML; first match wins."""
    assert [subclass.name for subclass in AMIDE.subclasses_b] == [
        "heteroaryl_amine",
        "aniline",
        "aliphatic_secondary",
        "aliphatic_primary",
    ]


def test_role_without_subclasses_has_none(tmp_path: Path) -> None:
    """A role with no ``subclasses`` in its YAML answers None (not "other"): nothing to choose from."""
    path = tmp_path / "plain.yml"
    path.write_text(
        "name: plain\n"
        "roles:\n"
        '  acid: {smarts: "[CX3:1](=O)[OX2H1]"}\n'
        '  amine: {smarts: "[NX3;H2:1]"}\n'
        "formed_bond: [acid, amine]\n"
        "allowed_dynamic_bonds: [2]\n",
        encoding="utf-8",
    )
    plain = load_class(path)
    assert plain.subclasses_a == ()
    assert groups.subclass_of("CC(=O)O", "acid", plain) is None


def test_acid_role_has_no_subclasses() -> None:
    """The acid subclasses were removed on 2026-10-22: the rooted Morgan fingerprint of the
    carboxyl carbon (features.yml, morgan_centre) carries the same information without hand-written
    SMARTS, and no baseline uses them. The amine subclasses stay (B4, within-class evaluation)."""
    assert AMIDE.subclasses_a == ()
    assert groups.subclass_of("CC(=O)O", "acid", AMIDE) is None


def test_piperidine_is_aliphatic_secondary() -> None:
    """C1CCNCC1: N-H between two sp3 carbons."""
    assert groups.subclass_of("C1CCNCC1", "amine", AMIDE) == "aliphatic_secondary"


def test_aniline_is_aniline() -> None:
    """Nc1ccccc1: N on a benzene carbon."""
    assert groups.subclass_of("Nc1ccccc1", "amine", AMIDE) == "aniline"


def test_2_aminopyridine_is_heteroaryl_amine() -> None:
    """Nc1ccccn1: the carbon bearing N is next to the ring n."""
    assert groups.subclass_of("Nc1ccccn1", "amine", AMIDE) == "heteroaryl_amine"


def test_2_aminothiazole_is_heteroaryl_amine() -> None:
    """Nc1nccs1: five-membered ring, the carbon bearing N is next to ring n (and s); one amine."""
    thiazole_mol = Chem.MolFromSmiles("Nc1nccs1")
    assert len(thiazole_mol.GetSubstructMatches(AMIDE.pattern_b)) == 1
    assert groups.subclass_of("Nc1nccs1", "amine", AMIDE) == "heteroaryl_amine"


def test_3_aminopyridine_is_aniline() -> None:
    """Nc1cccnc1: the ring n is two atoms away from the carbon bearing N, so it is an aniline."""
    assert groups.subclass_of("Nc1cccnc1", "amine", AMIDE) == "aniline"


def test_benzylamine_is_aliphatic_primary() -> None:
    """NCc1ccccc1: the aromatic ring is on the carbon, not on N."""
    assert groups.subclass_of("NCc1ccccc1", "amine", AMIDE) == "aliphatic_primary"


def test_amine_with_anilide_nh_is_aliphatic_primary() -> None:
    """NCCc1ccc(NC(C)=O)cc1 has an aryl N-H too (the anilide), but the reacting N is the CH2-NH2.

    Without the :1 anchor the "aniline" SMARTS matches the anilide N and gives the wrong class.
    """
    smiles = "NCCc1ccc(NC(C)=O)cc1"
    assert groups.subclass_of(smiles, "amine", AMIDE) == "aliphatic_primary"


def test_amine_on_phosphorus_is_other() -> None:
    """NP(=O)Ph2 matches the amine role (checked) but none of the four subclasses."""
    assert groups.subclass_of("NP(=O)(c1ccccc1)c1ccccc1", "amine", AMIDE) == "other"


def test_subclass_of_unknown_role_fails_loudly() -> None:
    """A role the class does not have is a typo, not "no subclass"."""
    with pytest.raises(ValueError):
        groups.subclass_of("NCc1ccccc1", "alcohol", AMIDE)


def test_subclass_smarts_without_mark_fails_loudly(tmp_path: Path) -> None:
    """A subclass SMARTS needs exactly one :1 atom, as the role SMARTS do."""
    path = tmp_path / "bad.yml"
    path.write_text(
        "name: bad\n"
        "roles:\n"
        '  acid: {smarts: "[CX3:1](=O)[OX2H1]"}\n'
        '  amine: {smarts: "[NX3;H2:1]", subclasses: [{name: x, smarts: "[NX3;H2]"}]}\n'
        "formed_bond: [acid, amine]\n"
        "allowed_dynamic_bonds: [2]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_class(path)
