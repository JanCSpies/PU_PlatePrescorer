"""Tests for scripts/explore_condition_bins.py on eight synthetic rows.

Every row is the same chemistry (acetic acid + benzylamine -> N-benzylacetamide) unless the
row tests an exclusion; only the conditions differ.
"""

import pandas as pd
import pytest
from explore_condition_bins import (
    BinCount,
    ConditionKey,
    canonical,
    condition_part,
    count_bins,
)

ACID = "CC(=O)O"
AMINE = "NCc1ccccc1"
PRODUCT = "CC(=O)NCc1ccccc1"
ACID_CHLORIDE = "CC(=O)Cl"

HATU = "CN(C)C(On1nnc2cccnc21)=[N+](C)C"  # HATU cation; ORDerly lists PF6- separately
PF6 = "F[P-](F)(F)(F)(F)F"
EDC = "CCN=C=NCCCN(C)C"
EDC_HCL = "CCN=C=NCCCN(C)C.Cl"  # EDC·HCl as one string (synonym -> EDC)
DIPEA = "CCN(C(C)C)C(C)C"
DMF = "CN(C)C=O"
MAPPED_DMF = "[CH3:1][N:2]([CH3:3])[CH:4]=[O:5]"
MAPPED_ACID = "[CH3:1][C:2](=[O:3])[OH:4]"

NONE = None
COLUMNS = ["reactant_000", "reactant_001", "product_000", "agent_000", "agent_001", "agent_002",
           "solvent_000", "solvent_001"]  # fmt: skip
ROWS = [
    # 1 and 2: same reagents, different order -> one bin
    [ACID, AMINE, PRODUCT, HATU, PF6, NONE, DIPEA, DMF],
    [AMINE, ACID, PRODUCT, PF6, HATU, NONE, DMF, DIPEA],
    # 3 and 4: EDC·HCl as one string, and EDC + separate Cl (ORDerly's way) -> one EDC bin
    [ACID, AMINE, PRODUCT, EDC_HCL, NONE, NONE, DMF, NONE],
    [ACID, AMINE, PRODUCT, EDC, "Cl", NONE, DMF, NONE],
    # 5: acid chloride instead of acid -> not an amide coupling
    [ACID_CHLORIDE, AMINE, PRODUCT, EDC, NONE, NONE, DMF, NONE],
    # 6: only one reactant -> excluded
    [ACID, NONE, PRODUCT, EDC, NONE, NONE, DMF, NONE],
    # 7: no solvent -> excluded, not an "unknown" bin
    [ACID, AMINE, PRODUCT, HATU, PF6, NONE, NONE, NONE],
    # 8: atom-mapped acid and solvent -> map numbers stripped, joins the EDC bin
    [MAPPED_ACID, AMINE, PRODUCT, EDC, NONE, NONE, MAPPED_DMF, NONE],
]


def key(agents: list[str], solvents: list[str]) -> ConditionKey:
    """Build the expected condition key the same way the script does, from raw SMILES."""
    return (
        tuple(sorted(str(canonical(s)) for s in agents)),
        tuple(sorted(str(canonical(s)) for s in solvents)),
    )


HATU_BIN = key([HATU], [DIPEA, DMF])  # PF6- is dropped as a counter-ion
EDC_BIN = key([EDC], [DMF])


@pytest.fixture(scope="module")
def result() -> BinCount:
    """Count the eight synthetic rows once."""
    return count_bins(pd.DataFrame(ROWS, columns=COLUMNS))


def test_rows_left_after_each_step(result: BinCount) -> None:
    """8 rows -> 7 with two reactants -> 6 amide couplings -> 5 with agents and solvents."""
    assert list(result.steps.values()) == [8, 7, 6, 5]


def test_same_reagents_in_another_order_share_one_bin(result: BinCount) -> None:
    """HATU + PF6 | DIPEA + DMF and PF6 + HATU | DMF + DIPEA are one bin of two (PF6 dropped)."""
    assert result.bins[HATU_BIN] == 2


def test_edc_hcl_and_edc_plus_cl_share_the_edc_bin(result: BinCount) -> None:
    """EDC·HCl (synonym) and EDC + separate Cl (dropped) both count as plain EDC."""
    assert result.bins[EDC_BIN] == 3
    assert all("Cl" not in agents for agents, _ in result.bins)


def test_acid_chloride_is_not_an_amide_coupling(result: BinCount) -> None:
    """The acid-chloride row drops out at the amide step (7 -> 6)."""
    assert result.steps["two reactants + product"] - result.steps["amide coupling"] == 1


def test_one_reactant_reaction_is_excluded(result: BinCount) -> None:
    """The one-reactant row drops out at the first step (8 -> 7)."""
    assert result.steps["all rows"] - result.steps["two reactants + product"] == 1


def test_missing_solvent_is_excluded_not_binned(result: BinCount) -> None:
    """HATU without a solvent is not counted anywhere, so only two bins exist."""
    assert set(result.bins) == {HATU_BIN, EDC_BIN}


def test_atom_map_numbers_are_stripped() -> None:
    """Mapped and unmapped SMILES of the same molecule canonicalise to one string."""
    assert canonical(MAPPED_DMF) == canonical(DMF)
    assert canonical(MAPPED_ACID) == canonical(ACID)


def test_hbtu_and_tbtu_share_one_key() -> None:
    """HBTU (cation + PF6-) and TBTU (same cation + BF4-) differ only in the dropped counter-ion."""
    cation = "CN(C)C(On1nnc2ccccc21)=[N+](C)C"
    assert condition_part([cation, PF6]) == condition_part([cation, "F[B-](F)(F)F"])


def test_ionic_chloride_is_dropped_like_hcl() -> None:
    """[Cl-] and Cl are both counter-ions; neither stays in a key."""
    assert (
        condition_part([EDC, "[Cl-]"])
        == condition_part([EDC, "Cl"])
        == condition_part([EDC])
    )
