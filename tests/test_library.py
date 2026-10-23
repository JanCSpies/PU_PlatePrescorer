"""Tests for the lab substrate library (U for the plate), on a made-up inventory with the real layout.

No internal inventory data is in the repo: the rows below are invented, the layout (two rows above the
header, German column names) is the one of the group's inventory export.
"""

from pathlib import Path

import pandas as pd
import pytest

from platepu.classify.engine import load_class
from platepu.library.inventory import library_table, read_inventory
from platepu.standardise.molecule import load_std_config

ROOT = Path(__file__).parents[1]
RULE = load_class(ROOT / "configs" / "classes" / "amide_coupling.yml")
STD = load_std_config(ROOT / "configs" / "standardise.yml")

ROWS = [
    ("benzylamine HCl", "[Cl-].NCc1ccccc1.[H+]"),  # amine as salt
    ("benzylamine", "NCc1ccccc1"),  # same compound, second container
    ("acetic acid sodium salt", "CC(=O)[O-].[Na+]"),  # acid as salt
    ("glycine", "NCC(=O)O"),  # both roles
    ("succinic acid", "OC(=O)CCC(=O)O"),  # two acid groups
    ("toluene", "Cc1ccccc1"),  # no role
    ("sodium cyclopentadienide", "c1cc[cH-]c1.[Na+]"),  # neutralisation fails
    ("unknown", None),  # no structure
]


@pytest.fixture
def inventory(tmp_path: Path) -> pd.DataFrame:
    """Write the made-up rows as an .xlsx with the export's layout and read them back."""
    table = pd.DataFrame(
        {
            "Name": [name for name, _ in ROWS],
            "stereo-SMILES": [smiles for _, smiles in ROWS],
            "CAS-Nr.": [None] * len(ROWS),
            "Barcode": [f"B{i}" for i in range(len(ROWS))],
            "Menge": ["1 g"] * len(ROWS),
        }
    )
    path = tmp_path / "inventory.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame([["x"], ["y"]]).to_excel(writer, index=False, header=False)
        table.to_excel(writer, index=False, startrow=2)
    return read_inventory(path)


def test_header_is_found_below_two_rows(inventory: pd.DataFrame) -> None:
    """The export has two rows above the header; every container becomes one row."""
    assert len(inventory) == len(ROWS)
    assert list(inventory.columns[:2]) == ["container_id", "smiles"]


def test_compounds_get_one_role_each(inventory: pd.DataFrame) -> None:
    """Salts reduce to the parent; the two benzylamine containers become one compound."""
    compounds, _ = library_table(inventory, RULE, STD)
    roles = dict(zip(compounds["smiles"], compounds["role"], strict=True))
    assert roles == {"NCc1ccccc1": "amine", "CC(=O)O": "acid"}
    assert compounds.set_index("smiles").loc["NCc1ccccc1", "n_containers"] == 2


def test_rejected_containers_have_reasons(inventory: pd.DataFrame) -> None:
    """Every container that is not a usable substrate is counted with why."""
    _, rejected = library_table(inventory, RULE, STD)
    assert sorted(rejected["reason"]) == [
        "both_roles",
        "multiple_groups",
        "neutralise_failed",
        "no_role",
        "no_smiles",
    ]


def test_library_rows_are_internal(inventory: pd.DataFrame) -> None:
    """ADR 004: compounds and rejected containers carry licence 'internal', so write_table guards them."""
    compounds, rejected = library_table(inventory, RULE, STD)
    assert set(compounds["licence"]) == {"internal"}
    assert set(rejected["licence"]) == {"internal"}


def test_library_compounds_carry_std_version(inventory: pd.DataFrame) -> None:
    """Compounds say which standardisation made their keys (CLAUDE.md section 6)."""
    compounds, _ = library_table(inventory, RULE, STD)
    assert set(compounds["std_version"]) == {STD.std_version}
