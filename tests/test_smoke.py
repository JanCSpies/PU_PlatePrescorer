"""Smoke tests: the platepu package is installed, importable, and leaves RDKit's logging alone."""

import subprocess
import sys
from pathlib import Path

import pytest

import platepu
from platepu.ingest.record import ReactionRecord
from platepu.reactions.smiles import reaction_smiles
from platepu.standardise.molecule import load_std_config, standardize


def test_import_platepu() -> None:
    """The package imports and resolves to the installed platepu module."""
    assert platepu.__name__ == "platepu"


def test_importing_platepu_leaves_rdkit_logging_on() -> None:
    """Importing the package must not switch RDKit's error log off for the whole process (tests,
    model code); only the functions that read data molecules keep RDKit quiet, and only while
    they run. Checked in a fresh interpreter, since other tests import platepu already."""
    code = (
        "import platepu.classify.engine, platepu.reactions.smiles, platepu.standardise.molecule;"
        "from rdkit import rdBase; print(rdBase.LogStatus())"
    )
    status = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert "rdApp.error:enabled" in status


def test_unreadable_data_molecule_is_quiet(capfd: pytest.CaptureFixture[str]) -> None:
    """An unparsable SMILES in the data is a counted reason, not a line on stderr per row."""
    config = load_std_config(Path(__file__).parents[1] / "configs" / "standardise.yml")
    assert standardize("C(=O", "substrate", config).reason == "unparsable"
    assert reaction_smiles(_record(), {}) is None
    assert "SMILES Parse Error" not in capfd.readouterr().err


def _record() -> ReactionRecord:
    """A record whose only reactant cannot be read."""
    return ReactionRecord(
        source="orderly",
        source_id="1",
        licence="open",
        reactants=("C(=O",),
        agents=(),
        solvents=(),
        products=("CC",),
        source_rxn=None,
        temperature_c=None,
        reaction_time_h=None,
        yield_pct=None,
        yield_kind=None,
        year=None,
        source_document=None,
    )
