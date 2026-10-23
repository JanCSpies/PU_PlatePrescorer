"""Shared fixture: a stand-in for rxnutils' mapping script, so the subprocess code runs for real.

``stub_rxnmapper(mode)`` returns the interpreter to hand to ``rxnmapper()`` or to a paths file. The
interpreter is the test one, but ``PYTHONPATH`` puts a fake ``rxnutils.data.mapping`` first, so
``python -m rxnutils.data.mapping`` runs the stub below instead of RXNMapper. Like rxnutils it writes
a TSV with the input ``rsmi`` column plus ``mapped_rxn`` and ``confidence``. Modes:

- ``ok``: every reaction mapped (confidence 0.9), rows in input order.
- ``reversed``: the right rows in the wrong order.
- ``padded``: as many rows as the input, but the last one is a copy of the first (dedup + padding).
- ``error``: exit 1 with a ModuleNotFoundError on stderr, as a missing rxnmapper would.
- ``failing_first:N``: the first N reactions come back unchanged with confidence 0.0, as rxnutils
  reports any failure of RXNMapper; the rest are mapped.
"""

import sys
from collections.abc import Callable
from pathlib import Path

import pytest

STUB = """
import argparse
import os
import sys

import pandas as pd

parser = argparse.ArgumentParser()
for name in ("--input", "--output", "--column", "--mapper_batch_size"):
    parser.add_argument(name)
args = parser.parse_args()
mode = os.environ["STUB_MODE"]
if mode == "error":
    sys.stderr.write("ModuleNotFoundError: No module named 'rxnmapper'\\n")
    sys.exit(1)
rsmis = list(pd.read_csv(args.input, sep="\\t")[args.column])
failing = int(mode.split(":")[1]) if mode.startswith("failing_first:") else 0
rows = [
    (rsmi, rsmi, 0.0) if position < failing else (rsmi, "[CH3:1]" + rsmi, 0.9)
    for position, rsmi in enumerate(rsmis)
]
if mode == "reversed":
    rows = rows[::-1]
if mode == "padded":
    rows = rows[:-1] + [rows[0]]
out = pd.DataFrame(rows, columns=["rsmi", "mapped_rxn", "confidence"])
out.to_csv(args.output, sep="\\t", index=False)
"""


@pytest.fixture
def stub_rxnmapper(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Callable[[str], Path]:
    """Return ``use(mode)``: select a stub mode and get the interpreter that runs it."""
    package = tmp_path_factory.mktemp("stub_rxnutils") / "rxnutils" / "data"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "mapping.py").write_text(STUB)
    monkeypatch.setenv("PYTHONPATH", str(package.parents[1]))

    def use(mode: str) -> Path:
        monkeypatch.setenv("STUB_MODE", mode)
        return Path(sys.executable)

    return use
