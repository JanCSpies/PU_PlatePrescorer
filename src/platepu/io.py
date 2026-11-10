"""Reading and writing stage tables: a licence guard on the way out, and a report with provenance.

Every stage writes ``X.parquet`` and ``X.report.json``, and ``X.rejected.parquet`` where it rejects
rows (docs/design/rebuild.md, "Principles"). Rows not licensed ``open`` -- Pistachio (``licensed``),
Chemlex (``noncommercial``, ADR 009) and the group inventory (``internal``, ADR 004), with everything
derived from each -- may only be written under a ``private/`` folder (CLAUDE.md §9); this module is
the one place that enforces it.
"""

import hashlib
import json
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

PACKAGES = ("platepu", "rdkit", "reaction-utils", "pandas", "pyarrow")
# Sources whose data and everything derived from it may only be written under private/. Named for
# what every reader does with it -- withhold this source's rows from the open outputs -- and not for
# one reason to do so: Pistachio is licensed, Chemlex is CC BY-NC-ND (ADR 009).
PRIVATE_SOURCES = frozenset({"pistachio", "chemlex"})
# Licence values, strictest first. Merged rows take the strictest of their inputs.
LICENCE_PRECEDENCE = ("internal", "licensed", "noncommercial", "open")
# The one licence value a table may carry outside private/. Everything else -- "licensed"
# (Pistachio), "internal" (the group inventory, ADR 004), "noncommercial" (Chemlex) and anything
# unrecognised -- is refused there by write_table, so no list of restricted values is needed.
OPEN_LICENCE = "open"
# Rejected rows of stages 2-5: the record and why it left the stage.
REJECTED_SCHEMA = pa.schema([("record_id", pa.string()), ("reason", pa.string())])


@dataclass(frozen=True)
class Paths:
    """Machine-specific paths from ``configs/paths/<machine>.yml``.

    Attributes:
        root: Output root for stage tables and mapping caches.
        rxnmapper_python: Interpreter of the separate RXNMapper environment (ADR 002).
        mapper_batch_size: Reactions per RXNMapper call.
    """

    root: Path
    rxnmapper_python: Path
    mapper_batch_size: int


def load_paths(path: Path) -> Paths:
    """Read a paths file; ``~`` is expanded. A missing key raises KeyError, an unknown one TypeError."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Paths(
        root=Path(config.pop("root")).expanduser(),
        rxnmapper_python=Path(config.pop("rxnmapper_python")).expanduser(),
        **config,
    )


def source_dir(root: Path, source: str) -> Path:
    """Folder for one source's stage tables; a private source goes under ``<root>/private/``."""
    return root / "private" / source if source in PRIVATE_SOURCES else root / source


def strictest_licence(values: Iterable[object]) -> str:
    """The strictest licence among ``values``, ordered by ``LICENCE_PRECEDENCE``.

    When rows merge, the result takes the strictest licence of its inputs: one inventory row makes a
    substrate ``internal`` (ADR 004) and one Chemlex row makes it ``noncommercial`` (ADR 009). This
    is the only place that rule lives; ``features`` and ``positives`` both call it, having each had
    their own copy until 2026-10-22, one of which knew nothing of ``internal``.

    Raises:
        ValueError: a value outside the precedence, including a null, or no values at all. An empty
            group means a table that has lost its licence column, and answering ``open`` for it
            would be exactly the fail-open the allow-list in ``write_table`` exists to prevent.
    """
    seen = set(values)
    if not seen:
        raise ValueError("no licence values to take the strictest of")
    unknown = seen - set(LICENCE_PRECEDENCE)
    if unknown:
        raise ValueError(
            f"unknown licence values {sorted(str(value) for value in unknown)}"
        )
    return next(licence for licence in LICENCE_PRECEDENCE if licence in seen)


def write_table(table: pa.Table, path: Path, root: Path) -> None:
    """Write a parquet table, refusing any row not licensed ``open`` outside a ``private/`` folder.

    Args:
        table: The table to write.
        path: Where to write it; must lie inside ``root``.
        root: The output root (e.g. ``Datasets/derived``). Only folders below it count, so a
            system folder such as macOS's ``/private/var`` never passes for ``private/``.

    Raises:
        ValueError: the table has a ``licence`` column holding any value other than ``"open"``, and
            the bytes would not land in a ``private/`` folder below ``root``.

    The test is an allow-list, not a list of values to refuse. Until 2026-10-22 it named
    ``licensed`` and ``internal``, so everything it did not recognise passed: a null licence,
    ``"Internal"``, ``"internal "``, or a value nobody had thought of yet. Asking instead that every
    value **be** ``open`` refuses all of those, and a new restricted value such as
    ``noncommercial`` needs no entry anywhere to be protected.

    A table with **no** licence column is still written. That is a known gap, decided and recorded
    on 2026-10-22, not an oversight: six ``*.rejected.parquet`` writers (every stage but ``library``,
    whose ``library_table`` adds the column), the RXNMapper cache parts and the ``map_missing``
    shards carry no such column, and giving them one is separate work. The benchmark labels and
    their splits **do** carry one, since ADR 009 item 3. ``tests/test_io.py`` pins the gap so that
    it is not tidied into a refusal unread.
    """
    licences = (
        set(pc.unique(table["licence"]).to_pylist())
        if "licence" in table.column_names
        else set()
    )
    if licences and licences != {OPEN_LICENCE} and not _under_private(path, root):
        raise ValueError(
            "only rows licensed open may be written outside private/; this table carries "
            f"{sorted(str(value) for value in licences)}: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _under_private(path: Path, root: Path) -> bool:
    """Do the bytes really land in a ``private/`` folder below ``root``?

    Asked of the resolved path, not of the string. Until 2026-10-22 this was a lexical test on
    ``path.relative_to(root).parts``, which two things defeated: ``root/private/../library/x`` reads
    as containing ``private`` while the file lands in ``root/library``, and a ``private`` symlink can
    point anywhere at all. Resolving both sides normalises ``..`` away and follows the symlink, so
    the question becomes the one we meant to ask. ``strict=False`` because the folder is created
    after this check. A path that resolves outside ``root`` is not private, which also keeps a
    system folder such as macOS's ``/private/var`` from passing: only folders below ``root`` count.
    """
    try:
        inside = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return "private" in inside.parts


def write_report(
    path: Path, report: Mapping[str, object], configs: Sequence[Path] = ()
) -> None:
    """Write the stage report: the given counts plus git commit, versions, time and config hashes."""
    full = {
        **report,
        "configs": {str(config): sha256(config) for config in configs},
        "git_commit": _git_commit(),
        "versions": {name: _version(name) for name in PACKAGES},
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    """Hash of a file's bytes, so a report says exactly which config produced it.

    Public because a stage that stamps the hash on its own rows needs the same function the report
    uses: ``scripts/score_baselines.py`` writes it into a ``config_sha256`` column, and had its own
    copy of this line until 2026-10-22.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    """The current commit, with ``-dirty`` when the working tree has changes; None outside git."""
    here = Path(__file__).parent
    try:
        commit = _git(here, "rev-parse", "HEAD")
        dirty = _git(here, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if dirty else commit


def _git(cwd: Path, *args: str) -> str:
    """Run one git command and return its stripped output."""
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _version(package: str) -> str | None:
    """Installed version of a package, or None when it is not installed."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def read_table(path: Path, columns: list[str], what: str) -> pd.DataFrame:
    """One input table, with an explicit column list, or a ValueError naming which input it was.

    The column list is as much the point as the reading: a stage that names its columns cannot be
    handed one it never asked for, which is how stage S keeps a benchmark label out of the process.

    pandas names the missing path on its own, which is the less useful half. Several of the paths a
    stage builds come from **config keys** -- a stage U stem, an S2a stem, a feature block -- so a
    reader needs to know which key sent it to a file that is not there. This lived in
    ``scripts/score_pu.py`` until 2026-11-02, when stage S2b needed the same thing and a second copy
    would have been the duplication §5 forbids.

    Raises:
        ValueError: no file at ``path``, naming ``what`` and the path.
    """
    if not path.exists():
        raise ValueError(f"{what} is missing: no file at {path}")
    # The columns are checked against the schema rather than left to the reader, because pyarrow's
    # own refusal is `No match for FieldRef.Name(morgan_near) in key_stereo: string\nmorgan: ...`,
    # an internal schema dump that names neither the input nor the config key that asked for the
    # column, and gives no hint that the stage which writes it must be re-run.
    absent = [name for name in columns if name not in set(pq.read_schema(path).names)]
    if absent:
        raise ValueError(f"{what} lacks column(s) {absent}: {path}")
    return pd.read_parquet(path, columns=columns)
