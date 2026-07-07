"""Split the curated BH-HTE dataset into test / literature / dropped partitions.

The raw file ``BH_HTE_Curated_Dataset_v20260219.csv`` bundles Buchwald–Hartwig
C–N coupling reactions from eight literature/HTE sources. For this project we
partition it into three disjoint groups, keyed purely on the ``Source`` column:

* **test** — ``JNJ HTE 2024`` (Neves et al.). Held out for final evaluation
  (and optionally an internal validation split). Never used for training.
* **literature** — every remaining source *except* Ahneman. These are the
  candidate literature-HTE positives for PU training.
* **dropped** — ``Dreher Doyle 2018``, i.e. Ahneman et al. *Science* 2018. This
  is the isoxazole-*additive* screen and it is the only source carrying an
  additive dimension, but the curated file has **no additive-identity column**
  (only engineered ``Additive_*`` flags). Twenty-three distinct additive
  experiments therefore collapse to identical reaction featurisations with
  different yields, making these rows irreducibly ambiguous. They are removed.

The splitter is deliberately conservative: it validates that the expected
sources are present, that the three partitions are disjoint and exhaustive
(no row is lost or duplicated), and that Ahneman is indeed the only
additive-bearing source before writing anything.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
from pydantic import BaseModel, Field

# --- Source-label constants (verbatim values in the raw ``Source`` column) ----

#: Source label of the held-out HTE test set (Neves et al. JnJ 2024 well plates).
TEST_SOURCE: Final[str] = "JNJ HTE 2024"

#: Source label of the Ahneman et al. Science 2018 additive screen, dropped
#: because the additive identity is missing from the curated file.
AHNEMAN_SOURCE: Final[str] = "Dreher Doyle 2018"

#: Raw ``Source`` values expected to remain as literature-HTE training data.
_EXPECTED_LITERATURE_SOURCES: Final[frozenset[str]] = frozenset(
    {
        "Merck 2025",
        "Roche 2023",
        "HiTEA 2024",
        "Dreher 2014",
        "Dreher 2016",
        "AZ ELN 2023",
    }
)

#: Column holding the (engineered) flag for whether an additive was present.
_ADDITIVE_PRESENT_COLUMN: Final[str] = "Additive_present"

#: Original (unnamed) index column produced by the upstream export.
_RAW_INDEX_HEADER: Final[str] = "Unnamed: 0"

#: Name we give to the preserved original row index for provenance.
_RAW_ROW_ID_COLUMN: Final[str] = "raw_row_id"


class DatasetSplit(BaseModel):
    """Per-source statistics for a single output partition.

    Attributes:
        name: Partition name (``"test"``, ``"literature"`` or ``"dropped"``).
        n_rows: Total number of reactions in the partition.
        sources: Mapping of ``Source`` label to row count.
        label_balance: Mapping of the ``labels`` value ("0"/"1") to row count.
    """

    name: str
    n_rows: int
    sources: dict[str, int] = Field(default_factory=dict)
    label_balance: dict[str, int] = Field(default_factory=dict)


class SplitReport(BaseModel):
    """Machine- and human-readable provenance record for a split run.

    Written next to the output partitions so a later session (human or agent)
    can audit exactly how ``datasets/processed`` was produced without re-running.

    Attributes:
        raw_path: Path of the source CSV.
        raw_sha256: SHA-256 of the source CSV (immutability check).
        raw_n_rows: Total rows in the source CSV.
        test_source: Source label routed to the test partition.
        dropped_source: Source label removed from training.
        partitions: One :class:`DatasetSplit` per output partition.
    """

    raw_path: str
    raw_sha256: str
    raw_n_rows: int
    test_source: str
    dropped_source: str
    partitions: list[DatasetSplit]


@dataclass(frozen=True, slots=True)
class RawDatasetSplitter:
    """Partition the curated BH-HTE CSV by ``Source`` into test/literature/dropped.

    The splitter is a pure, side-effect-free transformer until :meth:`write` is
    called. Construction only records configuration; :meth:`load` reads the CSV
    and :meth:`split` computes the partitions.

    Attributes:
        raw_csv_path: Path to ``BH_HTE_Curated_Dataset_v20260219.csv``.
        test_source: ``Source`` value to route to the test partition.
        ahneman_source: ``Source`` value to drop entirely.
    """

    raw_csv_path: Path
    test_source: str = TEST_SOURCE
    ahneman_source: str = AHNEMAN_SOURCE

    # -- Loading ------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Read the raw CSV, preserving the original row index for provenance.

        Returns:
            The full dataset with the upstream unnamed index renamed to
            ``raw_row_id`` so every processed row can be traced to the source.

        Raises:
            FileNotFoundError: If :attr:`raw_csv_path` does not exist.
        """
        if not self.raw_csv_path.is_file():
            raise FileNotFoundError(f"Raw dataset not found: {self.raw_csv_path}")
        frame = pd.read_csv(self.raw_csv_path, low_memory=False)
        if _RAW_INDEX_HEADER in frame.columns:
            frame = frame.rename(columns={_RAW_INDEX_HEADER: _RAW_ROW_ID_COLUMN})
        return frame

    # -- Splitting ----------------------------------------------------------

    def split(self, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Partition ``frame`` into ``test``, ``literature`` and ``dropped``.

        Args:
            frame: The full dataset as returned by :meth:`load`.

        Returns:
            Mapping with keys ``"test"``, ``"literature"`` and ``"dropped"``.

        Raises:
            ValueError: If expected sources are missing, if the additive
                invariant is violated, or if the partitions are not a disjoint
                exhaustive cover of the input rows.
        """
        self._validate_sources(frame)
        self._validate_additive_invariant(frame)

        source = frame["Source"]
        test_mask = source == self.test_source
        dropped_mask = source == self.ahneman_source
        literature_mask = ~(test_mask | dropped_mask)

        partitions = {
            "test": frame.loc[test_mask].copy(),
            "literature": frame.loc[literature_mask].copy(),
            "dropped": frame.loc[dropped_mask].copy(),
        }
        self._validate_partition_cover(frame, partitions)
        return partitions

    # -- Persistence --------------------------------------------------------

    def write(
        self,
        partitions: dict[str, pd.DataFrame],
        output_dir: Path,
        *,
        write_dropped: bool = True,
        write_parquet: bool = True,
    ) -> dict[str, Path]:
        """Write partitions to ``output_dir`` as CSV (and optionally Parquet).

        Args:
            partitions: Output of :meth:`split`.
            output_dir: Destination directory (created if absent).
            write_dropped: If ``True``, also write the dropped Ahneman rows
                (kept for transparency, not for training).
            write_parquet: If ``True``, additionally write Parquet copies.

        Returns:
            Mapping of partition name to the CSV path written.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        filenames = {
            "test": "test_jnj_hte_2024",
            "literature": "literature_hte",
            "dropped": "dropped_ahneman_dreher_doyle_2018",
        }
        written: dict[str, Path] = {}
        for name, frame in partitions.items():
            if name == "dropped" and not write_dropped:
                continue
            csv_path = output_dir / f"{filenames[name]}.csv"
            frame.to_csv(csv_path, index=False)
            written[name] = csv_path
            if write_parquet:
                frame.to_parquet(output_dir / f"{filenames[name]}.parquet", index=False)
        return written

    # -- Reporting ----------------------------------------------------------

    def report(self, frame: pd.DataFrame, partitions: dict[str, pd.DataFrame]) -> SplitReport:
        """Build a :class:`SplitReport` provenance record for this run.

        Args:
            frame: The full dataset (for row-count and checksum).
            partitions: Output of :meth:`split`.

        Returns:
            A populated :class:`SplitReport`.
        """
        return SplitReport(
            raw_path=str(self.raw_csv_path),
            raw_sha256=self._sha256(self.raw_csv_path),
            raw_n_rows=len(frame),
            test_source=self.test_source,
            dropped_source=self.ahneman_source,
            partitions=[self._describe_partition(name, part) for name, part in partitions.items()],
        )

    # -- Internal helpers ---------------------------------------------------

    def _validate_sources(self, frame: pd.DataFrame) -> None:
        """Ensure the test, Ahneman and literature sources are all present."""
        present = set(frame["Source"].unique())
        required = {self.test_source, self.ahneman_source}
        missing = required - present
        if missing:
            raise ValueError(f"Expected sources missing from raw data: {sorted(missing)}")
        missing_lit = _EXPECTED_LITERATURE_SOURCES - present
        if missing_lit:
            raise ValueError(f"Expected literature sources missing: {sorted(missing_lit)}")

    def _validate_additive_invariant(self, frame: pd.DataFrame) -> None:
        """Confirm Ahneman is the only source with additive-bearing rows.

        This is the concrete justification for dropping Ahneman: it is the sole
        source whose reactions vary an (unrecorded) additive, so its removal
        does not silently discard additive information from any other source.
        """
        if _ADDITIVE_PRESENT_COLUMN not in frame.columns:
            return
        additive_rows = frame[frame[_ADDITIVE_PRESENT_COLUMN].astype(str) == "1"]
        offending = set(additive_rows["Source"].unique()) - {self.ahneman_source}
        if offending:
            raise ValueError(
                "Additive-bearing rows found outside the dropped Ahneman source: "
                f"{sorted(offending)}. Review before dropping."
            )

    @staticmethod
    def _validate_partition_cover(frame: pd.DataFrame, partitions: dict[str, pd.DataFrame]) -> None:
        """Assert the partitions form a disjoint, exhaustive cover of ``frame``."""
        total = sum(len(part) for part in partitions.values())
        if total != len(frame):
            raise ValueError(
                f"Partition row counts sum to {total}, expected {len(frame)} "
                "(rows lost or duplicated)."
            )

    @staticmethod
    def _describe_partition(name: str, frame: pd.DataFrame) -> DatasetSplit:
        """Summarise one partition into a :class:`DatasetSplit`."""
        sources = frame["Source"].value_counts().to_dict()
        labels = frame["labels"].astype(str).value_counts().to_dict()
        return DatasetSplit(
            name=name,
            n_rows=len(frame),
            sources={str(k): int(v) for k, v in sources.items()},
            label_balance={str(k): int(v) for k, v in labels.items()},
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        """Return the hex SHA-256 digest of ``path`` (streamed, memory-safe)."""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
