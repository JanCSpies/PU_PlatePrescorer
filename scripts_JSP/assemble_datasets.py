"""Assemble the three condition-aware datasets (TEST / P / U) in one place.

This is the JSP branch's **single-file dataset assembly**. It produces the three
datasets the condition-aware transfer experiment needs, all sharing **one unified
condition schema** so positives, unlabeled candidates, and the evaluation plate live
in the same feature space:

* **TEST** — the JnJ HTE 2024 plate (held-out evaluation), with the substrate-disjoint
  VAL/TEST assignment of :class:`~bhpu.eval.protocol.FrozenEvalProtocol` annotated and
  its explicit ``Catalyst/Base/Solvent SMILES`` canonicalised into the unified schema.
* **P** — reported Pd-catalysed Buchwald–Hartwig couplings, **with conditions preserved**
  (catalyst/ligand/base/**solvent**), deduplicated per ``(pair, condition)``, and leakage-cleaned
  against JnJ substrates. Mined **per source** — ORD (open) and Pistachio (licensed) — into
  separate ``positives_<source>_bh.parquet`` files, then combined in a final
  :meth:`DatasetAssembler.merge_positives` step into ``positives_merged_bh.parquet``. Keeping the
  sources separate lets a consumer pick ORD-only / Pistachio-only / merged without re-mining.
* **U** — the condition-spanning unlabeled grid: unmeasured JnJ aryl×amine pairs crossed
  with conditions sampled from the JnJ condition distribution (mirrors ``scripts/17``).

Licensing: Pistachio is licensed, so Pistachio-derived and merged corpora are written to
``<out_dir>/private/`` (gitignored) and must never be published. Each source is mined once
(strict + relaxed variants share the scan); ``--jobs N`` parallelises detection.

Why this file exists (see ``JSP_WORKPLAN.md`` and ``docs/decisions/006-pd-bh-conditions.md``):
the shipped :func:`~bhpu.data.mining.iter_orderly` reader **drops the ``solvent_*`` columns**,
so mined conditions silently lose their solvent. ORDerly *forward* rows do carry
``agent_*`` (catalyst/ligand/base) **and** ``solvent_*`` **and** ``temperature`` as canonical
SMILES; this assembler pools reactants + agents + solvents into the molecule bag so the
canonicaliser (:func:`~bhpu.chem.conditions.condition_from_reagents`, invoked inside
:meth:`~bhpu.data.mining.BHCouplingDetector.detect`) recovers the full condition.

The heavy chemistry (product-confirmed BH detection, precatalyst⇄metal+ligand
canonicalisation, the frozen leakage protocol) is **reused from** ``src/bhpu`` — this file
only orchestrates it and owns the ORD reader + per-``(pair, condition)`` aggregation.

Run::

    # fast smoke test (few ORD rows, no Pistachio):
    uv run python scripts_JSP/assemble_datasets.py --limit 50000
    # full ORD only:
    uv run python scripts_JSP/assemble_datasets.py --jobs 16
    # full ORD + Pistachio (heavy → prefer SLURM: sbatch scripts_JSP/assemble_datasets.sbatch):
    uv run python scripts_JSP/assemble_datasets.py --pistachio <pistachio.smi.deduped.csv> --jobs 32
    # re-combine already-mined per-source corpora without re-mining:
    uv run python scripts_JSP/assemble_datasets.py --merge-only

By default a Pd reaction that also contains a Cu/Ni metal is thrown out (mixed additive system);
pass ``--keep-cu-ni`` to retain them.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from bhpu.chem.canonical import inchikey, strip_and_canonicalise
from bhpu.chem.conditions import CanonicalCondition, condition_from_columns
from bhpu.chem.reaction import build_cn_coupling_product
from bhpu.data import schema
from bhpu.data.mining import BHCouplingDetector, RawReaction, iter_pistachio_csv
from bhpu.eval.protocol import FrozenEvalProtocol

# --- Unified condition schema (added to all three datasets) -------------------

#: Metal(s) of the catalytic system, ``|``-joined element symbols (e.g. ``"Pd"``).
METAL: str = "metal"
#: Free-ligand (phosphine/donor) canonical SMILES, precatalyst⇄metal+ligand unified.
LIGAND_SMILES: str = "ligand_smiles"
#: Connectivity InChIKey of :data:`LIGAND_SMILES` (cross-source join key).
LIGAND_KEY: str = "ligand_key"
#: Base anion canonical SMILES (Cs2CO3 ≡ K2CO3 collapse to the anion).
COND_BASE_SMILES: str = "cond_base_smiles"
#: Connectivity InChIKey of :data:`COND_BASE_SMILES`.
COND_BASE_KEY: str = "cond_base_key"
#: Solvent canonical SMILES.
COND_SOLVENT_SMILES: str = "cond_solvent_smiles"
#: Connectivity InChIKey of :data:`COND_SOLVENT_SMILES`.
COND_SOLVENT_KEY: str = "cond_solvent_key"

#: Ordered unified condition columns, guaranteed present in TEST, P and U.
UNIFIED_CONDITION_COLUMNS: tuple[str, ...] = (
    METAL,
    LIGAND_SMILES,
    LIGAND_KEY,
    COND_BASE_SMILES,
    COND_BASE_KEY,
    COND_SOLVENT_SMILES,
    COND_SOLVENT_KEY,
)

#: Transition metals detectable in a reagent bag (raw scan). Matches the canonicaliser's set
#: (``bhpu.chem.conditions._METAL_ATOMIC_NUMBERS``).
_METAL_SYMBOLS: tuple[str, ...] = ("Pd", "Cu", "Ni", "Pt", "Au", "Rh", "Ru", "Co", "Fe")

#: Metals that actually catalyse an aryl C–N coupling. A metal outside this set that appears in
#: the bag is ligand-borne (the **Fe** of a dppf/dtbpf ferrocene) or a spectator — not the
#: catalyst — so the reported ``metal`` field is restricted to this set.
_CN_CATALYTIC_METALS: frozenset[str] = frozenset({"Pd", "Cu", "Ni"})

#: Non-Pd C–N-coupling metals. A Pd reaction that *also* contains one of these is a mixed
#: Cu/Ni-additive system; with ``reject_cu_ni`` it is thrown out so P stays cleanly Pd-only.
_NON_PD_METALS: frozenset[str] = frozenset({"Cu", "Ni"})

#: Default raw Pistachio reaction-SMILES CSV (licensed — outputs stay in ``private/``).
DEFAULT_PISTACHIO: Path = Path(
    "/mnt/datasets/pistachio/releases/latest/data/pistachio.smi.deduped.csv"
)

#: Positive sources that are open (publishable) vs licensed (gitignored, ``private/`` subdir).
_LICENSED_SOURCES: frozenset[str] = frozenset({"pistachio"})

#: The merged corpus is named for this pseudo-source.
_MERGED_SOURCE: str = "merged"

#: A positive-source reader: a zero-arg callable streaming ``(reaction, temperature)`` pairs.
_Reader = Callable[[], Iterator[tuple[RawReaction, float | None]]]


def _is_smiles(value: object) -> bool:
    """Return ``True`` if ``value`` is a non-empty, non-null SMILES-looking string.

    Args:
        value: A raw parquet cell.

    Returns:
        Whether the cell should be treated as a SMILES string.
    """
    return isinstance(value, str) and bool(value) and value.strip().lower() != "nan"


def _substrate_key(smiles: object) -> str | None:
    """Return the salt-stripped connectivity InChIKey identity for a substrate.

    Matches :func:`bhpu.eval.protocol._substrate_key` so leakage checks here use the
    same identity as the frozen protocol.

    Args:
        smiles: A substrate SMILES (or non-string / empty, which yields ``None``).

    Returns:
        The connectivity InChIKey, or ``None`` if unparsable.
    """
    if not isinstance(smiles, str) or not smiles:
        return None
    return inchikey(strip_and_canonicalise(smiles) or smiles)


def _metals_in_bag(molecules: tuple[str, ...]) -> frozenset[str]:
    """Return the transition metals present anywhere in a molecule bag.

    Metals only enter via the catalyst/precatalyst, never via substrates, so scanning
    the full bag is safe. Element symbols in SMILES are case-sensitive two-letter tokens
    (``Pd``/``Cu``/``Ni`` …), so a substring test is unambiguous here.

    Args:
        molecules: The reagent + reactant SMILES bag.

    Returns:
        The set of detected metal element symbols (empty if metal-free).
    """
    joined = " ".join(molecules)
    return frozenset(symbol for symbol in _METAL_SYMBOLS if symbol in joined)


def _catalytic_metals(molecules: tuple[str, ...]) -> frozenset[str]:
    """Return the metals in a bag that can catalyse an aryl C–N coupling (Pd/Cu/Ni).

    Restricting to the C–N-active metals excludes any metal that is present only as part of a
    *ligand* — most importantly the **Fe** of a ferrocene backbone (dppf/dtbpf), which is not
    catalytically active. (The canonicaliser's ``extract_ligand`` strips that Fe out of the
    ligand SMILES, so it cannot be recovered from the ligand — hence the intersection here.)
    For a Pd/dppf Buchwald–Hartwig system this yields ``{"Pd"}``, not the misleading ``Fe|Pd``.

    Args:
        molecules: The full reagent + reactant SMILES bag.

    Returns:
        The catalytic metal element symbols (a subset of {Pd, Cu, Ni}).
    """
    return _metals_in_bag(molecules) & _CN_CATALYTIC_METALS


def _condition_columns(condition: CanonicalCondition) -> dict[str, object]:
    """Project a :class:`CanonicalCondition` onto the unified condition columns.

    This single projection is applied to TEST, P and U alike, which is what guarantees
    the three datasets share one condition feature space.

    The reported ``metal`` is restricted to the C–N-catalytic set (:data:`_CN_CATALYTIC_METALS`)
    so that a ferrocene ligand's **Fe** never appears as a metal — applied here, in the single
    projection shared by TEST, P and U, so all three agree (JnJ's dppf/dtbpf/josiphos/qphos-Pd
    precatalysts would otherwise read ``Fe|Pd`` in TEST/U but ``Pd`` in P).

    Args:
        condition: A canonicalised condition (from columns or from a reagent bag).

    Returns:
        A mapping keyed by :data:`UNIFIED_CONDITION_COLUMNS`.
    """
    metals = condition.metals & _CN_CATALYTIC_METALS
    return {
        METAL: "|".join(sorted(metals)) if metals else None,
        LIGAND_SMILES: condition.ligand_smiles,
        LIGAND_KEY: condition.ligand_key,
        COND_BASE_SMILES: condition.base_smiles,
        COND_BASE_KEY: condition.base_key,
        COND_SOLVENT_SMILES: condition.solvent_smiles,
        COND_SOLVENT_KEY: condition.solvent_key,
    }


@dataclass(slots=True)
class AssemblyConfig:
    """Configuration for a dataset-assembly run.

    Attributes:
        ord_forward: Raw ORDerly *forward* parquet (carries conditions + solvents).
        pistachio: Raw Pistachio ``rxn_smiles`` CSV (licensed). ``None`` skips Pistachio.
        jnj_test: The processed JnJ HTE 2024 plate (TEST source, condition-labelled).
        out_dir: Destination directory for the datasets + provenance report. Licensed
            (Pistachio-derived) outputs are written to the ``private/`` subdirectory.
        require_pd: Keep only Pd-catalysed couplings as positives (reject Cu/Ni/SNAr).
        reject_cu_ni: Also throw out Pd reactions that *co-contain* a Cu or Ni metal (mixed
            Cu/Ni-additive systems), so P is cleanly Pd-only.
        require_phosphine: Additionally require a phosphorus-bearing ligand (Pd + phosphine).
        conditions_per_pair: Conditions sampled per unmeasured pair when building U.
        seed: RNG seed for the U condition sampling and the frozen split.
        val_fraction: Fraction of JnJ substrates assigned to VAL (rest TEST).
        limit: Optional cap on ORD rows scanned (for fast smoke tests); ``None`` = full.
        batch_size: ORD parquet rows per Arrow batch.
        jobs: Worker processes for the ORD detection scan (``1`` = serial). >1 uses a
            ``forkserver`` pool — safe with RDKit (see ``docs/slurm.md``).
        chunk_size: Reactions dispatched per worker task when ``jobs`` > 1.
    """

    ord_forward: Path = Path("/mnt/datasets/ORDerly/orderly_forward_train.parquet")
    pistachio: Path | None = None
    jnj_test: Path = Path("datasets/processed/test_jnj_hte_2024.parquet")
    out_dir: Path = Path("final_datasets")
    require_pd: bool = True
    reject_cu_ni: bool = True
    require_phosphine: bool = False
    conditions_per_pair: int = 8
    seed: int = 1234
    val_fraction: float = 0.5
    limit: int | None = None
    batch_size: int = 20_000
    jobs: int = 1
    chunk_size: int = 5_000


@dataclass(slots=True)
class PositivesReport:
    """Provenance + leakage accounting for one P (positives) build.

    Attributes:
        allow_shared_substrate: Leakage mode. ``False`` (strict) drops any detection sharing a
            JnJ aryl/amine; ``True`` (relaxed) keeps single-substrate overlaps and drops only
            exact JnJ pairs.
        n_scanned: ORD rows examined.
        n_detected: Rows confirmed as (Pd-)BH aryl C–N couplings.
        n_shares_substrate: Detections sharing at least one JnJ aryl/amine (informational).
        n_exact_jnj_pair_hits: Detections whose (aryl, amine) exactly equals a JnJ pair.
        n_dropped_leakage: Detections dropped by the active leakage rule.
        n_dropped_no_phosphine: Detections dropped by the optional phosphine filter.
        n_dropped_cu_ni: Detections dropped for co-containing a Cu/Ni metal (Pd-only cleanup).
        n_rows: Final ``(pair, condition)`` rows written.
        n_unique_pairs: Distinct substrate pairs among the written rows.
        n_with_solvent: Rows carrying a non-null solvent (solvent-capture sanity check).
        n_with_ligand: Rows carrying a non-null ligand.
        n_with_base: Rows carrying a non-null base.
        n_jnj_ligand_overlap: Written rows whose ligand key is also a JnJ ligand key.
    """

    allow_shared_substrate: bool = False
    n_scanned: int = 0
    n_detected: int = 0
    n_shares_substrate: int = 0
    n_exact_jnj_pair_hits: int = 0
    n_dropped_leakage: int = 0
    n_dropped_no_phosphine: int = 0
    n_dropped_cu_ni: int = 0
    n_rows: int = 0
    n_unique_pairs: int = 0
    n_with_solvent: int = 0
    n_with_ligand: int = 0
    n_with_base: int = 0
    n_jnj_ligand_overlap: int = 0


@dataclass(slots=True)
class _ScanStats:
    """Counters for a single source's detection pass (shared by both P variants)."""

    n_scanned: int = 0
    n_detected: int = 0
    n_dropped_no_phosphine: int = 0
    n_dropped_cu_ni: int = 0


@dataclass(slots=True)
class _Detection:
    """One confirmed Pd-BH detection, pre-leakage-filter, with its leakage flags.

    Detections are collected once and reused to build both the strict and relaxed P sets.
    """

    aryl_smiles: str
    amine_smiles: str
    product_smiles: str
    aryl_key: str
    amine_key: str
    condition: CanonicalCondition
    source: str
    yield_value: float | None
    temperature: float | None
    raw_id: str
    is_exact_pair: bool
    shares_substrate: bool


@dataclass(frozen=True, slots=True)
class _RawHit:
    """A picklable confirmed detection, as returned by a (possibly worker-process) scan.

    Substrate keys and leakage flags are *not* computed here — that needs the frozen
    protocol, which lives in the parent process — so a worker only returns the raw fields
    plus the already-restricted catalytic metals.
    """

    aryl_smiles: str
    amine_smiles: str
    product_smiles: str
    ligand_smiles: str | None
    base_smiles: str | None
    solvent_smiles: str | None
    metals: tuple[str, ...]
    source: str
    yield_value: float | None
    temperature: float | None
    raw_id: str


def _detect_reaction(
    reaction: RawReaction,
    temperature: float | None,
    *,
    require_pd: bool,
    require_phosphine: bool,
    reject_cu_ni: bool,
    detector: BHCouplingDetector,
) -> tuple[str, _RawHit | None]:
    """Run BH detection on one reaction and package the result.

    Args:
        reaction: The candidate reaction (solvents already pooled into the bag).
        temperature: The row temperature (carried through; the mining record has no slot).
        require_pd: Require a Pd species (reject Cu/Ni/metal-free).
        require_phosphine: Additionally require a phosphorus-bearing ligand.
        reject_cu_ni: Throw out a Pd reaction that also co-contains a Cu/Ni metal.
        detector: The BH coupling detector (one per process).

    Returns:
        ``("none", None)`` if not a coupling; ``("phosphine", None)`` / ``("cu_ni", None)`` if
        dropped by the respective filter; else ``("hit", raw_hit)``.
    """
    match = detector.detect(reaction, require_pd=require_pd)
    if match is None:
        return "none", None
    if require_phosphine and not _has_phosphine(match.ligand_smiles):
        return "phosphine", None
    metals = _catalytic_metals(reaction.molecules)
    if reject_cu_ni and metals & _NON_PD_METALS:
        return "cu_ni", None  # mixed Cu/Ni-additive system — not clean Pd-BH
    hit = _RawHit(
        aryl_smiles=match.aryl_smiles,
        amine_smiles=match.amine_smiles,
        product_smiles=match.product_smiles,
        ligand_smiles=match.ligand_smiles,
        base_smiles=match.base_smiles,
        solvent_smiles=match.solvent_smiles,
        metals=tuple(sorted(metals)),
        source=reaction.source,
        yield_value=match.yield_value,
        temperature=temperature,
        raw_id=match.raw_id or "",
    )
    return "hit", hit


def _detect_chunk(
    payload: tuple[list[tuple[RawReaction, float | None]], bool, bool, bool],
) -> tuple[int, int, int, list[_RawHit]]:
    """Worker entry point: detect couplings in a chunk of reactions (own detector).

    Args:
        payload: ``(reactions, require_pd, require_phosphine, reject_cu_ni)`` — a chunk + flags.

    Returns:
        ``(n_scanned, n_dropped_no_phosphine, n_dropped_cu_ni, hits)`` for the chunk.
    """
    reactions, require_pd, require_phosphine, reject_cu_ni = payload
    detector = BHCouplingDetector()
    hits: list[_RawHit] = []
    n_dropped_phosphine = 0
    n_dropped_cu_ni = 0
    for reaction, temperature in reactions:
        status, hit = _detect_reaction(
            reaction,
            temperature,
            require_pd=require_pd,
            require_phosphine=require_phosphine,
            reject_cu_ni=reject_cu_ni,
            detector=detector,
        )
        if status == "phosphine":
            n_dropped_phosphine += 1
        elif status == "cu_ni":
            n_dropped_cu_ni += 1
        elif hit is not None:
            hits.append(hit)
    return len(reactions), n_dropped_phosphine, n_dropped_cu_ni, hits


def _chunked(
    iterable: Iterable[tuple[RawReaction, float | None]], size: int
) -> Iterator[list[tuple[RawReaction, float | None]]]:
    """Yield successive ``size``-length chunks from a reaction stream (bounded memory)."""
    chunk: list[tuple[RawReaction, float | None]] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


@dataclass(slots=True)
class _ConditionAggregate:
    """Mutable accumulator for all records of one ``(pair, condition)`` signature."""

    aryl_smiles: str
    amine_smiles: str
    product_smiles: str
    condition: CanonicalCondition
    sources: set[str]
    raw_ids: set[str]
    max_yield: float | None
    max_temperature: float | None
    n_records: int


class DatasetAssembler:
    """Build the TEST / P / U datasets in the unified condition schema.

    The assembler owns the ORD reader and the per-``(pair, condition)`` aggregation, and
    delegates chemistry to ``src/bhpu``: product-confirmed BH detection + Pd filter
    (:class:`~bhpu.data.mining.BHCouplingDetector`), condition canonicalisation
    (:func:`~bhpu.chem.conditions.condition_from_columns`), and the substrate-disjoint
    leakage protocol (:class:`~bhpu.eval.protocol.FrozenEvalProtocol`).

    Attributes:
        config: The run configuration.
        protocol: The frozen JnJ eval protocol (split + leakage reference).
        positives_report: Provenance for the strict P build.
        positives_report_relaxed: Provenance for the relaxed (shared-substrate) P build.
    """

    def __init__(self, config: AssemblyConfig) -> None:
        """Initialise the assembler and load the JnJ plate + frozen protocol.

        Args:
            config: The assembly configuration.
        """
        self.config = config
        self._jnj: pd.DataFrame = pd.read_parquet(config.jnj_test).reset_index(drop=True)
        self.protocol = FrozenEvalProtocol(
            self._jnj, seed=config.seed, val_fraction=config.val_fraction
        )
        self._detector = BHCouplingDetector()
        self.positives_report = PositivesReport()
        self.positives_report_relaxed = PositivesReport(allow_shared_substrate=True)
        self._jnj_ligand_keys: frozenset[str] = self._compute_jnj_ligand_keys()
        # Detection scan is expensive → cache per source; both P variants of a source reuse it.
        self._detections_by_source: dict[str, list[_Detection]] = {}
        self._scan_stats_by_source: dict[str, _ScanStats] = {}
        # Per-(source, allow_shared_substrate) provenance, populated as corpora are built.
        self._reports: dict[tuple[str, bool], PositivesReport] = {}

    # -- TEST ---------------------------------------------------------------

    def build_test(self) -> pd.DataFrame:
        """Assemble the TEST dataset from the JnJ plate with unified conditions.

        The full JnJ plate is emitted (so VAL is available for model selection and TEST
        for the single final touch); each row is tagged with its frozen ``eval_split`` and
        its explicit ``Catalyst/Base/Solvent SMILES`` canonicalised into the unified schema.

        Returns:
            The JnJ plate augmented with ``eval_split`` and the unified condition columns.
        """
        frame = self._jnj.copy()
        split = pd.Series("none", index=frame.index, dtype=object)
        split.iloc[self.protocol.val_index] = "val"
        split.iloc[self.protocol.test_index] = "test"
        frame["eval_split"] = split.to_numpy()

        condition_rows = [
            _condition_columns(
                condition_from_columns(
                    row.get(schema.CATALYST_SMILES),
                    row.get(schema.BASE_SMILES),
                    row.get(schema.SOLVENT_SMILES),
                )
            )
            for row in frame.to_dict("records")
        ]
        conditions = pd.DataFrame(condition_rows, index=frame.index)
        return pd.concat([frame, conditions], axis=1)

    # -- P (positives) ------------------------------------------------------

    def _iter_ord(self) -> Iterator[tuple[RawReaction, float | None]]:
        """Stream ORDerly-forward rows as reactions, **pooling solvents into the bag**.

        Reactants + agents + solvents are pooled as candidate molecules so that the
        condition canonicaliser sees the catalyst/ligand/base *and* the solvent. The row
        temperature is yielded alongside (the mining record has no temperature slot).

        Yields:
            ``(reaction, temperature)`` for each ORD row, up to ``config.limit`` rows.
        """
        parquet = pq.ParquetFile(self.config.ord_forward)
        available = set(parquet.schema.names)
        reactant_cols = [
            c for c in ("reactant_000", "reactant_001", "reactant_002") if c in available
        ]
        agent_cols = [c for c in ("agent_000", "agent_001", "agent_002") if c in available]
        solvent_cols = [c for c in ("solvent_000", "solvent_001", "solvent_002") if c in available]
        product_cols = [c for c in ("product_000", "product_001") if c in available]
        id_col = "original_index" if "original_index" in available else "index"
        extra = [c for c in ("temperature", "yield_000") if c in available]
        columns = [*reactant_cols, *agent_cols, *solvent_cols, *product_cols, id_col, *extra]

        scanned = 0
        for batch in parquet.iter_batches(batch_size=self.config.batch_size, columns=columns):
            frame = batch.to_pandas()
            for row in frame.itertuples(index=False):
                data = row._asdict()
                molecules = tuple(
                    str(data[c])
                    for c in (*reactant_cols, *agent_cols, *solvent_cols)
                    if _is_smiles(data.get(c))
                )
                products = tuple(str(data[c]) for c in product_cols if _is_smiles(data.get(c)))
                reaction = RawReaction(
                    molecules=molecules,
                    products=products,
                    source="ORD",
                    raw_id=str(data.get(id_col)),
                    yield_value=_to_float(data.get("yield_000")),
                )
                yield reaction, _to_float(data.get("temperature"))
                scanned += 1
                if self.config.limit is not None and scanned >= self.config.limit:
                    return

    def _iter_pistachio(self) -> Iterator[tuple[RawReaction, float | None]]:
        """Stream raw Pistachio reactions (``reactants>agents>products`` SMILES).

        The agents block already pools catalyst/ligand/base/**solvent**, so conditions are
        captured natively; Pistachio records no temperature (yields ``None``). Atom maps are
        stripped downstream by the detector/canonicaliser.

        Yields:
            ``(reaction, None)`` for each parsable Pistachio reaction, up to ``config.limit``.
        """
        if self.config.pistachio is None:
            return
        for scanned, reaction in enumerate(
            iter_pistachio_csv(self.config.pistachio, source="Pistachio"), start=1
        ):
            yield reaction, None
            if self.config.limit is not None and scanned >= self.config.limit:
                return

    def _source_readers(self) -> dict[str, _Reader]:
        """Return the ``{source_name: reader_method}`` map of configured positive sources."""
        readers: dict[str, _Reader] = {"ord": self._iter_ord}
        if self.config.pistachio is not None:
            readers["pistachio"] = self._iter_pistachio
        return readers

    def _collect_detections(self, source: str, reader: _Reader) -> list[_Detection]:
        """Scan one source **once** (serial or parallel), returning detections + leakage flags.

        The result is cached per source so a source's strict and relaxed P variants share a
        single (expensive) pass. Detection runs serially or across ``config.jobs`` worker
        processes; substrate keys and the JnJ leakage flags are then attached in this process
        (they need the frozen protocol). The catalytic metal was already restricted to Pd/Cu/Ni
        during the scan, so ferrocene Fe never appears.

        Args:
            source: The positive source name (used as the cache key).
            reader: A zero-arg callable yielding ``(reaction, temperature)`` for the source.

        Returns:
            All detections passing detection + the optional phosphine filter.
        """
        if source in self._detections_by_source:
            return self._detections_by_source[source]
        hits, stats = self._scan(reader)
        jnj_pairs = self.protocol.jnj_pairs
        jnj_keys = self.protocol.jnj_substrate_keys
        detections: list[_Detection] = []
        for hit in hits:
            aryl_key = _substrate_key(hit.aryl_smiles)
            amine_key = _substrate_key(hit.amine_smiles)
            if aryl_key is None or amine_key is None:
                continue
            condition = CanonicalCondition(
                metals=frozenset(hit.metals),
                ligand_smiles=hit.ligand_smiles,
                base_smiles=hit.base_smiles,
                solvent_smiles=hit.solvent_smiles,
            )
            detections.append(
                _Detection(
                    aryl_smiles=hit.aryl_smiles,
                    amine_smiles=hit.amine_smiles,
                    product_smiles=hit.product_smiles,
                    aryl_key=aryl_key,
                    amine_key=amine_key,
                    condition=condition,
                    source=hit.source,
                    yield_value=hit.yield_value,
                    temperature=hit.temperature,
                    raw_id=hit.raw_id,
                    is_exact_pair=(aryl_key, amine_key) in jnj_pairs,
                    shares_substrate=aryl_key in jnj_keys or amine_key in jnj_keys,
                )
            )
        self._scan_stats_by_source[source] = stats
        self._detections_by_source[source] = detections
        return detections

    def _scan(self, reader: _Reader) -> tuple[list[_RawHit], _ScanStats]:
        """Detect couplings over a reaction stream, serially or via a worker pool."""
        if self.config.jobs > 1:
            return self._scan_parallel(reader)
        return self._scan_serial(reader)

    def _scan_serial(self, reader: _Reader) -> tuple[list[_RawHit], _ScanStats]:
        """Single-process detection scan over a reaction stream."""
        stats = _ScanStats()
        hits: list[_RawHit] = []
        for reaction, temperature in reader():
            stats.n_scanned += 1
            status, hit = _detect_reaction(
                reaction,
                temperature,
                require_pd=self.config.require_pd,
                require_phosphine=self.config.require_phosphine,
                reject_cu_ni=self.config.reject_cu_ni,
                detector=self._detector,
            )
            if status == "phosphine":
                stats.n_dropped_no_phosphine += 1
            elif status == "cu_ni":
                stats.n_dropped_cu_ni += 1
            elif hit is not None:
                stats.n_detected += 1
                hits.append(hit)
        return hits, stats

    def _scan_parallel(self, reader: _Reader) -> tuple[list[_RawHit], _ScanStats]:
        """Multi-process detection scan (``forkserver`` pool; RDKit-safe per docs/slurm.md)."""
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")  # avoid core oversubscription across workers
        stats = _ScanStats()
        hits: list[_RawHit] = []
        payloads = (
            (chunk, self.config.require_pd, self.config.require_phosphine, self.config.reject_cu_ni)
            for chunk in _chunked(reader(), self.config.chunk_size)
        )
        context = multiprocessing.get_context("forkserver")
        with context.Pool(processes=self.config.jobs) as pool:
            for n_scanned, n_phos, n_cu_ni, chunk_hits in pool.imap_unordered(
                _detect_chunk, payloads
            ):
                stats.n_scanned += n_scanned
                stats.n_dropped_no_phosphine += n_phos
                stats.n_dropped_cu_ni += n_cu_ni
                stats.n_detected += len(chunk_hits)
                hits.extend(chunk_hits)
        return hits, stats

    def build_positives(
        self, source: str = "ord", *, allow_shared_substrate: bool = False
    ) -> pd.DataFrame:
        """Assemble a P (positives) corpus for one source from its cached detections.

        Applies the active leakage rule and aggregates repeats per ``(pair, condition)``
        (max yield/temperature, union of sources). Records provenance in :attr:`_reports` and,
        for the default ``ord`` source, :attr:`positives_report` / ``positives_report_relaxed``.

        Args:
            source: Positive source name (``"ord"`` or ``"pistachio"``).
            allow_shared_substrate: If ``False`` (default, strict), drop any detection sharing
                a JnJ aryl or amine. If ``True`` (relaxed), keep single-substrate overlaps and
                drop only detections that reuse an **exact** JnJ pair (which would leak a
                measured test outcome).

        Returns:
            The positives corpus: one row per ``(aryl, amine, ligand, base, solvent)``.
        """
        readers = self._source_readers()
        if source not in readers:
            raise ValueError(f"Unknown positive source {source!r}; have {sorted(readers)}.")
        detections = self._collect_detections(source, readers[source])
        stats = self._scan_stats_by_source[source]
        report = PositivesReport(
            allow_shared_substrate=allow_shared_substrate,
            n_scanned=stats.n_scanned,
            n_detected=stats.n_detected,
            n_dropped_no_phosphine=stats.n_dropped_no_phosphine,
            n_dropped_cu_ni=stats.n_dropped_cu_ni,
            n_shares_substrate=sum(1 for d in detections if d.shares_substrate),
            n_exact_jnj_pair_hits=sum(1 for d in detections if d.is_exact_pair),
        )
        aggregates: dict[tuple[str | None, ...], _ConditionAggregate] = {}
        for det in detections:
            drop = det.is_exact_pair if allow_shared_substrate else det.shares_substrate
            if drop:
                report.n_dropped_leakage += 1
                continue
            signature = (
                det.aryl_key,
                det.amine_key,
                det.condition.ligand_key,
                det.condition.base_key,
                det.condition.solvent_key,
            )
            existing = aggregates.get(signature)
            if existing is None:
                aggregates[signature] = _ConditionAggregate(
                    aryl_smiles=det.aryl_smiles,
                    amine_smiles=det.amine_smiles,
                    product_smiles=det.product_smiles,
                    condition=det.condition,
                    sources={det.source},
                    raw_ids={det.raw_id} if det.raw_id else set(),
                    max_yield=det.yield_value,
                    max_temperature=det.temperature,
                    n_records=1,
                )
            else:
                existing.sources.add(det.source)
                if det.raw_id:
                    existing.raw_ids.add(det.raw_id)
                existing.max_yield = _max_optional(existing.max_yield, det.yield_value)
                existing.max_temperature = _max_optional(existing.max_temperature, det.temperature)
                existing.n_records += 1

        frame = self._positives_frame(aggregates)
        self._finalise_positives_report(report, frame)
        self._reports[(source, allow_shared_substrate)] = report
        if allow_shared_substrate:
            self.positives_report_relaxed = report
        else:
            self.positives_report = report
        return frame

    def _positives_frame(
        self, aggregates: dict[tuple[str | None, ...], _ConditionAggregate]
    ) -> pd.DataFrame:
        """Materialise the aggregated positives into a DataFrame in the unified schema."""
        rows: list[dict[str, object]] = []
        for aggregate in aggregates.values():
            row: dict[str, object] = {
                schema.ARYL_SMILES: aggregate.aryl_smiles,
                schema.AMINE_SMILES: aggregate.amine_smiles,
                "aryl_key": _substrate_key(aggregate.aryl_smiles),
                "amine_key": _substrate_key(aggregate.amine_smiles),
                schema.PRODUCT_SMILES: aggregate.product_smiles,
                "max_yield": aggregate.max_yield,
                "max_temperature": aggregate.max_temperature,
                "sources": "|".join(sorted(aggregate.sources)),
                "n_records": aggregate.n_records,
                "label": 1,
            }
            row.update(_condition_columns(aggregate.condition))
            rows.append(row)
        return pd.DataFrame(rows)

    def _finalise_positives_report(self, report: PositivesReport, frame: pd.DataFrame) -> None:
        """Fill the coverage counters of ``report`` from the assembled positives frame."""
        report.n_rows = len(frame)
        if frame.empty:
            return
        report.n_unique_pairs = int(
            frame[[schema.ARYL_SMILES, schema.AMINE_SMILES]].drop_duplicates().shape[0]
        )
        report.n_with_solvent = int(frame[COND_SOLVENT_SMILES].notna().sum())
        report.n_with_ligand = int(frame[LIGAND_SMILES].notna().sum())
        report.n_with_base = int(frame[COND_BASE_SMILES].notna().sum())
        report.n_jnj_ligand_overlap = int(frame[LIGAND_KEY].isin(self._jnj_ligand_keys).sum())

    # -- U (unlabeled grid) -------------------------------------------------

    def build_unlabeled(self) -> pd.DataFrame:
        """Assemble the condition-spanning unlabeled grid U.

        Mirrors ``scripts/17``'s ``_build_condition_grid``: every **unmeasured** JnJ
        aryl×amine pair is crossed with ``conditions_per_pair`` conditions sampled (without
        replacement, frequency-weighted) from the JnJ condition distribution, then each
        sampled condition is canonicalised into the unified schema.

        Returns:
            The unlabeled grid in the unified schema (no label column).
        """
        rng = np.random.default_rng(self.config.seed)
        aryls = sorted(self._jnj[schema.ARYL_SMILES].dropna().unique())
        amines = sorted(self._jnj[schema.AMINE_SMILES].dropna().unique())
        measured = set(
            zip(self._jnj[schema.ARYL_SMILES], self._jnj[schema.AMINE_SMILES], strict=True)
        )

        condition_cols = [schema.CATALYST_SMILES, schema.BASE_SMILES, schema.SOLVENT_SMILES]
        conditions = self._jnj[condition_cols].dropna().reset_index(drop=True)
        counts = conditions.groupby(condition_cols, sort=True).size()
        unique_conditions = counts.index.to_frame(index=False)
        weights = (counts / counts.sum()).to_numpy()
        canonical_cache: dict[tuple[str, str, str], dict[str, object]] = {}

        records: list[dict[str, object]] = []
        n_pick = min(self.config.conditions_per_pair, len(unique_conditions))
        for aryl in aryls:
            for amine in amines:
                if (aryl, amine) in measured:
                    continue
                product = build_cn_coupling_product(aryl, amine)  # persist so consumers need not re-derive
                picks = rng.choice(len(unique_conditions), size=n_pick, replace=False, p=weights)
                for idx in picks:
                    triple = unique_conditions.iloc[int(idx)]
                    key = (
                        triple[schema.CATALYST_SMILES],
                        triple[schema.BASE_SMILES],
                        triple[schema.SOLVENT_SMILES],
                    )
                    unified = canonical_cache.get(key)
                    if unified is None:
                        unified = _condition_columns(condition_from_columns(*key))
                        canonical_cache[key] = unified
                    records.append(
                        {
                            schema.ARYL_SMILES: aryl,
                            schema.AMINE_SMILES: amine,
                            schema.PRODUCT_SMILES: product,
                            **unified,
                        }
                    )
        return pd.DataFrame.from_records(records)

    # -- Orchestration ------------------------------------------------------

    def assemble(self) -> dict[str, object]:
        """Build all datasets, write per-source + merged corpora, and return a report.

        Writes (open) ``test.parquet`` and ``unlabeled_grid.parquet``; for each source,
        ``positives_<source>_bh.parquet`` (+ ``_relaxed``); and, when >1 source is configured,
        the combined ``positives_merged_bh.parquet`` (+ ``_relaxed``). **Licensed
        (Pistachio-derived) and merged corpora are written to ``out_dir/private/``.** Each source
        is mined independently and combined only in the final :meth:`merge_positives` step — so a
        consumer can use ORD-only, Pistachio-only, or merged without re-running the pipeline. Each
        variant's leakage contract is asserted before writing.

        Returns:
            The provenance report dict (also written as ``assembly_report.json``).
        """
        out_dir = self.config.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        test = self.build_test()
        unlabeled = self.build_unlabeled()
        test.to_parquet(out_dir / "test.parquet", index=False)
        unlabeled.to_parquet(out_dir / "unlabeled_grid.parquet", index=False)

        strict_by_source: dict[str, pd.DataFrame] = {}
        relaxed_by_source: dict[str, pd.DataFrame] = {}
        for source in self._source_readers():
            strict = self.build_positives(source, allow_shared_substrate=False)
            relaxed = self.build_positives(source, allow_shared_substrate=True)
            self._write_positives(source, strict, relaxed)
            strict_by_source[source] = strict
            relaxed_by_source[source] = relaxed

        merged_strict = self.merge_positives(list(strict_by_source.values()))
        merged_relaxed = self.merge_positives(list(relaxed_by_source.values()))
        if len(strict_by_source) > 1:  # a single source's "merge" is just itself — skip
            self._write_positives(_MERGED_SOURCE, merged_strict, merged_relaxed, force_private=True)

        report = self._report(
            test, unlabeled, strict_by_source, relaxed_by_source, merged_strict, merged_relaxed
        )
        (out_dir / "assembly_report.json").write_text(json.dumps(report, indent=2))
        return report

    def merge_existing(self) -> dict[str, object]:
        """Re-merge existing per-source corpora on disk **without re-mining**.

        Reads every ``positives_<source>_bh.parquet`` (and ``_relaxed``) under ``out_dir`` and
        ``out_dir/private`` (excluding any previously-merged file), merges them, and writes the
        combined corpus to ``out_dir/private``. Lets a consumer recombine sources cheaply after a
        full run (the point of keeping the sources separate).

        Returns:
            A small report of what was merged (also written as ``merge_report.json``).
        """
        strict_paths = self._existing_source_files(relaxed=False)
        relaxed_paths = self._existing_source_files(relaxed=True)
        merged_strict = self.merge_positives([pd.read_parquet(p) for p in strict_paths])
        merged_relaxed = self.merge_positives([pd.read_parquet(p) for p in relaxed_paths])
        self._write_positives(_MERGED_SOURCE, merged_strict, merged_relaxed, force_private=True)
        report: dict[str, object] = {
            "merged_from_strict": [str(p) for p in strict_paths],
            "merged_from_relaxed": [str(p) for p in relaxed_paths],
            "merged_strict_rows": len(merged_strict),
            "merged_relaxed_rows": len(merged_relaxed),
        }
        (self.config.out_dir / "merge_report.json").write_text(json.dumps(report, indent=2))
        return report

    @staticmethod
    def merge_positives(frames: list[pd.DataFrame]) -> pd.DataFrame:
        """Merge per-source positive corpora into one, deduping per ``(pair, condition)``.

        A ``(pair, condition)`` observed in multiple sources collapses to one row with the union
        of ``sources``, the max yield/temperature and the summed record count. Operates on
        already-mined frames (no re-mining), so sources can be recombined cheaply.

        Args:
            frames: Per-source positive corpora (identical unified schema).

        Returns:
            The merged corpus, or an empty frame if there are no non-empty inputs.
        """
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame()
        if len(frames) == 1:
            return frames[0].reset_index(drop=True)
        combined = pd.concat(frames, ignore_index=True)
        signature = [
            schema.ARYL_SMILES,
            schema.AMINE_SMILES,
            LIGAND_KEY,
            COND_BASE_KEY,
            COND_SOLVENT_KEY,
        ]
        first_cols = [
            "aryl_key",
            "amine_key",
            schema.PRODUCT_SMILES,
            "label",
            METAL,
            LIGAND_SMILES,
            COND_BASE_SMILES,
            COND_SOLVENT_SMILES,
        ]
        agg_spec: dict[str, tuple[str, object]] = {col: (col, "first") for col in first_cols}
        agg_spec["max_yield"] = ("max_yield", "max")
        agg_spec["max_temperature"] = ("max_temperature", "max")
        agg_spec["n_records"] = ("n_records", "sum")
        agg_spec["sources"] = ("sources", _merge_sources)
        merged = combined.groupby(signature, dropna=False, sort=False).agg(**agg_spec).reset_index()
        return merged

    def _write_positives(
        self,
        source: str,
        strict: pd.DataFrame,
        relaxed: pd.DataFrame,
        *,
        force_private: bool = False,
    ) -> None:
        """Assert each variant's leakage contract, then write the strict + relaxed corpora."""
        if not strict.empty:
            self.protocol.assert_training_clean(strict)  # strict: no shared substrate/pair
        if not relaxed.empty:
            self._assert_no_exact_pairs(relaxed)  # relaxed: no exact JnJ pair
        target = self._positives_dir(source, force_private=force_private)
        target.mkdir(parents=True, exist_ok=True)
        strict.to_parquet(target / f"positives_{source}_bh.parquet", index=False)
        relaxed.to_parquet(target / f"positives_{source}_bh_relaxed.parquet", index=False)

    def _positives_dir(self, source: str, *, force_private: bool = False) -> Path:
        """Return a source's output dir (``private/`` if licensed/merged, else ``out_dir``)."""
        if force_private or source in _LICENSED_SOURCES:
            return self.config.out_dir / "private"
        return self.config.out_dir

    def _existing_source_files(self, *, relaxed: bool) -> list[Path]:
        """List existing per-source positive parquets (excluding the merged corpus)."""
        pattern = "positives_*_bh_relaxed.parquet" if relaxed else "positives_*_bh.parquet"
        found: list[Path] = []
        for base in (self.config.out_dir, self.config.out_dir / "private"):
            if not base.exists():
                continue
            found.extend(p for p in sorted(base.glob(pattern)) if _MERGED_SOURCE not in p.name)
        return found

    def _assert_no_exact_pairs(self, frame: pd.DataFrame) -> None:
        """Raise if any row reuses an exact JnJ (aryl, amine) pair (measured-outcome leak).

        Args:
            frame: A positives corpus (the relaxed variant).

        Raises:
            AssertionError: If any row's substrate pair equals a JnJ pair.
        """
        pairs = self.protocol.jnj_pairs
        leaks = sum(
            1
            for aryl, amine in zip(
                frame[schema.ARYL_SMILES], frame[schema.AMINE_SMILES], strict=True
            )
            if (_substrate_key(aryl), _substrate_key(amine)) in pairs
        )
        if leaks:
            raise AssertionError(f"Relaxed positives leak {leaks} exact JnJ pairs.")

    def _report(
        self,
        test: pd.DataFrame,
        unlabeled: pd.DataFrame,
        strict_by_source: dict[str, pd.DataFrame],
        relaxed_by_source: dict[str, pd.DataFrame],
        merged_strict: pd.DataFrame,
        merged_relaxed: pd.DataFrame,
    ) -> dict[str, object]:
        """Build the provenance report dict from the assembled frames."""
        positives: dict[str, object] = {
            source: {
                "strict": asdict(self._reports[(source, False)]),
                "relaxed": asdict(self._reports[(source, True)]),
            }
            for source in strict_by_source
        }
        if len(strict_by_source) > 1:
            positives[_MERGED_SOURCE] = {
                "strict_rows": len(merged_strict),
                "relaxed_rows": len(merged_relaxed),
                "strict_unique_pairs": _n_unique_pairs(merged_strict),
            }
        return {
            "config": {
                "ord_forward": str(self.config.ord_forward),
                "pistachio": str(self.config.pistachio) if self.config.pistachio else None,
                "jnj_test": str(self.config.jnj_test),
                "require_pd": self.config.require_pd,
                "reject_cu_ni": self.config.reject_cu_ni,
                "require_phosphine": self.config.require_phosphine,
                "conditions_per_pair": self.config.conditions_per_pair,
                "seed": self.config.seed,
                "val_fraction": self.config.val_fraction,
                "limit": self.config.limit,
                "jobs": self.config.jobs,
            },
            "unified_condition_columns": list(UNIFIED_CONDITION_COLUMNS),
            "test": {
                "n_rows": len(test),
                "n_val": int((test["eval_split"] == "val").sum()),
                "n_test": int((test["eval_split"] == "test").sum()),
                "n_positive_labels": int((test[schema.LABEL] == 1).sum()),
                "n_with_solvent": int(test[COND_SOLVENT_SMILES].notna().sum()),
            },
            "positives": positives,
            "unlabeled": {
                "n_rows": len(unlabeled),
                "n_unique_pairs": _n_unique_pairs(unlabeled),
                "n_unique_conditions": int(
                    unlabeled[list(UNIFIED_CONDITION_COLUMNS)].drop_duplicates().shape[0]
                )
                if not unlabeled.empty
                else 0,
            },
            "split_summary": self.protocol.summary(),
        }

    def _compute_jnj_ligand_keys(self) -> frozenset[str]:
        """Return the set of JnJ ligand InChIKeys (for ORD↔JnJ ligand-overlap accounting)."""
        keys: set[str] = set()
        for catalyst in self._jnj[schema.CATALYST_SMILES].dropna().unique():
            condition = condition_from_columns(catalyst, None, None)
            if condition.ligand_key is not None:
                keys.add(condition.ligand_key)
        return frozenset(keys)


def _merge_sources(values: pd.Series) -> str:
    """Union of ``|``-joined source tokens across a group (e.g. ``ORD`` + ``Pistachio``)."""
    tokens = {token for value in values for token in str(value).split("|") if token}
    return "|".join(sorted(tokens))


def _n_unique_pairs(frame: pd.DataFrame) -> int:
    """Number of distinct (aryl, amine) substrate pairs in a positives/unlabeled frame."""
    if frame.empty:
        return 0
    return int(frame[[schema.ARYL_SMILES, schema.AMINE_SMILES]].drop_duplicates().shape[0])


def _has_phosphine(ligand_smiles: str | None) -> bool:
    """Return ``True`` if the (metal-stripped) ligand SMILES contains phosphorus.

    Args:
        ligand_smiles: A free-ligand SMILES with metals already removed, or ``None``.

    Returns:
        Whether the ligand is phosphorus-bearing (a phosphine).
    """
    return bool(ligand_smiles) and "P" in ligand_smiles  # type: ignore[operator]


def _to_float(value: object) -> float | None:
    """Coerce a cell to a finite float, or ``None`` (handles NaN/None/non-numeric)."""
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # reject NaN


def _max_optional(current: float | None, candidate: float | None) -> float | None:
    """Return the maximum of two optional floats (ignoring ``None``)."""
    values = [v for v in (current, candidate) if isinstance(v, (int, float))]
    return max(float(v) for v in values) if values else None


def _parse_args(argv: list[str] | None = None) -> tuple[AssemblyConfig, bool]:
    """Parse CLI arguments into an :class:`AssemblyConfig` and a ``merge_only`` flag.

    Args:
        argv: Argument vector (defaults to ``sys.argv``).

    Returns:
        ``(config, merge_only)``.
    """
    parser = argparse.ArgumentParser(description="Assemble TEST / P / U condition-aware datasets.")
    defaults = AssemblyConfig()
    parser.add_argument("--ord-forward", type=Path, default=defaults.ord_forward)
    parser.add_argument(
        "--pistachio",
        type=Path,
        default=defaults.pistachio,
        help=f"Raw Pistachio rxn_smiles CSV (licensed). Use {DEFAULT_PISTACHIO} to include it.",
    )
    parser.add_argument("--jnj-test", type=Path, default=defaults.jnj_test)
    parser.add_argument("--out-dir", type=Path, default=defaults.out_dir)
    parser.add_argument("--conditions-per-pair", type=int, default=defaults.conditions_per_pair)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--val-fraction", type=float, default=defaults.val_fraction)
    parser.add_argument("--limit", type=int, default=defaults.limit)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--jobs",
        type=int,
        default=defaults.jobs,
        help="Worker processes for the ORD scan (1 = serial; >1 uses a forkserver pool).",
    )
    parser.add_argument("--chunk-size", type=int, default=defaults.chunk_size)
    parser.add_argument(
        "--no-require-pd",
        dest="require_pd",
        action="store_false",
        help="Do not require a Pd species (keep Cu/Ni/metal-free too).",
    )
    parser.add_argument(
        "--require-phosphine",
        action="store_true",
        help="Additionally require a phosphorus-bearing ligand (Pd + phosphine).",
    )
    parser.add_argument(
        "--keep-cu-ni",
        dest="reject_cu_ni",
        action="store_false",
        help="Keep Pd reactions that also contain a Cu/Ni additive (default: throw them out).",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip mining: re-merge existing per-source positive parquets in --out-dir.",
    )
    args = parser.parse_args(argv)
    config = AssemblyConfig(
        ord_forward=args.ord_forward,
        pistachio=args.pistachio,
        jnj_test=args.jnj_test,
        out_dir=args.out_dir,
        require_pd=args.require_pd,
        reject_cu_ni=args.reject_cu_ni,
        require_phosphine=args.require_phosphine,
        conditions_per_pair=args.conditions_per_pair,
        seed=args.seed,
        val_fraction=args.val_fraction,
        limit=args.limit,
        batch_size=args.batch_size,
        jobs=args.jobs,
        chunk_size=args.chunk_size,
    )
    return config, bool(args.merge_only)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: assemble the datasets (or re-merge) and print the report.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).
    """
    config, merge_only = _parse_args(argv)
    assembler = DatasetAssembler(config)
    report = assembler.merge_existing() if merge_only else assembler.assemble()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
