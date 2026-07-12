"""The per-reaction parse and a multiprocessed source-parsing driver that writes parquet.

:func:`parse_reaction` turns one :class:`SourceReaction` into a :class:`ParsedReaction` -- canonical
dedup key, named-reaction class + reaction-center signature, coupling partners, and the binned
condition fields. :func:`parse_source` streams a whole source through a worker pool (the classifier
and condition tables are process-global singletons, compiled once per worker) and appends the rows
to ``<source>_parsed.parquet`` in bounded memory.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from itertools import islice
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bhpu.chem.canonical import strip_and_canonicalise
from pubench.rxngroup.classify import NamedReactionLibrary, reaction_center_signature
from pubench.rxngroup.conditions import ConditionExtractor, is_reagent, metals_in
from pubench.rxngroup.records import ParsedReaction, SourceReaction

logger = logging.getLogger(__name__)

_LIBRARY: NamedReactionLibrary | None = None
_EXTRACTOR: ConditionExtractor | None = None


def _library() -> NamedReactionLibrary:
    """Return the process-global reaction library (built once per worker)."""
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = NamedReactionLibrary()
    return _LIBRARY


def _extractor() -> ConditionExtractor:
    """Return the process-global condition extractor (built once per worker)."""
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = ConditionExtractor()
    return _EXTRACTOR


def parse_reaction(sr: SourceReaction) -> ParsedReaction | None:
    """Process one raw reaction into a :class:`ParsedReaction`, or ``None`` if unusable.

    Args:
        sr: The raw source reaction (roles kept separate).

    Returns:
        The processed reaction row, or ``None`` if no reactant/product survives canonicalisation.
    """
    reactants = [c for r in sr.reactants if (c := strip_and_canonicalise(r)) is not None]
    products = [c for p in sr.products if (c := strip_and_canonicalise(p)) is not None]
    if not reactants or not products:
        return None

    # Split substrates from pooled reagents (catalyst/base/solvent/gas): CHORISO/CJHIF dump
    # everything into one reactant list; clean sources keep the agent bag separate, so partitioning
    # is a near-noop there and keeps the dedup key consistent across sources.
    substrates = [s for s in reactants if not is_reagent(s)]
    reagents = [s for s in reactants if is_reagent(s)]
    if not substrates:  # everything looked like a reagent -> keep the raw reactants
        substrates = reactants
    agent_bag = tuple(sr.agents) + tuple(reagents)

    rxn_key = f"{'.'.join(sorted(set(substrates)))}>>{'.'.join(sorted(set(products)))}"
    metals: frozenset[str] = (
        frozenset().union(*(metals_in(a) for a in agent_bag)) if agent_bag else frozenset()
    )

    cls = _library().classify(substrates, products, agent_bag, metals)
    metal, ligand_key, base_family, solvent_family, tbin = _extractor().extract(
        agent_bag, sr.temperature
    )
    center = reaction_center_signature(sr.mapped_rxn) if sr.mapped_rxn else None

    return ParsedReaction(
        rxn_key=rxn_key,
        reaction_class=cls.reaction_class,
        center_sig=center,
        partner_a=cls.partner_a,
        partner_b=cls.partner_b,
        metal=metal,
        ligand_key=ligand_key,
        base_family=base_family,
        solvent_family=solvent_family,
        temp_bin=tbin,
        yield_value=sr.yield_value,
        source=sr.source,
        raw_id=sr.raw_id,
    )


def _parse_chunk(chunk: list[SourceReaction]) -> list[dict[str, object]]:
    """Worker: parse a chunk of raw reactions into row dicts (drops ``None``)."""
    rows: list[dict[str, object]] = []
    for sr in chunk:
        parsed = parse_reaction(sr)
        if parsed is not None:
            rows.append(parsed.as_row())
    return rows


def _chunks[T](iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield successive ``size``-length lists from ``iterable``."""
    it = iter(iterable)
    while batch := list(islice(it, size)):
        yield batch


def parse_source(
    reactions: Iterable[SourceReaction],
    out_path: Path,
    *,
    workers: int = 16,
    chunk_size: int = 5_000,
    log_every: int = 200_000,
) -> int:
    """Parse a source's reaction stream in parallel and append rows to a parquet file.

    Args:
        reactions: The raw reaction stream (from a :mod:`pubench.rxngroup.sources` reader).
        out_path: Destination ``.parquet`` path (parent dirs are created).
        workers: Worker-process count.
        chunk_size: Reactions per worker task.
        log_every: Emit a progress log every this many written rows.

    Returns:
        The number of parsed rows written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    written = 0
    try:
        with Pool(processes=workers) as pool:
            for rows in pool.imap_unordered(_parse_chunk, _chunks(reactions, chunk_size)):
                if not rows:
                    continue
                table = pa.Table.from_pylist(rows)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema)
                writer.write_table(table)
                written += len(rows)
                if written // log_every != (written - len(rows)) // log_every:
                    logger.info("parsed %s rows -> %s", f"{written:,}", out_path.name)
    finally:
        if writer is not None:
            writer.close()
    logger.info("done: %s rows -> %s", f"{written:,}", out_path)
    return written
