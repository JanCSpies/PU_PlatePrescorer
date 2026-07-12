"""Streaming readers that turn each raw source into a common :class:`SourceReaction` stream.

Readers do **only cheap string work** (split ``reactants>agents>products``, pick columns) and leave
all RDKit canonicalisation to the multiprocessed pipeline, so a single reader process can feed many
worker processes without becoming the bottleneck. Three sources are supported:

* **Pistachio** -- a one-column ``rxn_smiles`` CSV of atom-mapped reactions (line-iterated).
* **USPTO** -- the DynaMAIK ``ReactionSmiles`` CSVs (+ patent no. / text-mined yield), pandas-read.
* **ORDerly** -- the cleaned ORD parquet with role-typed ``reactant_/agent_/solvent_/temperature/
  yield`` columns, read in Arrow batches; conditions come straight from the role columns.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

from pubench.rxngroup.records import SourceReaction

_REACTANT_COLS = ("reactant_000", "reactant_001", "reactant_002")
_PRODUCT_COLS = ("product_000", "product_001")
_AGENT_COLS = ("agent_000", "agent_001", "agent_002", "solvent_000", "solvent_001", "solvent_002")


def _split_rxn(rxn: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Split an atom-mapped ``reactants>agents>products`` SMILES into three non-empty tuples."""
    parts = rxn.split(">")
    if len(parts) < 3:
        return (), (), ()
    reactants = tuple(s for s in parts[0].split(".") if s)
    agents = tuple(s for s in parts[1].split(".") if s)
    products = tuple(s for s in parts[-1].split(".") if s)
    return reactants, agents, products


def iter_pistachio(path: Path, *, limit: int | None = None) -> Iterator[SourceReaction]:
    """Stream Pistachio reactions from the one-column ``rxn_smiles`` CSV.

    Args:
        path: Path to ``pistachio.smi.deduped.csv``.
        limit: Optional cap on the number of reactions yielded (for smoke tests).

    Yields:
        One :class:`SourceReaction` per parsable line.
    """
    with path.open() as handle:
        next(handle, None)  # header: rxn_smiles
        for n, line in enumerate(handle):
            if limit is not None and n >= limit:
                return
            rxn = line.strip()
            reactants, agents, products = _split_rxn(rxn)
            if not reactants or not products:
                continue
            yield SourceReaction(
                reactants=reactants,
                agents=agents,
                products=products,
                mapped_rxn=rxn,
                source="Pistachio",
            )


def iter_uspto(
    path: Path, *, limit: int | None = None, chunksize: int = 100_000
) -> Iterator[SourceReaction]:
    """Stream USPTO reactions from a DynaMAIK ``ReactionSmiles`` CSV.

    Args:
        path: Path to a ``uspto_*.csv`` file.
        limit: Optional cap on reactions yielded.
        chunksize: Pandas read chunk size.

    Yields:
        One :class:`SourceReaction` per row (yield from text-mined/calculated columns).
    """
    import pandas as pd

    usecols = ["ReactionSmiles", "PatentNumber", "TextMinedYield", "CalculatedYield"]
    n = 0
    for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=chunksize):
        for row in chunk.itertuples(index=False):
            if limit is not None and n >= limit:
                return
            n += 1
            rxn = str(getattr(row, "ReactionSmiles", "") or "")
            reactants, agents, products = _split_rxn(rxn)
            if not reactants or not products:
                continue
            yield SourceReaction(
                reactants=reactants,
                agents=agents,
                products=products,
                mapped_rxn=rxn,
                source="USPTO",
                raw_id=str(getattr(row, "PatentNumber", "") or "") or None,
                yield_value=_first_yield(
                    getattr(row, "TextMinedYield", None), getattr(row, "CalculatedYield", None)
                ),
            )


def iter_orderly(
    path: Path, *, limit: int | None = None, batch_size: int = 20_000
) -> Iterator[SourceReaction]:
    """Stream ORDerly reactions from the cleaned ORD parquet (role-typed conditions).

    Args:
        path: Path to ``orderly_condition_train.parquet``.
        limit: Optional cap on reactions yielded.
        batch_size: Arrow record-batch size.

    Yields:
        One :class:`SourceReaction` per row; agents pool the agent+solvent columns, temperature and
        yield come straight from the cleaned columns, and ``mapped_rxn`` is set only when mapped.
    """
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)  # condition_train has fewer role columns than forward
    reactant_cols = [c for c in _REACTANT_COLS if c in available]
    product_cols = [c for c in _PRODUCT_COLS if c in available]
    agent_cols = [c for c in _AGENT_COLS if c in available]
    extra = [
        c
        for c in ("temperature", "yield_000", "rxn_str", "is_mapped", "original_index")
        if c in available
    ]
    cols = reactant_cols + product_cols + agent_cols + extra
    n = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        d = batch.to_pydict()
        for i in range(len(batch)):
            if limit is not None and n >= limit:
                return
            n += 1
            reactants = tuple(s for c in reactant_cols if (s := d[c][i]))
            products = tuple(s for c in product_cols if (s := d[c][i]))
            if not reactants or not products:
                continue
            agents = tuple(s for c in agent_cols if (s := d[c][i]))
            mapped = d["rxn_str"][i] if d.get("is_mapped", [None] * len(batch))[i] else None
            raw = d.get("original_index", [None] * len(batch))[i]
            yield SourceReaction(
                reactants=reactants,
                agents=agents,
                products=products,
                mapped_rxn=mapped,
                source="ORDerly",
                raw_id=str(raw) if raw is not None else None,
                yield_value=_to_float(d.get("yield_000", [None] * len(batch))[i]),
                temperature=_to_float(d.get("temperature", [None] * len(batch))[i]),
            )


def _to_float(value: object) -> float | None:
    """Coerce a cell to float, or ``None`` if missing/unparsable."""
    if value is None:
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # drop NaN


def iter_choriso(
    path: Path, *, limit: int | None = None, chunksize: int = 100_000
) -> Iterator[SourceReaction]:
    """Stream CHORISO reactions (open, USPTO-derived, condition-annotated + rxnmapper-mapped).

    The SMILES reagents/catalyst/solvent are pooled into the ``reactants`` column (the ``reagent``/
    ``solvent``/``catalyst`` columns are English names, not SMILES), so the pipeline's reagent
    partition separates them. The atom-mapped ``rxnmapper_aam`` gives the reaction-center signature.

    Args:
        path: Path to ``choriso_extracted.csv``.
        limit: Optional cap on reactions yielded.
        chunksize: Pandas read chunk size.

    Yields:
        One :class:`SourceReaction` per row.
    """
    import pandas as pd

    usecols = ["reactants", "products", "rxnmapper_aam", "yield"]
    n = 0
    for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=chunksize):
        for row in chunk.itertuples(index=False):
            if limit is not None and n >= limit:
                return
            n += 1
            reactants = tuple(s for s in str(getattr(row, "reactants", "") or "").split(".") if s)
            products = tuple(s for s in str(getattr(row, "products", "") or "").split(".") if s)
            if not reactants or not products:
                continue
            mapped = getattr(row, "rxnmapper_aam", None)
            yield SourceReaction(
                reactants=reactants,
                agents=(),
                products=products,
                mapped_rxn=str(mapped) if isinstance(mapped, str) and ">" in mapped else None,
                source="CHORISO",
                yield_value=_first_yield(getattr(row, "yield", None)),
            )


def iter_cjhif(
    path: Path, *, limit: int | None = None, batch_size: int = 100_000
) -> Iterator[SourceReaction]:
    """Stream the genuinely-journal (``SOURCE == "CJHIF"``) rows of the CJHIF_USPTO_ORD parquet.

    The bundled file is 96 % CHORISO/ORD/USPTO (already ingested separately), so only the ~13k
    ``CJHIF``-tagged rows -- non-patent journal literature -- are read here. Reactants pool reagents
    (partitioned downstream); no atom map or yield is available.

    Args:
        path: Path to ``final_merged_dataset_train.parquet`` (or the test parquet).
        limit: Optional cap on reactions yielded.
        batch_size: Arrow record-batch size.

    Yields:
        One :class:`SourceReaction` per CJHIF-origin row.
    """
    pf = pq.ParquetFile(path)
    n = 0
    for batch in pf.iter_batches(
        batch_size=batch_size, columns=["REACTANTS", "PRODUCTS", "SOURCE"]
    ):
        d = batch.to_pydict()
        for i in range(len(batch)):
            if d["SOURCE"][i] != "CJHIF":
                continue
            if limit is not None and n >= limit:
                return
            n += 1
            reactants = tuple(s for s in str(d["REACTANTS"][i] or "").split(".") if s)
            products = tuple(s for s in str(d["PRODUCTS"][i] or "").split(".") if s)
            if not reactants or not products:
                continue
            yield SourceReaction(
                reactants=reactants, agents=(), products=products, mapped_rxn=None, source="CJHIF"
            )


def _first_yield(*values: object) -> float | None:
    """Return the first parsable, in-range (0-100] yield among ``values``."""
    for v in values:
        f = _to_float(v)
        if f is not None and 0.0 < f <= 100.0:
            return f
    return None


#: Sources whose data may be redistributed (open); Pistachio is the only licensed one.
PUBLIC_SOURCES: frozenset[str] = frozenset({"USPTO", "ORDerly", "CHORISO", "CJHIF"})

#: Source name -> (reader, default path). The default paths are the verified `/mnt/datasets` files.
SOURCES: dict[str, tuple[object, str]] = {
    "pistachio": (
        iter_pistachio,
        "/mnt/datasets/pistachio/releases/latest/data/pistachio.smi.deduped.csv",
    ),
    "uspto_76_16": (iter_uspto, "/mnt/datasets/DynaMAIK/uspto_76_16.csv"),
    "uspto_01_16": (iter_uspto, "/mnt/datasets/DynaMAIK/uspto_01_16.csv"),
    "orderly": (iter_orderly, "/mnt/datasets/ORDerly/orderly_condition_train.parquet"),
    "choriso": (iter_choriso, "/mnt/datasets/DynaMAIK/choriso_extracted.csv"),
    "cjhif": (iter_cjhif, "/mnt/datasets/CJHIF_USPTO_ORD/final_merged_dataset_train.parquet"),
}
