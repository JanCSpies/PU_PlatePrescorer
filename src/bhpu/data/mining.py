"""Mine Buchwald–Hartwig (aryl C–N) couplings from large reaction corpora (ORD, …).

The literature positive corpus for PU learning is built by scanning millions of reactions and
keeping those that *provably* formed an aryl C–N bond. Detection is **high precision**: a reaction
counts only if some (aryl halide, N–H amine) pair among its molecules produces — via the RDKit
BH template — one of the reaction's own recorded products. That confirmation (rather than a loose
SMARTS on the product) rejects SNAr-on-a-different-site, reductive aminations, and mis-parses.

Design:

* :class:`BHCouplingDetector` — structure detector over a bag of molecule SMILES + product SMILES.
* Source readers (:func:`iter_orderly`, :func:`iter_ord_csv`) — stream normalised raw records.
* :class:`CNCouplingMiner` — orchestrates detection, **cross-source dedup** to unique substrate
  pairs, and a **leakage audit** against JnJ (so ORD/Pistachio cannot smuggle the test set in).

Every mined pair is a *positive* (a reported, product-confirmed coupling) — the SCAR-violating,
positive-only signal the PU pipeline consumes.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

from bhpu.chem.canonical import canonical_smiles, inchikey, strip_and_canonicalise
from bhpu.chem.reaction import enumerate_cn_coupling_products

RDLogger.DisableLog("rdApp.*")  # silence per-molecule parse warnings during bulk mining

#: Aryl (pseudo)halide: aromatic carbon bearing a leaving halide (matches the product template).
_ARYL_HALIDE = Chem.MolFromSmarts("[c][F,Cl,Br,I]")
#: N–H amine able to undergo C–N coupling (primary/secondary; not imine/nitrile/hydrazine/oxime).
_NH_AMINE = Chem.MolFromSmarts("[N;H1,H2;!$(N=*);!$(N#*);!$(N-[#7,#8])]")


@dataclass(frozen=True)
class RawReaction:
    """A source-agnostic reaction record fed to the detector.

    Attributes:
        molecules: SMILES of all reactant/reagent molecules (order irrelevant).
        products: SMILES of the recorded product(s).
        source: Provenance tag (e.g. ``"ORDerly-forward"``).
        raw_id: Source-native identifier, if any.
        yield_value: Reported yield in percent, if any.
    """

    molecules: tuple[str, ...]
    products: tuple[str, ...]
    source: str
    raw_id: str | None = None
    yield_value: float | None = None


@dataclass(frozen=True)
class MinedReaction:
    """A product-confirmed aryl C–N coupling extracted from a raw reaction.

    Attributes:
        aryl_smiles: Canonical SMILES of the aryl-halide electrophile.
        amine_smiles: Canonical SMILES of the amine nucleophile.
        product_smiles: Canonical SMILES of the matched coupling product.
        source: Provenance tag.
        raw_id: Source-native identifier.
        yield_value: Reported yield (percent), if any.
        has_pd: Whether a palladium species was present (BH-consistent conditions).
    """

    aryl_smiles: str
    amine_smiles: str
    product_smiles: str
    source: str
    raw_id: str | None
    yield_value: float | None
    has_pd: bool


def _clean_smiles(smiles: str) -> str | None:
    """Canonicalise a SMILES and strip atom-map numbers (some ORD reactants are mapped).

    Atom maps survive :func:`canonical_smiles` and would pollute the stored corpus and its
    fingerprints. InChIKey-based dedup already ignores them, but the persisted SMILES must be clean.

    Args:
        smiles: Raw SMILES (possibly atom-mapped).

    Returns:
        Canonical, unmapped SMILES, or ``None`` if unparsable.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    cleaned: str = Chem.MolToSmiles(mol)
    return cleaned


class BHCouplingDetector:
    """Detect and extract aryl C–N couplings from a bag of molecules + products."""

    def detect(self, reaction: RawReaction) -> MinedReaction | None:
        """Return the extracted coupling for ``reaction``, or ``None`` if it is not one.

        Args:
            reaction: A raw reaction record.

        Returns:
            A :class:`MinedReaction` for the first (aryl halide, amine) pair whose enumerated
            BH product matches a recorded product, else ``None``.
        """
        mols = [(s, Chem.MolFromSmiles(s)) for s in reaction.molecules]
        parsed = [(s, m) for s, m in mols if m is not None]
        aryl_halides = [s for s, m in parsed if m.HasSubstructMatch(_ARYL_HALIDE)]
        amines = [s for s, m in parsed if m.HasSubstructMatch(_NH_AMINE)]
        if not aryl_halides or not amines:
            return None

        product_keys = {
            key
            for p in reaction.products
            if (key := inchikey(canonical_smiles(p) or p)) is not None
        }
        if not product_keys:
            return None
        has_pd = any(s.count("Pd") > 0 or "[Pd" in s for s in reaction.molecules)

        for aryl_raw in aryl_halides:
            aryl = _clean_smiles(aryl_raw)
            if aryl is None:
                continue
            for amine_raw in amines:
                if amine_raw == aryl_raw:
                    continue  # require an intermolecular coupling
                amine = _clean_smiles(amine_raw)
                if amine is None:
                    continue
                for product in enumerate_cn_coupling_products(aryl, amine):
                    if inchikey(product) in product_keys:
                        return MinedReaction(
                            aryl_smiles=aryl,
                            amine_smiles=amine,
                            product_smiles=product,
                            source=reaction.source,
                            raw_id=reaction.raw_id,
                            yield_value=reaction.yield_value,
                            has_pd=has_pd,
                        )
        return None


def detect_batch(reactions: list[RawReaction]) -> tuple[int, list[MinedReaction]]:
    """Detect couplings in a batch of reactions (a picklable worker for multiprocessing).

    Args:
        reactions: Raw reactions to scan.

    Returns:
        ``(n_scanned, matches)`` — the number examined and the confirmed couplings
        (typically ~10 % of the input for ORD).
    """
    detector = BHCouplingDetector()
    matches = [m for r in reactions if (m := detector.detect(r)) is not None]
    return len(reactions), matches


def _split_smiles(value: object) -> tuple[str, ...]:
    """Split a dot-joined SMILES cell into components (empty tuple if missing)."""
    if not isinstance(value, str) or not value:
        return ()
    return tuple(part for part in value.split(".") if part)


def iter_ord_csv(
    path: Path, *, source: str = "ORD", chunksize: int = 50_000
) -> Iterator[RawReaction]:
    """Stream reactions from a DynaMAIK-style ``ord.csv`` (reactants/products columns).

    Args:
        path: Path to ``ord.csv`` (columns ``reaction_id, reactants, products, reaction_smiles``).
        source: Provenance tag to stamp on each record.
        chunksize: Rows per pandas chunk (keeps memory bounded on the 2 GB file).

    Yields:
        One :class:`RawReaction` per row.
    """
    reader = pd.read_csv(
        path, usecols=["reaction_id", "reactants", "products"], chunksize=chunksize, dtype=str
    )
    for chunk in reader:
        for row in chunk.itertuples(index=False):
            yield RawReaction(
                molecules=_split_smiles(row.reactants),
                products=_split_smiles(row.products),
                source=source,
                raw_id=str(row.reaction_id),
            )


def iter_orderly(
    path: Path, *, source: str, batch_size: int = 20_000
) -> Iterator[RawReaction]:
    """Stream reactions from an ORDerly parquet (separate reactant/agent/product columns).

    Args:
        path: Path to an ``orderly_*.parquet`` file.
        source: Provenance tag.
        batch_size: Rows per Arrow batch.

    Yields:
        One :class:`RawReaction` per row (reactants + agents pooled as candidate molecules).
    """
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema.names)
    reactant_cols = [c for c in ("reactant_000", "reactant_001", "reactant_002") if c in available]
    agent_cols = [c for c in ("agent_000", "agent_001", "agent_002") if c in available]
    product_cols = [c for c in ("product_000", "product_001") if c in available]
    yield_col = "yield_000" if "yield_000" in available else None
    id_col = "original_index" if "original_index" in available else "index"
    columns = [*reactant_cols, *agent_cols, *product_cols, id_col]
    if yield_col:
        columns.append(yield_col)

    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        frame = batch.to_pandas()
        for row in frame.itertuples(index=False):
            data = row._asdict()
            molecules = tuple(
                str(data[c]) for c in (*reactant_cols, *agent_cols) if _is_smiles(data.get(c))
            )
            products = tuple(str(data[c]) for c in product_cols if _is_smiles(data.get(c)))
            raw_yield = data.get(yield_col) if yield_col else None
            yield RawReaction(
                molecules=molecules,
                products=products,
                source=source,
                raw_id=str(data.get(id_col)),
                yield_value=_to_float(raw_yield),
            )


def _is_smiles(value: object) -> bool:
    """True if ``value`` is a non-empty, non-null SMILES-looking string."""
    return isinstance(value, str) and bool(value) and value.lower() != "nan"


def _to_float(value: object) -> float | None:
    """Coerce a cell to a finite float, or ``None`` (handles NaN/None/non-numeric)."""
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # reject NaN


@dataclass
class _PairAggregate:
    """Mutable accumulator for all records of one (aryl, amine) substrate pair."""

    aryl_smiles: str
    amine_smiles: str
    product_smiles: str
    sources: set[str]
    max_yield: float | None
    has_pd: bool
    n_records: int


@dataclass
class MiningReport:
    """Provenance + leakage accounting for a mining run.

    Attributes:
        n_scanned: Raw reactions examined.
        n_detected: Reactions confirmed as aryl C–N couplings.
        n_unique_pairs: Distinct (aryl, amine) substrate pairs after dedup.
        n_pairs_dropped_substrate_leak: Pairs removed for sharing a JnJ aryl/amine.
        n_exact_jnj_pair_hits: Mined pairs that exactly equal a JnJ (aryl, amine) pair
            (the "is the test set inside this corpus?" number).
        per_source: Detected-count per source tag.
    """

    n_scanned: int = 0
    n_detected: int = 0
    n_unique_pairs: int = 0
    n_pairs_dropped_substrate_leak: int = 0
    n_exact_jnj_pair_hits: int = 0
    per_source: dict[str, int] = field(default_factory=dict)


class CNCouplingMiner:
    """Detect, dedup, and leakage-audit aryl C–N couplings into a positive corpus.

    Attributes:
        report: Running :class:`MiningReport` populated by :meth:`mine`.
    """

    def __init__(
        self,
        jnj_substrate_keys: frozenset[str],
        jnj_pairs: frozenset[tuple[str, str]],
        *,
        detector: BHCouplingDetector | None = None,
    ) -> None:
        """Initialise the miner with the JnJ leakage reference.

        Args:
            jnj_substrate_keys: InChIKeys of every JnJ aryl and amine (substrate-level guard).
            jnj_pairs: InChIKey (aryl, amine) pairs of JnJ (exact-overlap accounting).
            detector: Detector to use (defaults to a fresh :class:`BHCouplingDetector`).
        """
        self._jnj_keys = jnj_substrate_keys
        self._jnj_pairs = jnj_pairs
        self._detector = detector or BHCouplingDetector()
        self.report = MiningReport()
        # pair-key -> aggregated record
        self._pairs: dict[tuple[str, str], _PairAggregate] = {}

    def mine(self, reactions: Iterator[RawReaction]) -> None:
        """Detect couplings in a reaction stream and accumulate unique, clean pairs.

        Single-process convenience path; for bulk mining use :meth:`note_scanned` +
        :meth:`add_match` fed by parallel :func:`detect_batch` workers.

        Args:
            reactions: A stream of raw reactions (from one or more source readers).
        """
        for reaction in reactions:
            self.report.n_scanned += 1
            match = self._detector.detect(reaction)
            if match is not None:
                self.add_match(match)

    def note_scanned(self, count: int) -> None:
        """Record that ``count`` raw reactions were scanned (for parallel accounting)."""
        self.report.n_scanned += count

    def add_match(self, match: MinedReaction) -> None:
        """Fold one detected coupling into the corpus (dedup + leakage audit).

        Args:
            match: A confirmed coupling (e.g. from :func:`detect_batch`).
        """
        self.report.n_detected += 1
        self.report.per_source[match.source] = self.report.per_source.get(match.source, 0) + 1
        self._accumulate(match)

    def _accumulate(self, match: MinedReaction) -> None:
        """Fold a detected coupling into the unique-pair table with leakage accounting."""
        aryl_key = inchikey(strip_and_canonicalise(match.aryl_smiles) or match.aryl_smiles)
        amine_key = inchikey(strip_and_canonicalise(match.amine_smiles) or match.amine_smiles)
        if aryl_key is None or amine_key is None:
            return
        if (aryl_key, amine_key) in self._jnj_pairs:
            self.report.n_exact_jnj_pair_hits += 1
        if aryl_key in self._jnj_keys or amine_key in self._jnj_keys:
            self.report.n_pairs_dropped_substrate_leak += 1
            return  # frozen-protocol guard: never train on a JnJ substrate

        key = (aryl_key, amine_key)
        record = self._pairs.get(key)
        if record is None:
            self._pairs[key] = _PairAggregate(
                aryl_smiles=match.aryl_smiles,
                amine_smiles=match.amine_smiles,
                product_smiles=match.product_smiles,
                sources={match.source},
                max_yield=match.yield_value,
                has_pd=match.has_pd,
                n_records=1,
            )
        else:
            record.sources.add(match.source)
            record.n_records += 1
            record.has_pd = record.has_pd or match.has_pd
            record.max_yield = _max_optional(record.max_yield, match.yield_value)

    def to_frame(self) -> pd.DataFrame:
        """Return the deduped positive corpus as a DataFrame (one row per substrate pair)."""
        self.report.n_unique_pairs = len(self._pairs)
        rows = [
            {
                "aryl_key": aryl_key,
                "amine_key": amine_key,
                "aryl_smiles": rec.aryl_smiles,
                "amine_smiles": rec.amine_smiles,
                "product_smiles": rec.product_smiles,
                "sources": "|".join(sorted(rec.sources)),
                "max_yield": rec.max_yield,
                "has_pd": rec.has_pd,
                "n_records": rec.n_records,
            }
            for (aryl_key, amine_key), rec in self._pairs.items()
        ]
        return pd.DataFrame(rows)


def _max_optional(a: object, b: float | None) -> float | None:
    """Maximum of two optional yields (ignoring ``None``)."""
    values = [x for x in (a, b) if isinstance(x, (int, float))]
    return max(float(v) for v in values) if values else None
