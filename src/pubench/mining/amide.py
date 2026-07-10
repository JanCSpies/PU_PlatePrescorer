"""Mine amide (carboxylic acid + amine -> amide) couplings from ORD / Pistachio.

The design mirrors :class:`bhpu.data.mining.BHCouplingDetector`: detection is **high precision** —
a reaction counts only if some (carboxylic-acid, N-H amine) pair among its molecules produces, via
an RDKit amide-condensation template, one of the reaction's own recorded products. That
product-confirmation (rather than a loose SMARTS on the product) rejects esterifications,
N-acylations onto a pre-existing amide, and mis-parses. The template reproduces 91 % of the Chemlex
wet-lab set's own products (the residual are exotic N-P / N-N / N-O nucleophiles this filter
deliberately excludes to keep the corpus clean); on standard literature amines recall is higher.

Reuses the source readers and the :class:`~bhpu.data.mining.RawReaction` record from
:mod:`bhpu.data.mining`; only the detector, the product template, and the miner's leakage rule are
amide-specific.

Leakage rule (transductive): :class:`AmideMiner` **drops** positive rows whose exact (acid, amine)
identity pair equals a Test pair, but keeps rows that merely reuse a Test substrate -- those
substrates legitimately populate the transductive unlabeled grid U (the plate-design use case scores
a *known* substrate matrix). This differs from the BH miner's substrate-level guard, which targeted
the strictly inductive JnJ protocol.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from bhpu.chem.canonical import canonical_smiles, inchikey, strip_and_canonicalise
from bhpu.data.mining import RawReaction

RDLogger.DisableLog("rdApp.*")  # silence per-molecule parse warnings during bulk mining

#: Carboxylic acid acyl donor: an sp2 carbon bearing =O and a hydroxyl (free acid, not carboxylate).
_CARBOXYLIC_ACID = Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1]")
#: N-H amine nucleophile (primary/secondary); excludes imine/nitrile N, N bonded to a heteroatom
#: (hydrazide/hydroxylamine/sulfonamide), and N already part of an amide -- mirrors the BH amine
#: filter's precision philosophy.
_NH_AMINE = Chem.MolFromSmarts(
    "[NX3;H1,H2;!$([NX3]=*);!$([NX3]#*);!$([NX3]-[!#6;!#1]);!$([NX3][CX3]=[OX1])]"
)
#: Amide-condensation template: acid C(=O)OH + amine N-H -> amide C(=O)N (the hydroxyl leaves).
_AMIDE_SMARTS = (
    "[CX3:1](=[OX1:2])[OX2H1]."
    "[NX3;H1,H2;!$([NX3]=*);!$([NX3]#*);!$([NX3]-[!#6;!#1]);!$([NX3][CX3]=[OX1]):4]"
    ">>[CX3:1](=[OX1:2])[N:4]"
)
_AMIDE_RXN = AllChem.ReactionFromSmarts(_AMIDE_SMARTS)


@dataclass(frozen=True)
class MinedAmide:
    """A product-confirmed amide coupling extracted from a raw reaction.

    Attributes:
        acid_smiles: Canonical SMILES of the carboxylic-acid acyl donor.
        amine_smiles: Canonical SMILES of the amine nucleophile.
        product_smiles: Canonical SMILES of the matched amide product.
        source: Provenance tag.
        raw_id: Source-native identifier, if any.
        yield_value: Reported yield (percent), if any.
    """

    acid_smiles: str
    amine_smiles: str
    product_smiles: str
    source: str
    raw_id: str | None
    yield_value: float | None


def _clean_smiles(smiles: str) -> str | None:
    """Canonicalise a SMILES, stripping atom-map numbers (ORD/Pistachio reactants may be mapped)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    cleaned: str = Chem.MolToSmiles(mol)
    return cleaned


@lru_cache(maxsize=200_000)
def enumerate_amide_products(acid_smiles: str, amine_smiles: str) -> frozenset[str]:
    """Enumerate all distinct amide products of a carboxylic acid and an amine.

    Args:
        acid_smiles: SMILES of the carboxylic-acid acyl donor.
        amine_smiles: SMILES of the amine nucleophile.

    Returns:
        A frozenset of canonical product SMILES (empty if the template does not apply or either
        input is unparsable).
    """
    acid = Chem.MolFromSmiles(acid_smiles)
    amine = Chem.MolFromSmiles(amine_smiles)
    if acid is None or amine is None:
        return frozenset()
    products: set[str] = set()
    for outcome in _AMIDE_RXN.RunReactants((acid, amine)):
        product = outcome[0]
        try:
            Chem.SanitizeMol(product)
        except (Chem.AtomValenceException, Chem.KekulizeException, ValueError):
            continue
        products.add(Chem.MolToSmiles(product))
    return frozenset(products)


class AmideCouplingDetector:
    """Detect and extract amide couplings from a bag of molecules + products."""

    def detect(self, reaction: RawReaction) -> MinedAmide | None:
        """Return the extracted amide coupling for ``reaction``, or ``None`` if it is not one.

        Args:
            reaction: A raw reaction record (reactants + agents pooled as candidate molecules).

        Returns:
            A :class:`MinedAmide` for the first (acid, amine) pair whose enumerated amide product
            matches a recorded product, else ``None``.
        """
        parsed = [(s, m) for s in reaction.molecules if (m := Chem.MolFromSmiles(s)) is not None]
        acids = [s for s, m in parsed if m.HasSubstructMatch(_CARBOXYLIC_ACID)]
        amines = [s for s, m in parsed if m.HasSubstructMatch(_NH_AMINE)]
        if not acids or not amines:
            return None

        product_keys = {
            key
            for p in reaction.products
            if (key := inchikey(canonical_smiles(p) or p)) is not None
        }
        if not product_keys:
            return None

        for acid_raw in acids:
            acid = _clean_smiles(acid_raw)
            if acid is None:
                continue
            for amine_raw in amines:
                if amine_raw == acid_raw:
                    continue  # require an intermolecular coupling
                amine = _clean_smiles(amine_raw)
                if amine is None:
                    continue
                for product in enumerate_amide_products(acid, amine):
                    if inchikey(product) in product_keys:
                        return MinedAmide(
                            acid_smiles=acid,
                            amine_smiles=amine,
                            product_smiles=product,
                            source=reaction.source,
                            raw_id=reaction.raw_id,
                            yield_value=reaction.yield_value,
                        )
        return None


def detect_amide_batch(reactions: list[RawReaction]) -> tuple[int, list[MinedAmide]]:
    """Detect amide couplings in a batch (a picklable worker for multiprocessing).

    Args:
        reactions: Raw reactions to scan.

    Returns:
        ``(n_scanned, matches)`` -- the number examined and the confirmed amide couplings.
    """
    detector = AmideCouplingDetector()
    matches = [m for r in reactions if (m := detector.detect(r)) is not None]
    return len(reactions), matches


@dataclass
class _PairAggregate:
    """Mutable accumulator for all records of one (acid, amine) substrate pair."""

    acid_smiles: str
    amine_smiles: str
    product_smiles: str
    sources: set[str]
    max_yield: float | None
    n_records: int


@dataclass
class AmideMiningReport:
    """Provenance + leakage accounting for an amide mining run.

    Attributes:
        n_scanned: Raw reactions examined.
        n_detected: Reactions confirmed as amide couplings.
        n_unique_pairs: Distinct (acid, amine) substrate pairs kept after dedup + leakage drop.
        n_exact_test_pair_hits: Detected pairs that exactly equal a Test (acid, amine) pair.
        n_pairs_dropped_test_leak: Unique pairs removed because they exactly matched a Test pair.
        per_source: Detected-count per source tag.
    """

    n_scanned: int = 0
    n_detected: int = 0
    n_unique_pairs: int = 0
    n_exact_test_pair_hits: int = 0
    n_pairs_dropped_test_leak: int = 0
    per_source: dict[str, int] = field(default_factory=dict)


class AmideMiner:
    """Detect, dedup, and test-leakage-audit amide couplings into a positive corpus.

    Attributes:
        report: Running :class:`AmideMiningReport` populated by :meth:`add_match`.
    """

    def __init__(
        self,
        test_pairs: frozenset[tuple[str, str]],
        *,
        detector: AmideCouplingDetector | None = None,
    ) -> None:
        """Initialise the miner with the Test leakage reference.

        Args:
            test_pairs: InChIKey-14 (acid, amine) pairs of the held-out Test set. A mined pair
                equal to one of these is dropped (exact-pair leakage) but counted.
            detector: Detector to use (defaults to a fresh :class:`AmideCouplingDetector`).
        """
        self._test_pairs = test_pairs
        self._detector = detector or AmideCouplingDetector()
        self.report = AmideMiningReport()
        self._pairs: dict[tuple[str, str], _PairAggregate] = {}

    def mine(self, reactions: Iterator[RawReaction]) -> None:
        """Detect couplings in a reaction stream and accumulate unique, deduped, clean pairs."""
        for reaction in reactions:
            self.report.n_scanned += 1
            match = self._detector.detect(reaction)
            if match is not None:
                self.add_match(match)

    def note_scanned(self, count: int) -> None:
        """Record that ``count`` raw reactions were scanned (for parallel accounting)."""
        self.report.n_scanned += count

    def add_match(self, match: MinedAmide) -> None:
        """Fold one detected coupling into the corpus (dedup + exact-Test-pair leakage drop)."""
        self.report.n_detected += 1
        self.report.per_source[match.source] = self.report.per_source.get(match.source, 0) + 1
        acid_key = inchikey(strip_and_canonicalise(match.acid_smiles) or match.acid_smiles)
        amine_key = inchikey(strip_and_canonicalise(match.amine_smiles) or match.amine_smiles)
        if acid_key is None or amine_key is None:
            return
        key = (acid_key, amine_key)
        if key in self._test_pairs:
            self.report.n_exact_test_pair_hits += 1
            return  # transductive leakage guard: never train on an exact Test pair

        record = self._pairs.get(key)
        if record is None:
            self._pairs[key] = _PairAggregate(
                acid_smiles=match.acid_smiles,
                amine_smiles=match.amine_smiles,
                product_smiles=match.product_smiles,
                sources={match.source},
                max_yield=match.yield_value,
                n_records=1,
            )
        else:
            record.sources.add(match.source)
            record.n_records += 1
            record.max_yield = _max_optional(record.max_yield, match.yield_value)

    @property
    def n_pairs(self) -> int:
        """Number of unique (acid, amine) pairs accumulated so far (for progress readouts)."""
        return len(self._pairs)

    def to_frame(self) -> pd.DataFrame:
        """Return the deduped positive corpus as a DataFrame (one row per acid/amine pair)."""
        self.report.n_unique_pairs = len(self._pairs)
        self.report.n_pairs_dropped_test_leak = self.report.n_exact_test_pair_hits
        rows = [
            {
                "acid_key": acid_key,
                "amine_key": amine_key,
                "acid_smiles": rec.acid_smiles,
                "amine_smiles": rec.amine_smiles,
                "product_smiles": rec.product_smiles,
                "sources": "|".join(sorted(rec.sources)),
                "max_yield": rec.max_yield,
                "n_records": rec.n_records,
            }
            for (acid_key, amine_key), rec in self._pairs.items()
        ]
        return pd.DataFrame(rows)


def _max_optional(a: float | None, b: float | None) -> float | None:
    """Maximum of two optional yields (ignoring ``None``)."""
    values = [x for x in (a, b) if isinstance(x, (int, float))]
    return max(float(v) for v in values) if values else None
