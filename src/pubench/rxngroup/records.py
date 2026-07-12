"""Typed records that flow through the reaction-grouping pipeline.

The pipeline has two record types:

* :class:`SourceReaction` -- the *raw* reaction as read from one source, before any chemistry. It
  keeps reactants, condition agents and products separate (so conditions can be extracted from the
  agent bag rather than pooled with substrates), plus the atom-mapped reaction string (for the
  reaction-center signature) and any reported temperature / yield.
* :class:`ParsedReaction` -- one fully processed reaction row: a canonical dedup key, its reaction
  class (named-reaction label + reaction-center signature), its binned condition cluster, the two
  coupling partners, and provenance. These become the columns of ``<source>_parsed.parquet``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class SourceReaction:
    """A raw reaction streamed from a source, with roles kept separate.

    Attributes:
        reactants: SMILES of the reactant molecules (the coupling-partner candidates). May be
            atom-mapped (Pistachio / USPTO) or plain (some ORDerly rows).
        agents: SMILES of the condition bag -- catalyst / precatalyst / ligand / base / solvent /
            other reagents. Conditions are extracted from here, not from ``reactants``.
        products: SMILES of the product molecules.
        mapped_rxn: The full atom-mapped ``reactants>agents>products`` reaction SMILES, or ``None``
            if the source row is not atom-mapped (then no reaction-center signature is computed).
        source: Provenance tag (``"Pistachio"`` / ``"USPTO"`` / ``"ORDerly"``).
        raw_id: Source-native identifier (patent no., ORD id, row index), if any.
        yield_value: Reported yield in percent, if any.
        temperature: Reported temperature in degrees Celsius, if any (ORDerly only).
    """

    reactants: tuple[str, ...]
    agents: tuple[str, ...]
    products: tuple[str, ...]
    mapped_rxn: str | None
    source: str
    raw_id: str | None = None
    yield_value: float | None = None
    temperature: float | None = None


@dataclass(frozen=True, slots=True)
class ParsedReaction:
    """One processed reaction: dedup key + class + condition cluster + partners + provenance.

    Attributes:
        rxn_key: Canonical, atom-map-stripped ``sorted(reactants)>>sorted(products)`` key used to
            deduplicate the same reaction across sources.
        reaction_class: Named-reaction label (e.g. ``"Suzuki-Miyaura"``) or ``"other"``.
        center_sig: Atom-mapped reaction-center signature (formed/broken bond environments), or
            ``None`` if the row was not mapped. Used to sub-split the ``"other"`` bucket.
        partner_a: Canonical SMILES of the first coupling partner for the matched class (e.g. the
            aryl halide), or ``None`` if the class does not define ordered partners.
        partner_b: Canonical SMILES of the second coupling partner (e.g. the boron / amine).
        metal: Catalyst-metal family (``"Pd"`` / ``"Ni"`` / ``"Cu"`` / ``"Pd+Cu"`` / ``"none"``).
        ligand_key: Connectivity InChIKey of the free ligand (precatalyst == free ligand), or
            ``None``.
        base_family: Base family (``"carbonate"`` / ``"phosphate"`` / ``"amine"`` ...), or ``None``.
        solvent_family: Solvent class (``"ethereal"`` / ``"polar_aprotic"`` / ``"protic"`` ...), or
            ``None``.
        temp_bin: Temperature bin label (``"rt"`` / ``"40-80"`` / ``"80-120"`` / ``">120"`` /
            ``"unknown"``).
        yield_value: Reported yield in percent, if any.
        source: Provenance tag.
        raw_id: Source-native identifier, if any.
    """

    rxn_key: str
    reaction_class: str
    center_sig: str | None
    partner_a: str | None
    partner_b: str | None
    metal: str
    ligand_key: str | None
    base_family: str | None
    solvent_family: str | None
    temp_bin: str
    yield_value: float | None
    source: str
    raw_id: str | None

    def as_row(self) -> dict[str, object]:
        """Return the record as a flat dict for a pandas/parquet row."""
        return asdict(self)
