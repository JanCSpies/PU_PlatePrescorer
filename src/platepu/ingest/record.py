"""The common record every source adapter produces (see docs/design/rebuild.md, Stage 1)."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ReactionRecord:
    """One literature reaction, reshaped from its source but chemically untouched.

    Adapters only read and reshape: every SMILES is kept exactly as the source gives it,
    so any later result can be traced back to the original row. Missing values are
    ``None``, never ``""``, ``"empty"`` or ``"nan"``.

    Attributes:
        source: Short key of the source file, e.g. ``"orderly"``.
        source_id: ID of the row in that source (for ORDerly: the row ``index``).
        licence: ``"open"`` or ``"licensed"``; licensed rows may only be written under
            a ``private/`` folder.
        reactants: Reactant SMILES as given by the source.
        agents: Catalyst and reagent SMILES as given; empty if the source pools them
            with the reactants.
        solvents: Solvent SMILES as given; empty if the source does not separate them.
        products: Product SMILES as given.
        mapped_rxn: Atom-mapped reaction string, if the source has one.
        temperature_c: Reaction temperature in °C.
        reaction_time_h: Reaction time in hours.
        yield_pct: Yield in percent as given; may be above 100 (filtered later, with a
            reason, not here).
        year: Year of the record's date; for patents this is the grant year, not the
            year of the experiment.
    """

    source: str
    source_id: str
    licence: Literal["open", "licensed"]
    reactants: tuple[str, ...]
    agents: tuple[str, ...]
    solvents: tuple[str, ...]
    products: tuple[str, ...]
    mapped_rxn: str | None
    temperature_c: float | None
    reaction_time_h: float | None
    yield_pct: float | None
    year: int | None


@dataclass(frozen=True)
class RejectedRow:
    """A source row an adapter could not turn into a ``ReactionRecord``.

    Rejected rows are saved and counted in the stage report, never dropped silently.

    Attributes:
        source: Short key of the source file, e.g. ``"orderly"``.
        source_id: ID of the row in that source.
        reason: Short, fixed reason such as ``"no reactants"``, so rows can be counted
            by reason.
    """

    source: str
    source_id: str
    reason: str
