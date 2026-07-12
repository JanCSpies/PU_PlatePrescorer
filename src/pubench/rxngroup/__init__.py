"""Reaction-class x condition clustering of large reaction corpora (ORD/Pistachio/USPTO).

Parses each source into a common schema (canonical dedup key, named-reaction class + reaction-center
signature, binned catalyst/base/solvent/temperature condition fields, coupling partners), so the
merged, deduplicated corpus can be grouped into *"same reaction, commonish conditions"* candidate
plates. See ``PROSPECTIVE_REACTION_CONDITION_GROUPING.md``.
"""

from __future__ import annotations

from pubench.rxngroup.classify import NamedReactionLibrary, reaction_center_signature
from pubench.rxngroup.conditions import ConditionExtractor
from pubench.rxngroup.pipeline import parse_reaction, parse_source
from pubench.rxngroup.records import ParsedReaction, SourceReaction

__all__ = [
    "ConditionExtractor",
    "NamedReactionLibrary",
    "ParsedReaction",
    "SourceReaction",
    "parse_reaction",
    "parse_source",
    "reaction_center_signature",
]
