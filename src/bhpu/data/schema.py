"""Central schema: column names, the feasibility rule, and condition-set definitions.

Keeping every raw column name and domain constant in one place avoids magic strings
scattered across featurisers, the leakage auditor, and the space generator, and gives a
single point of truth if the upstream file schema changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# --- Raw component columns ----------------------------------------------------

SOURCE: Final[str] = "Source"
ARYL_SMILES: Final[str] = "Aryl SMILES"
AMINE_SMILES: Final[str] = "Amine SMILES"
CATALYST_NAME: Final[str] = "Catalyst Name"
CATALYST_SMILES: Final[str] = "Catalyst SMILES"
BASE_NAME: Final[str] = "Base Name"
BASE_SMILES: Final[str] = "Base SMILES"
SOLVENT_NAME: Final[str] = "Solvent Name"
SOLVENT_SMILES: Final[str] = "Solvent SMILES"
PRODUCT_SMILES: Final[str] = "Product SMILES"
YIELD: Final[str] = "Yield"
LABEL: Final[str] = "labels"

#: The two substrate columns (define an aryl-halide × amine pair).
SUBSTRATE_COLUMNS: Final[tuple[str, str]] = (ARYL_SMILES, AMINE_SMILES)

#: The three reaction-condition (reagent) SMILES columns.
CONDITION_SMILES_COLUMNS: Final[tuple[str, str, str]] = (
    CATALYST_SMILES,
    BASE_SMILES,
    SOLVENT_SMILES,
)

#: The three reaction-condition name columns (human-readable categoricals).
CONDITION_NAME_COLUMNS: Final[tuple[str, str, str]] = (
    CATALYST_NAME,
    BASE_NAME,
    SOLVENT_NAME,
)

# --- Feasibility rule ---------------------------------------------------------

#: A reaction is "feasible" iff assay yield is at least this many percent.
#: Verified to reproduce the ``labels`` column with 99.9 % agreement.
FEASIBILITY_YIELD_THRESHOLD_PCT: Final[float] = 10.0


# --- Condition sets -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConditionSet:
    """A named (catalyst, base, solvent) reaction-condition triple.

    Attributes:
        catalyst_name: Pd pre-catalyst / ligand name (e.g. ``"ruphos-pd-g4"``).
        base_name: Base name (e.g. ``"P2Et"``).
        solvent_name: Solvent name (e.g. ``"t-AmOH"``).
    """

    catalyst_name: str
    base_name: str
    solvent_name: str

    def as_tuple(self) -> tuple[str, str, str]:
        """Return the ``(catalyst, base, solvent)`` name triple."""
        return (self.catalyst_name, self.base_name, self.solvent_name)


#: The four dominant "general conditions" screened broadly across the JnJ substrate
#: matrix (Rinehart–Doyle substrate-adaptive Pd C–N sets; all base = P2Et). See
#: docs/literature/notes/bh-JnJ-testset-spec.md §4.
GENERAL_CONDITIONS: Final[tuple[ConditionSet, ...]] = (
    ConditionSet("ruphos-pd-g4", "P2Et", "t-AmOH"),
    ConditionSet("brettphos-pd-g4", "P2Et", "t-AmOH"),
    ConditionSet("xphos-pd-g4", "P2Et", "DMA"),
    ConditionSet("p(tbu)3-pd-g4", "P2Et", "DMSO"),
)
