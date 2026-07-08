"""The explicit on-disk data contract for the ``final_datasets`` splits.

These constants name the columns the pipeline reads from the assembled parquet files
(``scripts_JSP/assemble_datasets.py`` output). All four splits share the substrate and unified
condition columns; only the JnJ-derived splits (val/test) carry a ground-truth ``labels`` column.

On-disk files (under the datasets root):

======  ==================================  ================================================
role    file                                notes
======  ==================================  ================================================
P       ``private/positives_merged_bh.parquet``  positives (ORD+Pistachio), all ``label == 1``
U       ``unlabeled_grid.parquet``          unlabeled pool; ``Product SMILES`` is derived
val     ``test.parquet`` (``eval_split=="val"``)   labelled; model selection / thresholding
test    ``test.parquet`` (``eval_split=="test"``)  labelled; final evaluation only
======  ==================================  ================================================
"""

from __future__ import annotations

from typing import Final

# --- Substrate + product columns (present in P and test; derived for U) -------
ARYL_SMILES: Final[str] = "Aryl SMILES"
AMINE_SMILES: Final[str] = "Amine SMILES"
PRODUCT_SMILES: Final[str] = "Product SMILES"

# --- Unified condition columns (present in P, U and test) ---------------------
METAL: Final[str] = "metal"
LIGAND_SMILES: Final[str] = "ligand_smiles"
BASE_SMILES: Final[str] = "cond_base_smiles"
SOLVENT_SMILES: Final[str] = "cond_solvent_smiles"

#: The condition SMILES columns fingerprinted by the ECFP featurizer.
CONDITION_SMILES_COLUMNS: Final[tuple[str, str, str]] = (
    LIGAND_SMILES,
    BASE_SMILES,
    SOLVENT_SMILES,
)

# --- Ground-truth + split columns (JnJ / val + test) --------------------------
#: Binary feasibility label on the labelled splits (1 = reaction works, 0 = does not).
LABEL: Final[str] = "labels"
#: PU-positive marker column present in the mined positives corpus (always 1).
POSITIVE_MARKER: Final[str] = "label"
#: Column of the JnJ plate assigning each row to the frozen ``val`` / ``test`` / ``none`` split.
EVAL_SPLIT: Final[str] = "eval_split"

# --- Default file names under the datasets root -------------------------------
DEFAULT_POSITIVES: Final[dict[str, str]] = {
    "merged": "private/positives_merged_bh.parquet",
    "ord": "positives_ord_bh.parquet",
}
DEFAULT_UNLABELED: Final[str] = "unlabeled_grid.parquet"
DEFAULT_EVAL: Final[str] = "test.parquet"
