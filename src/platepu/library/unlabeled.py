"""The unlabeled space U: every library compound of role A paired with every one of role B, minus
the positives of the family (ADR 003). Every pair is internal data (ADR 004)."""

import pandas as pd

from platepu.library.inventory import INTERNAL

PAIR = ["key_a_stereo", "key_b_stereo"]


def unlabeled_pairs(
    library: pd.DataFrame, positives: pd.DataFrame, role_a: str, role_b: str
) -> pd.DataFrame:
    """All role-A × role-B pairs of ``library`` that are not in ``positives`` (by stereo keys)."""
    side = ["key_stereo", "key_flat", "smiles"]
    a = library.loc[library["role"] == role_a, side].add_suffix("_a")
    b = library.loc[library["role"] == role_b, side].add_suffix("_b")
    pairs = a.merge(b, how="cross").rename(
        columns={
            "key_stereo_a": "key_a_stereo",
            "key_flat_a": "key_a_flat",
            "smiles_a": "substrate_a",
            "key_stereo_b": "key_b_stereo",
            "key_flat_b": "key_b_flat",
            "smiles_b": "substrate_b",
        }
    )
    known = pairs.merge(
        positives[PAIR].drop_duplicates(), on=PAIR, how="left", indicator=True
    )
    unlabeled = pairs[(known["_merge"] == "left_only").to_numpy()]
    return unlabeled.reset_index(drop=True).assign(licence=INTERNAL)
