"""The group's chemical inventory as a substrate library for one reaction class.

The export (``Export Inventory Aktuell - Group.xlsx``) has two rows above the header; one row is one
container. Counted 2026-10-09: 10,212 containers, 9,635 with a SMILES; barcodes are not unique, so a
container is identified by its row. SMILES often carry counter-ions and ``[H+]``; ``standardize()``
reduces a substrate to its neutral largest fragment. Internal group data: every output row carries
``licence = "internal"``, so ``io.write_table`` only writes it under private/ (ADR 004).
"""

from pathlib import Path

import pandas as pd

from platepu.classify.engine import ClassRule
from platepu.classify.groups import group_counts
from platepu.standardise.molecule import StdConfig, standardize

# Licence of every table built from the inventory (ADR 004).
INTERNAL = "internal"

COLUMNS = {
    "stereo-SMILES": "smiles",
    "Name": "name",
    "CAS-Nr.": "cas",
    "Barcode": "barcode",
    "Menge": "amount",
}


def read_inventory(path: Path) -> pd.DataFrame:
    """One row per container: ``container_id`` (row number), smiles, name, cas, barcode, amount."""
    table = pd.read_excel(path, header=2, usecols=list(COLUMNS)).rename(columns=COLUMNS)
    table.insert(0, "container_id", [str(index) for index in table.index])
    return table[["container_id", *COLUMNS.values()]]


def library_table(
    inventory: pd.DataFrame, rule: ClassRule, std: StdConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compounds usable as one of the class's roles, and the containers that are not.

    A compound plays role A when it has exactly one role-A group and no role-B group, and the other
    way round. Reasons for rejected containers: ``no_smiles``, ``unparsable``, ``neutralise_failed``,
    ``no_role``, ``both_roles``, ``multiple_groups``.

    Returns:
        One row per compound (by ``key_stereo``): role, smiles, keys, number and ids of containers,
        first name and CAS; and the rejected containers with a reason. Both carry ``licence``.
    """
    records = inventory.to_dict(orient="records")
    rows = [
        _container({str(k): v for k, v in row.items()}, rule, std) for row in records
    ]
    table = pd.DataFrame(rows)
    usable = table[table["reason"] == ""]
    compounds = (
        usable.groupby("key_stereo", sort=False)
        .agg(
            role=("role", "first"),
            smiles=("std_smiles", "first"),
            key_flat=("key_flat", "first"),
            n_containers=("container_id", "size"),
            container_ids=("container_id", list),
            name=("name", "first"),
            cas=("cas", "first"),
        )
        .reset_index()
        .assign(licence=INTERNAL, std_version=std.std_version)
    )
    rejected = table.loc[table["reason"] != "", ["container_id", "reason"]]
    return compounds, rejected.reset_index(drop=True).assign(licence=INTERNAL)


def _container(
    row: dict[str, object], rule: ClassRule, std: StdConfig
) -> dict[str, object]:
    """Standardise one container's SMILES and give it a role, or a reason why not."""
    out = {
        **row,
        "std_smiles": None,
        "key_stereo": None,
        "key_flat": None,
        "role": None,
    }
    if not isinstance(row["smiles"], str):
        return out | {"reason": "no_smiles"}
    result = standardize(row["smiles"], "substrate", std)
    if result.smiles is None:
        return out | {"reason": result.reason}
    groups_a, groups_b = group_counts(result.smiles, rule)
    reason = _role_problem(groups_a, groups_b)
    role = None if reason else (rule.role_a if groups_a == 1 else rule.role_b)
    return out | {
        "std_smiles": result.smiles,
        "key_stereo": result.key_stereo,
        "key_flat": result.key_flat,
        "role": role,
        "reason": reason,
    }


def _role_problem(groups_a: int, groups_b: int) -> str:
    """Why a molecule cannot play exactly one role ("" when it can)."""
    if groups_a == 0 and groups_b == 0:
        return "no_role"
    if groups_a > 0 and groups_b > 0:
        return "both_roles"
    if groups_a > 1 or groups_b > 1:
        return "multiple_groups"
    return ""
