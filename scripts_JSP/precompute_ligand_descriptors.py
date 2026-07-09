"""Precompute physical descriptors for every unique ligand and cache them to parquet.

Ligand descriptors (RDKit bulk + morfeus %V_bur/SASA at phosphorus + GFN2-xTB electronics) are
expensive, so we compute them once per unique ligand identity across P ∪ U ∪ test and write a cache
keyed by :data:`schema.LIGAND_KEY`. The :class:`ConditionDescriptorFeaturizer` then does a fast
lookup. Base and solvent descriptors are static curated tables, so only ligands need precomputing.

Run (needs the ``qm`` extra):

    uv run python scripts_JSP/precompute_ligand_descriptors.py --jobs 16
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from pubench.data import schema
from pubench.features.condition_descriptors import (
    LIGAND_DESCRIPTOR_NAMES,
    compute_ligand_descriptors,
)

_LOGGER = logging.getLogger("precompute_ligands")

_DEFAULT_INPUTS: tuple[str, ...] = (
    "final_datasets/private/positives_merged_bh.parquet",
    "final_datasets/unlabeled_grid.parquet",
    "final_datasets/test.parquet",
)
_DEFAULT_OUT = "datasets/processed/features/ligand_descriptors.parquet"


def _strip_atom_maps(smiles: str) -> str:
    """Return the SMILES with atom-map numbers removed (they are corpus noise, not chemistry)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return str(Chem.MolToSmiles(mol))


def _representative_smiles(frames: list[pd.DataFrame]) -> dict[str, str]:
    """Map each ligand key to its most common atom-map-stripped SMILES across the inputs."""
    rows = pd.concat(
        [f[[schema.LIGAND_KEY, schema.LIGAND_SMILES]].dropna() for f in frames], ignore_index=True
    )
    rows = rows[rows[schema.LIGAND_SMILES].str.len() > 0]
    rows["clean"] = rows[schema.LIGAND_SMILES].map(_strip_atom_maps)
    representative: dict[str, str] = {}
    for key, group in rows.groupby(schema.LIGAND_KEY):
        representative[str(key)] = group["clean"].value_counts().idxmax()
    return representative


def _worker(item: tuple[str, str]) -> tuple[str, str, np.ndarray]:
    """Compute the descriptor vector for one ``(ligand_key, smiles)`` pair (nan on failure)."""
    key, smiles = item
    vector = compute_ligand_descriptors(smiles)
    if vector is None:
        vector = np.full(len(LIGAND_DESCRIPTOR_NAMES), np.nan)
    return key, smiles, vector


def main() -> None:
    """Compute and cache ligand descriptors for all unique ligands in the assembled datasets."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=list(_DEFAULT_INPUTS))
    parser.add_argument("--out", default=_DEFAULT_OUT)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker processes.")
    args = parser.parse_args()

    frames = [pd.read_parquet(p) for p in args.inputs if Path(p).exists()]
    representative = _representative_smiles(frames)
    items = sorted(representative.items())
    _LOGGER.info("computing descriptors for %d unique ligands (jobs=%d)", len(items), args.jobs)

    if args.jobs > 1:
        ctx = mp.get_context("forkserver")  # RDKit-safe (docs/slurm.md)
        with ctx.Pool(args.jobs) as pool:
            results = pool.map(_worker, items)
    else:
        results = [_worker(item) for item in items]

    matrix = np.vstack([vec for _, _, vec in results])
    frame = pd.DataFrame(matrix, columns=list(LIGAND_DESCRIPTOR_NAMES))
    frame.insert(0, schema.LIGAND_KEY, [key for key, _, _ in results])
    frame.insert(1, "representative_smiles", [smi for _, smi, _ in results])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)

    n_failed = int(np.isnan(matrix).all(axis=1).sum())
    n_partial = int((np.isnan(matrix).any(axis=1) & ~np.isnan(matrix).all(axis=1)).sum())
    _LOGGER.info(
        "wrote %d ligand descriptors to %s (%d full failures, %d partial-nan)",
        len(frame),
        out_path,
        n_failed,
        n_partial,
    )


if __name__ == "__main__":
    main()
