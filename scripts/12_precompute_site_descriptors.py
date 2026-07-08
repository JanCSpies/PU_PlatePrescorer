"""Precompute reactive-site descriptors (morfeus sterics + g-xTB electronics), SLURM-parallel.

For each unique aryl/amine substrate this embeds a conformer ensemble once, computes the
per-conformer site descriptors, and writes **three aggregation variants** — Boltzmann-weighted,
lowest-energy, and plain mean — so the downstream evaluation can compare conformer-handling
strategies without recomputing the (costly) g-xTB single points. Aryls and amines carry
different descriptor sets and are cached separately, one file per (method, role, mode):

    datasets/processed/features/morfeus_aryl_boltzmann.parquet   ... _lowest / _mean
    datasets/processed/features/gxtb_amine_boltzmann.parquet     ... etc.

Use ``--datasets lit,jnj,grid`` (the default) for a cheap subset-first run that excludes the
large ORD corpus; add ``ord`` once the descriptors are shown to help.

Usage:
    uv run python scripts/12_precompute_site_descriptors.py --method both \
        --datasets lit,jnj,grid --n-conformers 20 [--workers 90] [--limit N]
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from bhpu.chem.canonical import canonical_smiles
from bhpu.chem.sites import AGGREGATION_MODES, aggregate_conformer_descriptors
from bhpu.data import schema
from bhpu.features.gxtb import (
    GXTB_DESCRIPTORS,
    compute_amine_gxtb_conformer_matrix,
    compute_aryl_gxtb_conformer_matrix,
)
from bhpu.features.morfeus_desc import (
    AMINE_STERIC_DESCRIPTORS,
    ARYL_STERIC_DESCRIPTORS,
    compute_amine_conformer_matrix,
    compute_aryl_conformer_matrix,
)

FEATURES_DIR = Path("datasets/processed/features")
CORPUS = Path("datasets/processed/cn_corpus_ord.parquet")

#: A conformer-matrix function: (SMILES, n_conformers=...) -> (matrix, energies) or None.
MatrixFn = Callable[..., "tuple[np.ndarray, list[float]] | None"]

#: (method, role) -> (conformer-matrix fn, descriptor names).
_JOBS: dict[tuple[str, str], tuple[MatrixFn, tuple[str, ...]]] = {
    ("morfeus", "aryl"): (compute_aryl_conformer_matrix, ARYL_STERIC_DESCRIPTORS),
    ("morfeus", "amine"): (compute_amine_conformer_matrix, AMINE_STERIC_DESCRIPTORS),
    ("gxtb", "aryl"): (compute_aryl_gxtb_conformer_matrix, GXTB_DESCRIPTORS),
    ("gxtb", "amine"): (compute_amine_gxtb_conformer_matrix, GXTB_DESCRIPTORS),
}

_DATASET_PATHS = {
    "lit": FEATURES_DIR / "literature_clean.parquet",
    "jnj": FEATURES_DIR / "test_jnj.parquet",
    "grid": FEATURES_DIR / "unlabeled_grid.parquet",
}


def _unique_substrates(role: str, datasets: list[str]) -> list[str]:
    """Collect canonical SMILES for one substrate role across the requested datasets."""
    column = schema.ARYL_SMILES if role == "aryl" else schema.AMINE_SMILES
    substrates: set[str] = set()
    for name in datasets:
        if name == "ord":
            if CORPUS.exists():
                corpus = pd.read_parquet(CORPUS)
                col = "aryl_smiles" if role == "aryl" else "amine_smiles"
                substrates.update(c for s in corpus[col] if (c := canonical_smiles(s)) is not None)
        elif (path := _DATASET_PATHS.get(name)) is not None and path.exists():
            frame = pd.read_parquet(path)
            substrates.update(c for s in frame[column] if (c := canonical_smiles(s)) is not None)
    return sorted(substrates)


# Module-level worker holders (picklable): set per job before the pool starts.
_MATRIX_FN: MatrixFn | None = None
_N_CONFORMERS: int = 20


def _init_worker(matrix_fn: MatrixFn, n_conformers: int) -> None:
    """Pool initialiser: bind the per-job matrix function + conformer count into each worker."""
    global _MATRIX_FN, _N_CONFORMERS
    _MATRIX_FN, _N_CONFORMERS = matrix_fn, n_conformers


def _worker(smiles: str) -> tuple[str, np.ndarray | None, list[float] | None]:
    """Compute one substrate's per-conformer descriptor matrix (picklable pool worker)."""
    assert _MATRIX_FN is not None
    result = _MATRIX_FN(smiles, n_conformers=_N_CONFORMERS)
    if result is None:
        return smiles, None, None
    matrix, energies = result
    return smiles, matrix, energies


def _run_job(
    method: str,
    role: str,
    *,
    datasets: list[str],
    n_conformers: int,
    workers: int,
    limit: int | None,
) -> None:
    """Compute per-conformer matrices for one (method, role) and write all aggregation caches."""
    matrix_fn, names = _JOBS[(method, role)]
    substrates = _unique_substrates(role, datasets)
    if limit is not None:
        substrates = substrates[:limit]
    print(f"\n[{method}/{role}] {len(substrates)} substrates x {n_conformers} conformers ...")

    start = time.time()
    per_mode: dict[str, list[dict[str, object]]] = {mode: [] for mode in AGGREGATION_MODES}
    n_failed = 0
    ctx = get_context("forkserver")
    with ctx.Pool(
        processes=workers, initializer=_init_worker, initargs=(matrix_fn, n_conformers)
    ) as pool:
        for i, (smiles, matrix, energies) in enumerate(
            pool.imap_unordered(_worker, substrates, chunksize=8)
        ):
            if matrix is None or energies is None:
                n_failed += 1
                continue
            for mode in AGGREGATION_MODES:
                vector = aggregate_conformer_descriptors(matrix, energies, mode=mode)
                per_mode[mode].append({"smiles": smiles, **dict(zip(names, vector, strict=True))})
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.time() - start)
                print(f"  {i + 1}/{len(substrates)}  ({rate:.0f}/s, {n_failed} failed)")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    for mode, rows in per_mode.items():
        out = FEATURES_DIR / f"{method}_{role}_{mode}.parquet"
        pd.DataFrame(rows).to_parquet(out, index=False)
    print(
        f"[{method}/{role}] wrote {len(AGGREGATION_MODES)} modes: {len(per_mode['mean'])} ok, "
        f"{n_failed} failed, {time.time() - start:.0f}s"
    )


def main() -> None:
    """Entry point: precompute site descriptors (all aggregation modes) for the given method(s)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("morfeus", "gxtb", "both"), default="both")
    parser.add_argument(
        "--datasets", default="lit,jnj,grid", help="Comma list of lit,jnj,grid,ord."
    )
    parser.add_argument("--n-conformers", type=int, default=20)
    parser.add_argument("--workers", type=int, default=90)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    methods = ("morfeus", "gxtb") if args.method == "both" else (args.method,)
    print(f"datasets={datasets}  methods={methods}  n_conformers={args.n_conformers}")
    for method in methods:
        for role in ("aryl", "amine"):
            _run_job(
                method,
                role,
                datasets=datasets,
                n_conformers=args.n_conformers,
                workers=args.workers,
                limit=args.limit,
            )


if __name__ == "__main__":
    main()
