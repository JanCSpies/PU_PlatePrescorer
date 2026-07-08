"""Precompute GFN2-xTB descriptors for every unique substrate (SLURM-parallel, cached).

Gathers all aryl/amine substrates from the literature, JnJ, grid, and mined-ORD sets, computes
their GFN2-xTB electronic descriptors once each (forkserver workers), and writes a cache parquet
(``smiles`` + descriptor columns) consumed by :class:`bhpu.features.xtb.XTBDescriptorFeaturizer`.

Usage:
    uv run python scripts/10_precompute_xtb.py [--workers 60] [--limit N] [--include-ord]
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from bhpu.chem.canonical import canonical_smiles
from bhpu.features.xtb import XTB_DESCRIPTORS, compute_xtb_descriptors

FEATURES_DIR = Path("datasets/processed/features")
CORPUS = Path("datasets/processed/cn_corpus_ord.parquet")
OUT = FEATURES_DIR / "xtb_substrate_cache.parquet"


def _unique_substrates(*, include_ord: bool) -> list[str]:
    """Collect all canonical substrate SMILES across the datasets."""
    frames = {
        "lit": (FEATURES_DIR / "literature_clean.parquet", "Aryl SMILES", "Amine SMILES"),
        "jnj": (FEATURES_DIR / "test_jnj.parquet", "Aryl SMILES", "Amine SMILES"),
        "grid": (FEATURES_DIR / "unlabeled_grid.parquet", "Aryl SMILES", "Amine SMILES"),
    }
    substrates: set[str] = set()
    for path, aryl_col, amine_col in frames.values():
        frame = pd.read_parquet(path)
        for column in (aryl_col, amine_col):
            substrates.update(
                c for s in frame[column] if (c := canonical_smiles(s)) is not None
            )
    if include_ord and CORPUS.exists():
        corpus = pd.read_parquet(CORPUS)
        for column in ("aryl_smiles", "amine_smiles"):
            substrates.update(
                c for s in corpus[column] if (c := canonical_smiles(s)) is not None
            )
    return sorted(substrates)


def _worker(smiles: str) -> tuple[str, np.ndarray | None]:
    """Compute descriptors for one SMILES (picklable pool worker)."""
    return smiles, compute_xtb_descriptors(smiles)


def main() -> None:
    """Entry point: compute + cache xTB descriptors for all unique substrates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-ord", action="store_true")
    args = parser.parse_args()

    substrates = _unique_substrates(include_ord=args.include_ord)
    if args.limit is not None:
        substrates = substrates[: args.limit]
    print(f"Computing GFN2-xTB descriptors for {len(substrates)} unique substrates...")

    start = time.time()
    rows: list[dict[str, object]] = []
    n_failed = 0
    ctx = get_context("forkserver")
    with ctx.Pool(processes=args.workers) as pool:
        for i, (smiles, vector) in enumerate(pool.imap_unordered(_worker, substrates, chunksize=8)):
            if vector is None:
                n_failed += 1
                continue
            rows.append({"smiles": smiles, **dict(zip(XTB_DESCRIPTORS, vector, strict=True))})
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.time() - start)
                print(f"  {i + 1}/{len(substrates)}  ({rate:.0f}/s, {n_failed} failed)")

    frame = pd.DataFrame(rows)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT, index=False)
    print(
        f"\nWrote {OUT}: {len(frame)} substrates "
        f"({n_failed} failed, {time.time() - start:.0f}s total)"
    )


if __name__ == "__main__":
    main()
