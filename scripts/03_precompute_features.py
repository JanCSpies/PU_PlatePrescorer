"""Precompute and cache reaction features in parallel (built for SLURM CPU nodes).

DRFP featurisation is the pipeline bottleneck (single-threaded ~30 ms/reaction). This
script parallelises it across many CPU cores with a process pool and caches the results
so downstream model/benchmark runs load features instantly instead of recomputing.

For each dataset (held-out JnJ test, leakage-filtered literature, and the enumerated
unlabeled combinatorial grid) it computes DRFP and ECFP feature matrices and writes:

    datasets/processed/features/<dataset>.parquet         # aligned rows + labels
    datasets/processed/features/<dataset>__<feat>.npz     # X (float32), row-aligned
    datasets/processed/features/manifest.json             # shapes, timings, config

Usage:
    uv run python scripts/03_precompute_features.py --workers 64
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bhpu.features import DRFPFeaturizer, ECFPFeaturizer, ReactionFeaturizer
from bhpu.leakage import LeakageAuditor
from bhpu.space import CombinatorialSpaceGenerator

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
PROCESSED: Path = REPO_ROOT / "datasets" / "processed"
FEATURE_DIR: Path = PROCESSED / "features"
CHUNK_SIZE: int = 200


@dataclass(frozen=True, slots=True)
class FeaturizerSpec:
    """A featuriser identity and its bit width.

    Attributes:
        kind: ``"drfp"`` or ``"ecfp"``.
        n_bits: Folded fingerprint length.
    """

    kind: str
    n_bits: int

    @property
    def name(self) -> str:
        """A short cache key like ``drfp2048`` / ``ecfp2048``."""
        return f"{self.kind}{self.n_bits}"


def _build_featurizer(spec: FeaturizerSpec) -> ReactionFeaturizer:
    """Instantiate a featuriser from ``spec`` (top-level for picklability)."""
    if spec.kind == "drfp":
        return DRFPFeaturizer(n_bits=spec.n_bits)
    if spec.kind == "ecfp":
        return ECFPFeaturizer(n_bits=spec.n_bits)
    raise ValueError(f"Unknown featuriser kind: {spec.kind!r}")


def _featurize_chunk(payload: tuple[FeaturizerSpec, pd.DataFrame]) -> np.ndarray:
    """Featurise one chunk in a worker process."""
    spec, chunk = payload
    return _build_featurizer(spec).transform(chunk)


def _parallel_featurize(spec: FeaturizerSpec, frame: pd.DataFrame, workers: int) -> np.ndarray:
    """Featurise ``frame`` with ``spec`` across a process pool, preserving row order."""
    chunks = [frame.iloc[i : i + CHUNK_SIZE] for i in range(0, len(frame), CHUNK_SIZE)]
    payloads = [(spec, chunk) for chunk in chunks]
    # forkserver avoids the fork-after-threads deadlock that numpy/rdkit can trigger.
    context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        blocks = list(pool.map(_featurize_chunk, payloads))
    return np.vstack(blocks).astype(np.float32)


def _build_datasets() -> dict[str, pd.DataFrame]:
    """Assemble the frames to featurise (JnJ, clean literature, unlabeled grid)."""
    jnj = pd.read_parquet(PROCESSED / "test_jnj_hte_2024.parquet").reset_index(drop=True)
    literature = pd.read_parquet(PROCESSED / "literature_hte.parquet").reset_index(drop=True)
    literature_clean = (
        LeakageAuditor(jnj).filter(literature, level="substrate").reset_index(drop=True)
    )
    unlabeled = CombinatorialSpaceGenerator(jnj, add_products=True).generate(
        exclude="measured_pairs"
    )
    return {
        "test_jnj": jnj,
        "literature_clean": literature_clean,
        "unlabeled_grid": unlabeled,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)),
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--bits", type=int, default=2048, help="Fingerprint length for DRFP and ECFP."
    )
    args = parser.parse_args()

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        FeaturizerSpec("drfp", args.bits),
        FeaturizerSpec("ecfp", args.bits),
    ]
    datasets = _build_datasets()

    manifest: dict[str, object] = {"workers": args.workers, "bits": args.bits, "entries": []}
    entries: list[dict[str, object]] = []
    for name, frame in datasets.items():
        frame.to_parquet(FEATURE_DIR / f"{name}.parquet", index=False)
        for spec in specs:
            start = time.perf_counter()
            features = _parallel_featurize(spec, frame, args.workers)
            elapsed = time.perf_counter() - start
            out_path = FEATURE_DIR / f"{name}__{spec.name}.npz"
            np.savez_compressed(out_path, X=features)
            entry = {
                "dataset": name,
                "featurizer": spec.name,
                "shape": list(features.shape),
                "seconds": round(elapsed, 1),
                "path": str(out_path.relative_to(REPO_ROOT)),
            }
            entries.append(entry)
            print(f"[{name}::{spec.name}] {features.shape} in {elapsed:.1f}s -> {out_path.name}")

    manifest["entries"] = entries
    (FEATURE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(entries)} feature matrices + manifest to {FEATURE_DIR.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
