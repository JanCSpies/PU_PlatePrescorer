"""Mine a positive corpus of Buchwald–Hartwig C–N couplings from ORD (open) sources.

Streams the DynaMAIK ``ord.csv`` and the ORDerly parquets, detects product-confirmed aryl C–N
couplings in parallel (RDKit in forkserver workers), deduplicates to unique substrate pairs across
all sources, and audits every pair against the JnJ test set (dropping any that reuse a JnJ
substrate; separately counting exact JnJ-pair overlaps — the "is the test set inside ORD?" check).

Writes ``datasets/processed/cn_corpus_ord.parquet`` (+ ``.csv``) and a provenance JSON. Never
modifies the shared ``/mnt`` sources.

Usage:
    uv run python scripts/07_mine_cn_couplings.py [--limit N] [--workers 30] [--sources ord,orderly]
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections.abc import Iterator
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

from bhpu.data.mining import (
    CNCouplingMiner,
    MinedReaction,
    RawReaction,
    detect_batch,
    iter_ord_csv,
    iter_orderly,
    iter_pistachio_csv,
)
from bhpu.eval.protocol import FrozenEvalProtocol

JNJ_PATH = Path("datasets/processed/test_jnj_hte_2024.parquet")
ORD_CSV = Path("/mnt/datasets/DynaMAIK/ord.csv")
ORDERLY_DIR = Path("/mnt/datasets/ORDerly")
# Pistachio is LICENSED: read the shared copy only; its mined corpus must stay gitignored
# (write with --out private/... ) and must never be published.
PISTACHIO_CSV = Path("/mnt/datasets/pistachio/releases/latest/data/pistachio.smi.deduped.csv")
OUT_DIR = Path("datasets/processed")
ORDERLY_FILES = {
    "ORDerly-forward": "orderly_forward_train.parquet",
    "ORDerly-condition": "orderly_condition_train.parquet",
}


def _raw_stream(sources: list[str]) -> Iterator[RawReaction]:
    """Chain the requested source readers into a single raw-reaction stream."""
    if "ord" in sources:
        yield from iter_ord_csv(ORD_CSV, source="ORD")
    if "orderly" in sources:
        for tag, filename in ORDERLY_FILES.items():
            path = ORDERLY_DIR / filename
            if path.exists():
                yield from iter_orderly(path, source=tag)
    if "pistachio" in sources and PISTACHIO_CSV.exists():
        yield from iter_pistachio_csv(PISTACHIO_CSV, source="Pistachio")


def _batched(stream: Iterator[RawReaction], size: int) -> Iterator[list[RawReaction]]:
    """Group a raw-reaction stream into lists of ``size`` for pool dispatch."""
    while batch := list(itertools.islice(stream, size)):
        yield batch


def main() -> None:
    """Entry point: parallel-mine ORD into a deduped, leakage-audited positive corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Cap raw reactions (debug).")
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--sources", default="ord,orderly")
    parser.add_argument("--out", default="cn_corpus_ord")
    args = parser.parse_args()
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    jnj = pd.read_parquet(JNJ_PATH)
    protocol = FrozenEvalProtocol(jnj)
    miner = CNCouplingMiner(protocol.jnj_substrate_keys, protocol.jnj_pairs)
    n_sub, n_pair = len(protocol.jnj_substrate_keys), len(protocol.jnj_pairs)
    print(f"JnJ guard: {n_sub} substrates, {n_pair} pairs")
    print(f"Sources: {sources} | workers={args.workers} | batch={args.batch_size}")

    stream: Iterator[RawReaction] = _raw_stream(sources)
    if args.limit is not None:
        stream = itertools.islice(stream, args.limit)

    start = time.time()
    ctx = get_context("forkserver")
    with ctx.Pool(processes=args.workers) as pool:
        last_report = 0
        for n_scanned, matches in pool.imap_unordered(
            detect_batch, _batched(stream, args.batch_size)
        ):
            miner.note_scanned(n_scanned)
            for match in matches:
                assert isinstance(match, MinedReaction)
                miner.add_match(match)
            if miner.report.n_scanned - last_report >= 200_000:
                last_report = miner.report.n_scanned
                elapsed = time.time() - start
                print(
                    f"  scanned={miner.report.n_scanned} detected={miner.report.n_detected} "
                    f"pairs={len(miner.to_frame())} "
                    f"leak_drop={miner.report.n_pairs_dropped_substrate_leak} "
                    f"exact_jnj={miner.report.n_exact_jnj_pair_hits} "
                    f"t={elapsed:.0f}s rate={miner.report.n_scanned / max(elapsed, 1):.0f}/s"
                )

    frame = miner.to_frame()
    elapsed = time.time() - start
    out_base = OUT_DIR / args.out
    out_base.parent.mkdir(parents=True, exist_ok=True)  # supports gitignored private/ subdir
    frame.to_parquet(out_base.with_suffix(".parquet"), index=False)
    frame.to_csv(out_base.with_suffix(".csv"), index=False)

    report = miner.report
    provenance = {
        "sources": sources,
        "n_detected": report.n_detected,
        "n_unique_pairs": len(frame),
        "n_pairs_dropped_substrate_leak": report.n_pairs_dropped_substrate_leak,
        "n_exact_jnj_pair_hits": report.n_exact_jnj_pair_hits,
        "per_source_detected": report.per_source,
        "n_with_pd": int(frame["has_pd"].sum()) if len(frame) else 0,
        "elapsed_seconds": round(elapsed, 1),
        "jnj_substrates": len(protocol.jnj_substrate_keys),
        "jnj_pairs": len(protocol.jnj_pairs),
    }
    (OUT_DIR / f"{args.out}_provenance.json").write_text(json.dumps(provenance, indent=2))
    print("\n=== mining complete ===")
    print(json.dumps(provenance, indent=2))
    print(f"Wrote {OUT_DIR / f'{args.out}.parquet'}  ({len(frame)} unique pairs)")


if __name__ == "__main__":
    main()
