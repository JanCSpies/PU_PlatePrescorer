"""Mine a positive corpus of amide (acid + amine -> amide) couplings from ORD / Pistachio.

Streams the DynaMAIK ``ord.csv``, the ORDerly parquets, and (optionally) Pistachio, detects
product-confirmed amide couplings in parallel (RDKit in forkserver workers), deduplicates to unique
(acid, amine) substrate pairs across all sources, and drops any pair that exactly equals a Chemlex
Test pair (transductive leakage guard -- shared *substrates* are kept, only exact pairs removed).

Writes ``datasets/processed/<out>.parquet`` (+ ``.csv``) and a provenance JSON. Pistachio is
LICENSED: when mining it, write to a ``private/`` out path (gitignored) and never publish. Never
modifies the shared ``/mnt`` sources.

Usage:
    uv run python scripts_JSP/mine_amide.py --sources ord,orderly --out amide_corpus_ord
    uv run python scripts_JSP/mine_amide.py --sources pistachio --out private/amide_corpus_pistachio
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

from bhpu.chem.canonical import inchikey
from bhpu.data.mining import RawReaction, iter_ord_csv, iter_orderly, iter_pistachio_csv
from pubench.mining import AmideMiner, MinedAmide, detect_amide_batch

CHEMLEX_PATH = Path("datasets/raw/Chemlex/Chemlex_Acidamine_Wetlab_Data.xlsx")
ORD_CSV = Path("/mnt/datasets/DynaMAIK/ord.csv")
ORDERLY_DIR = Path("/mnt/datasets/ORDerly")
PISTACHIO_CSV = Path("/mnt/datasets/pistachio/releases/latest/data/pistachio.smi.deduped.csv")
OUT_DIR = Path("datasets/processed")
ORDERLY_FILES = {
    "ORDerly-forward": "orderly_forward_train.parquet",
    "ORDerly-condition": "orderly_condition_train.parquet",
}


def _chemlex_test_pairs() -> frozenset[tuple[str, str]]:
    """Return the InChIKey-14 (acid, amine) pairs of the Chemlex set (the leakage reference).

    Uses *all* Chemlex pairs (conservative): no observed acid/amine pair may leak into P, whichever
    Chemlex split we ultimately evaluate on.
    """
    chemlex = pd.read_excel(CHEMLEX_PATH)
    pairs: set[tuple[str, str]] = set()
    for acid, amine in zip(chemlex["Acid"], chemlex["Amine"], strict=True):
        ak, nk = inchikey(str(acid)), inchikey(str(amine))
        if ak and nk:
            pairs.add((ak, nk))
    return frozenset(pairs)


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
    """Entry point: parallel-mine amide couplings into a deduped, leakage-audited corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Cap raw reactions (debug).")
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--sources", default="ord,orderly")
    parser.add_argument("--out", default="amide_corpus_ord")
    args = parser.parse_args()
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    test_pairs = _chemlex_test_pairs()
    miner = AmideMiner(test_pairs)
    print(f"Chemlex leakage guard: {len(test_pairs)} exact (acid, amine) pairs")
    print(f"Sources: {sources} | workers={args.workers} | batch={args.batch_size}")

    stream: Iterator[RawReaction] = _raw_stream(sources)
    if args.limit is not None:
        stream = itertools.islice(stream, args.limit)

    start = time.time()
    ctx = get_context("forkserver")
    with ctx.Pool(processes=args.workers) as pool:
        last_report = 0
        for n_scanned, matches in pool.imap_unordered(
            detect_amide_batch, _batched(stream, args.batch_size)
        ):
            miner.note_scanned(n_scanned)
            for match in matches:
                assert isinstance(match, MinedAmide)
                miner.add_match(match)
            if miner.report.n_scanned - last_report >= 200_000:
                last_report = miner.report.n_scanned
                elapsed = time.time() - start
                print(
                    f"  scanned={miner.report.n_scanned} detected={miner.report.n_detected} "
                    f"pairs={miner.n_pairs} "
                    f"exact_test={miner.report.n_exact_test_pair_hits} "
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
        "reaction_class": "amide_coupling",
        "sources": sources,
        "n_detected": report.n_detected,
        "n_unique_pairs": len(frame),
        "n_exact_test_pair_hits": report.n_exact_test_pair_hits,
        "n_pairs_dropped_test_leak": report.n_pairs_dropped_test_leak,
        "per_source_detected": report.per_source,
        "chemlex_leakage_pairs": len(test_pairs),
        "elapsed_seconds": round(elapsed, 1),
    }
    (OUT_DIR / f"{args.out}_provenance.json").write_text(json.dumps(provenance, indent=2))
    print("\n=== mining complete ===")
    print(json.dumps(provenance, indent=2))
    print(f"Wrote {out_base.with_suffix('.parquet')}  ({len(frame)} unique pairs)")


if __name__ == "__main__":
    main()
