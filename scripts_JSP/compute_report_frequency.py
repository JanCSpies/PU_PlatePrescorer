"""Compute per-substrate *report frequency* from the ORDerly forward corpus.

Report frequency = how often a building block appears as a **reactant** across all reaction types in
the literature/patent corpus (ORDerly forward, ~832k reactions). It is a **non-chemical**
reportability / popularity proxy (a common, cheap, well-known building block is reported a lot; an
obscure one is not), and — crucially — it is defined on the *same* InChIKey-14 identity used for
``aryl_key`` / ``amine_key``, so it is comparable for **both** literature and JnJ substrates. It is
the first input to the Stream-D propensity work: it lets us model *"was this reported because it is
feasible, or just because it is popular?"* (the beyond-SCAR / SAR selection-bias correction).

We count across *all* reaction types (not BH only) on purpose: a BH-specific count would entangle
popularity with BH feasibility; cross-reaction usage is closer to pure availability/popularity.

Run (SLURM preferred — see docs/slurm.md):

    sbatch --partition=CPU --nice=1000000 --cpus-per-task=32 --mem=32G \
        --wrap="uv run python scripts_JSP/compute_report_frequency.py --jobs 32"
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from bhpu.chem.canonical import inchikey, strip_and_canonicalise

_LOGGER = logging.getLogger("report_frequency")

_DEFAULT_CORPUS = "/mnt/datasets/ORDerly/orderly_forward_train.parquet"
_REACTANT_COLS = ("reactant_000", "reactant_001", "reactant_002")
_DEFAULT_OUT = "datasets/processed/features/report_frequency.parquet"


def _key_for(smiles: str) -> str | None:
    """Return the connectivity InChIKey-14 for a reactant SMILES (matches the substrate keys)."""
    return inchikey(strip_and_canonicalise(smiles) or smiles)


def main() -> None:
    """Scan the corpus reactants, map each to an InChIKey-14, and sum occurrence counts per key."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=_DEFAULT_CORPUS)
    parser.add_argument("--out", default=_DEFAULT_OUT)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    available = set(pq.ParquetFile(args.corpus).schema.names)
    cols = [c for c in _REACTANT_COLS if c in available]
    if not cols:
        raise ValueError(f"No reactant columns {_REACTANT_COLS} found in {args.corpus}.")
    _LOGGER.info("reading reactant columns %s from %s", cols, args.corpus)
    frame = pd.read_parquet(args.corpus, columns=list(cols))

    # SMILES -> occurrence count (dedup so we canonicalise each unique SMILES once).
    smiles_counts = pd.concat([frame[c] for c in cols], ignore_index=True).dropna().value_counts()
    smiles_counts = smiles_counts[smiles_counts.index.str.len() > 0]
    unique = list(smiles_counts.index)
    _LOGGER.info(
        "%d reactant occurrences over %d unique SMILES", int(smiles_counts.sum()), len(unique)
    )

    if args.jobs > 1:
        ctx = mp.get_context("forkserver")  # RDKit-safe (docs/slurm.md)
        with ctx.Pool(args.jobs) as pool:
            keys = pool.map(_key_for, unique, chunksize=256)
    else:
        keys = [_key_for(s) for s in unique]

    # Aggregate occurrence counts and distinct-SMILES-variant counts per InChIKey-14.
    key_count: Counter[str] = Counter()
    key_variants: Counter[str] = Counter()
    for smiles, key in zip(unique, keys, strict=True):
        if key is None:
            continue
        key_count[key] += int(smiles_counts[smiles])
        key_variants[key] += 1

    out = pd.DataFrame(
        {
            "substrate_key": list(key_count.keys()),
            "report_count": [key_count[k] for k in key_count],
            "n_smiles_variants": [key_variants[k] for k in key_count],
        }
    ).sort_values("report_count", ascending=False, ignore_index=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    _LOGGER.info(
        "wrote %d unique substrate keys to %s (max count %d, median %d)",
        len(out),
        out_path,
        int(out["report_count"].max()),
        int(out["report_count"].median()),
    )


if __name__ == "__main__":
    main()
