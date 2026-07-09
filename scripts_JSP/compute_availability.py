"""Build the commercial-availability lookup from the ZINC in-stock set.

No real per-compound *price* catalogue is mounted (Enamine REAL is make-on-demand products with no
price), so we use **purchasability** as the availability axis: a substrate that is a commercially
in-stock compound (present in ZINC in-stock, ~13.2M compounds) is cheap/available and therefore more
likely to be used and reported. This is a cleaner propensity signal than report-frequency because
availability is set by market factors, largely independent of whether a BH coupling works.
Output = the set of purchasable InChIKey-14 substrate keys (the same identity as
``aryl_key``/``amine_key``), so it joins directly onto P/U/test substrates.

Run on SLURM (cpus-per-task 32, mem 64G):
    uv run python scripts_JSP/compute_availability.py --jobs 32
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
from pathlib import Path

import pandas as pd

from bhpu.chem.canonical import inchikey, strip_and_canonicalise

_LOGGER = logging.getLogger("availability")

_DEFAULT_SRC = "/mnt/datasets/ZINC_instock/combined.smi"
_DEFAULT_OUT = "datasets/processed/features/availability_zinc_instock.parquet"


def _key_for(smiles: str) -> str | None:
    """Connectivity InChIKey-14 for a ZINC SMILES (matches the substrate keys)."""
    return inchikey(strip_and_canonicalise(smiles) or smiles)


def main() -> None:
    """Read the ZINC in-stock SMILES, map to InChIKey-14, and write the purchasable-key set."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=_DEFAULT_SRC)
    parser.add_argument("--out", default=_DEFAULT_OUT)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    _LOGGER.info("reading %s", args.src)
    frame = pd.read_csv(args.src, sep=r"\s+", usecols=["smiles"])
    unique = frame["smiles"].dropna().unique().tolist()
    _LOGGER.info("%d rows, %d unique SMILES", len(frame), len(unique))

    if args.jobs > 1:
        ctx = mp.get_context("forkserver")  # RDKit-safe (docs/slurm.md)
        with ctx.Pool(args.jobs) as pool:
            keys = pool.map(_key_for, unique, chunksize=512)
    else:
        keys = [_key_for(s) for s in unique]

    purchasable = sorted({k for k in keys if k is not None})
    out = pd.DataFrame({"substrate_key": purchasable})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    _LOGGER.info("wrote %d purchasable substrate keys to %s", len(out), out_path)


if __name__ == "__main__":
    main()
