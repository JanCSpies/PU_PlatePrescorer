"""Validate the per-substrate reactive-site locator against product-confirmed couplings.

Site-specific descriptors are only meaningful if we describe the *right* atom. Our locators pick
one canonical site per substrate from the BH template (the first enumerated outcome), which is
unambiguous for mono-functional substrates but a guess for poly-halide aryls / poly-N–H amines.

This script measures how often that guess is correct, using the mined corpus as ground truth:
each mined pair stores the *observed* coupling product, so the locator's chosen site is correct
iff :func:`build_cn_coupling_product` (which uses the same first-outcome site) reproduces that
product. It reports overall accuracy and — the number that matters — accuracy restricted to the
ambiguous (multi-site) substrates.

Usage:
    uv run python scripts/13_validate_reaction_sites.py [--corpus <mined_corpus.parquet>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bhpu.chem.canonical import inchikey
from bhpu.chem.reaction import build_cn_coupling_product, enumerate_cn_coupling_products


def main() -> None:
    """Entry point: report reactive-site accuracy against the product-confirmed corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="datasets/processed/cn_corpus_ord.parquet")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    corpus = pd.read_parquet(Path(args.corpus))
    if args.limit is not None:
        corpus = corpus.head(args.limit)

    n_total = n_single = n_multi = 0
    correct_single = correct_multi = 0
    for row in corpus.itertuples(index=False):
        aryl, amine = row.aryl_smiles, row.amine_smiles
        observed = inchikey(row.product_smiles)
        all_products = enumerate_cn_coupling_products(aryl, amine)
        n_sites = len(all_products)
        if n_sites == 0 or observed is None:
            continue
        n_total += 1
        first = build_cn_coupling_product(aryl, amine)  # the locator's canonical site
        picked_correct = first is not None and inchikey(first) == observed
        if n_sites == 1:
            n_single += 1
            correct_single += int(picked_correct)
        else:
            n_multi += 1
            correct_multi += int(picked_correct)

    correct_total = correct_single + correct_multi
    print(f"corpus pairs evaluated : {n_total}")
    print(f"  single-site (unambiguous): {n_single}  ({100 * n_single / n_total:.1f}%)")
    print(f"  multi-site  (ambiguous)  : {n_multi}  ({100 * n_multi / n_total:.1f}%)")
    print("\nlocator picks the observed-reaction site:")
    print(f"  overall     : {correct_total}/{n_total}  = {100 * correct_total / n_total:.1f}%")
    if n_single:
        print(
            f"  single-site : {correct_single}/{n_single}  = {100 * correct_single / n_single:.1f}%"
        )
    if n_multi:
        print(
            f"  multi-site  : {correct_multi}/{n_multi}  = {100 * correct_multi / n_multi:.1f}%"
            "   <- the number that matters"
        )


if __name__ == "__main__":
    main()
