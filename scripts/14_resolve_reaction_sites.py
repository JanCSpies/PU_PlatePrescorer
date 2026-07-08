"""Resolve each substrate's *observed* reacting atom from product-confirmed reactions.

The template locator picks a canonical site per substrate (correct for single-site substrates,
but only 37% accurate on ambiguous poly-halide / poly-N-H substrates — see scripts/13). This
script fixes that for every substrate that appears in a reaction with a known product: it finds
which template outcome reproduces the observed product and records that reacting atom, then takes
a majority vote per substrate. The resulting maps are consumed as ``site_override`` by the
descriptor precompute (scripts/12), so descriptors describe the atom that actually reacted.

Writes ``datasets/processed/features/aryl_site_map.parquet`` (``smiles``, ``carbon_idx``) and
``amine_site_map.parquet`` (``smiles``, ``nitrogen_idx``). Only single, unambiguous outcomes are
needed for most substrates; the maps override the template only where they disagree.

Usage:
    uv run python scripts/14_resolve_reaction_sites.py [--include-pistachio]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from bhpu.chem.canonical import canonical_smiles, inchikey
from bhpu.chem.reaction import coupling_site_indices_for_product
from bhpu.data import schema

PROCESSED = Path("datasets/processed")
FEATURES_DIR = PROCESSED / "features"
PRIVATE = PROCESSED / "private"

#: (path, aryl col, amine col, product col) for every product-bearing reaction source.
_SOURCES = [
    (
        PROCESSED / "test_jnj_hte_2024.parquet",
        schema.ARYL_SMILES,
        schema.AMINE_SMILES,
        schema.PRODUCT_SMILES,
    ),
    (
        PROCESSED / "literature_hte.parquet",
        schema.ARYL_SMILES,
        schema.AMINE_SMILES,
        schema.PRODUCT_SMILES,
    ),
    (PROCESSED / "cn_corpus_ord.parquet", "aryl_smiles", "amine_smiles", "product_smiles"),
]


def _iter_reactions(include_pistachio: bool) -> list[tuple[str, str, str]]:
    """Collect (aryl, amine, product) triples from all product-bearing sources."""
    sources = list(_SOURCES)
    if include_pistachio:
        sources.append(
            (
                PRIVATE / "cn_corpus_pistachio.parquet",
                "aryl_smiles",
                "amine_smiles",
                "product_smiles",
            )
        )
    rows: list[tuple[str, str, str]] = []
    for path, aryl_col, amine_col, product_col in sources:
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=[aryl_col, amine_col, product_col])
        for aryl, amine, product in frame.itertuples(index=False):
            if isinstance(aryl, str) and isinstance(amine, str) and isinstance(product, str):
                rows.append((aryl, amine, product))
    return rows


def main() -> None:
    """Entry point: resolve and write per-substrate reacted-site maps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-pistachio", action="store_true")
    args = parser.parse_args()

    reactions = _iter_reactions(args.include_pistachio)
    print(f"resolving reacting sites from {len(reactions)} product-bearing reactions ...")

    aryl_votes: dict[str, Counter[int]] = {}
    amine_votes: dict[str, Counter[int]] = {}
    n_resolved = 0
    for aryl_raw, amine_raw, product_raw in reactions:
        aryl = canonical_smiles(aryl_raw)
        amine = canonical_smiles(amine_raw)
        product_key = inchikey(canonical_smiles(product_raw) or product_raw)
        if aryl is None or amine is None or product_key is None:
            continue
        sites = coupling_site_indices_for_product(aryl, amine, product_key)
        if sites is None:
            continue
        carbon_idx, _, nitrogen_idx = sites
        aryl_votes.setdefault(aryl, Counter())[carbon_idx] += 1
        amine_votes.setdefault(amine, Counter())[nitrogen_idx] += 1
        n_resolved += 1

    aryl_map = pd.DataFrame(
        [{"smiles": s, "carbon_idx": votes.most_common(1)[0][0]} for s, votes in aryl_votes.items()]
    )
    amine_map = pd.DataFrame(
        [
            {"smiles": s, "nitrogen_idx": votes.most_common(1)[0][0]}
            for s, votes in amine_votes.items()
        ]
    )
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    aryl_map.to_parquet(FEATURES_DIR / "aryl_site_map.parquet", index=False)
    amine_map.to_parquet(FEATURES_DIR / "amine_site_map.parquet", index=False)
    print(
        f"resolved {n_resolved} reactions -> {len(aryl_map)} aryl sites, "
        f"{len(amine_map)} amine sites (majority vote); wrote site maps."
    )


if __name__ == "__main__":
    main()
