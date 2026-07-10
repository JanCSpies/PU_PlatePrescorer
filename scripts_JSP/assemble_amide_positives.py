"""Merge the ORD and Pistachio amide corpora into the final amide positives P + overlap report.

Combines ``amide_corpus_ord.parquet`` (open) and ``private/amide_corpus_pistachio.parquet``
(licensed) into one deduplicated positives corpus keyed on the (acid, amine) InChIKey-14 pair
(union of sources, max yield, summed record counts), and reports the final substrate-space overlap
against the Chemlex test set. Output goes to ``final_datasets/private/`` because it contains
Pistachio-derived rows (gitignored, never published).

Usage:
    uv run python scripts_JSP/assemble_amide_positives.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bhpu.chem.canonical import inchikey

ORD_CORPUS = Path("datasets/processed/amide_corpus_ord.parquet")
PISTACHIO_CORPUS = Path("datasets/processed/private/amide_corpus_pistachio.parquet")
CHEMLEX = Path("datasets/raw/Chemlex/Chemlex_Acidamine_Wetlab_Data.xlsx")
OUT_P = Path("final_datasets/private/positives_amide_merged.parquet")
OUT_REPORT = Path("artifacts/dataset_switch/amide_overlap.json")


def _merge_sources(values: pd.Series) -> str:
    """Union pipe-joined source tags across the duplicate rows of one pair."""
    parts: set[str] = set()
    for value in values.dropna():
        parts.update(str(value).split("|"))
    return "|".join(sorted(parts))


def _keyset(smiles: pd.Series) -> set[str]:
    """Connectivity InChIKey-14 set for a SMILES column (drops un-parseable / missing)."""
    out: set[str] = set()
    for s in smiles.dropna().astype(str):
        if k := inchikey(s):
            out.add(k)
    return out


def main() -> None:
    """Merge the two amide corpora and report the final Chemlex overlap."""
    ord_p = pd.read_parquet(ORD_CORPUS)
    pistachio = pd.read_parquet(PISTACHIO_CORPUS)
    combined = pd.concat([ord_p, pistachio], ignore_index=True)
    merged = combined.groupby(["acid_key", "amine_key"], as_index=False).agg(
        acid_smiles=("acid_smiles", "first"),
        amine_smiles=("amine_smiles", "first"),
        product_smiles=("product_smiles", "first"),
        sources=("sources", _merge_sources),
        max_yield=("max_yield", "max"),
        n_records=("n_records", "sum"),
    )
    OUT_P.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_P, index=False)

    chemlex = pd.read_excel(CHEMLEX)
    c_acid, c_amine = _keyset(chemlex["Acid"]), _keyset(chemlex["Amine"])
    c_pairs = {
        (inchikey(str(a)), inchikey(str(n)))
        for a, n in zip(chemlex["Acid"], chemlex["Amine"], strict=True)
    }
    c_pairs = {p for p in c_pairs if p[0] and p[1]}
    p_acid, p_amine = set(merged["acid_key"]), set(merged["amine_key"])
    p_pairs = set(zip(merged["acid_key"], merged["amine_key"], strict=True))
    both_in = sum(
        1
        for a, n in zip(chemlex["Acid"], chemlex["Amine"], strict=True)
        if inchikey(str(a)) in p_acid and inchikey(str(n)) in p_amine
    )

    report = {
        "amide_P_merged": {
            "unique_pairs": len(merged),
            "from_ord": len(ord_p),
            "from_pistachio": len(pistachio),
            "unique_acids": int(merged["acid_key"].nunique()),
            "unique_amines": int(merged["amine_key"].nunique()),
        },
        "chemlex_overlap": {
            "acids_in_P": [len(c_acid & p_acid), len(c_acid)],
            "amines_in_P": [len(c_amine & p_amine), len(c_amine)],
            "exact_pairs_in_P": [len(c_pairs & p_pairs), len(c_pairs)],
            "rows_both_substrates_in_P": [both_in, len(chemlex)],
        },
    }
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {OUT_P} ({len(merged)} pairs) and {OUT_REPORT}")


if __name__ == "__main__":
    main()
