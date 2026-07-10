"""Profile the two candidate test sets (HiTEA BH, Chemlex amide) and their P↔Test overlap.

Pre-run analysis for the dataset-switch branch. For each candidate we report substrate-space and
condition-space diversity, and how much the positives corpus P overlaps the test set:

* **HiTEA (Buchwald-Hartwig)** -- P already exists (``positives_merged_bh.parquet``, ORD+Pistachio
  BH). We compute exact substrate/condition overlap against it (this is the leakage + domain-gap
  diagnostic, the analogue of the JnJ ADR-007 analysis).
* **Chemlex (amide coupling)** -- an amide P is *not yet mined*, so full P↔Test overlap is deferred.
  As a cheap substrate-space proxy we test how many Chemlex acids/amines appear as a reactant
  anywhere in ORDerly (``report_frequency.parquet``, 530k InChIKey-14 reactant keys) -- i.e. how
  much substrate coverage an amide mine could recover.

Identity keys are connectivity InChIKey-14 via :func:`bhpu.chem.canonical.inchikey`, matching the
``*_key`` columns already in the positives corpus.

Usage:
    uv run python scripts_JSP/analyze_candidate_datasets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bhpu.chem.canonical import inchikey

HITEA = Path("datasets/raw/HiTEA/buchwald.csv")
CHEMLEX = Path("datasets/raw/Chemlex/Chemlex_Acidamine_Wetlab_Data.xlsx")
MERGED_BH = Path("final_datasets/private/positives_merged_bh.parquet")
ORD_INDEX = Path("datasets/processed/features/report_frequency.parquet")
OUT = Path("artifacts/dataset_switch/overlap_analysis.json")

#: HiTEA solvent names -> SMILES (single-component only; mixtures are reported but not keyed).
HITEA_SOLVENT_SMILES: dict[str, str] = {
    "Dioxane": "O1CCOCC1",
    "t-AmOH": "CCC(C)(C)O",
    "THF": "C1CCOC1",
    "Toluene": "Cc1ccccc1",
}


def _keys(smiles: pd.Series) -> set[str]:
    """Connectivity InChIKey-14 set for a SMILES column (drops un-parseable / missing)."""
    out: set[str] = set()
    for s in smiles.dropna().astype(str):
        k = inchikey(s)
        if k:
            out.add(k)
    return out


def _frac(n: int, d: int) -> float:
    """Safe ``n/d`` as a float (0.0 when ``d == 0``)."""
    return float(n) / d if d else 0.0


def analyze_hitea() -> dict[str, object]:
    """Profile HiTEA BH and compute its overlap against the existing ORD+Pistachio BH positives."""
    h = pd.read_csv(HITEA)
    p = pd.read_parquet(MERGED_BH)

    # --- substrate identity ---
    h_aryl = h["halide"].map(lambda s: inchikey(str(s)) if pd.notna(s) else None)
    h_amine = h["nuc"].map(lambda s: inchikey(str(s)) if pd.notna(s) else None)
    h_pairs = {(a, n) for a, n in zip(h_aryl, h_amine, strict=True) if a and n}
    h_aryl_set, h_amine_set = {a for a in h_aryl if a}, {n for n in h_amine if n}

    p_aryl_set = set(p["aryl_key"].dropna())
    p_amine_set = set(p["amine_key"].dropna())
    p_pairs = set(zip(p["aryl_key"], p["amine_key"], strict=True)) - {(None, None)}

    # --- condition identity ---
    h_ligand_keys = _keys(h["ligand"])
    p_ligand_keys = set(p["ligand_key"].dropna())
    h_solvent_keys = {inchikey(smi) for smi in HITEA_SOLVENT_SMILES.values()}
    h_solvent_keys = {k for k in h_solvent_keys if k}
    p_solvent_keys = set(p["cond_solvent_key"].dropna())
    p_base_keys = set(p["cond_base_key"].dropna())

    bases = h["Reagent_1_Short_Hand"].dropna()
    phosphazene = bases.str.contains("P2-Et|P4-t-Bu", case=False, regex=True).sum()
    hmds = bases.str.contains("HMDS", case=False).sum()

    aryl_hit = len(h_aryl_set & p_aryl_set)
    amine_hit = len(h_amine_set & p_amine_set)
    pair_hit = len(h_pairs & p_pairs)
    return {
        "profile": {
            "rows": len(h),
            "unique_aryl_halide": len(h_aryl_set),
            "unique_amine": len(h_amine_set),
            "unique_pairs": len(h_pairs),
            "unique_ligands": int(h["ligand"].dropna().nunique()),
            "unique_bases": int(h["Reagent_1_Short_Hand"].dropna().nunique()),
            "unique_solvents": int(h["Solvent_1_Name"].dropna().nunique()),
            "unique_catalyst_shorthand": int(h["Catalyst_1_Short_Hand"].dropna().nunique()),
            "pos_rate_yield_ge_10": float((h["Product_Yield_PCT_Area_UV"] >= 10).mean()),
            "pos_rate_yield_gt_0": float((h["Product_Yield_PCT_Area_UV"] > 0).mean()),
            "n_phosphazene_base_rows": int(phosphazene),
            "n_hmds_base_rows": int(hmds),
        },
        "P_corpus": {
            "rows": len(p),
            "unique_aryl": len(p_aryl_set),
            "unique_amine": len(p_amine_set),
            "ligand_key_coverage": _frac(int(p["ligand_key"].notna().sum()), len(p)),
            "base_key_coverage": _frac(int(p["cond_base_key"].notna().sum()), len(p)),
            "solvent_key_coverage": _frac(int(p["cond_solvent_key"].notna().sum()), len(p)),
        },
        "substrate_overlap": {
            "aryl_in_P": [aryl_hit, len(h_aryl_set), _frac(aryl_hit, len(h_aryl_set))],
            "amine_in_P": [amine_hit, len(h_amine_set), _frac(amine_hit, len(h_amine_set))],
            "pairs_in_P (exact leakage to drop)": [
                pair_hit,
                len(h_pairs),
                _frac(pair_hit, len(h_pairs)),
            ],
        },
        "condition_overlap": {
            "ligand_keys_in_P": [
                len(h_ligand_keys & p_ligand_keys),
                len(h_ligand_keys),
                _frac(len(h_ligand_keys & p_ligand_keys), len(h_ligand_keys)),
            ],
            "solvent_keys_in_P": [
                len(h_solvent_keys & p_solvent_keys),
                len(h_solvent_keys),
                _frac(len(h_solvent_keys & p_solvent_keys), len(h_solvent_keys)),
            ],
            "n_P_base_keys_total": len(p_base_keys),
            "note": "P base/solvent keys are sparsely mined (see P_corpus coverage); "
            "HiTEA bases include JnJ-style exotic phosphazenes (P2-Et/P4-t-Bu) & HMDS.",
        },
    }


def analyze_chemlex() -> dict[str, object]:
    """Profile Chemlex amide and estimate substrate coverage against the ORDerly reactant index."""
    c = pd.read_excel(CHEMLEX)
    idx = set(pd.read_parquet(ORD_INDEX)["substrate_key"])

    acid_keys = _keys(c["Acid"])
    amine_keys = _keys(c["Amine"])
    acid_hit = len(acid_keys & idx)
    amine_hit = len(amine_keys & idx)
    return {
        "profile": {
            "rows": len(c),
            "unique_acid": int(c["Acid"].nunique()),
            "unique_amine": int(c["Amine"].nunique()),
            "unique_pairs": int(c.groupby(["Acid", "Amine"]).ngroups),
            "unique_reagents": int(c["Reagents"].nunique()),
            "unique_solvents": int(c["Solvent"].nunique()),
            "pos_rate_conversion_ge_20": float((c["Conversion"] >= 20).mean()),
            "predefined_both_unseen_split": c["Stratified_Split_Both_Unseen"]
            .value_counts()
            .to_dict(),
        },
        "substrate_coverage_in_ORDerly_reactants (proxy; amide P not yet mined)": {
            "acid_keys_in_index": [acid_hit, len(acid_keys), _frac(acid_hit, len(acid_keys))],
            "amine_keys_in_index": [
                amine_hit,
                len(amine_keys),
                _frac(amine_hit, len(amine_keys)),
            ],
            "note": "ORDerly-forward reactant index only (530k keys); full Pistachio amide mine "
            "would raise coverage. This is substrate presence anywhere, not amide-reaction P.",
        },
    }


def main() -> None:
    """Run both analyses, print a readable report, and persist JSON."""
    result = {"hitea_bh": analyze_hitea(), "chemlex_amide": analyze_chemlex()}
    print(json.dumps(result, indent=2))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
