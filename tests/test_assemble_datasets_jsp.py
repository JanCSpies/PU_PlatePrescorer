"""Unit tests for the JSP single-file dataset assembler (synthetic parquets; fast).

Focus on the net-new behaviour: the ORD reader must pool ``solvent_*`` into the molecule
bag (the shipped reader dropped it), the Pd filter must reject non-Pd couplings, the
substrate leakage guard must drop JnJ substrates, and the unified condition schema must be
identical across TEST / P / U.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_JSP"))

import assemble_datasets as ad

from bhpu.chem.conditions import CanonicalCondition, condition_from_columns
from bhpu.chem.reaction import build_cn_coupling_product
from bhpu.data import schema

# A known Buchwald–Hartwig coupling: bromobenzene + morpholine -> N-phenylmorpholine.
_ARYL = "Brc1ccccc1"
_AMINE = "C1COCCN1"
_DIOXANE = "C1COCCO1"
_LIGAND = "c1ccc(P(c2ccccc2)c2ccccc2)cc1"  # PPh3


def _make_jnj(tmp_path: Path) -> Path:
    """Write a minimal synthetic JnJ plate (distinct substrates, real conditions)."""
    aryls = [f"Brc1ccc(C{'C' * i})cc1" for i in range(6)]
    amines = [f"C1CCN(C{'C' * i})CC1" for i in range(6)]
    rows = [
        {
            schema.ARYL_SMILES: aryls[i],
            schema.AMINE_SMILES: amines[i],
            schema.CATALYST_SMILES: "CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1",
            schema.BASE_SMILES: "CC(C)(C)[O-].[Na+]",
            schema.SOLVENT_SMILES: "Cc1ccccc1",
            schema.PRODUCT_SMILES: build_cn_coupling_product(aryls[i], amines[i]) or "",
            schema.LABEL: i % 2,
            schema.YIELD: 50.0 * (i % 2),
        }
        for i in range(6)
    ]
    path = tmp_path / "jnj.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _make_ord(tmp_path: Path, jnj_aryl: str) -> Path:
    """Write a synthetic ORD-forward parquet with a Pd-BH row, a Cu row, and a leak row."""
    product = build_cn_coupling_product(_ARYL, _AMINE)
    assert product is not None
    leak_product = build_cn_coupling_product(jnj_aryl, _AMINE)
    rows = [
        {  # detectable Pd-BH coupling with the solvent in solvent_000
            "reactant_000": _ARYL,
            "reactant_001": _AMINE,
            "agent_000": "[Pd]",
            "agent_001": _LIGAND,
            "agent_002": "CC(C)(C)[O-].[Na+]",
            "solvent_000": _DIOXANE,
            "solvent_001": None,
            "solvent_002": None,
            "product_000": product,
            "product_001": None,
            "original_index": "rxn-pd",
            "temperature": 90.0,
            "yield_000": 75.0,
        },
        {  # same coupling but Cu-catalysed -> rejected by require_pd
            "reactant_000": _ARYL,
            "reactant_001": _AMINE,
            "agent_000": "[Cu]",
            "agent_001": None,
            "agent_002": None,
            "solvent_000": _DIOXANE,
            "solvent_001": None,
            "solvent_002": None,
            "product_000": product,
            "product_001": None,
            "original_index": "rxn-cu",
            "temperature": 90.0,
            "yield_000": 10.0,
        },
        {  # uses a JnJ aryl -> dropped by the substrate leakage guard
            "reactant_000": jnj_aryl,
            "reactant_001": _AMINE,
            "agent_000": "[Pd]",
            "agent_001": _LIGAND,
            "agent_002": None,
            "solvent_000": _DIOXANE,
            "solvent_001": None,
            "solvent_002": None,
            "product_000": leak_product,
            "product_001": None,
            "original_index": "rxn-leak",
            "temperature": 90.0,
            "yield_000": 42.0,
        },
    ]
    path = tmp_path / "ord.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_reader_pools_solvent_into_bag(tmp_path: Path) -> None:
    """The ORD reader must place ``solvent_000`` into the reaction's molecule bag."""
    jnj = _make_jnj(tmp_path)
    ord_path = _make_ord(tmp_path, jnj_aryl="Brc1ccc(C)cc1")
    config = ad.AssemblyConfig(ord_forward=ord_path, jnj_test=jnj)
    assembler = ad.DatasetAssembler(config)
    reactions = list(assembler._iter_ord())
    assert reactions, "reader yielded no reactions"
    first_reaction, temperature = reactions[0]
    assert _DIOXANE in first_reaction.molecules  # solvent reached the bag (the bug fix)
    assert temperature == 90.0


def test_positives_capture_conditions_and_apply_filters(tmp_path: Path) -> None:
    """P must keep the Pd-BH row with its solvent, reject Cu, and drop the JnJ leak."""
    jnj_aryl = "Brc1ccc(C)cc1"
    jnj = _make_jnj(tmp_path)
    ord_path = _make_ord(tmp_path, jnj_aryl=jnj_aryl)
    config = ad.AssemblyConfig(ord_forward=ord_path, jnj_test=jnj)
    assembler = ad.DatasetAssembler(config)

    positives = assembler.build_positives()
    report = assembler.positives_report

    assert report.n_detected == 2  # Pd row + leak row are both couplings...
    assert report.n_dropped_leakage == 1  # ...but the leak row is guarded out (strict)
    assert len(positives) == 1  # only the clean Pd-BH row survives

    row = positives.iloc[0]
    assert row[ad.METAL] == "Pd"
    assert row[ad.COND_SOLVENT_SMILES] is not None  # solvent captured, not dropped
    assert row["max_temperature"] == 90.0
    assert row["label"] == 1
    # every unified condition column is present on the positives frame
    assert set(ad.UNIFIED_CONDITION_COLUMNS).issubset(positives.columns)


def test_relaxed_positives_keep_shared_substrate_but_not_exact_pair(tmp_path: Path) -> None:
    """The relaxed P set keeps the JnJ-substrate-sharing row (its pair is not a JnJ pair)."""
    jnj_aryl = "Brc1ccc(C)cc1"  # a JnJ aryl, paired in ORD with morpholine (not a JnJ pair)
    jnj = _make_jnj(tmp_path)
    ord_path = _make_ord(tmp_path, jnj_aryl=jnj_aryl)
    config = ad.AssemblyConfig(ord_forward=ord_path, jnj_test=jnj)
    assembler = ad.DatasetAssembler(config)

    strict = assembler.build_positives(allow_shared_substrate=False)
    relaxed = assembler.build_positives(allow_shared_substrate=True)

    assert len(strict) == 1  # leak row dropped
    assert len(relaxed) == 2  # leak row kept (shares only a substrate, not the pair)
    assert assembler.positives_report_relaxed.n_dropped_leakage == 0
    assert assembler.positives_report_relaxed.n_shares_substrate == 1


def test_reject_cu_ni_additive(tmp_path: Path) -> None:
    """A Pd reaction that co-contains a Cu additive is thrown out (default reject_cu_ni)."""
    jnj = _make_jnj(tmp_path)
    ord_path = _make_ord(tmp_path, jnj_aryl="Brc1ccc(C)cc1")
    # Add a Cu additive ([Cu]I) to the clean Pd-BH row so its metals become {Cu, Pd}.
    frame = pd.read_parquet(ord_path)
    frame.loc[frame["original_index"] == "rxn-pd", "solvent_002"] = "[Cu]I"
    frame.to_parquet(ord_path, index=False)

    rejected = ad.DatasetAssembler(ad.AssemblyConfig(ord_forward=ord_path, jnj_test=jnj))
    kept = ad.DatasetAssembler(
        ad.AssemblyConfig(ord_forward=ord_path, jnj_test=jnj, reject_cu_ni=False)
    )
    assert rejected.build_positives().empty  # Pd+Cu thrown out
    assert rejected.positives_report.n_dropped_cu_ni == 1
    kept_positives = kept.build_positives()  # kept when the filter is disabled
    assert len(kept_positives) == 1
    assert kept_positives.iloc[0][ad.METAL] == "Cu|Pd"


def test_catalytic_metal_excludes_ferrocene_iron() -> None:
    """Fe from a ferrocene (dppf-style) ligand is not reported as a catalytic metal."""
    bag = ("[Pd]", "[Fe]")  # Pd catalyst + an Fe-bearing (ferrocene) ligand
    assert ad._catalytic_metals(bag) == frozenset({"Pd"})  # Fe not C–N-catalytic → excluded
    # the raw scan still sees both metals; only the catalytic filter drops Fe
    assert ad._metals_in_bag(bag) == frozenset({"Pd", "Fe"})


def test_condition_columns_restrict_metal_for_all_datasets() -> None:
    """The shared projection strips ferrocene Fe, so TEST/U agree with P on the metal."""
    ferrocene_pd = CanonicalCondition(
        metals=frozenset({"Fe", "Pd"}),  # what the canonicaliser emits for a dppf-Pd catalyst
        ligand_smiles=_LIGAND,
        base_smiles=None,
        solvent_smiles=None,
    )
    assert ad._condition_columns(ferrocene_pd)[ad.METAL] == "Pd"


def _make_pistachio(tmp_path: Path) -> Path:
    """Write a synthetic Pistachio ``rxn_smiles`` CSV with one detectable Pd-BH reaction."""
    product = build_cn_coupling_product(_ARYL, _AMINE)
    assert product is not None
    # reactants > agents (Pd + PPh3 + toluene solvent) > product
    rxn = f"{_ARYL}.{_AMINE}>[Pd].{_LIGAND}.Cc1ccccc1>{product}"
    path = tmp_path / "pistachio.csv"
    path.write_text("rxn_smiles\n" + rxn + "\n")
    return path


def test_pistachio_reader_captures_conditions(tmp_path: Path) -> None:
    """Pistachio's agents block (with solvent) flows into a condition-resolved positive."""
    jnj = _make_jnj(tmp_path)
    pistachio = _make_pistachio(tmp_path)
    config = ad.AssemblyConfig(
        ord_forward=tmp_path / "unused.parquet", pistachio=pistachio, jnj_test=jnj
    )
    assembler = ad.DatasetAssembler(config)
    positives = assembler.build_positives("pistachio")
    assert len(positives) == 1
    row = positives.iloc[0]
    assert row[ad.METAL] == "Pd"
    assert row[ad.COND_SOLVENT_SMILES] is not None  # solvent from the agents block
    assert row["sources"] == "Pistachio"


def test_merge_positives_dedups_and_unions_sources() -> None:
    """A (pair, condition) seen in two sources merges to one row with unioned sources."""

    def _row(aryl: str, amine: str, ligand_key: str, source: str, yld: float | None) -> dict:
        return {
            schema.ARYL_SMILES: aryl,
            schema.AMINE_SMILES: amine,
            "aryl_key": f"{aryl}K",
            "amine_key": f"{amine}K",
            schema.PRODUCT_SMILES: "P",
            "max_yield": yld,
            "max_temperature": None,
            "sources": source,
            "n_records": 1,
            "label": 1,
            ad.METAL: "Pd",
            ad.LIGAND_SMILES: f"{ligand_key}S",
            ad.LIGAND_KEY: ligand_key,
            ad.COND_BASE_SMILES: None,
            ad.COND_BASE_KEY: None,
            ad.COND_SOLVENT_SMILES: None,
            ad.COND_SOLVENT_KEY: None,
        }

    ord_frame = pd.DataFrame([_row("A", "B", "L1", "ORD", 50.0), _row("C", "D", "L2", "ORD", 10.0)])
    pist_frame = pd.DataFrame(
        [_row("A", "B", "L1", "Pistachio", 70.0), _row("E", "F", "L3", "Pistachio", None)]
    )
    merged = ad.DatasetAssembler.merge_positives([ord_frame, pist_frame])

    assert len(merged) == 3  # A/B/L1 collapses; C/D/L2 and E/F/L3 distinct
    shared = merged[merged[schema.ARYL_SMILES] == "A"].iloc[0]
    assert shared["sources"] == "ORD|Pistachio"  # union of sources
    assert shared["max_yield"] == 70.0  # max across sources
    assert shared["n_records"] == 2  # summed


def test_parallel_scan_matches_serial(tmp_path: Path) -> None:
    """A 2-worker scan yields the same positives as the serial scan on the synthetic ORD."""
    jnj = _make_jnj(tmp_path)
    ord_path = _make_ord(tmp_path, jnj_aryl="Brc1ccc(C)cc1")
    serial = ad.DatasetAssembler(ad.AssemblyConfig(ord_forward=ord_path, jnj_test=jnj))
    parallel = ad.DatasetAssembler(
        ad.AssemblyConfig(ord_forward=ord_path, jnj_test=jnj, jobs=2, chunk_size=1)
    )
    p_serial = serial.build_positives()
    p_parallel = parallel.build_positives()
    assert len(p_serial) == len(p_parallel) == 1
    assert serial.positives_report.n_detected == parallel.positives_report.n_detected


def test_phosphine_filter_rejects_ligandless_pd(tmp_path: Path) -> None:
    """With ``require_phosphine`` the ligand-free Pd row is dropped."""
    jnj = _make_jnj(tmp_path)
    ord_path = _make_ord(tmp_path, jnj_aryl="Brc1ccc(C)cc1")
    # Strip the ligand from the Pd row so only a bare [Pd] remains.
    frame = pd.read_parquet(ord_path)
    frame.loc[frame["original_index"] == "rxn-pd", "agent_001"] = None
    frame.to_parquet(ord_path, index=False)

    config = ad.AssemblyConfig(ord_forward=ord_path, jnj_test=jnj, require_phosphine=True)
    assembler = ad.DatasetAssembler(config)
    positives = assembler.build_positives()
    assert positives.empty
    assert assembler.positives_report.n_dropped_no_phosphine == 1


def test_unified_condition_columns_align_across_datasets() -> None:
    """``_condition_columns`` yields exactly the unified schema for both condition sources."""
    from_columns = ad._condition_columns(
        condition_from_columns("[Pd]", "CC(C)(C)[O-].[Na+]", "Cc1ccccc1")
    )
    from_bag = ad._condition_columns(
        CanonicalCondition(
            metals=frozenset({"Pd"}),
            ligand_smiles=_LIGAND,
            base_smiles="CC(C)(C)[O-]",
            solvent_smiles=_DIOXANE,
        )
    )
    assert set(from_columns) == set(ad.UNIFIED_CONDITION_COLUMNS)
    assert set(from_bag) == set(ad.UNIFIED_CONDITION_COLUMNS)


def test_pure_helpers() -> None:
    """The small structural helpers behave as documented."""
    assert ad._is_smiles("CCO")
    assert not ad._is_smiles("nan")
    assert not ad._is_smiles(None)
    assert ad._metals_in_bag(("[Pd]", "CCO")) == frozenset({"Pd"})
    assert ad._metals_in_bag(("CCO",)) == frozenset()
    assert ad._has_phosphine(_LIGAND)
    assert not ad._has_phosphine("CCN(CC)CC")
    assert not ad._has_phosphine(None)
