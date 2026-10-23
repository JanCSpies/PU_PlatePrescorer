"""Tests for stage F, substrate features (docs/design/modeling.md, "Stage F - features").

Every SMILES is hand-written and its numbers were worked out by hand on 2026-10-21 (R = reacting
atom, G = all atoms matched by the role SMARTS, A = heavy neighbours of R outside G; distances are
counted from R, atoms of G are never counted). The tautomer pair has one InChIKey, checked with RDKit.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from platepu.classify.engine import load_class
from platepu.features.substrate import (
    DESCRIPTORS,
    load_feature_config,
    substrate_features,
    substrate_table,
)

ROOT = Path(__file__).parents[1]
RULE = load_class(ROOT / "configs" / "classes" / "amide_coupling.yml")
CONFIG = load_feature_config(ROOT / "configs" / "features.yml")

FORMIC_ACID = "O=CO"
ACETIC_ACID = "CC(=O)O"
PIVALIC_ACID = "CC(C)(C)C(=O)O"
BENZOIC_ACID = "OC(=O)c1ccccc1"
BOC_PROLINE = "CC(C)(C)OC(=O)N1CCC[C@H]1C(=O)O"
TERT_BUTYLAMINE = "CC(C)(C)N"
MORPHOLINE = "C1COCCN1"
ANILINE = "Nc1ccccc1"
SUCCINIC_ACID = "OC(=O)CCC(=O)O"
ETHANOL = "CCO"
L_ALANINE = "C[C@H](N)C(=O)O"
D_ALANINE = "C[C@@H](N)C(=O)O"
# 2-hydroxypyridine-4-carboxylic acid and its pyridone tautomer: one InChIKey, two SMILES.
HYDROXYPYRIDINE_ACID = "O=C(O)c1ccnc(O)c1"
PYRIDONE_ACID = "O=C(O)c1cc[nH]c(=O)c1"
TAUTOMER_KEY = "BXHCJLRTXPHUGH-UHFFFAOYSA-N"


def descriptors(smiles: str, role: str) -> dict[str, object]:
    """The ten descriptors of one substrate, without Morgan and subclass."""
    features = substrate_features(smiles, role, RULE, CONFIG)
    return {name: features[name] for name in DESCRIPTORS}


def test_feature_config_is_read_from_the_yaml() -> None:
    """configs/features.yml: radius 2, 2,048 bits (as PAYN)."""
    assert (CONFIG.morgan_radius, CONFIG.morgan_bits) == (2, 2048)


def test_feature_config_reads_the_centre_block() -> None:
    """configs/features.yml, morgan_centre: radius 2, 2,048 bits."""
    assert (CONFIG.centre_radius, CONFIG.centre_bits) == (2, 2048)


def test_five_electron_withdrawing_patterns_are_parsed_mols() -> None:
    """The config has five patterns, each a compiled RDKit Mol. What each one matches is tested
    below, one chemistry test per group (nitro, nitrile, trihalomethyl, sulfonyl, carbonyl)."""
    assert len(CONFIG.electron_withdrawing) == 5
    assert all(isinstance(pattern, Chem.Mol) for pattern in CONFIG.electron_withdrawing)


def test_unparsable_electron_withdrawing_smarts_stops_loading(tmp_path: Path) -> None:
    """A typo in an EWG SMARTS is a ValueError at load, as for the class SMARTS."""
    path = tmp_path / "features.yml"
    path.write_text(
        "morgan: {radius: 2, n_bits: 2048}\n"
        "morgan_centre: {radius: 2, n_bits: 2048}\n"
        "electron_withdrawing:\n"
        '  - {name: nitrile, smarts: "[CX2#N"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_feature_config(path)


def test_formic_acid_has_no_alpha_atom() -> None:
    """O=CO: 3 heavy atoms, R = C, G = all three, so A is empty and every shell count is 0."""
    assert descriptors(FORMIC_ACID, "acid") == dict.fromkeys(DESCRIPTORS, 0) | {
        "n_heavy_atoms": 3
    }


def test_acetic_acid_has_one_methyl_alpha() -> None:
    """CC(=O)O: 4 heavy atoms; A = {CH3} (heavy degree 1, bond to R only); nothing beyond it."""
    assert descriptors(ACETIC_ACID, "acid") == {
        "n_heavy_atoms": 4,
        "centre_in_ring": 0,
        "n_alpha": 1,
        "alpha_n_aromatic": 0,
        "alpha_n_in_ring": 0,
        "alpha_max_heavy_degree": 1,
        "alpha_sum_heavy_degree": 1,
        "n_heavy_at_2_bonds": 0,
        "n_heavy_at_3_bonds": 0,
        "n_ewg_on_ring": 0,
    }


def test_pivalic_acid_alpha_is_quaternary() -> None:
    """CC(C)(C)C(=O)O: 7 heavy atoms; A = {C} with degree 4; 3 methyls at 2 bonds, none at 3."""
    assert descriptors(PIVALIC_ACID, "acid") == {
        "n_heavy_atoms": 7,
        "centre_in_ring": 0,
        "n_alpha": 1,
        "alpha_n_aromatic": 0,
        "alpha_n_in_ring": 0,
        "alpha_max_heavy_degree": 4,
        "alpha_sum_heavy_degree": 4,
        "n_heavy_at_2_bonds": 3,
        "n_heavy_at_3_bonds": 0,
        "n_ewg_on_ring": 0,
    }


def test_benzoic_acid_alpha_is_aromatic_ring_atom() -> None:
    """OC(=O)c1ccccc1: 9 heavy atoms; A = {ipso c}: aromatic, in ring, degree 3 (R and 2 ring
    neighbours); 2 ortho carbons at 2 bonds, 2 meta carbons at 3 bonds (para is at 4)."""
    assert descriptors(BENZOIC_ACID, "acid") == {
        "n_heavy_atoms": 9,
        "centre_in_ring": 0,
        "n_alpha": 1,
        "alpha_n_aromatic": 1,
        "alpha_n_in_ring": 1,
        "alpha_max_heavy_degree": 3,
        "alpha_sum_heavy_degree": 3,
        "n_heavy_at_2_bonds": 2,
        "n_heavy_at_3_bonds": 2,
        "n_ewg_on_ring": 0,
    }


def test_carboxyl_oxygens_are_not_alpha_or_shell_atoms() -> None:
    """In acetic acid the two oxygens are neighbours of R but belong to G: not alpha, not at 2
    bonds, and the only heavy atom counted at distance 1 is the methyl. Swapping CH3 for H (formic
    acid) must therefore give n_alpha 0, not 2."""
    assert descriptors(ACETIC_ACID, "acid")["n_alpha"] == 1
    assert descriptors(FORMIC_ACID, "acid")["n_alpha"] == 0


def test_n_boc_proline_alpha_in_ring() -> None:
    """CC(C)(C)OC(=O)N1CCC[C@H]1C(=O)O: 15 heavy atoms (tBu 4, O, C=O 2, N, 3 CH2, CH, COOH 3).
    The Boc O-tBu is not an acid (no H on O), so one acid group. R = COOH carbon, not in a ring.
    A = {C-alpha}: in ring, not aromatic, degree 3 (R, N, ring CH2). At 2 bonds: N and the ring CH2
    next to C-alpha = 2. At 3 bonds: Boc carbonyl C, the CH2 next to N, the middle CH2 = 3."""
    assert descriptors(BOC_PROLINE, "acid") == {
        "n_heavy_atoms": 15,
        "centre_in_ring": 0,
        "n_alpha": 1,
        "alpha_n_aromatic": 0,
        "alpha_n_in_ring": 1,
        "alpha_max_heavy_degree": 3,
        "alpha_sum_heavy_degree": 3,
        "n_heavy_at_2_bonds": 2,
        "n_heavy_at_3_bonds": 3,
        "n_ewg_on_ring": 0,
    }


def test_tert_butylamine_alpha_is_quaternary() -> None:
    """CC(C)(C)N: 5 heavy atoms; R = N (G = {N}), not in ring; A = {C} with degree 4;
    3 methyls at 2 bonds, none at 3."""
    assert descriptors(TERT_BUTYLAMINE, "amine") == {
        "n_heavy_atoms": 5,
        "centre_in_ring": 0,
        "n_alpha": 1,
        "alpha_n_aromatic": 0,
        "alpha_n_in_ring": 0,
        "alpha_max_heavy_degree": 4,
        "alpha_sum_heavy_degree": 4,
        "n_heavy_at_2_bonds": 3,
        "n_heavy_at_3_bonds": 0,
        "n_ewg_on_ring": 0,
    }


def test_morpholine_n_in_ring_oxygen_at_three_bonds() -> None:
    """C1COCCN1: 6 heavy atoms; R = N is in the ring; A = the two CH2 next to N (in ring, degree 2
    each, sum 4); the two CH2 next to O are at 2 bonds; O is at 3 bonds, reached by both paths but
    counted once."""
    assert descriptors(MORPHOLINE, "amine") == {
        "n_heavy_atoms": 6,
        "centre_in_ring": 1,
        "n_alpha": 2,
        "alpha_n_aromatic": 0,
        "alpha_n_in_ring": 2,
        "alpha_max_heavy_degree": 2,
        "alpha_sum_heavy_degree": 4,
        "n_heavy_at_2_bonds": 2,
        "n_heavy_at_3_bonds": 1,
        "n_ewg_on_ring": 0,
    }


def test_aniline_alpha_is_aromatic() -> None:
    """Nc1ccccc1: 7 heavy atoms; R = N, not in ring; A = {ipso c}: aromatic, in ring, degree 3;
    2 ortho at 2 bonds, 2 meta at 3 bonds."""
    assert descriptors(ANILINE, "amine") == {
        "n_heavy_atoms": 7,
        "centre_in_ring": 0,
        "n_alpha": 1,
        "alpha_n_aromatic": 1,
        "alpha_n_in_ring": 1,
        "alpha_max_heavy_degree": 3,
        "alpha_sum_heavy_degree": 3,
        "n_heavy_at_2_bonds": 2,
        "n_heavy_at_3_bonds": 2,
        "n_ewg_on_ring": 0,
    }


def test_subclass_is_part_of_the_features() -> None:
    """Aniline gets the subclass from the class rule; the acid role has none since 2026-10-22."""
    assert substrate_features(ANILINE, "amine", RULE, CONFIG)["subclass"] == "aniline"
    assert substrate_features(ACETIC_ACID, "acid", RULE, CONFIG)["subclass"] is None


# n_ewg_on_ring: distinct attachment atoms of electron-withdrawing groups on the aromatic ring
# system (fused rings, not biaryl) of the neighbour of R. Worked out by hand and checked with RDKit
# on 2026-10-21. The reacting carboxyl C of an acid is R itself, so it is never counted.
@pytest.mark.parametrize(
    ("smiles", "role", "expected"),
    [
        pytest.param(ANILINE, "amine", 0, id="aniline_has_no_ewg"),
        pytest.param("Nc1ccc([N+](=O)[O-])cc1", "amine", 1, id="4_nitroaniline"),
        pytest.param(
            "Nc1cc(C(F)(F)F)cc(C(F)(F)F)c1",
            "amine",
            2,
            id="3_5_bis_trifluoromethyl_aniline",
        ),
        pytest.param(
            "NCc1ccccc1", "amine", 0, id="benzylamine_n_has_no_aromatic_neighbour"
        ),
        pytest.param(
            "Nc1ccc2cc(C#N)ccc2c1",
            "amine",
            1,
            id="6_aminonaphthalene_2_carbonitrile_one_system",
        ),
        pytest.param(
            "Nc1ccc(-c2ccc([N+](=O)[O-])cc2)cc1",
            "amine",
            0,
            id="4_prime_nitrobiphenyl_4_amine_nitro_on_other_ring",
        ),
        pytest.param(
            "Nc1ccc2[nH]c(=O)ccc2c1",
            "amine",
            0,
            id="6_aminoquinolin_2_one_carbonyl_in_ring",
        ),
        pytest.param(
            "OC(=O)c1ccc([N+](=O)[O-])cc1", "acid", 1, id="4_nitrobenzoic_acid"
        ),
    ],
)
def test_n_ewg_on_ring(smiles: str, role: str, expected: int) -> None:
    """Expected counts per case id; each is the number of distinct EWG attachment atoms on the ring."""
    assert descriptors(smiles, role)["n_ewg_on_ring"] == expected


def test_4_aminobenzamide_counts_attachment_atoms_not_matches() -> None:
    """Nc1ccc(C(N)=O)cc1: the amide matches `[CX3](=O)[#6,#7,#8]` twice (through the ring C and
    through N) but has one attachment atom, the carbonyl C, so the value is 1. Counting matches
    would give 2 (checked with RDKit)."""
    assert descriptors("Nc1ccc(C(N)=O)cc1", "amine")["n_ewg_on_ring"] == 1


# Diarylamine: R = N has two aromatic neighbours, so both ring systems carry substituents of R.
# Checked with RDKit on 2026-10-21: each SMILES below has exactly one amine reacting atom.
# Note: "COc1ccc(Nc2ccc([N+](=O)[O-])cc2)cc1" (4-methoxy-4'-nitrodiphenylamine) and
# "O=[N+]([O-])c1ccc(Nc2ccccc2)cc1" (4-nitrodiphenylamine) are NOT the same molecule.
METHOXY_NITRO_DIARYLAMINE = "COc1ccc(Nc2ccc([N+](=O)[O-])cc2)cc1"
METHOXY_NITRO_DIARYLAMINE_REORDERED = "O=[N+]([O-])c1ccc(Nc2ccc(OC)cc2)cc1"
NITRO_CYANO_DIARYLAMINE = "N#Cc1ccc(Nc2ccc([N+](=O)[O-])cc2)cc1"
METHYLSULFONYL_ANILINE = "Nc1ccc(S(C)(=O)=O)cc1"
AMINOTHIOPHENE_NITRILE = "Nc1sccc1C#N"


def test_diarylamine_counts_the_ewg_on_either_ring() -> None:
    """4-methoxy-4'-nitrodiphenylamine written with the nitro ring first or second is one molecule
    (same canonical SMILES, checked with RDKit; one amine match each) and has one nitro, so
    n_ewg_on_ring is 1 for both spellings, whichever ring comes first in the atom order."""
    first = Chem.MolToSmiles(Chem.MolFromSmiles(METHOXY_NITRO_DIARYLAMINE))
    second = Chem.MolToSmiles(Chem.MolFromSmiles(METHOXY_NITRO_DIARYLAMINE_REORDERED))
    assert first == second
    assert descriptors(METHOXY_NITRO_DIARYLAMINE, "amine")["n_ewg_on_ring"] == 1
    assert (
        descriptors(METHOXY_NITRO_DIARYLAMINE_REORDERED, "amine")["n_ewg_on_ring"] == 1
    )


def test_diarylamine_counts_both_rings() -> None:
    """4-cyano-4'-nitrodiphenylamine (one amine match, checked with RDKit): the nitrile sits on
    one ring and the nitro on the other, both rings belong to R, so n_ewg_on_ring is 2."""
    assert descriptors(NITRO_CYANO_DIARYLAMINE, "amine")["n_ewg_on_ring"] == 2


def test_methylsulfonyl_aniline_counts_the_sulfonyl() -> None:
    """4-(methylsulfonyl)aniline (one amine match): the S of S(=O)(=O) is the one attachment atom
    on the ring, and no other pattern matches, so n_ewg_on_ring is 1 (sulfonyl line only)."""
    assert descriptors(METHYLSULFONYL_ANILINE, "amine")["n_ewg_on_ring"] == 1


def test_aminothiophene_carbonitrile_counts_the_nitrile_on_the_heteroaromatic_ring() -> (
    None
):
    """2-aminothiophene-3-carbonitrile (one amine match): R = N is on the thiophene C2, the nitrile
    C is on C3 of the same ring, so n_ewg_on_ring is 1."""
    assert descriptors(AMINOTHIOPHENE_NITRILE, "amine")["n_ewg_on_ring"] == 1


def test_diacid_fails_loudly() -> None:
    """Succinic acid has two acid groups, so there is no single reacting atom."""
    with pytest.raises(ValueError):
        substrate_features(SUCCINIC_ACID, "acid", RULE, CONFIG)


def test_no_reacting_atom_fails_loudly() -> None:
    """Ethanol has no carboxyl group."""
    with pytest.raises(ValueError):
        substrate_features(ETHANOL, "acid", RULE, CONFIG)


def test_unparsable_smiles_fails_loudly() -> None:
    """A SMILES RDKit cannot read stops the run."""
    with pytest.raises(ValueError):
        substrate_features("C(=O)(O)(", "acid", RULE, CONFIG)


def test_morgan_counts_equal_rdkit_count_fingerprint() -> None:
    """The Morgan column is RDKit's count fingerprint (radius 2, 2,048 bits) as uint8."""
    morgan = substrate_features(BOC_PROLINE, "acid", RULE, CONFIG)["morgan"]
    expected = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048
    ).GetCountFingerprintAsNumPy(Chem.MolFromSmiles(BOC_PROLINE))
    assert isinstance(morgan, np.ndarray)
    assert morgan.dtype == np.uint8
    assert morgan.shape == (2048,)
    assert morgan.tolist() == expected.tolist()


# --- morgan_centre: the Morgan environments of the reacting atom R alone (fromAtoms=[R]) -------------
# RDKit sets one bit per radius 0..r for the single root atom. Every expectation below was checked
# with RDKit on 2026-10-22 (radius 2, 2,048 bits).
BENZYLAMINE = "NCc1ccccc1"
AMINOPYRIDINE = "Nc1ccccn1"
CHLOROANILINE = "Nc1ccc(Cl)cc1"
METHYLANILINE = "Nc1ccc(C)cc1"
BUTANOIC_ACID = "CCCC(=O)O"
HEXANOIC_ACID = "CCCCCC(=O)O"


def centre_bits(smiles: str, role: str, config: object = CONFIG) -> set[int]:
    """The positions set in morgan_centre."""
    centre = substrate_features(smiles, role, RULE, config)["morgan_centre"]  # type: ignore[arg-type]
    return set(np.flatnonzero(centre).tolist())


def test_centre_fingerprint_sets_one_bit_per_radius() -> None:
    """Rooted at one atom with radius 2: one environment per radius 0, 1, 2, so at most 3 bits.
    Aniline sets exactly 3 (N alone; N with its ipso c; N out to the ortho c's), each count 1."""
    features = substrate_features(ANILINE, "amine", RULE, CONFIG)
    centre = features["morgan_centre"]
    assert isinstance(centre, np.ndarray)
    assert centre.dtype == np.uint8
    assert centre.shape == (2048,)
    assert np.count_nonzero(centre) == 3
    assert int(centre.sum()) == 3


def test_centre_fingerprint_separates_aniline_from_benzylamine() -> None:
    """N-c(aromatic) against N-C(aliphatic): the radius-1 environment of N already differs, which is
    what the aniline / aliphatic_primary subclasses encode by hand. Radius 0 (bare NH2) is shared."""
    radius_1 = replace(CONFIG, centre_radius=1)
    aniline = centre_bits(ANILINE, "amine", radius_1)
    benzylamine = centre_bits(BENZYLAMINE, "amine", radius_1)
    assert aniline != benzylamine
    assert len(aniline & benzylamine) == 1


def test_centre_fingerprint_separates_2_aminopyridine_from_aniline() -> None:
    """Both N sit on an aromatic c, so radius 1 agrees (the ring n is two bonds from N); at radius 2
    the ortho atom is n against c and the vectors differ. That is the level the heteroaryl_amine
    rule looks at: 3 bits each, 2 shared."""
    radius_1 = replace(CONFIG, centre_radius=1)
    assert centre_bits(ANILINE, "amine", radius_1) == centre_bits(
        AMINOPYRIDINE, "amine", radius_1
    )
    aniline = centre_bits(ANILINE, "amine")
    aminopyridine = centre_bits(AMINOPYRIDINE, "amine")
    assert aniline != aminopyridine
    assert len(aniline & aminopyridine) == 2


def test_centre_fingerprint_ignores_a_remote_substituent() -> None:
    """4-chloro- and 4-methylaniline: the para substituent is four bonds from N and radius 2 never
    reaches it, so the rooted fingerprints are equal. This is the price of the rooted view: the
    full `morgan` still tells them apart."""
    assert centre_bits(CHLOROANILINE, "amine") == centre_bits(METHYLANILINE, "amine")
    chloro = substrate_features(CHLOROANILINE, "amine", RULE, CONFIG)["morgan"]
    methyl = substrate_features(METHYLANILINE, "amine", RULE, CONFIG)["morgan"]
    assert chloro.tolist() != methyl.tolist()  # type: ignore[attr-defined]


def rooted_and_whole_bits(smiles: str, role: str) -> tuple[set[int], set[int]]:
    """Bits of the rooted fingerprint and of the whole-molecule one (default generator)."""
    features = substrate_features(smiles, role, RULE, CONFIG)
    rooted = set(np.flatnonzero(features["morgan_centre"]).tolist())  # type: ignore[arg-type]
    whole = set(np.flatnonzero(features["morgan"]).tolist())  # type: ignore[arg-type]
    return rooted, whole


def test_centre_fingerprint_adds_an_environment_the_full_fingerprint_drops() -> None:
    """Aniline: the radius-2 environment of the terminal N covers bonds N-c, c-c, c-c, all inside
    the radius-2 environment of the ipso carbon, so RDKit drops it from the whole-molecule
    fingerprint (default includeRedundantEnvironments=False). Rooted at N nothing contains it, so it
    keeps its own bit. For a terminal reacting atom the column carries something that is missing."""
    rooted, whole = rooted_and_whole_bits(ANILINE, "amine")
    assert len(rooted) == 3
    assert len(rooted - whole) == 1


def test_centre_fingerprint_is_a_subset_once_redundant_environments_are_kept() -> None:
    """With includeRedundantEnvironments=True the whole-molecule fingerprint keeps the dropped
    environment, so every rooted bit is also a whole-molecule bit (aniline). Boc-proline: R is the
    carboxyl carbon, not terminal, nothing swallows its environments, and the same holds."""
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeRedundantEnvironments=True
    )
    for smiles, role in ((ANILINE, "amine"), (BOC_PROLINE, "acid")):
        rooted, _ = rooted_and_whole_bits(smiles, role)
        whole = set(
            np.flatnonzero(
                generator.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(smiles))
            ).tolist()
        )
        assert rooted <= whole


def test_centre_fingerprint_of_a_non_terminal_centre_is_a_subset_by_default() -> None:
    """Boc-proline: the carboxyl carbon has three neighbours, so no environment of it is contained
    in a neighbour's and the subset already holds with the default generator."""
    rooted, whole = rooted_and_whole_bits(BOC_PROLINE, "acid")
    assert rooted <= whole


def test_centre_fingerprint_of_an_acid_is_rooted_at_the_carboxyl_carbon() -> None:
    """Butanoic and hexanoic acid: seen from the carboxyl C out to radius 2 both show C(=O)O, a CH2
    and a second CH2 (degree 2); the chain length beyond is invisible, so the rooted fingerprints
    are equal (3 bits) while the whole-molecule fingerprints differ."""
    assert centre_bits(BUTANOIC_ACID, "acid") == centre_bits(HEXANOIC_ACID, "acid")
    assert len(centre_bits(BUTANOIC_ACID, "acid")) == 3
    butanoic = substrate_features(BUTANOIC_ACID, "acid", RULE, CONFIG)["morgan"]
    hexanoic = substrate_features(HEXANOIC_ACID, "acid", RULE, CONFIG)["morgan"]
    assert butanoic.tolist() != hexanoic.tolist()  # type: ignore[attr-defined]


def test_enantiomers_get_equal_features() -> None:
    """L- and D-alanine: no chirality in the model features (stereo lives in key_stereo)."""
    l_features = substrate_features(L_ALANINE, "acid", RULE, CONFIG)
    d_features = substrate_features(D_ALANINE, "acid", RULE, CONFIG)
    assert l_features["morgan"].tolist() == d_features["morgan"].tolist()  # type: ignore[attr-defined]
    for name in (*DESCRIPTORS, "subclass"):
        assert l_features[name] == d_features[name]


def test_long_chain_overflowing_uint8_fails_loudly() -> None:
    """A C300 chain acid: one Morgan bit is counted 299 times (checked with RDKit), above 255."""
    with pytest.raises(ValueError):
        substrate_features("C" * 300 + "C(=O)O", "acid", RULE, CONFIG)


# --- the table: one row per key_stereo ---------------------------------------------------------

STD = "std1"


def positives_row(
    key_a: str, smiles_a: str, key_b: str, smiles_b: str, licence: str = "open"
) -> dict[str, str]:
    """One positive pair; the flat key is the stereo key in lower case (made-up, enough here)."""
    return {
        "key_a_stereo": key_a,
        "key_a_flat": key_a.lower(),
        "substrate_a": smiles_a,
        "key_b_stereo": key_b,
        "key_b_flat": key_b.lower(),
        "substrate_b": smiles_b,
        "licence": licence,
        "std_version": STD,
    }


def library_row(
    key: str, smiles: str, role: str, std_version: str = STD
) -> dict[str, str]:
    """One library substrate (licence internal)."""
    return {
        "key_stereo": key,
        "key_flat": key.lower(),
        "smiles": smiles,
        "role": role,
        "licence": "internal",
        "std_version": std_version,
    }


POSITIVES_COLUMNS = list(positives_row("A", "x", "B", "y"))
LIBRARY_COLUMNS = list(library_row("A", "x", "acid"))
VALIDATION_COLUMNS = (
    POSITIVES_COLUMNS  # ADR 009: the validation table carries its own licence
)


def frames(
    positives: list[dict[str, str]],
    library: list[dict[str, str]],
    validation: list[dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The three input tables, with their columns even when a list is empty."""
    return (
        pd.DataFrame(positives, columns=POSITIVES_COLUMNS),
        pd.DataFrame(library, columns=LIBRARY_COLUMNS),
        pd.DataFrame(validation, columns=VALIDATION_COLUMNS),
    )


def build(
    positives: list[dict[str, str]],
    library: list[dict[str, str]],
    validation: list[dict[str, str]],
) -> pd.DataFrame:
    """Run the table builder on small hand-made inputs."""
    return substrate_table(*frames(positives, library, validation), RULE, CONFIG)


def test_same_key_in_positives_and_library_gives_one_internal_row() -> None:
    """Acetic acid (key A1) is a positive substrate and a library acid: one row, licence internal."""
    table = build(
        [positives_row("A1", ACETIC_ACID, "N1", "NCc1ccccc1")],
        [library_row("A1", ACETIC_ACID, "acid")],
        [],
    )
    row = table[table["key_stereo"] == "A1"]
    assert len(row) == 1
    assert row.iloc[0]["licence"] == "internal"
    assert bool(row.iloc[0]["in_positives"]) and bool(row.iloc[0]["in_library"])
    assert not bool(row.iloc[0]["in_validation"])
    assert row.iloc[0]["role"] == "acid"
    assert row.iloc[0]["n_heavy_atoms"] == 4


def test_licensed_positive_without_library_is_licensed() -> None:
    """A licensed positive row that is not in the library keeps its licence on the substrate."""
    table = build(
        [positives_row("A1", ACETIC_ACID, "N1", "NCc1ccccc1", licence="licensed")],
        [],
        [],
    )
    assert set(table["licence"]) == {"licensed"}


def test_unknown_licence_fails_loudly() -> None:
    """A licence outside internal / licensed / open must stop the run, not be guessed."""
    with pytest.raises(ValueError, match="unknown licence"):
        build(
            [positives_row("A1", ACETIC_ACID, "N1", "NCc1ccccc1", licence="cc_by")],
            [],
            [],
        )


def test_tautomer_key_uses_smallest_smiles() -> None:
    """Two SMILES under the key BXHCJLRTXPHUGH-UHFFFAOYSA-N: the row has the smaller string
    (the [nH] pyridone, since 'c' sorts before 'n'), n_smiles 2, and its features."""
    table = build(
        [
            positives_row(TAUTOMER_KEY, HYDROXYPYRIDINE_ACID, "N1", "NCc1ccccc1"),
            positives_row(TAUTOMER_KEY, PYRIDONE_ACID, "N2", "NCC"),
        ],
        [],
        [],
    )
    row = table[table["key_stereo"] == TAUTOMER_KEY].iloc[0]
    assert min(HYDROXYPYRIDINE_ACID, PYRIDONE_ACID) == PYRIDONE_ACID
    assert row["smiles"] == PYRIDONE_ACID
    assert row["n_smiles"] == 2
    expected = substrate_features(PYRIDONE_ACID, "acid", RULE, CONFIG)
    assert row["morgan"].tolist() == expected["morgan"].tolist()  # type: ignore[attr-defined]
    assert row["n_heavy_atoms"] == expected["n_heavy_atoms"]


def test_key_in_both_roles_fails_loudly() -> None:
    """The same key as an acid and as an amine (glycine-like mix-up) stops the run."""
    with pytest.raises(ValueError):
        build([positives_row("K1", ACETIC_ACID, "K1", "NCc1ccccc1")], [], [])


def test_library_role_outside_the_class_fails_loudly() -> None:
    """A library row with role 'alcohol' is not acid or amine."""
    with pytest.raises(ValueError):
        build(
            [positives_row("A1", ACETIC_ACID, "N1", "NCc1ccccc1")],
            [library_row("X1", ETHANOL, "alcohol")],
            [],
        )


def test_mixed_std_version_fails_loudly() -> None:
    """One library row from another standardisation version stops the run."""
    with pytest.raises(ValueError):
        build(
            [positives_row("A1", ACETIC_ACID, "N1", "NCc1ccccc1")],
            [library_row("A2", BENZOIC_ACID, "acid", std_version="std0")],
            [],
        )


def validation_row(licence: str = "noncommercial") -> dict[str, str]:
    """A validation pair as the benchmark writes it: Chemlex is noncommercial (ADR 009)."""
    return positives_row("A9", PIVALIC_ACID, "N9", ANILINE, licence=licence)


def test_validation_only_key_carries_the_licence_of_the_validation_table() -> None:
    """A key seen only in the validation set takes the validation table's own licence
    (noncommercial for Chemlex), read from the column, not stamped open (ADR 009 item 5)."""
    table = build(
        [positives_row("A1", ACETIC_ACID, "N1", "NCc1ccccc1")], [], [validation_row()]
    )
    row = table[table["key_stereo"] == "A9"].iloc[0]
    assert row["licence"] == "noncommercial"
    assert bool(row["in_validation"])
    assert not bool(row["in_positives"]) and not bool(row["in_library"])


def test_validation_licence_is_read_not_stamped() -> None:
    """The same row labelled "open" in the table stays open: the column decides, not the code."""
    table = build(
        [positives_row("A1", ACETIC_ACID, "N1", "NCc1ccccc1")],
        [],
        [validation_row(licence="open")],
    )
    assert table.loc[table["key_stereo"] == "A9", "licence"].iloc[0] == "open"


def test_key_in_library_and_validation_is_internal() -> None:
    """Strictest wins: an inventory acid (internal) that Chemlex also measured (noncommercial)
    comes out internal."""
    table = build(
        [positives_row("A1", ACETIC_ACID, "N1", "NCc1ccccc1")],
        [library_row("A9", PIVALIC_ACID, "acid")],
        [validation_row()],
    )
    assert table.loc[table["key_stereo"] == "A9", "licence"].iloc[0] == "internal"


def test_key_in_positives_and_validation_is_noncommercial() -> None:
    """An open positive substrate that Chemlex also measured must not stay open: the row now
    carries a Chemlex measurement."""
    table = build(
        [positives_row("A9", PIVALIC_ACID, "N1", "NCc1ccccc1")], [], [validation_row()]
    )
    assert table.loc[table["key_stereo"] == "A9", "licence"].iloc[0] == "noncommercial"


def test_validation_without_a_licence_column_fails_loudly() -> None:
    """The old Chemlex table had no licence column and the code stamped it open. That is gone:
    a validation table without the column is an error, not a guess."""
    validation = pd.DataFrame([validation_row()]).drop(columns=["licence"])
    positives, library, _ = frames(
        [positives_row("A1", ACETIC_ACID, "N1", "NCc1ccccc1")], [], []
    )
    with pytest.raises((KeyError, ValueError), match="licence"):
        substrate_table(positives, library, validation, RULE, CONFIG)


def plain(table: pd.DataFrame) -> pd.DataFrame:
    """Morgan arrays as lists so that two tables can be compared with pandas."""
    return table.assign(
        morgan=table["morgan"].map(lambda counts: counts.tolist()),
        morgan_centre=table["morgan_centre"].map(lambda counts: counts.tolist()),
    )


def test_table_is_sorted_and_repeatable() -> None:
    """Rows are sorted by (role, key_stereo), and reversed input rows give the same table."""
    positives = [
        positives_row("A2", BENZOIC_ACID, "N2", ANILINE),
        positives_row("A1", ACETIC_ACID, "N1", "NCc1ccccc1"),
    ]
    first = build(positives, [library_row("N3", MORPHOLINE, "amine")], [])
    second = build(positives[::-1], [library_row("N3", MORPHOLINE, "amine")], [])
    assert list(zip(first["role"], first["key_stereo"], strict=True)) == [
        ("acid", "A1"),
        ("acid", "A2"),
        ("amine", "N1"),
        ("amine", "N2"),
        ("amine", "N3"),
    ]
    pd.testing.assert_frame_equal(plain(first), plain(second))


def test_table_columns() -> None:
    """The columns of the design table."""
    table = build([positives_row("A1", ACETIC_ACID, "N1", "NCc1ccccc1")], [], [])
    assert set(table.columns) == {
        "key_stereo",
        "key_flat",
        "role",
        "smiles",
        "n_smiles",
        "std_version",
        "licence",
        "in_positives",
        "in_library",
        "in_validation",
        "subclass",
        "morgan",
        "morgan_centre",
        *DESCRIPTORS,
    }
