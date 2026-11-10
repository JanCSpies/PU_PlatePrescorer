"""Hard cases for molecule standardisation and condition keys (docs/design/rebuild.md, Stage 2).

Every case comes from the data (counted 2026-10-09): HATU drawn two ways (Chemlex), DIPEA filed as a
solvent (ORDerly) or a reagent (Chemlex), carboxylates and ammonium salts as substrates (ORDerly),
PyBrOP's PF6 written as a cation (Chemlex), stereocentres in substrates (both).
"""

from pathlib import Path

import pytest

from platepu.standardise.molecule import (
    StdConfig,
    condition_from_text,
    condition_key,
    condition_text,
    in_family,
    load_std_config,
    standardize,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "standardise.yml"

HATU_O_FORM = "CN(C)C(On1nnc2cccnc21)=[N+](C)C"
HATU_N_OXIDE = "CN(C)C(n1n[n+]([O-])c2ncccc21)=[N+](C)C"
HBTU_TBTU_CATION = "CN(C)C(On1nnc2ccccc21)=[N+](C)C"
PF6 = "F[P-](F)(F)(F)(F)F"
BF4 = "F[B-](F)(F)F"
DIPEA = "CCN(C(C)C)C(C)C"
DMF = "CN(C)C=O"


@pytest.fixture(scope="module")
def config() -> StdConfig:
    """The repository's standardisation config."""
    return load_std_config(CONFIG_PATH)


def test_both_hatu_drawings_give_one_agent(config: StdConfig) -> None:
    """Chemlex draws HATU as the O-form and as the N-oxide; both become the N-oxide."""
    o_form = standardize(HATU_O_FORM, "agent", config)
    n_oxide = standardize(HATU_N_OXIDE, "agent", config)
    assert o_form.smiles == n_oxide.smiles
    assert (
        o_form.smiles is not None and "[O-]" in o_form.smiles
    )  # the N-oxide, not the O-form


def test_hbtu_and_tbtu_give_one_condition(config: StdConfig) -> None:
    """HBTU (PF6-) and TBTU (BF4-) share one cation; the counter-ions are dropped."""
    hbtu = condition_key((HBTU_TBTU_CATION, PF6), (DIPEA, DMF), config)
    tbtu = condition_key((HBTU_TBTU_CATION, BF4), (DIPEA, DMF), config)
    assert hbtu == tbtu


def test_hatu_dipea_dmf_is_one_key_in_both_sources(config: StdConfig) -> None:
    """ORDerly: HATU (O-form) + PF6 as agents, DIPEA and DMF as solvents. Chemlex: DIEA, HATU
    (N-oxide) and PF6 as reagents, DMF as solvent. Same condition, same key."""
    orderly = condition_key((HATU_O_FORM, PF6), (DIPEA, DMF), config)
    chemlex = condition_key((DIPEA, HATU_N_OXIDE, PF6), (DMF,), config)
    assert orderly == chemlex
    assert orderly is not None and len(orderly) == 3


def test_key_ignores_order_and_repeats(config: StdConfig) -> None:
    """A key is a sorted set: the order in the source and listing a molecule twice do not matter."""
    once = condition_key((HATU_O_FORM,), (DIPEA, DMF), config)
    shuffled = condition_key((HATU_O_FORM, HATU_N_OXIDE), (DMF, DIPEA), config)
    assert once == shuffled


def test_carboxylate_substrate_becomes_the_acid(config: StdConfig) -> None:
    """ORDerly splits a lithium carboxylate into the anion (reactant) and Li+ (agent)."""
    assert (
        standardize("COC(=O)CCC(=O)[O-]", "substrate", config).smiles
        == "COC(=O)CCC(=O)O"
    )


def test_ammonium_substrate_becomes_the_amine(config: StdConfig) -> None:
    """An amine hydrochloride as the protonated amine becomes the free amine; its Cl is dropped
    from the condition key."""
    assert standardize("CC[NH3+]", "substrate", config).smiles == "CCN"
    assert condition_key((HATU_O_FORM, "Cl"), (DMF,), config) == condition_key(
        (HATU_O_FORM,), (DMF,), config
    )


def test_permanent_cation_stays_charged(config: StdConfig) -> None:
    """A quaternary N (thiazolium in ORDerly, N-aminopyridinium in Chemlex) cannot be neutralised."""
    assert standardize("N[n+]1ccccc1", "substrate", config).smiles == "N[n+]1ccccc1"


def test_nitro_group_is_left_alone(config: StdConfig) -> None:
    """Nitro is neutral overall; its charge separation is how it is drawn, not a salt."""
    nitro = "C[C@H](N)c1cccc([N+](=O)[O-])c1"
    assert standardize(nitro, "substrate", config).smiles == nitro


def test_agents_keep_their_charge(config: StdConfig) -> None:
    """Only substrates are neutralised: an acetate agent stays acetate (Cs2CO3 is not K2CO3)."""
    assert standardize("CC(=O)[O-]", "agent", config).smiles == "CC(=O)[O-]"


def test_atom_map_numbers_are_dropped(config: StdConfig) -> None:
    """Mapped and unmapped SMILES of one molecule give one result."""
    assert standardize("[CH3:1][OH:2]", "solvent", config).smiles == "CO"


def test_stereo_key_and_flat_key(config: StdConfig) -> None:
    """R and S differ in key_stereo but share key_flat, used to match P against TEST."""
    r = standardize("C[C@@H](N)c1ccccc1", "substrate", config)
    s = standardize("C[C@H](N)c1ccccc1", "substrate", config)
    assert r.key_stereo != s.key_stereo
    assert r.key_flat == s.key_flat == "RQEUFEKYXDPUSK"


def test_unparsable_gives_none_with_reason(config: StdConfig) -> None:
    """PyBrOP's PF6 written as a cation is not valid; it fails with a reason, never a guess."""
    result = standardize("F[P+](F)(F)(F)(F)F", "agent", config)
    assert result.smiles is None
    assert result.reason == "unparsable"
    assert result.original == "F[P+](F)(F)(F)(F)F"


def test_unparsable_part_gives_no_condition_key(config: StdConfig) -> None:
    """One unreadable reagent makes the whole condition unknown, so it is never matched."""
    assert condition_key((DIPEA, "F[P+](F)(F)(F)(F)F"), (DMF,), config) is None


@pytest.mark.parametrize(
    ("salt", "parent"),
    [
        (
            "[Cl-].N(Cc1ccccc1)Cc2ccccc2.[H+]",
            "c1ccc(CNCc2ccccc2)cc1",
        ),  # inventory amine salt
        (
            "CC(=O)[O-].[H+]",
            "CC(=O)O",
        ),  # inventory acid written as carboxylate + proton
        ("O=C([O-])c1ccccc1.[Na+]", "O=C(O)c1ccccc1"),
    ],
)
def test_substrate_salt_gives_the_parent(
    config: StdConfig, salt: str, parent: str
) -> None:
    """Lab inventory SMILES carry counter-ions and protons: a substrate is its largest fragment, neutral."""
    assert standardize(salt, "substrate", config).smiles == parent


def test_failed_neutralisation_gives_none_with_reason(config: StdConfig) -> None:
    """A cyclopentadienide (inventory) neutralises to an unreadable SMILES: no result, never a crash."""
    result = standardize("c1cc[cH-]c1", "substrate", config)
    assert result.smiles is None
    assert result.reason == "neutralise_failed"


def test_results_carry_the_config_version(config: StdConfig) -> None:
    """Every result says which standardisation produced it."""
    assert standardize(DMF, "solvent", config).std_version == config.std_version == "2"


def test_dummy_atom_gives_no_key_and_a_reason(config: StdConfig) -> None:
    """RDKit writes no InChIKey for a dummy atom (*); without a key a molecule never matches or
    deduplicates, so it fails with a reason instead of passing with an empty key."""
    result = standardize("*C(=O)O", "substrate", config)
    assert (result.smiles, result.reason, result.key_stereo) == (
        None,
        "no_inchikey",
        None,
    )


def test_family_needs_every_member_and_allows_extras() -> None:
    """Decided 2026-10-09: a condition is in a family when it contains every member; extras allowed."""
    family = frozenset({HATU_N_OXIDE, DIPEA, DMF})
    assert in_family((DIPEA, DMF, HATU_N_OXIDE, "CN(C)c1ccncc1"), family)
    assert not in_family((DIPEA, DMF), family)
    assert not in_family(None, family)  # an unreadable condition is in no family


def test_condition_text_marks_an_unreadable_condition() -> None:
    """Reports and benchmark labels write a condition as one string; unreadable ones are marked."""
    assert condition_text((DIPEA, DMF)) == f"{DIPEA} + {DMF}"
    assert condition_text(None) == "(unreadable)"


@pytest.mark.parametrize("key", [(DIPEA, DMF, HATU_N_OXIDE), (), None])
def test_condition_text_reads_back(key: tuple[str, ...] | None) -> None:
    """Benchmark labels store the condition as text; reading it back gives the key, the empty
    condition (no reagent reported) and the unreadable one included. SMILES hold no spaces, so
    " + " cannot occur inside one."""
    assert condition_from_text(condition_text(key)) == key
