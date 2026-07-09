"""Physics-based condition descriptors (ligand sterics+electronics, base pKa, solvent parameters).

ADR-005 showed that *structural* condition fingerprints crash cross-domain transfer: literature and
JnJ share few catalyst SMILES, so a condition fingerprint becomes a literature-vs-JnJ domain tag.
The fix is a **shared physical space** in which a bulky electron-rich literature ligand and a
different-but-similar JnJ ligand map close. This module encodes each reaction's conditions as
continuous descriptors:

* **ligand** -- RDKit bulk/shape, morfeus %V_bur and SASA at the phosphorus donor (the metal-free
  steric proxy that has replaced the Tolman cone angle), and GFN2-xTB electronics (HOMO/LUMO/gap,
  dipole, partial-charge summaries -- the sigma/pi donor-acceptor signal). Cached per unique ligand
  identity to disk (a few hundred ligands), since it is expensive.
* **base** -- approximate aqueous pKaH of the conjugate acid, anion charge, and a neutral/anionic
  flag, from a small curated table (13 bases).
* **solvent** -- dielectric constant, dipole moment, ET(30) polarity and the Kamlet-Taft pi*/alpha/
  beta parameters, from a small curated table (18 solvents).

The lookup tables and the :class:`ConditionDescriptorFeaturizer` import cleanly without the ``qm``
extra; only :func:`compute_ligand_descriptors` (used by the one-off precompute) needs morfeus and
tblite, imported lazily. Descriptor axes have full support on this corpus (every test
ligand/base/solvent also appears in training), which is exactly what the fingerprint route lacked.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import ClassVar, Final

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from pubench.data import schema
from pubench.features.base import FEATURIZERS, Featurizer

# --- Ligand descriptor layout ------------------------------------------------
#: RDKit whole-molecule descriptors (always computable; robust size/shape/flexibility).
LIGAND_RDKIT_NAMES: Final[tuple[str, ...]] = (
    "mol_wt",
    "heavy_atoms",
    "n_aromatic_rings",
    "n_rot_bonds",
    "tpsa",
    "fraction_csp3",
    "n_phosphorus",
    "n_hba",
    "labute_asa",
)
#: morfeus steric descriptors averaged over the phosphorus donor atom(s).
LIGAND_STERIC_NAMES: Final[tuple[str, ...]] = ("vbur_p", "sasa_p")
#: GFN2-xTB electronic descriptors of the whole ligand.
LIGAND_XTB_NAMES: Final[tuple[str, ...]] = (
    "homo",
    "lumo",
    "gap",
    "dipole",
    "q_min",
    "q_max",
    "q_absmean",
    "energy_per_atom",
)
#: Full ordered ligand descriptor vector.
LIGAND_DESCRIPTOR_NAMES: Final[tuple[str, ...]] = (
    *LIGAND_RDKIT_NAMES,
    *LIGAND_STERIC_NAMES,
    *LIGAND_XTB_NAMES,
)

BASE_DESCRIPTOR_NAMES: Final[tuple[str, ...]] = (
    "base_pkah",
    "base_anion_charge",
    "base_is_neutral",
)
SOLVENT_DESCRIPTOR_NAMES: Final[tuple[str, ...]] = (
    "solvent_dielectric",
    "solvent_dipole",
    "solvent_et30",
    "solvent_kt_pi",
    "solvent_kt_alpha",
    "solvent_kt_beta",
)

_VBUR_RADIUS: Final[float] = 3.5
_CONFORMER_SEED: Final[int] = 42

# --- Curated base table: InChIKey-14 -> (approx aqueous pKaH, anion charge, is_neutral) -------
#: Superbase pK\ :sub:`aH` values are approximate aqueous-equivalent (extrapolated); they encode
#: relative base strength, which is what the model consumes.
BASE_TABLE: Final[dict[str, tuple[float, float, float]]] = {
    "BVKZGUZCCUSVTD": (10.3, 2.0, 0.0),  # carbonate (Cs2CO3 / K2CO3 / bicarbonate)
    "CFUKEHPEQCSIOM": (21.0, 0.0, 1.0),  # P2-Et phosphazene (general-condition base)
    "NBIIXXVUZAFLBC": (12.3, 3.0, 0.0),  # phosphate (K3PO4)
    "XGIUDIMNNMKGDE": (22.0, 1.0, 0.0),  # HMDS anion (LiHMDS / NaHMDS / KHMDS)
    "SDTMFDGELKWGFT": (18.0, 1.0, 0.0),  # tert-butoxide (NaOtBu / KOtBu)
    "NLZUEZXRPGMBCV": (12.0, 1.0, 0.0),  # 2,6-di-tBu-4-Me-phenoxide (from BHT)
    "YQHJFPFNGVDEDT": (14.0, 0.0, 1.0),  # BTMG (Barton-type guanidine)
    "XLYOFNOQVPJJNP": (15.7, 1.0, 0.0),  # hydroxide (curated file labels it O / [OH-])
    "LFPXTSYMGPLZQV": (13.0, 1.0, 0.0),  # trimethylsilanolate (TMSOK)
    "ZMANZCXQSJIPKH": (10.75, 0.0, 1.0),  # triethylamine
    "GQHTUMJGOHRCHB": (13.5, 0.0, 1.0),  # DBU (amidine)
    "KRHYYFGTRYWZRS": (3.2, 1.0, 0.0),  # fluoride (CsF / KF)
    "GETQZCLCWQTVFV": (9.8, 0.0, 1.0),  # trimethylamine
}

# --- Curated solvent table: InChIKey-14 -> (ε, dipole/D, ET(30), KT π*, KT α, KT β) -----------
SOLVENT_TABLE: Final[dict[str, tuple[float, float, float, float, float, float]]] = {
    "RYHBNJHYFVUHQT": (2.25, 0.45, 36.0, 0.55, 0.00, 0.37),  # 1,4-dioxane
    "YXFVVABEGXRONW": (2.38, 0.36, 33.9, 0.54, 0.00, 0.11),  # toluene
    "IAZDPXIOMUYVGZ": (46.7, 3.96, 45.1, 1.00, 0.00, 0.76),  # DMSO
    "MSXVEPNJUHWQHW": (5.80, 1.70, 41.0, 0.40, 0.30, 0.93),  # 2-methyl-2-butanol (t-amyl-OH)
    "FXHOOIRPVKKKFG": (37.8, 3.72, 42.9, 0.88, 0.00, 0.76),  # N,N-dimethylacetamide (DMAc)
    "JWUJQDFVADABEY": (6.97, 1.38, 36.5, 0.53, 0.00, 0.58),  # 2-methyltetrahydrofuran (2-MeTHF)
    "WEVYAHXRMPXWCK": (37.5, 3.92, 45.6, 0.75, 0.19, 0.40),  # acetonitrile
    "CTQNGGLPUBDAKN": (2.57, 0.62, 34.3, 0.47, 0.00, 0.12),  # o-xylene
    "WYURNTSHIVDZCO": (7.58, 1.75, 37.4, 0.58, 0.00, 0.55),  # THF
    "ZMXDDKWLCZADIW": (36.7, 3.86, 43.2, 0.88, 0.00, 0.69),  # DMF
    "DKGAVHZHDRPRBM": (12.5, 1.70, 43.3, 0.41, 0.42, 0.93),  # tert-butanol
    "OKKJLVBELUTLKV": (32.7, 1.70, 55.4, 0.60, 0.98, 0.66),  # methanol
    "XLYOFNOQVPJJNP": (80.1, 1.85, 63.1, 1.09, 1.17, 0.47),  # water
    "VLKZOEOYAKHREP": (1.88, 0.00, 31.0, -0.04, 0.00, 0.00),  # hexane
    "LFQSCWFLJHTTHZ": (24.5, 1.69, 51.9, 0.54, 0.86, 0.75),  # ethanol
    "KFZMGEQAYNKOFK": (19.9, 1.66, 48.4, 0.48, 0.76, 0.84),  # isopropanol
    "YMWUJEATGCHHMB": (8.93, 1.60, 40.7, 0.82, 0.13, 0.10),  # dichloromethane
    "XDTMQSROBMDMFD": (2.02, 0.00, 30.9, 0.00, 0.00, 0.00),  # cyclohexane
}


def compute_ligand_descriptors(smiles: str, *, seed: int = _CONFORMER_SEED) -> np.ndarray | None:
    """Compute the full ligand descriptor vector for one ligand SMILES.

    Concatenates RDKit whole-molecule descriptors, morfeus %V_bur/SASA at the phosphorus donor
    (averaged over P atoms), and GFN2-xTB electronics. Individual blocks degrade to ``nan`` if their
    tool fails; the RDKit block is always populated for a parseable molecule.

    Args:
        smiles: Ligand SMILES (free phosphine; precatalyst metals should already be stripped).
        seed: RNG seed for conformer embedding, fixed for reproducibility.

    Returns:
        A ``(len(LIGAND_DESCRIPTOR_NAMES),)`` float array, or ``None`` if the SMILES is unparseable.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rdkit_block = np.array(
        [
            Descriptors.MolWt(mol),
            float(mol.GetNumHeavyAtoms()),
            float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            float(Descriptors.NumRotatableBonds(mol)),
            Descriptors.TPSA(mol),
            Descriptors.FractionCSP3(mol),
            float(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 15)),
            float(Lipinski.NumHAcceptors(mol)),
            rdMolDescriptors.CalcLabuteASA(mol),
        ],
        dtype=np.float64,
    )

    steric_block = _ligand_steric_block(smiles, seed=seed)
    xtb_block = _ligand_xtb_block(smiles)
    return np.concatenate([rdkit_block, steric_block, xtb_block])


def _ligand_steric_block(smiles: str, *, seed: int) -> np.ndarray:
    """%V_bur and SASA at the phosphorus donor(s), averaged (``nan`` block on failure)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    width = len(LIGAND_STERIC_NAMES)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full(width, np.nan)
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        return np.full(width, np.nan)
    with contextlib.suppress(RuntimeError, ValueError):
        AllChem.MMFFOptimizeMolecule(mol)

    phosphorus = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 15]
    if not phosphorus:
        return np.full(width, np.nan)
    from morfeus import SASA, BuriedVolume

    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = np.asarray(mol.GetConformer().GetPositions())
    atom_areas = None
    with contextlib.suppress(ValueError, np.linalg.LinAlgError):
        atom_areas = SASA(elements, coords).atom_areas  # 1-indexed per-atom SASA
    vburs: list[float] = []
    sasas: list[float] = []
    for idx in phosphorus:
        one = idx + 1  # morfeus is 1-indexed
        with contextlib.suppress(ValueError, IndexError, np.linalg.LinAlgError):
            buried = BuriedVolume(elements, coords, one, radius=_VBUR_RADIUS)
            vburs.append(float(buried.fraction_buried_volume) * 100.0)
        if atom_areas is not None:
            with contextlib.suppress(KeyError, IndexError):
                sasas.append(float(atom_areas[one]))
    return np.array(
        [
            float(np.mean(vburs)) if vburs else np.nan,
            float(np.mean(sasas)) if sasas else np.nan,
        ],
        dtype=np.float64,
    )


def _ligand_xtb_block(smiles: str) -> np.ndarray:
    """GFN2-xTB electronics via the shared bhpu routine (``nan`` block on failure)."""
    from bhpu.features.xtb import compute_xtb_descriptors

    width = len(LIGAND_XTB_NAMES)
    full = compute_xtb_descriptors(smiles)
    if full is None:
        return np.full(width, np.nan)
    # bhpu order: n_heavy, energy_per_atom, homo, lumo, gap, dipole, q_min, q_max, q_absmean.
    homo, lumo, gap, dipole, q_min, q_max, q_absmean = full[2:9]
    energy_per_atom = full[1]
    return np.array(
        [homo, lumo, gap, dipole, q_min, q_max, q_absmean, energy_per_atom], dtype=np.float64
    )


@FEATURIZERS.register("condition_descriptors")
class ConditionDescriptorFeaturizer(Featurizer):
    """Reaction conditions as physical descriptors: ligand ⊕ base ⊕ solvent ⊕ metal one-hot.

    Ligand descriptors come from a precomputed cache (keyed by :data:`schema.LIGAND_KEY`); base and
    solvent descriptors from the curated in-module tables. Missing entries become ``nan`` and are
    median-imputed (the imputer + standardiser are fit on the runner's P+U pool, label-free). The
    representation lives in a shared physical space, so it can bridge the literature↔JnJ condition
    gap that structural condition fingerprints cannot.

    Parameters
    ----------
    ligand_cache:
        Path to the ligand descriptor parquet (a ``ligand_key`` column plus
        :data:`LIGAND_DESCRIPTOR_NAMES`), or a prebuilt ``key -> vector`` mapping, or ``None`` (all
        ligand descriptors missing -> imputed; only useful for tests / base+solvent-only ablations).
    metal_vocab:
        Catalytic metals encoded as a one-hot block (order fixed for reproducibility).
    standardize:
        If true, standardise the imputed descriptors (recommended for the downstream learners).
    """

    name: ClassVar[str] = "condition_descriptors"

    def __init__(
        self,
        *,
        ligand_cache: str | dict[str, np.ndarray] | None = None,
        metal_vocab: tuple[str, ...] = ("Pd", "Cu", "Ni"),
        standardize: bool = True,
    ) -> None:
        self.ligand_cache = ligand_cache
        self.metal_vocab = metal_vocab
        self.standardize = standardize
        self._ligand: dict[str, np.ndarray] = (
            self.load_ligand_cache(ligand_cache)
            if isinstance(ligand_cache, str)
            else (ligand_cache or {})
        )
        self._imputer: SimpleImputer | None = None
        self._scaler: StandardScaler | None = None
        self._n_out: int | None = None

    @staticmethod
    def load_ligand_cache(path: str) -> dict[str, np.ndarray]:
        """Load a ligand descriptor parquet into a ``ligand_key -> descriptor vector`` mapping."""
        frame = pd.read_parquet(path)
        values = frame[list(LIGAND_DESCRIPTOR_NAMES)].to_numpy(dtype=np.float64)
        return {key: values[i] for i, key in enumerate(frame[schema.LIGAND_KEY])}

    def fit(self, records: pd.DataFrame) -> ConditionDescriptorFeaturizer:  # noqa: D102 (inherited)
        raw = self._raw_block(records)
        self._imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(raw)
        imputed = self._imputer.transform(raw)
        if self.standardize:
            self._scaler = StandardScaler().fit(imputed)
            imputed = self._scaler.transform(imputed)
        self._n_out = int(imputed.shape[1])
        return self

    def transform(self, records: pd.DataFrame) -> np.ndarray:  # noqa: D102 (inherited)
        if self._imputer is None:
            raise RuntimeError("ConditionDescriptorFeaturizer.transform called before fit().")
        imputed = self._imputer.transform(self._raw_block(records))
        if self._scaler is not None:
            imputed = self._scaler.transform(imputed)
        return np.asarray(imputed, dtype=np.float32)

    @property
    def n_features(self) -> int:
        """Total descriptor width (available after :meth:`fit`)."""
        if self._n_out is None:
            raise RuntimeError("n_features is only defined after fit().")
        return self._n_out

    @property
    def feature_names(self) -> list[str]:
        """Ordered descriptor names for the assembled block."""
        return [
            *(f"ligand__{n}" for n in LIGAND_DESCRIPTOR_NAMES),
            *(f"base__{n}" for n in BASE_DESCRIPTOR_NAMES),
            *(f"solvent__{n}" for n in SOLVENT_DESCRIPTOR_NAMES),
            *(f"metal__{m}" for m in self.metal_vocab),
        ]

    def _raw_block(self, records: pd.DataFrame) -> np.ndarray:
        """Assemble the unstandardised condition block (``nan`` for unknown identities)."""
        ligand = np.vstack([self._ligand_vec(k) for k in records[schema.LIGAND_KEY]])
        base = np.vstack(
            [
                _table_vec(k, BASE_TABLE, len(BASE_DESCRIPTOR_NAMES))
                for k in records[schema.BASE_KEY]
            ]
        )
        solvent = np.vstack(
            [
                _table_vec(k, SOLVENT_TABLE, len(SOLVENT_DESCRIPTOR_NAMES))
                for k in records[schema.SOLVENT_KEY]
            ]
        )
        metal = self._metal_onehot(records[schema.METAL])
        return np.hstack([ligand, base, solvent, metal]).astype(np.float64)

    def _ligand_vec(self, key: object) -> np.ndarray:
        """Cached ligand descriptor for a ligand key (``nan`` vector if absent)."""
        width = len(LIGAND_DESCRIPTOR_NAMES)
        if isinstance(key, str):
            cached = self._ligand.get(key)
            if cached is not None:
                return cached
        return np.full(width, np.nan)

    def _metal_onehot(self, values: pd.Series) -> np.ndarray:
        """One-hot encode the catalytic metal against :attr:`metal_vocab`."""
        index = {metal: i for i, metal in enumerate(self.metal_vocab)}
        out = np.zeros((len(values), len(self.metal_vocab)), dtype=np.float64)
        for row, value in enumerate(values):
            for token in str(value).split("|") if isinstance(value, str) else ():
                if token in index:
                    out[row, index[token]] = 1.0
        return out


def _table_vec(key: object, table: Mapping[str, tuple[float, ...]], width: int) -> np.ndarray:
    """Look up a curated descriptor tuple by key (``nan`` vector if absent)."""
    if isinstance(key, str):
        row = table.get(key)
        if row is not None:
            return np.array(row, dtype=np.float64)
    return np.full(width, np.nan)
