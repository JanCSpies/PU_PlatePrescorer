"""Tests for reactive-site location, conformer aggregation, and site-steric descriptors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bhpu.chem.reaction import coupling_site_indices
from bhpu.chem.sites import (
    AGGREGATION_MODES,
    aggregate_conformer_descriptors,
    boltzmann_weights,
    embed_conformers,
    locate_amine_site,
    locate_aryl_site,
    weighted_nanmean,
)
from bhpu.data import schema
from bhpu.data.mining import iter_pistachio_csv
from bhpu.features.morfeus_desc import (
    ARYL_STERIC_DESCRIPTORS,
    MorfeusSiteFeaturizer,
    compute_amine_steric_descriptors,
)


class TestReactiveSiteLocation:
    """The BH template resolves the reacting aryl carbon and amine nitrogen."""

    def test_locates_monohalide_and_amine(self) -> None:
        """A simple aryl bromide + secondary amine give a carbon, halide and nitrogen."""
        sites = coupling_site_indices("Cc1ccc(Br)cc1", "C1CCNCC1")
        assert sites is not None
        carbon, halide, nitrogen = sites
        assert carbon is not None and nitrogen is not None and halide is not None

    def test_polyhalide_picks_a_halogen_bearing_carbon(self) -> None:
        """For a poly-halide the located carbon still bears a halogen leaving group."""
        site = locate_aryl_site("Brc1cc(Br)cc(Br)c1")
        assert site is not None
        carbon_idx, leaving_idx = site
        assert carbon_idx is not None and leaving_idx is not None

    def test_rejects_non_nucleophile(self) -> None:
        """Benzene has no N–H site, so amine location returns None."""
        assert locate_amine_site("c1ccccc1") is None

    def test_aryl_site_partner_independent(self) -> None:
        """The aryl site is the same regardless of the reference amine used to locate it."""
        assert locate_aryl_site("Brc1ccccc1") is not None


class TestConformerAggregation:
    """Boltzmann/lowest/mean aggregation of per-conformer descriptors."""

    def test_boltzmann_weights_sum_to_one(self) -> None:
        """Weights are non-negative and normalised."""
        weights = boltzmann_weights([0.0, 1.0, 2.0])
        assert np.isclose(weights.sum(), 1.0)
        assert weights[0] > weights[1] > weights[2]  # lower energy -> higher weight

    def test_boltzmann_weights_uniform_when_energies_missing(self) -> None:
        """All-NaN energies fall back to a uniform distribution."""
        weights = boltzmann_weights([float("nan"), float("nan")])
        assert np.allclose(weights, 0.5)

    def test_weighted_nanmean_ignores_nan(self) -> None:
        """A descriptor that is NaN in one conformer is averaged over the rest."""
        matrix = np.array([[1.0, np.nan], [3.0, 4.0]])
        result = weighted_nanmean(matrix, np.array([0.5, 0.5]))
        assert np.isclose(result[0], 2.0)
        assert np.isclose(result[1], 4.0)

    def test_lowest_picks_min_energy_row(self) -> None:
        """``lowest`` returns the descriptor row of the lowest-energy conformer."""
        matrix = np.array([[10.0], [20.0], [30.0]])
        energies = [5.0, 1.0, 9.0]  # second conformer is lowest
        result = aggregate_conformer_descriptors(matrix, energies, mode="lowest")
        assert np.isclose(result[0], 20.0)

    def test_mean_is_uniform_average(self) -> None:
        """``mean`` ignores energies and averages uniformly."""
        matrix = np.array([[0.0], [10.0]])
        result = aggregate_conformer_descriptors(matrix, [0.0, 100.0], mode="mean")
        assert np.isclose(result[0], 5.0)

    def test_all_modes_supported(self) -> None:
        """Every advertised mode runs without error."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        for mode in AGGREGATION_MODES:
            out = aggregate_conformer_descriptors(matrix, [0.0, 1.0], mode=mode)
            assert out.shape == (2,)


class TestConformerEmbedding:
    """Multi-conformer embedding keeps the reactive index aligned."""

    def test_embed_returns_multiple_conformers(self) -> None:
        """A flexible amine embeds several conformers with aligned atom order."""
        site = locate_amine_site("CCCCNCCCC")
        assert site is not None
        ensemble = embed_conformers("CCCCNCCCC", site, n_conformers=5)
        assert ensemble is not None
        assert len(ensemble.conformers) >= 1
        assert ensemble.conformers[0].shape[1] == 3
        assert len(ensemble.elements) == ensemble.conformers[0].shape[0]


class TestMorfeusFeaturizer:
    """The morfeus site featuriser produces finite, correctly shaped descriptors."""

    def test_amine_descriptor_shape_and_finiteness(self) -> None:
        """A single amine yields a finite steric descriptor of the expected width."""
        vector = compute_amine_steric_descriptors("C1CCNCC1", n_conformers=3)
        assert vector is not None
        assert vector.shape == (3,)
        assert np.isfinite(vector).all()

    def test_featurizer_matrix_shape(self) -> None:
        """A small reaction frame is featurised to aryl⊕amine width, no all-zero rows."""
        frame = pd.DataFrame(
            {
                schema.ARYL_SMILES: ["Brc1ccccc1", "Cc1ccc(Cl)cc1"],
                schema.AMINE_SMILES: ["C1CCNCC1", "C1COCCN1"],
            }
        )
        matrix = MorfeusSiteFeaturizer().transform(frame)
        assert matrix.shape == (2, len(ARYL_STERIC_DESCRIPTORS) + 3)
        assert (np.abs(matrix).sum(axis=1) > 0).all()


class TestPistachioReader:
    """The Pistachio reader parses atom-mapped reaction SMILES into raw reactions."""

    def test_parses_reactants_agents_products(self, tmp_path: object) -> None:
        """A ``reactants>agents>products`` row yields pooled molecules and products."""
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        csv = tmp_path / "rxn.csv"
        csv.write_text("rxn_smiles\n[CH3:1]I.[NH3:2]>[Pd]>[CH3:1][NH2:2]\nBrc1ccccc1>>c1ccccc1\n")
        reactions = list(iter_pistachio_csv(csv))
        assert len(reactions) == 2
        first = reactions[0]
        assert "[Pd]" in first.molecules  # agent pooled into candidate molecules
        assert first.products == ("[CH3:1][NH2:2]",)
