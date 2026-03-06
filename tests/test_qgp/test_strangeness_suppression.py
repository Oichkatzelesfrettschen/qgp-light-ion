"""
test_strangeness_suppression.py - Canonical ensemble and strangeness suppression tests.

Tests the CanonicalEnsemble class and suppression factor computation against
thermodynamic consistency and empirical strangeness yield ratios.
"""

from __future__ import annotations

import numpy as np
import pytest

from qgp.strangeness_suppression import (
    CanonicalEnsemble,
    compute_suppression_factor,
    strangeness_threshold,
)


class TestCanonicalEnsemble:
    """Canonical ensemble initialization and properties."""

    def test_ensemble_initialization(self) -> None:
        """Ensemble should initialize with valid thermodynamic parameters."""
        ensemble = CanonicalEnsemble(T=156.5, mu_B=100.0, V=1.0, s_density=0.1)

        assert ensemble.T == 156.5
        assert ensemble.mu_B == 100.0
        assert ensemble.V == 1.0
        assert ensemble.s_density == 0.1

    def test_chemical_potentials_scale_correctly(self) -> None:
        """Chemical potentials should scale with baryon chemical potential."""
        ensemble = CanonicalEnsemble(T=156.5, mu_B=200.0)

        # mu_S ~ mu_B / 3 (SU(3) flavor symmetry)
        assert abs(ensemble.mu_S - 200.0 / 3.0) < 1e-6
        # mu_Q ~ mu_B / 2
        assert abs(ensemble.mu_Q - 200.0 / 2.0) < 1e-6

    def test_partition_function_decreases_with_strangeness(self) -> None:
        """Partition function ratio should decrease with increasing strange content."""
        ensemble = CanonicalEnsemble(T=156.5, mu_B=100.0)

        ratio_0 = ensemble.partition_function_ratio(0)
        ratio_1 = ensemble.partition_function_ratio(1)
        ratio_2 = ensemble.partition_function_ratio(2)

        # Higher strangeness should have lower weight
        assert ratio_0 > ratio_1 > ratio_2
        assert ratio_0 == pytest.approx(1.0)

    def test_suppression_factor_decreases_with_temperature(self) -> None:
        """Suppression factor should decrease (become stronger) as T decreases."""
        mu_B = 100.0

        ensemble_high_T = CanonicalEnsemble(T=200.0, mu_B=mu_B)
        ensemble_low_T = CanonicalEnsemble(T=100.0, mu_B=mu_B)

        gamma_high = ensemble_high_T.suppression_factor("kaon")
        gamma_low = ensemble_low_T.suppression_factor("kaon")

        # Lower T → more suppression → smaller γ_s
        assert gamma_low < gamma_high

    def test_suppression_factor_kaon_vs_lambda(self) -> None:
        """Lambda suppression should be stronger than kaon (extra exponent)."""
        ensemble = CanonicalEnsemble(T=156.5, mu_B=100.0)

        gamma_kaon = ensemble.suppression_factor("kaon")
        gamma_lambda = ensemble.suppression_factor("lambda")

        # Lambda: γ_s^(3/2) vs Kaon: γ_s
        # Since 0 < γ_s < 1, we have γ_s^(3/2) < γ_s
        assert gamma_lambda < gamma_kaon

    def test_suppression_factor_xi_vs_lambda(self) -> None:
        """Xi suppression should be strongest (two strange quarks)."""
        ensemble = CanonicalEnsemble(T=156.5, mu_B=100.0)

        gamma_lambda = ensemble.suppression_factor("lambda")
        gamma_xi = ensemble.suppression_factor("xi")

        # Xi: γ_s^3 vs Lambda: γ_s^(3/2)
        assert gamma_xi < gamma_lambda

    def test_threshold_below_crossover_temperature(self) -> None:
        """Below T_c, threshold should be positive and decrease with T."""
        T_c = 156.5

        ensemble_cold = CanonicalEnsemble(T=100.0, mu_B=100.0)
        ensemble_warm = CanonicalEnsemble(T=140.0, mu_B=100.0)

        threshold_cold = ensemble_cold.threshold_density()
        threshold_warm = ensemble_warm.threshold_density()

        # Both should be positive
        assert threshold_cold > 0
        assert threshold_warm > 0
        # Colder → higher threshold (more suppression)
        assert threshold_cold > threshold_warm

    def test_invalid_particle_type_raises_error(self) -> None:
        """Unknown particle type should raise ValueError."""
        ensemble = CanonicalEnsemble(T=156.5, mu_B=100.0)

        with pytest.raises(ValueError, match="Unknown particle type"):
            ensemble.suppression_factor("gluon")


class TestSuppressionFactorGrid:
    """Suppression factor computation on phase diagram grids."""

    def test_suppression_factor_1d_arrays(self) -> None:
        """Should handle 1D temperature and baryon chemical potential arrays."""
        T = np.linspace(100.0, 200.0, 10)
        mu_B = np.linspace(50.0, 300.0, 15)

        suppression = compute_suppression_factor(T, mu_B, particle_type="kaon")

        assert suppression.shape == (10, 15)
        assert np.all(suppression >= 0)
        assert np.all(suppression <= 1.0)

    def test_suppression_factor_2d_arrays(self) -> None:
        """Should handle pre-gridded 2D arrays."""
        T_1d = np.linspace(100.0, 200.0, 8)
        mu_B_1d = np.linspace(50.0, 300.0, 12)
        T, mu_B = np.meshgrid(T_1d, mu_B_1d, indexing="ij")

        suppression = compute_suppression_factor(T, mu_B, particle_type="lambda")

        assert suppression.shape == (8, 12)
        assert np.all(suppression >= 0)
        assert np.all(suppression <= 1.0)

    def test_suppression_decreases_toward_high_temperature(self) -> None:
        """At high T and low mu_B, suppression should be weak (γ_s ~ 1)."""
        T = np.array([100.0, 150.0, 200.0])
        mu_B = np.array([10.0])

        suppression = compute_suppression_factor(T, mu_B, particle_type="kaon")

        # High T → weak suppression → γ_s increases toward 1
        assert suppression[0, 0] < suppression[1, 0] < suppression[2, 0]

    def test_different_particle_types_ordered_correctly(self) -> None:
        """Suppression factors should be ordered: kaon > lambda > xi."""
        T = np.array([150.0])
        mu_B = np.array([100.0])

        gamma_K = compute_suppression_factor(T, mu_B, particle_type="kaon")[0, 0]
        gamma_L = compute_suppression_factor(T, mu_B, particle_type="lambda")[0, 0]
        gamma_Xi = compute_suppression_factor(T, mu_B, particle_type="xi")[0, 0]

        assert gamma_K > gamma_L > gamma_Xi
        # All should be in physical range
        assert 0.2 < gamma_K < 1.0
        assert 0.1 < gamma_L < gamma_K
        assert 0.05 < gamma_Xi < gamma_L


class TestStrangenessThreshold:
    """Strangeness threshold behavior."""

    def test_threshold_positive_everywhere(self) -> None:
        """Threshold density should be positive at all temperatures."""
        T = np.linspace(50.0, 400.0, 100)
        threshold = strangeness_threshold(T)

        assert np.all(threshold > 0)

    def test_threshold_minimum_near_crossover(self) -> None:
        """Threshold should have minimum near T_c ~ 156.5 MeV."""
        T = np.linspace(100.0, 200.0, 100)
        threshold = strangeness_threshold(T)

        # Find minimum
        min_idx = np.argmin(threshold)
        T_min = T[min_idx]

        # Should be near crossover temperature
        assert 140.0 < T_min < 170.0

    def test_threshold_increases_away_from_crossover(self) -> None:
        """Threshold should increase both above and below T_c."""
        T_low = 120.0
        T_high = 180.0

        threshold_low = strangeness_threshold(np.array([T_low]))[0]
        threshold_high = strangeness_threshold(np.array([T_high]))[0]

        # Both should be higher than the minimum (which is near T_c)
        threshold_Tc = strangeness_threshold(np.array([156.5]))[0]

        assert threshold_low > threshold_Tc
        assert threshold_high > threshold_Tc

    def test_threshold_array_shapes(self) -> None:
        """Should preserve input array shape."""
        T_1d = np.linspace(100.0, 200.0, 50)
        threshold_1d = strangeness_threshold(T_1d)

        assert threshold_1d.shape == T_1d.shape

        T_2d = np.array([[100.0, 150.0], [160.0, 200.0]])
        threshold_2d = strangeness_threshold(T_2d)

        assert threshold_2d.shape == T_2d.shape
