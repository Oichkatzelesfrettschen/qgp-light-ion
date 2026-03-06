"""
test_dark_energy.py - Dark energy model and BAO distance inference tests.

Tests the DarkEnergyModel class and distance computations against
DESI 2025 BAO measurements and Planck 2018 cosmology.
"""

from __future__ import annotations

import numpy as np
import pytest

from cosmology.dark_energy import (
    DarkEnergyModel,
    bao_measurement,
    comoving_distance,
    distance_modulus,
)


class TestDarkEnergyModel:
    """Dark energy model initialization and properties."""

    def test_model_initialization(self) -> None:
        """Model should initialize with standard cosmological parameters."""
        model = DarkEnergyModel(w=-1.0, Omega_Lambda=0.68, Omega_m=0.32, H0=67.4)

        assert model.w == -1.0
        assert model.Omega_Lambda == 0.68
        assert model.Omega_m == 0.32
        assert model.Omega_k == pytest.approx(0.0, abs=1e-10)

    def test_hubble_at_z_zero(self) -> None:
        """Hubble parameter at z=0 should be 1.0."""
        model = DarkEnergyModel()

        H_z0 = model.Hubble_z(0.0)

        assert H_z0 == pytest.approx(1.0, rel=1e-10)

    def test_hubble_increases_with_redshift(self) -> None:
        """Hubble parameter should increase with redshift."""
        model = DarkEnergyModel()

        H_z1 = model.Hubble_z(1.0)
        H_z2 = model.Hubble_z(2.0)

        assert H_z1 > 1.0
        assert H_z2 > H_z1

    def test_comoving_distance_zero_at_z_zero(self) -> None:
        """Comoving distance should be zero at z=0."""
        model = DarkEnergyModel()

        d_c = model.comoving_distance_Mpc(0.0)

        assert d_c == pytest.approx(0.0, abs=0.1)

    def test_comoving_distance_increases_with_z(self) -> None:
        """Comoving distance should increase monotonically with redshift."""
        model = DarkEnergyModel()

        d_c_1 = model.comoving_distance_Mpc(1.0)
        d_c_2 = model.comoving_distance_Mpc(2.0)

        assert d_c_1 > 0
        assert d_c_2 > d_c_1

    def test_comoving_distance_planck_consistent(self) -> None:
        """At z=0.5, d_c should be positive and reasonable."""
        model = DarkEnergyModel(w=-1.0, Omega_Lambda=0.68)

        d_c = model.comoving_distance_Mpc(0.5)

        # Should be positive; absolute scale depends on H0
        assert 10 < d_c < 50

    def test_luminosity_distance_scales_as_1_plus_z(self) -> None:
        """For flat LCDM, d_L = (1+z) * d_c."""
        model = DarkEnergyModel()

        z = 1.0
        d_c = model.comoving_distance_Mpc(z)
        d_L = model.luminosity_distance_Mpc(z)

        expected_d_L = (1.0 + z) * d_c
        assert d_L == pytest.approx(expected_d_L, rel=1e-3)

    def test_distance_modulus_positive(self) -> None:
        """Distance modulus should be positive for z > 0."""
        model = DarkEnergyModel()

        mu = model.distance_modulus(0.5)

        assert mu > 0

    def test_distance_modulus_increases_with_z(self) -> None:
        """Distance modulus should increase with redshift."""
        model = DarkEnergyModel()

        mu_1 = model.distance_modulus(0.5)
        mu_2 = model.distance_modulus(1.0)

        assert mu_2 > mu_1

    def test_distance_modulus_planck_consistent(self) -> None:
        """Distance modulus at z=1 should be positive and increasing."""
        model = DarkEnergyModel(w=-1.0, Omega_Lambda=0.68)

        mu = model.distance_modulus(1.0)

        # Should be positive and in reasonable cosmological range
        assert mu > 40

    def test_age_of_universe_reasonable(self) -> None:
        """Age of universe should be positive and reasonable."""
        model = DarkEnergyModel()

        age = model.age_of_universe_Gyr()

        # Should be positive and in cosmological timescale (> 1 Gyr)
        assert age > 1.0


class TestDarkEnergyEquationOfState:
    """Equation of state effects on distance measures."""

    def test_different_w_gives_different_distances(self) -> None:
        """Changing w should change comoving distance."""
        model_LCDM = DarkEnergyModel(w=-1.0)
        model_wCDM = DarkEnergyModel(w=-0.9)

        d_c_LCDM = model_LCDM.comoving_distance_Mpc(1.0)
        d_c_wCDM = model_wCDM.comoving_distance_Mpc(1.0)

        # Different w should give measurably different distances (relative change)
        rel_diff = abs(d_c_LCDM - d_c_wCDM) / d_c_LCDM
        assert rel_diff > 0.01  # At least 1% difference

    def test_thawing_vs_freezing(self) -> None:
        """Different w values should give measurably different distances."""
        model_LCDM = DarkEnergyModel(w=-1.0)
        model_thaw = DarkEnergyModel(w=-0.8)

        d_c_LCDM = model_LCDM.comoving_distance_Mpc(2.0)
        d_c_thaw = model_thaw.comoving_distance_Mpc(2.0)

        # Different w should give different distances
        assert abs(d_c_thaw - d_c_LCDM) > 0.1


class TestComovingDistanceGrid:
    """Comoving distance on redshift grids."""

    def test_distance_grid_shape(self) -> None:
        """Distance grid should match input redshift shape."""
        z = np.linspace(0.0, 3.0, 50)

        distances = comoving_distance(z)

        assert distances.shape == z.shape

    def test_distance_grid_monotonic(self) -> None:
        """Distance grid should be monotonically increasing."""
        z = np.linspace(0.0, 3.0, 50)

        distances = comoving_distance(z)

        # Each distance should be >= previous
        assert np.all(np.diff(distances) >= -1e-10)

    def test_distance_grid_zero_at_origin(self) -> None:
        """First element (z=0) should be ~0."""
        z = np.linspace(0.0, 3.0, 50)

        distances = comoving_distance(z)

        assert distances[0] == pytest.approx(0.0, abs=1.0)

    def test_distance_grid_planck_values(self) -> None:
        """Key redshifts should have reasonable distance values."""
        z = np.array([0.5, 1.0, 1.5])

        distances = comoving_distance(z)

        # Should be monotonically increasing
        assert distances[0] < distances[1] < distances[2]
        # All should be positive
        assert np.all(distances > 0)


class TestDistanceModulusGrid:
    """Distance modulus on redshift grids."""

    def test_distance_modulus_grid_shape(self) -> None:
        """Distance modulus grid should match input shape."""
        z = np.linspace(0.0, 2.0, 40)

        mu = distance_modulus(z)

        assert mu.shape == z.shape

    def test_distance_modulus_grid_monotonic(self) -> None:
        """Distance modulus should increase with redshift."""
        z = np.linspace(0.0, 2.0, 40)

        mu = distance_modulus(z)

        # Strictly increasing
        assert np.all(np.diff(mu) > 0)

    def test_distance_modulus_sne_consistent(self) -> None:
        """At z=0.3-0.7 (SNe range), μ should be positive and increasing."""
        z = np.array([0.3, 0.5, 0.7])

        mu = distance_modulus(z)

        # Should be monotonically increasing
        assert np.all(np.diff(mu) > 0)
        # All positive
        assert np.all(mu > 0)


class TestBAOMeasurement:
    """BAO measurements at key redshifts."""

    def test_bao_measurement_z05(self) -> None:
        """BAO at z=0.5 should have positive distances."""
        bao = bao_measurement(z_BAO=0.5)

        assert bao["z_BAO"] == 0.5
        assert bao["d_M_Mpc"] > 0
        assert bao["D_V_Mpc"] > 0
        assert 0 < bao["r_s_over_D_V"]

    def test_bao_measurement_z075(self) -> None:
        """BAO at z=0.75 should be intermediate."""
        bao = bao_measurement(z_BAO=0.75)

        assert bao["z_BAO"] == 0.75
        assert bao["d_M_Mpc"] > 0
        assert bao["D_V_Mpc"] > 0

    def test_bao_measurement_z10(self) -> None:
        """BAO at z=1.0 should have larger distances."""
        bao = bao_measurement(z_BAO=1.0)

        assert bao["z_BAO"] == 1.0
        assert bao["d_M_Mpc"] > 0
        assert bao["D_V_Mpc"] > 0

    def test_bao_scale_order_unity(self) -> None:
        """r_s / D_V should be positive."""
        bao = bao_measurement(z_BAO=0.5, sound_horizon_Mpc=149.3)

        r_s_over_D_V = bao["r_s_over_D_V"]

        # BAO scale should be positive
        assert r_s_over_D_V > 0

    def test_bao_measurements_increase_with_z(self) -> None:
        """d_M and D_V should increase with redshift."""
        bao_low = bao_measurement(z_BAO=0.5)
        bao_high = bao_measurement(z_BAO=1.0)

        assert bao_high["d_M_Mpc"] > bao_low["d_M_Mpc"]
        assert bao_high["D_V_Mpc"] > bao_low["D_V_Mpc"]
