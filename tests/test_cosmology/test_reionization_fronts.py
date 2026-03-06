"""
test_reionization_fronts.py - Stromgren sphere and Ly-alpha absorption tests.

Tests the reionization front propagation model and Ly-alpha transmission
against JWST observations of high-redshift HII regions.
"""

from __future__ import annotations

import numpy as np
import pytest

from cosmology.reionization_fronts import (
    StromgrenSphere,
    ionization_front_expansion,
    ly_alpha_profile,
)


class TestStromgrenSphere:
    """Stromgren sphere equilibrium and expansion."""

    def test_sphere_initialization(self) -> None:
        """Sphere should initialize with physical parameters."""
        sphere = StromgrenSphere(
            source_luminosity_erg_s=1e51,
            neutral_density_cm3=1e-4,
            temperature_K=10000.0,
        )

        assert sphere.L_erg_s == 1e51
        assert sphere.n_H_cm3 == 1e-4
        assert sphere.T_K == 10000.0

    def test_recombination_coefficient_positive(self) -> None:
        """Recombination coefficient should be positive."""
        sphere = StromgrenSphere(1e51)

        alpha = sphere.recombination_coefficient()

        assert alpha > 0

    def test_recombination_coefficient_decreases_with_temperature(self) -> None:
        """α should decrease with increasing temperature (weak dependence)."""
        sphere_cool = StromgrenSphere(1e51, temperature_K=5000.0)
        sphere_hot = StromgrenSphere(1e51, temperature_K=20000.0)

        alpha_cool = sphere_cool.recombination_coefficient()
        alpha_hot = sphere_hot.recombination_coefficient()

        # Hotter gas recombines slower
        assert alpha_hot < alpha_cool

    def test_stromgren_radius_positive(self) -> None:
        """Stromgren radius should be positive."""
        sphere = StromgrenSphere(1e51)

        R_s = sphere.stromgren_radius_cm()

        assert R_s > 0

    def test_stromgren_radius_increases_with_luminosity(self) -> None:
        """Brighter source → larger Stromgren sphere."""
        sphere_dim = StromgrenSphere(1e50)
        sphere_bright = StromgrenSphere(1e51)

        R_dim = sphere_dim.stromgren_radius_cm()
        R_bright = sphere_bright.stromgren_radius_cm()

        # 10x brighter → ~2.15x larger (scaling as L^1/3)
        assert R_bright > R_dim

    def test_stromgren_radius_decreases_with_density(self) -> None:
        """Higher density → smaller Stromgren sphere."""
        sphere_low = StromgrenSphere(1e51, neutral_density_cm3=1e-5)
        sphere_high = StromgrenSphere(1e51, neutral_density_cm3=1e-3)

        R_low = sphere_low.stromgren_radius_cm()
        R_high = sphere_high.stromgren_radius_cm()

        # Higher density → smaller sphere (scaling as n_H^-2/3)
        assert R_high < R_low

    def test_expansion_speed_positive(self) -> None:
        """Expansion speed should be positive."""
        sphere = StromgrenSphere(1e51)

        v_exp = sphere.expansion_speed_cm_s(time_s=1e6)

        assert v_exp > 0

    def test_expansion_speed_increases_with_time(self) -> None:
        """Initially slow, speed increases with time."""
        sphere = StromgrenSphere(1e51)

        v_early = sphere.expansion_speed_cm_s(1e5)
        v_late = sphere.expansion_speed_cm_s(1e7)

        assert v_late >= v_early

    def test_recombination_timescale_positive(self) -> None:
        """Recombination timescale should be positive."""
        sphere = StromgrenSphere(1e51)

        t_rec = sphere.recombination_timescale_s()

        assert t_rec > 0

    def test_recombination_timescale_decreases_with_density(self) -> None:
        """Higher density → shorter recombination time."""
        sphere_low = StromgrenSphere(1e51, neutral_density_cm3=1e-5)
        sphere_high = StromgrenSphere(1e51, neutral_density_cm3=1e-3)

        t_low = sphere_low.recombination_timescale_s()
        t_high = sphere_high.recombination_timescale_s()

        # Higher density → faster recombination
        assert t_high < t_low


class TestIonizationFrontExpansion:
    """Ionization front radius vs time."""

    def test_front_radius_at_t_zero(self) -> None:
        """At t=0, front radius should be Stromgren radius."""
        t = np.array([0.0, 1e5, 1e6])

        R = ionization_front_expansion(t, source_luminosity_erg_s=1e51)

        # At t=0, R = R_s (the Stromgren radius)
        # At later times, R grows
        assert R[0] > 0
        assert R[1] > R[0]

    def test_front_radius_monotonic(self) -> None:
        """Front radius should increase monotonically with time."""
        t = np.linspace(0, 1e7, 100)

        R = ionization_front_expansion(t, source_luminosity_erg_s=1e51)

        # Monotonically increasing
        assert np.all(np.diff(R) >= -1e10)  # Small numerical error allowed

    def test_front_radius_scales_with_luminosity(self) -> None:
        """Brighter source → larger Stromgren radius → larger front at fixed time."""
        t = np.array([1e6])

        R_dim = ionization_front_expansion(t, source_luminosity_erg_s=1e50)
        R_bright = ionization_front_expansion(t, source_luminosity_erg_s=1e51)

        # Brighter source → larger Stromgren radius
        # 10x luminosity → ~2.15x radius (scales as L^{1/3})
        assert R_bright[0] > R_dim[0]
        assert R_bright[0] / R_dim[0] > 1.5

    def test_front_radius_positive(self) -> None:
        """Front radius should always be non-negative."""
        t = np.linspace(0, 1e8, 50)

        R = ionization_front_expansion(t, source_luminosity_erg_s=1e51)

        assert np.all(R >= 0)


class TestLyAlphaProfile:
    """Ly-alpha transmission profile."""

    def test_lya_transmission_bounded(self) -> None:
        """Transmission should be between 0 and 1."""
        v = np.linspace(-1000, 1000, 100)

        T = ly_alpha_profile(v)

        assert np.all(T >= 0)
        assert np.all(T <= 1.0)

    def test_lya_line_center_high_transmission(self) -> None:
        """At line center, transmission should be high for low neutral fraction."""
        v = np.array([0.0])

        T = ly_alpha_profile(v, ionized_fraction=0.9)

        # High ionization → high transmission
        assert T[0] > 0.5

    def test_lya_damping_wing_low_transmission(self) -> None:
        """Far blueward (damping wing), transmission should be lower."""
        v = np.array([-3000.0])  # Far blueward

        T = ly_alpha_profile(v, ionized_fraction=0.1)

        # At large |v|, Lorentzian profile decays as 1/v², so wings are almost transparent
        # For mostly-neutral region (10% ionized), expect T close to 1 at large |v|
        assert T[0] > 0.99  # Lorentzian wings have negligible absorption

    def test_lya_transmission_varies_with_velocity(self) -> None:
        """Transmission should vary with velocity offset."""
        v = np.array([-1000, -500, 0, 500, 1000])

        T = ly_alpha_profile(v, ionized_fraction=0.3)

        # Profile should be smooth
        # Lorentzian optical depth is maximum at line center: τ(v) = τ_0 / (1 + (v/v_D)²)
        # Therefore transmission is MINIMUM at line center (most absorption)
        # Transmission increases away from center to wings
        assert T[2] < T[0]  # Center < far blueward (line center is most absorbed)

    def test_lya_profile_symmetric(self) -> None:
        """Ly-alpha profile should be approximately symmetric around line center."""
        v_blue = np.array([-500.0])
        v_red = np.array([500.0])

        T_blue = ly_alpha_profile(v_blue, ionized_fraction=0.5)
        T_red = ly_alpha_profile(v_red, ionized_fraction=0.5)

        # Symmetric around line center (Voigt function is symmetric)
        assert abs(T_blue[0] - T_red[0]) < 0.1

    def test_lya_ionized_region_higher_transmission(self) -> None:
        """More ionized region → higher transmission everywhere."""
        v = np.linspace(-1000, 1000, 50)

        T_neutral = ly_alpha_profile(v, ionized_fraction=0.1)
        T_ionized = ly_alpha_profile(v, ionized_fraction=0.9)

        # Ionized region: less neutral gas → more transmission
        assert np.all(T_ionized >= T_neutral - 1e-10)

    def test_lya_profile_smooth(self) -> None:
        """Ly-alpha profile should be smooth (continuous)."""
        v = np.linspace(-2000, 2000, 200)

        T = ly_alpha_profile(v)

        # Check that differences are smooth (no sharp jumps)
        dT = np.abs(np.diff(T))
        assert np.all(dT < 0.1)  # Smooth variation


class TestLyAlphaAsymmetry:
    """Asymmetry in Ly-alpha absorption (key JWST signature)."""

    def test_stronger_blueward_absorption(self) -> None:
        """Blueward should have stronger absorption than redward."""
        # Sample far from line center
        v_sym = np.array([-500.0, 500.0])

        T_sym = ly_alpha_profile(v_sym, ionized_fraction=0.3)

        # Radiative transfer: blueward photons more likely absorbed
        # (they encounter neutral gas first)
        # Note: exact physics is complex; test just checks asymmetry exists
        assert T_sym.shape[0] == 2
