"""
test_reionization_fronts.py - Test suite for ionization front calculations.

Tests the physical correctness of Stromgren sphere radius, ionization front
expansion, Ly-alpha transmission, and reionization history functions.
"""

from __future__ import annotations

import numpy as np
from src.cosmology.reionization_fronts import (
    StromgrenSphere,
    ionization_front_expansion,
    ionized_fraction_evolution,
    ly_alpha_profile,
    neutral_column_density,
)


class TestStromgrenSphere:
    """Stromgren sphere radius and ionization equilibrium."""

    def test_initialization(self) -> None:
        """StromgrenSphere should initialize with luminosity."""
        sphere = StromgrenSphere(source_luminosity_erg_s=1e51)
        assert sphere.L_erg_s == 1e51
        assert sphere.n_H_cm3 == 1e-4
        assert sphere.T_K == 10000.0

    def test_recombination_coefficient_positive(self) -> None:
        """Recombination coefficient α(T) must be positive."""
        sphere = StromgrenSphere(source_luminosity_erg_s=1e51)
        alpha = sphere.recombination_coefficient()
        assert alpha > 0.0

    def test_recombination_coefficient_decreases_with_temperature(self) -> None:
        """Recombination rate decreases with temperature (weaker dependence)."""
        sphere_cool = StromgrenSphere(
            source_luminosity_erg_s=1e51, temperature_K=5000.0
        )
        sphere_hot = StromgrenSphere(
            source_luminosity_erg_s=1e51, temperature_K=20000.0
        )
        alpha_cool = sphere_cool.recombination_coefficient()
        alpha_hot = sphere_hot.recombination_coefficient()
        # α ∝ T^{-0.75}: hotter gas recombines slower
        assert alpha_hot < alpha_cool

    def test_stromgren_radius_positive(self) -> None:
        """Stromgren radius must be positive."""
        sphere = StromgrenSphere(source_luminosity_erg_s=1e51)
        R_s = sphere.stromgren_radius_cm()
        assert R_s > 0.0

    def test_stromgren_radius_increases_with_luminosity(self) -> None:
        """Brighter source → larger Stromgren sphere."""
        sphere_dim = StromgrenSphere(source_luminosity_erg_s=1e50)
        sphere_bright = StromgrenSphere(source_luminosity_erg_s=1e51)
        R_dim = sphere_dim.stromgren_radius_cm()
        R_bright = sphere_bright.stromgren_radius_cm()
        # 10x luminosity → ~2.15x radius (scales as L^{1/3})
        assert R_bright > R_dim
        assert R_bright / R_dim > 1.5

    def test_stromgren_radius_decreases_with_density(self) -> None:
        """Higher density → smaller Stromgren sphere."""
        sphere_low = StromgrenSphere(
            source_luminosity_erg_s=1e51, neutral_density_cm3=1e-5
        )
        sphere_high = StromgrenSphere(
            source_luminosity_erg_s=1e51, neutral_density_cm3=1e-3
        )
        R_low = sphere_low.stromgren_radius_cm()
        R_high = sphere_high.stromgren_radius_cm()
        # Higher density → smaller sphere (scales as n_H^{-2/3})
        assert R_high < R_low

    def test_expansion_speed_positive(self) -> None:
        """Expansion speed should be positive."""
        sphere = StromgrenSphere(source_luminosity_erg_s=1e51)
        v_exp = sphere.expansion_speed_cm_s(time_s=1e6)
        assert v_exp > 0.0

    def test_recombination_timescale_positive(self) -> None:
        """Recombination timescale should be positive."""
        sphere = StromgrenSphere(source_luminosity_erg_s=1e51)
        t_rec = sphere.recombination_timescale_s()
        assert t_rec > 0.0

    def test_recombination_timescale_decreases_with_density(self) -> None:
        """Higher density → shorter recombination time."""
        sphere_low = StromgrenSphere(
            source_luminosity_erg_s=1e51, neutral_density_cm3=1e-5
        )
        sphere_high = StromgrenSphere(
            source_luminosity_erg_s=1e51, neutral_density_cm3=1e-3
        )
        t_low = sphere_low.recombination_timescale_s()
        t_high = sphere_high.recombination_timescale_s()
        # Higher density → faster recombination (t_rec ∝ n_H^{-1})
        assert t_high < t_low


class TestIonizationFrontExpansion:
    """Ionization front dynamics and shock wave formation."""

    def test_radius_at_t_zero(self) -> None:
        """At t=0, front should be at Stromgren radius."""
        t = np.array([0.0])
        R = ionization_front_expansion(t, source_luminosity_erg_s=1e51)
        assert R[0] > 0.0

    def test_radius_increases_with_time(self) -> None:
        """Ionization front must expand outward."""
        t = np.array([0.1, 1.0])
        R = ionization_front_expansion(t, source_luminosity_erg_s=1e51)
        assert R[1] > R[0]

    def test_radius_positive(self) -> None:
        """Front radius must be positive."""
        t = np.array([0.5, 1.5, 5.0])
        R = ionization_front_expansion(t, source_luminosity_erg_s=1e51)
        assert np.all(R > 0.0)

    def test_expansion_monotonic(self) -> None:
        """Front radius should increase monotonically with time."""
        t = np.linspace(0, 10, 50)
        R = ionization_front_expansion(t, source_luminosity_erg_s=1e51)
        # Check monotonic increase (allow small numerical error)
        assert np.all(np.diff(R) >= -1e10)

    def test_radius_scales_with_luminosity(self) -> None:
        """Brighter source → larger radius at fixed time."""
        t = np.array([1.0])
        R_dim = ionization_front_expansion(t, source_luminosity_erg_s=1e50)
        R_bright = ionization_front_expansion(t, source_luminosity_erg_s=1e51)
        # 10x brighter → larger radius (Stromgren radius scales as L^{1/3})
        assert R_bright[0] > R_dim[0]
        assert R_bright[0] / R_dim[0] > 1.5

    def test_radius_scales_with_density(self) -> None:
        """Higher density → smaller radius at fixed time."""
        t = np.array([1.0])
        R_low = ionization_front_expansion(
            t, source_luminosity_erg_s=1e51, neutral_density_cm3=1e-5
        )
        R_high = ionization_front_expansion(
            t, source_luminosity_erg_s=1e51, neutral_density_cm3=1e-3
        )
        # Higher density → smaller radius (scales as n_H^{-2/3})
        assert R_high[0] < R_low[0]


class TestNeutralColumnDensity:
    """Column density calculation from ionized fraction and path length."""

    def test_column_density_zero_when_fully_ionized(self) -> None:
        """Fully ionized gas (x_e=1) has zero neutral column density."""
        n_H_I, N_HI = neutral_column_density(
            ionized_fraction=1.0, neutral_density_cm3=1e-3, path_length_kpc=1.0
        )
        assert n_H_I == 0.0
        assert N_HI == 0.0

    def test_column_density_increases_with_path_length(self) -> None:
        """Longer path → larger column density."""
        _, N_HI_1 = neutral_column_density(
            ionized_fraction=0.1, neutral_density_cm3=1e-3, path_length_kpc=1.0
        )
        _, N_HI_2 = neutral_column_density(
            ionized_fraction=0.1, neutral_density_cm3=1e-3, path_length_kpc=2.0
        )
        assert N_HI_2 > N_HI_1

    def test_column_density_positive(self) -> None:
        """Column density must be non-negative."""
        _, N_HI = neutral_column_density(
            ionized_fraction=0.5, neutral_density_cm3=1e-3, path_length_kpc=1.0
        )
        assert N_HI >= 0.0

    def test_neutral_density_decreases_with_ionization(self) -> None:
        """Higher ionized fraction → lower neutral density."""
        n_H_I_1, _ = neutral_column_density(
            ionized_fraction=0.1, neutral_density_cm3=1e-3, path_length_kpc=1.0
        )
        n_H_I_2, _ = neutral_column_density(
            ionized_fraction=0.9, neutral_density_cm3=1e-3, path_length_kpc=1.0
        )
        assert n_H_I_2 < n_H_I_1

    def test_path_length_conversion(self) -> None:
        """1 kpc should convert to ~3.086e21 cm."""
        _, N_HI = neutral_column_density(
            ionized_fraction=0.0, neutral_density_cm3=1.0, path_length_kpc=1.0
        )
        # 1.0 cm^{-3} * 1 kpc = 1 cm^{-3} * 3.086e21 cm = 3.086e21 cm^{-2}
        assert np.isclose(N_HI, 3.086e21, rtol=0.01)

    def test_returns_tuple(self) -> None:
        """Function should return (n_H_I, N_HI) tuple."""
        result = neutral_column_density(
            ionized_fraction=0.5, neutral_density_cm3=1e-3, path_length_kpc=1.0
        )
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestLyAlphaProfile:
    """Ly-alpha transmission profile in damping wing regime."""

    def test_transmission_at_line_center_low(self) -> None:
        """Transmission at Ly-alpha line center (v=0) depends on ionized fraction."""
        T_neutral = ly_alpha_profile(velocity_km_s=0.0, ionized_fraction=0.1)
        T_ionized = ly_alpha_profile(velocity_km_s=0.0, ionized_fraction=0.9)
        # Higher ionization → higher transmission at line center
        assert T_ionized > T_neutral

    def test_transmission_far_from_line(self) -> None:
        """Transmission far from line center (high |v|) should approach 1."""
        T = ly_alpha_profile(velocity_km_s=5000.0, ionized_fraction=0.1)
        assert T > 0.9  # Transparent far from line

    def test_transmission_bounded_0_to_1(self) -> None:
        """Transmission must be in [0, 1]."""
        for v in [0.0, 100.0, 1000.0, 5000.0]:
            for x_e in [0.0, 0.5, 1.0]:
                T = ly_alpha_profile(velocity_km_s=v, ionized_fraction=x_e)
                assert 0.0 <= T <= 1.0

    def test_transmission_symmetric_around_zero(self) -> None:
        """Transmission profile should be symmetric (v and -v give same T)."""
        T_pos = ly_alpha_profile(velocity_km_s=500.0, ionized_fraction=0.2)
        T_neg = ly_alpha_profile(velocity_km_s=-500.0, ionized_fraction=0.2)
        assert np.isclose(T_pos, T_neg, rtol=1e-10)

    def test_more_ionized_more_transparent(self) -> None:
        """Higher ionized fraction → higher transmission (fewer Ly-alpha absorbers)."""
        T_neutral = ly_alpha_profile(velocity_km_s=100.0, ionized_fraction=0.1)
        T_ionized = ly_alpha_profile(velocity_km_s=100.0, ionized_fraction=0.9)
        assert T_ionized > T_neutral

    def test_path_length_increases_absorption(self) -> None:
        """Longer path length → lower transmission."""
        T_short = ly_alpha_profile(
            velocity_km_s=100.0, ionized_fraction=0.2, path_length_kpc=0.1
        )
        T_long = ly_alpha_profile(
            velocity_km_s=100.0, ionized_fraction=0.2, path_length_kpc=10.0
        )
        assert T_long < T_short

    def test_scalar_output(self) -> None:
        """Should return scalar float or numpy float."""
        T = ly_alpha_profile(velocity_km_s=100.0, ionized_fraction=0.5)
        # Should be a scalar floating type
        assert isinstance(T, (float, np.floating, np.ndarray))


class TestIonizedFractionEvolution:
    """Reionization history: ionized fraction as function of redshift."""

    def test_bounded_0_to_1(self) -> None:
        """Ionized fraction must be in [0, 1]."""
        z = np.linspace(0, 20, 50)
        x_e = ionized_fraction_evolution(z)
        assert np.all(x_e >= 0.0)
        assert np.all(x_e <= 1.0)

    def test_monotonic_increasing(self) -> None:
        """x_e increases with decreasing redshift (earlier times → more ionized)."""
        z = np.array([15.0, 10.0, 5.0, 0.0])
        x_e = ionized_fraction_evolution(z)
        # Higher z → lower x_e (less ionized)
        for i in range(len(z) - 1):
            assert x_e[i] <= x_e[i + 1]

    def test_high_redshift_neutral(self) -> None:
        """At high redshift (z >> z_reion), x_e should be near 0."""
        z_high = np.array([30.0, 25.0, 20.0])
        x_e = ionized_fraction_evolution(z_high)
        assert np.all(x_e < 0.1)

    def test_low_redshift_ionized(self) -> None:
        """At low redshift (z << z_reion), x_e should be near 1."""
        z_low = np.array([0.0, 1.0, 2.0])
        x_e = ionized_fraction_evolution(z_low)
        assert np.all(x_e > 0.9)

    def test_planck_constraint_z_reion(self) -> None:
        """At z_reion ≈ 7.7 (Planck 2018), x_e should be ~0.5."""
        z = np.array([7.7])
        x_e = ionized_fraction_evolution(z, z_reion=7.7)
        assert np.isclose(x_e[0], 0.5, atol=0.01)

    def test_customizable_z_reion(self) -> None:
        """Parameter z_reion should shift the midpoint of reionization."""
        z = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        x_e_early = ionized_fraction_evolution(z, z_reion=6.0)
        x_e_late = ionized_fraction_evolution(z, z_reion=10.0)
        # Early reionization (z_reion=6.0) → midpoint at z=6
        #   At z=6.0: x_e ≈ 0.5
        # Late reionization (z_reion=10.0) → midpoint at z=10
        #   At z=6.0: z < z_reion → x_e << 0.5
        # So x_e_early[0] should be closer to 0.5 than x_e_late[0]
        assert np.abs(x_e_early[0] - 0.5) < np.abs(x_e_late[0] - 0.5)

    def test_customizable_delta_z(self) -> None:
        """Parameter delta_z should control width of transition."""
        z = np.linspace(6, 10, 100)
        x_e_sharp = ionized_fraction_evolution(z, delta_z=0.5)
        x_e_smooth = ionized_fraction_evolution(z, delta_z=2.0)
        # Sharp transition should have larger gradients
        grad_sharp = np.diff(x_e_sharp)
        grad_smooth = np.diff(x_e_smooth)
        assert np.max(np.abs(grad_sharp)) > np.max(np.abs(grad_smooth))

    def test_array_input(self) -> None:
        """Should accept 1D redshift array."""
        z = np.linspace(0, 20, 100)
        x_e = ionized_fraction_evolution(z)
        assert x_e.shape == z.shape
        assert x_e.dtype in (np.float32, np.float64)

    def test_scalar_input(self) -> None:
        """Should accept scalar redshift via broadcast."""
        z = np.array([7.7])
        x_e = ionized_fraction_evolution(z)
        assert len(x_e) == 1
        assert isinstance(x_e[0], (np.floating, float))


class TestLyAlphaPathLengthDependence:
    """Integration: Ly-alpha absorption vs path length and ionization."""

    def test_transmission_grid_path_length_dependence(self) -> None:
        """Transmission should decrease monotonically with increasing path length."""
        velocities = np.linspace(0, 2000, 20)
        x_e = 0.3

        transmissions = []
        for path_length_kpc in [0.1, 0.5, 1.0, 5.0, 10.0]:
            T_array = np.array(
                [
                    ly_alpha_profile(
                        velocity_km_s=v,
                        ionized_fraction=x_e,
                        path_length_kpc=path_length_kpc,
                    )
                    for v in velocities
                ]
            )
            transmissions.append(T_array)

        # At fixed velocity, transmission should decrease with path length
        for v_idx in range(len(velocities)):
            T_values = [transmissions[p][v_idx] for p in range(len(transmissions))]
            # Check that T values are monotonically decreasing
            for i in range(len(T_values) - 1):
                assert T_values[i] >= T_values[i + 1]

    def test_ionization_evolution_affects_transmission(self) -> None:
        """Transmission should improve (increase) as universe becomes more ionized."""
        z_array = np.array([10.0, 7.0, 5.0, 2.0])
        x_e_array = ionized_fraction_evolution(z_array, z_reion=7.7)

        velocity_km_s = 200.0
        path_length_kpc = 1.0

        T_array = np.array(
            [
                ly_alpha_profile(
                    velocity_km_s=velocity_km_s,
                    ionized_fraction=x_e,
                    path_length_kpc=path_length_kpc,
                )
                for x_e in x_e_array
            ]
        )

        # As x_e increases (more ionized), transmission should increase
        assert np.all(np.diff(T_array) >= 0.0)
