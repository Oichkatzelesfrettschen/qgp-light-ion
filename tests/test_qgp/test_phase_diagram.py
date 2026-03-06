"""
test_phase_diagram.py - Unit tests for src/phase_diagram/ subpackage.

Tests all public functions from the 6 submodules with physics-motivated assertions:
- crossover.py: lattice QCD crossover line
- critical_point.py: exclusion region and FRG consensus
- first_order.py: first-order transition lines from multiple models
- freeze_out.py: chemical freeze-out parametrization
- trajectories.py: isentropic, early universe, neutron star trajectories
- params.py: dataclass defaults match constants.py
"""

from __future__ import annotations

import numpy as np
import pytest

from qgp.constants import KAPPA2, KAPPA2_ERR, KAPPA4, T_C0_MEV, T_C0_MEV_ERR
from qgp.phase_diagram.critical_point import (
    critical_point_box_excluded,
    critical_point_ellipse_excluded,
    critical_point_exclusion_boundary,
    critical_point_exclusion_region,
    critical_point_frg_box,
    critical_point_frg_ellipse,
)
from qgp.phase_diagram.crossover import crossover_temperature, crossover_uncertainty_band
from qgp.phase_diagram.first_order import (
    first_order_consensus_band,
    first_order_frg,
    first_order_line,
    first_order_njl,
    first_order_pqm,
)
from qgp.phase_diagram.freeze_out import (
    freeze_out_from_sqrt_s,
    freeze_out_parametrization,
    freeze_out_uncertainty_band,
)
from qgp.phase_diagram.params import PhaseTransitionParams
from qgp.phase_diagram.trajectories import (
    color_superconductivity_region,
    cooling_trajectory,
    early_universe_trajectory,
    isentropic_trajectory,
    neutron_star_trajectory,
)

# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

class TestCrossover:
    """Crossover temperature from lattice QCD."""

    def test_T_c_at_zero_mu_B(self):
        """T_c(mu_B=0) must equal T_c0."""
        params = PhaseTransitionParams()
        mu_B = np.array([0.0])
        T = crossover_temperature(mu_B, params)
        assert abs(T[0] - params.T_c0) < 1e-10

    def test_monotonically_decreasing(self):
        """Crossover temperature must decrease with mu_B."""
        params = PhaseTransitionParams()
        mu_B = np.linspace(0, 300, 100)
        T = crossover_temperature(mu_B, params)
        assert np.all(np.diff(T) <= 0)

    def test_uncertainty_band_contains_central(self):
        """Upper and lower bands must bracket the central value."""
        params = PhaseTransitionParams()
        mu_B = np.linspace(0, 250, 50)
        T_central = crossover_temperature(mu_B, params)
        T_upper, T_lower = crossover_uncertainty_band(mu_B, params)
        assert np.all(T_upper >= T_central)
        assert np.all(T_lower <= T_central)

    def test_uncertainty_band_width_increases(self):
        """Uncertainty grows with mu_B (lattice extrapolation becomes less reliable)."""
        params = PhaseTransitionParams()
        mu_B = np.linspace(0, 300, 50)
        T_upper, T_lower = crossover_uncertainty_band(mu_B, params)
        width = T_upper - T_lower
        assert width[-1] > width[0]


# ---------------------------------------------------------------------------
# Critical point
# ---------------------------------------------------------------------------

class TestCriticalPoint:
    """Exclusion region and FRG consensus estimate."""

    def test_exclusion_region_covers_450_MeV(self):
        """Exclusion region must extend to mu_B = 450 MeV."""
        mu_B, _T = critical_point_exclusion_region()
        assert np.max(mu_B) >= 450.0

    def test_exclusion_boundary_at_450_MeV(self):
        """Boundary line is at mu_B = 450 MeV (constant)."""
        mu_B, _T = critical_point_exclusion_boundary()
        assert np.all(np.abs(mu_B - 450.0) < 1e-10)

    def test_frg_ellipse_centered_at_consensus(self):
        """FRG ellipse must be centered at (630, 110) MeV."""
        mu_B, T = critical_point_frg_ellipse()
        mu_center = (np.max(mu_B) + np.min(mu_B)) / 2
        T_center = (np.max(T) + np.min(T)) / 2
        assert abs(mu_center - 630) < 1.0
        assert abs(T_center - 110) < 1.0

    def test_frg_box_contains_center(self):
        """FRG box must contain the central point."""
        mu_min, mu_max, T_min, T_max = critical_point_frg_box()
        assert mu_min < 630 < mu_max
        assert T_min < 110 < T_max

    def test_excluded_ellipse_returns_arrays(self):
        """Deprecated Clarke ellipse still returns valid arrays."""
        mu_B, T = critical_point_ellipse_excluded()
        assert len(mu_B) > 0
        assert len(T) > 0
        assert np.all(np.isfinite(mu_B))

    def test_excluded_box_returns_tuple(self):
        """Deprecated Clarke box returns 4-element tuple."""
        result = critical_point_box_excluded()
        assert len(result) == 4
        mu_min, mu_max, T_min, T_max = result
        assert mu_min < mu_max
        assert T_min < T_max


# ---------------------------------------------------------------------------
# First order
# ---------------------------------------------------------------------------

class TestFirstOrder:
    """First-order phase transition lines from multiple theoretical models."""

    def test_first_order_line_starts_at_CP(self):
        """First-order line T at the CP mu_B should equal T_CP."""
        params = PhaseTransitionParams()
        mu_B = np.array([params.mu_B_cp_frg])
        T = first_order_line(mu_B, params)
        assert abs(T[0] - params.T_cp_frg) < 1e-10

    def test_first_order_line_reaches_zero_T(self):
        """At nuclear matter mu_B=930, T should be near 0."""
        params = PhaseTransitionParams()
        mu_B = np.array([930.0])
        T = first_order_line(mu_B, params)
        assert T[0] == pytest.approx(0.0, abs=1.0)

    def test_first_order_line_nan_below_CP(self):
        """Below the CP, first-order line should be NaN."""
        params = PhaseTransitionParams()
        mu_B = np.array([100.0, 300.0, 500.0])
        T = first_order_line(mu_B, params)
        assert np.all(np.isnan(T))

    def test_njl_returns_finite_above_cp(self):
        """NJL model returns finite T above its CP (~330 MeV)."""
        params = PhaseTransitionParams()
        mu_B = np.array([400.0, 500.0, 600.0])
        T = first_order_njl(mu_B, params)
        assert np.all(np.isfinite(T))

    def test_pqm_non_negative(self):
        """PQM first-order line T must be >= 0."""
        params = PhaseTransitionParams()
        mu_B = np.linspace(370, 930, 100)
        T = first_order_pqm(mu_B, params)
        valid = ~np.isnan(T)
        assert np.all(T[valid] >= 0)

    def test_frg_non_negative(self):
        """FRG first-order line T must be >= 0."""
        params = PhaseTransitionParams()
        mu_B = np.linspace(630, 930, 100)
        T = first_order_frg(mu_B, params)
        valid = ~np.isnan(T)
        assert np.all(T[valid] >= 0)

    def test_consensus_band_returns_valid_data(self):
        """Consensus band must return finite, non-negative arrays."""
        mu_B, T_frg, T_njl = first_order_consensus_band()
        assert len(mu_B) > 0
        assert np.all(np.isfinite(T_frg))
        assert np.all(np.isfinite(T_njl))
        assert np.all(T_frg >= 0)
        assert np.all(T_njl >= 0)
        # FRG starts at the CP (T~110 MeV); NJL starts lower (~60 MeV at mu_B=630)
        assert T_frg[0] > T_njl[0]


# ---------------------------------------------------------------------------
# Freeze-out
# ---------------------------------------------------------------------------

class TestFreezeOut:
    """Chemical freeze-out parametrizations."""

    def test_freeze_out_at_zero_mu_B(self):
        """At mu_B=0, freeze-out T should be near 166 MeV (Andronic 2018)."""
        mu_B = np.array([0.0])
        T = freeze_out_parametrization(mu_B)
        assert 160 < T[0] < 170

    def test_freeze_out_from_sqrt_s_at_LHC(self):
        """At LHC sqrt_s ~ 5020 GeV, mu_B should be near zero."""
        T, mu_B = freeze_out_from_sqrt_s(np.array([5020.0]))
        assert mu_B[0] < 5.0  # Nearly baryon-free
        assert T[0] > 150  # Near T_c

    def test_freeze_out_from_sqrt_s_at_AGS(self):
        """At AGS sqrt_s ~ 5 GeV, mu_B should be large."""
        T, mu_B = freeze_out_from_sqrt_s(np.array([5.0]))
        assert mu_B[0] > 200  # Large baryon density
        assert T[0] < 160

    def test_uncertainty_band_positive_width(self):
        """Uncertainty band must have positive width everywhere."""
        mu_B = np.linspace(0, 500, 50)
        T_upper, T_lower = freeze_out_uncertainty_band(mu_B)
        assert np.all(T_upper > T_lower)

    def test_freeze_out_decreasing(self):
        """Freeze-out temperature decreases with mu_B."""
        mu_B = np.linspace(0, 500, 100)
        T = freeze_out_parametrization(mu_B)
        assert np.all(np.diff(T) <= 0)


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------

class TestTrajectories:
    """Phase diagram trajectories and astrophysical regions."""

    def test_isentropic_bounded(self):
        """Isentropic trajectories must stay within physical bounds."""
        mu_B, T = isentropic_trajectory(s_over_nB=420)
        assert np.all(mu_B >= 0)
        assert np.all(T > 0)
        assert np.all(mu_B < 600)

    def test_isentropic_high_s_lower_mu(self):
        """Higher s/n_B means lower mu_B at same T (more baryon-free)."""
        mu_high, _T_high = isentropic_trajectory(s_over_nB=420)
        mu_low, _T_low = isentropic_trajectory(s_over_nB=51)
        # At equivalent T, high s/n_B should have lower mu_B
        assert np.mean(mu_high) < np.mean(mu_low)

    def test_early_universe_near_zero_mu_B(self):
        """Early universe trajectory has nearly zero baryon chemical potential."""
        mu_B, T = early_universe_trajectory()
        assert np.all(mu_B < 1.0)  # Effectively zero
        assert T[0] > T[-1]  # Starts hot

    def test_neutron_star_high_mu_B(self):
        """Neutron star probes high mu_B, low T."""
        mu_B, T_center, _T_upper, _T_lower = neutron_star_trajectory()
        assert np.all(mu_B > 800)
        assert np.all(T_center < 50)

    def test_cooling_starts_hot(self):
        """Cooling trajectory starts at high T, ends at low T."""
        _mu_B, T = cooling_trajectory()
        assert T[0] > T[-1]
        assert T[0] > 300  # Starts above T_c
        assert T[-1] < 150  # Ends at freeze-out

    def test_color_superconductivity_low_T(self):
        """Color superconductivity region is at low T, high mu_B."""
        mu_B, T_boundary = color_superconductivity_region()
        assert np.all(mu_B >= 500)
        assert np.all(T_boundary < 100)


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

class TestParams:
    """PhaseTransitionParams defaults must match constants.py."""

    def test_T_c0_matches_constants(self):
        params = PhaseTransitionParams()
        assert params.T_c0 == T_C0_MEV

    def test_T_c0_err_matches_constants(self):
        params = PhaseTransitionParams()
        assert params.T_c0_err == T_C0_MEV_ERR

    def test_kappa2_matches_constants(self):
        params = PhaseTransitionParams()
        assert params.kappa2 == KAPPA2

    def test_kappa2_err_matches_constants(self):
        params = PhaseTransitionParams()
        assert params.kappa2_err == KAPPA2_ERR

    def test_kappa4_matches_constants(self):
        params = PhaseTransitionParams()
        assert params.kappa4 == KAPPA4

    def test_frg_consensus_values(self):
        params = PhaseTransitionParams()
        assert params.T_cp_frg == 110.0
        assert params.mu_B_cp_frg == 630.0
