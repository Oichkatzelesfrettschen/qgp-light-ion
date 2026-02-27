"""
test_qgp_physics.py - Unit tests for src/qgp_physics.py.

Covers every public function with physics-motivated assertions:
- Woods-Saxon density profile
- Glauber geometry (eccentricities, N_part)
- QCD crossover line
- Flow from eccentricity and centrality
- BDMPS-Z energy loss
- R_AA model
- Canonical strangeness suppression
- Temperature evolution
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from constants import KAPPA2, T_C0_GEV

# conftest.py inserts src/ into sys.path.
from qgp_physics import (
    NUCLEI,
    azimuthal_distribution,
    bdmps_energy_loss,
    bjorken_energy_density,
    calculate_eccentricities,
    canonical_suppression_factor,
    flow_from_eccentricity,
    generate_flow_vs_centrality,
    generate_raa_data,
    oxygen_alpha_cluster_positions,
    qcd_crossover_line,
    qcd_phase_boundaries,
    raa_model,
    sample_nucleon_positions,
    strangeness_enhancement_curve,
    temperature_evolution,
    woods_saxon,
)

# ---------------------------------------------------------------------------
# Woods-Saxon density profile
# ---------------------------------------------------------------------------

class TestWoodsSaxon:
    """Woods-Saxon nuclear density must satisfy basic physical properties."""

    def test_center_is_maximum(self):
        """Density at r=0 should exceed density at r=R0."""
        nucleus = NUCLEI["O"]
        rho_center = woods_saxon(np.array([0.0]), nucleus)[0]
        rho_surface = woods_saxon(np.array([nucleus.R0]), nucleus)[0]
        assert rho_center > rho_surface

    def test_half_density_at_R0(self):
        """At r = R0, the Woods-Saxon profile gives rho_0 / (1+e^0) = rho_0/2."""
        nucleus = NUCLEI["O"]
        rho_0 = woods_saxon(np.array([0.0]), nucleus)[0]
        rho_R0 = woods_saxon(np.array([nucleus.R0]), nucleus)[0]
        # rho(R0) / rho(0) should be close to 1/(1+1) = 0.5 for a=0
        # With finite a it deviates, but should be in (0.3, 0.7)
        ratio = rho_R0 / rho_0
        assert 0.3 < ratio < 0.7, f"rho(R0)/rho(0)={ratio:.3f} not near 0.5"

    def test_density_non_negative(self):
        r = np.linspace(0, 15, 200)
        nucleus = NUCLEI["O"]
        rho = woods_saxon(r, nucleus)
        assert np.all(rho >= 0)

    def test_density_decreasing(self):
        """Density must be monotonically decreasing with r (no deformation)."""
        r = np.linspace(0, 12, 100)
        nucleus = NUCLEI["O"]
        rho = woods_saxon(r, nucleus)
        assert np.all(np.diff(rho) <= 1e-12), "Woods-Saxon density not monotonically decreasing"

    def test_pb208_broader_than_o16(self):
        """Pb-208 has larger radius; density falls off more slowly."""
        r = np.array([5.0])
        rho_O = woods_saxon(r, NUCLEI["O"])[0]
        rho_Pb = woods_saxon(r, NUCLEI["Pb"])[0]
        assert rho_Pb > rho_O, "Pb-208 should have higher density at 5 fm than O-16"


# ---------------------------------------------------------------------------
# Nucleon sampling
# ---------------------------------------------------------------------------

class TestSampleNucleons:
    """sample_nucleon_positions must return physically reasonable configurations."""

    def test_output_shape(self):
        nucleus = NUCLEI["O"]
        positions = sample_nucleon_positions(nucleus, n_events=1)
        # Shape: (n_events, A, 3) - squeeze last dim for single event
        positions = np.squeeze(positions, axis=0) if positions.ndim == 3 else positions
        assert positions.shape == (nucleus.A, 3), f"Unexpected shape: {positions.shape}"

    def test_nucleons_inside_nucleus(self):
        """Most nucleons should be within 4*R0 of center (generous bound for rejection sampling)."""
        nucleus = NUCLEI["O"]
        positions = sample_nucleon_positions(nucleus, n_events=1)
        positions = np.squeeze(positions, axis=0) if positions.ndim == 3 else positions
        r = np.sqrt(np.sum(positions**2, axis=1))
        assert np.all(r < 4 * nucleus.R0), (
            f"Nucleon at r={r.max():.2f} fm > 4*R0={4*nucleus.R0:.2f} fm"
        )


# ---------------------------------------------------------------------------
# Eccentricities
# ---------------------------------------------------------------------------

class TestEccentricities:
    """Spatial eccentricities must satisfy basic bounds."""

    def test_output_keys(self):
        positions = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                               [0, 0, 1], [0, 0, -1], [0.5, 0.5, 0], [-0.5, -0.5, 0]],
                              dtype=float)
        result = calculate_eccentricities(positions)
        for n in [2, 3, 4, 5]:
            assert f"epsilon_{n}" in result

    def test_eccentricities_bounded(self):
        """epsilon_n must be in [0, 1]."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(-3, 3, size=(16, 3))
        result = calculate_eccentricities(positions)
        for n in [2, 3, 4, 5]:
            eps = result[f"epsilon_{n}"]
            assert 0 <= eps <= 1.0, f"epsilon_{n}={eps:.3f} outside [0, 1]"

    def test_symmetric_arrangement_low_eccentricity(self):
        """Nucleons on a circle should yield low eccentricity."""
        theta = np.linspace(0, 2 * math.pi, 16, endpoint=False)
        positions = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(16)])
        result = calculate_eccentricities(positions)
        # epsilon_2 for a ring of 16 equally spaced points is very close to 0
        assert result["epsilon_2"] < 0.05, f"epsilon_2={result['epsilon_2']:.4f} for ring"


# ---------------------------------------------------------------------------
# QCD crossover line
# ---------------------------------------------------------------------------

class TestQcdCrossoverLine:
    """QCD crossover temperature must match lattice QCD parametrization."""

    def test_T_at_zero_mu(self):
        """T_c(mu_B=0) must equal T_C0_GEV."""
        T = qcd_crossover_line(np.array([0.0]))[0]
        assert abs(T - T_C0_GEV) < 1e-10, f"T_c(0)={T:.6f} != T_C0_GEV={T_C0_GEV}"

    def test_monotonically_decreasing(self):
        """Crossover temperature decreases with mu_B."""
        mu_B = np.linspace(0, 0.3, 50)
        T = qcd_crossover_line(mu_B)
        assert np.all(np.diff(T) <= 0), "Crossover line not monotonically decreasing"

    def test_kappa2_curvature(self):
        """At mu_B = T_c0, T_c should decrease by kappa2 (to first order)."""
        # T_c(mu_B=T_c0) / T_c(0) = 1 - kappa2*(1)^2 - ...
        T0 = T_C0_GEV
        T_at_Tc0 = qcd_crossover_line(np.array([T0]))[0]
        ratio = T_at_Tc0 / T0
        expected = 1 - KAPPA2  # leading order
        assert abs(ratio - expected) < 0.001, f"Curvature ratio={ratio:.4f}, expected~{expected:.4f}"

    def test_phase_boundaries_dict(self):
        """qcd_phase_boundaries() must return required keys."""
        result = qcd_phase_boundaries()
        for key in ("mu_B_crossover", "T_crossover", "mu_B_critical", "T_critical"):
            assert key in result, f"Key {key} missing from qcd_phase_boundaries()"

    def test_phase_boundaries_critical_T_consistent(self):
        """Critical point temperature must match crossover line at critical mu_B."""
        result = qcd_phase_boundaries()
        mu_B_crit = result["mu_B_critical"]
        T_crit_expected = qcd_crossover_line(np.array([mu_B_crit]))[0]
        assert abs(result["T_critical"] - T_crit_expected) < 1e-8


# ---------------------------------------------------------------------------
# Hydrodynamic flow
# ---------------------------------------------------------------------------

class TestFlow:
    """Flow response must be physical."""

    def test_flow_positive(self):
        """v_n must be non-negative for positive eccentricity."""
        vn = flow_from_eccentricity(0.3, n=2)
        assert vn >= 0

    def test_flow_zero_for_zero_eccentricity(self):
        assert flow_from_eccentricity(0.0, n=2) == pytest.approx(0.0)

    def test_flow_hierarchy(self):
        """v_2 response coefficient > v_3 > v_4 for same eccentricity."""
        eps = 0.3
        v2 = flow_from_eccentricity(eps, n=2)
        v3 = flow_from_eccentricity(eps, n=3)
        v4 = flow_from_eccentricity(eps, n=4)
        assert v2 > v3 > v4, f"Expected v2>v3>v4, got {v2:.4f},{v3:.4f},{v4:.4f}"

    def test_generate_flow_shape(self):
        """generate_flow_vs_centrality must return correct array shapes."""
        nucleus = NUCLEI["O"]
        centrality = np.array([5, 10, 20, 30, 40, 50, 60])
        result = generate_flow_vs_centrality(nucleus, centrality)
        for key in ("v2", "v3", "v4"):
            assert key in result, f"Key {key} missing"
            assert len(result[key]) == len(centrality)

    def test_generate_flow_bounded(self):
        """All v_n values must satisfy |v_n| < 0.5."""
        nucleus = NUCLEI["O"]
        centrality = np.linspace(0, 80, 20)
        result = generate_flow_vs_centrality(nucleus, centrality)
        for key in ("v2", "v3", "v4"):
            assert np.all(np.abs(result[key]) < 0.5), (
                f"{key} exceeds 0.5: max={np.abs(result[key]).max():.3f}"
            )

    def test_azimuthal_distribution_normalized(self):
        """Azimuthal distribution should be positive and finite."""
        phi = np.linspace(0, 2 * math.pi, 1000)
        dN = azimuthal_distribution(phi, v2=0.05, v3=0.02)
        integral = np.trapezoid(dN, phi)
        assert integral > 0
        assert np.isfinite(integral)

    def test_azimuthal_distribution_no_v2_isotropic(self):
        """With v2=v3=0, azimuthal distribution should be uniform."""
        phi = np.linspace(0, 2 * math.pi, 100)
        dN = azimuthal_distribution(phi, v2=0.0, v3=0.0)
        std_relative = np.std(dN) / np.mean(dN)
        assert std_relative < 0.01, f"dN/dphi not uniform for v2=v3=0 (rel_std={std_relative:.4f})"


# ---------------------------------------------------------------------------
# BDMPS-Z energy loss
# ---------------------------------------------------------------------------

class TestBdmpsEnergyLoss:
    """BDMPS-Z energy loss must satisfy physical constraints."""

    def test_positive_energy_loss(self):
        assert bdmps_energy_loss(E=50.0, L=5.0, qhat=2.0) > 0

    def test_cannot_exceed_input_energy(self):
        """Delta_E must be less than the initial parton energy E."""
        for E in (5.0, 20.0, 100.0):
            delta_E = bdmps_energy_loss(E=E, L=8.0, qhat=5.0)
            assert delta_E < E, f"Delta_E={delta_E:.3f} >= E={E}"

    def test_l_squared_scaling(self):
        """Energy loss scales as L^2 (BDMPS-Z leading term)."""
        E = 100.0
        qhat = 1.0
        # Use short L to avoid energy cap
        L1, L2 = 1.0, 2.0
        dE1 = bdmps_energy_loss(E=E, L=L1, qhat=qhat)
        dE2 = bdmps_energy_loss(E=E, L=L2, qhat=qhat)
        # If both are below the 0.9*E cap, ratio should be 4
        if dE2 < 0.85 * E:
            ratio = dE2 / dE1
            assert abs(ratio - 4.0) < 0.5, f"L^2 scaling: ratio={ratio:.2f}, expected 4"

    def test_zero_path_length(self):
        """Zero path length means no energy loss."""
        delta_E = bdmps_energy_loss(E=50.0, L=0.0, qhat=2.0)
        assert delta_E == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# R_AA model
# ---------------------------------------------------------------------------

class TestRaaModel:
    """R_AA must satisfy physical constraints."""

    def test_raa_positive(self):
        pT = np.logspace(0, 2, 30)
        R_AA = raa_model(pT, L_eff=5.0, qhat=2.0)
        assert np.all(R_AA > 0)

    def test_raa_bounded(self):
        pT = np.logspace(0, 2, 30)
        R_AA = raa_model(pT, L_eff=5.0, qhat=2.0)
        assert np.all(R_AA <= 1.2)

    def test_generate_raa_oo_minimum(self):
        """O-O R_AA minimum should be near 0.69 (CMS arXiv:2510.09864)."""
        result = generate_raa_data("OO")
        assert "pT" in result and "R_AA" in result
        raa_min = result["R_AA"].min()
        assert 0.55 <= raa_min <= 0.85, (
            f"O-O R_AA min={raa_min:.3f} outside expected [0.55, 0.85]"
        )

    def test_generate_raa_pp_near_unity(self):
        """pp R_AA should be close to 1 (no medium)."""
        result = generate_raa_data("pp")
        # Most values should be close to 1
        assert np.mean(result["R_AA"]) > 0.9


# ---------------------------------------------------------------------------
# Canonical strangeness suppression
# ---------------------------------------------------------------------------

class TestStrangeness:
    """Canonical suppression must approach 1 in the thermodynamic limit."""

    def test_suppression_bounded(self):
        """Suppression factor must be in [0, 1]."""
        for s in [1, 2, 3]:
            for x in [0.1, 1.0, 5.0, 20.0]:
                gamma = canonical_suppression_factor(s, x)
                assert 0 <= gamma <= 1.0, f"suppression({s},{x})={gamma:.4f} not in [0,1]"

    def test_grand_canonical_limit(self):
        """At large x (system size), suppression factor approaches 1."""
        for s in [1, 2, 3]:
            gamma_large = canonical_suppression_factor(s, 50.0)
            assert gamma_large > 0.85, (
                f"gamma_s({s}, x=50)={gamma_large:.4f} should be close to 1"
            )

    def test_canonical_suppression_small_system(self):
        """In small systems (x small), suppression factor < 1."""
        gamma_small = canonical_suppression_factor(3, 0.5)
        assert gamma_small < 0.5, (
            f"gamma_s(3, 0.5)={gamma_small:.4f} should be significantly < 1"
        )

    def test_strangeness_enhancement_curve_shape(self):
        """Enhancement factor must increase monotonically with multiplicity."""
        dNch = np.linspace(1, 100, 50)
        result = strangeness_enhancement_curve(dNch)
        # strangeness_enhancement_curve returns keys like enhancement_K, _Lambda, _Xi, _Omega
        assert "enhancement_K" in result, f"Keys: {list(result.keys())}"
        enh = result["enhancement_K"]
        assert enh[-1] > enh[0], "K enhancement should increase with multiplicity"
        assert np.all(np.array(enh) >= 0)


# ---------------------------------------------------------------------------
# Temperature evolution
# ---------------------------------------------------------------------------

class TestTemperatureEvolution:
    """Bjorken cooling: T ∝ tau^(-1/3)."""

    def test_temperature_decreasing(self):
        tau = np.linspace(1, 20, 100)
        T = temperature_evolution(tau, T_0=0.5, tau_0=1.0)
        assert np.all(np.diff(T) <= 0), "Temperature should decrease with tau"

    def test_initial_temperature(self):
        """T(tau_0) must equal T_0."""
        T_0 = 0.5
        tau_0 = 1.0
        T = temperature_evolution(np.array([tau_0]), T_0=T_0, tau_0=tau_0)
        assert abs(T[0] - T_0) < 1e-10

    def test_bjorken_scaling(self):
        """T(2*tau_0) / T(tau_0) = 2^(-1/3) for Bjorken cooling."""
        T_0 = 0.5
        tau_0 = 1.0
        T_at_tau0 = temperature_evolution(np.array([tau_0]), T_0=T_0, tau_0=tau_0)[0]
        T_at_2tau0 = temperature_evolution(np.array([2 * tau_0]), T_0=T_0, tau_0=tau_0)[0]
        ratio = T_at_2tau0 / T_at_tau0
        expected = 2 ** (-1.0 / 3.0)
        assert abs(ratio - expected) < 0.01, f"Bjorken ratio={ratio:.4f}, expected {expected:.4f}"

    def test_temperature_positive(self):
        tau = np.linspace(0.5, 50, 200)
        T = temperature_evolution(tau, T_0=0.4, tau_0=0.5)
        assert np.all(T > 0)


# ---------------------------------------------------------------------------
# Oxygen alpha-cluster positions
# ---------------------------------------------------------------------------

class TestOxygenAlphaClusters:
    """oxygen_alpha_cluster_positions must return 4 alpha clusters."""

    def test_returns_16_nucleons(self):
        positions = oxygen_alpha_cluster_positions()
        assert positions.shape == (16, 3), f"Expected (16, 3), got {positions.shape}"

    def test_positions_finite(self):
        positions = oxygen_alpha_cluster_positions()
        assert np.all(np.isfinite(positions))

    def test_positions_within_nucleus(self):
        """All nucleons should be within reasonable radius (~5 fm for O-16)."""
        positions = oxygen_alpha_cluster_positions()
        r = np.sqrt(np.sum(positions**2, axis=1))
        assert np.all(r < 10.0), f"Nucleon at r={r.max():.2f} fm seems too far"


# ---------------------------------------------------------------------------
# Bjorken energy density
# ---------------------------------------------------------------------------

class TestBjorkenEnergyDensity:
    """Bjorken formula sanity check."""

    def test_positive(self):
        eps = bjorken_energy_density(dET_dy=1000, A_perp=50, tau_0=1.0)
        assert eps > 0

    def test_scales_with_dET_dy(self):
        """Doubling dET/dy should double energy density."""
        eps1 = bjorken_energy_density(dET_dy=1000, A_perp=50, tau_0=1.0)
        eps2 = bjorken_energy_density(dET_dy=2000, A_perp=50, tau_0=1.0)
        assert abs(eps2 / eps1 - 2.0) < 1e-10
