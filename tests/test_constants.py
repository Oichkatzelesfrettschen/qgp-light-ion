"""
test_constants.py - Unit tests for src/constants.py.

Validates:
- Physical consistency of constant values
- Internal unit coherence (MeV/GeV conversions)
- Literature bounds for key QCD observables
"""

from __future__ import annotations

import math

# conftest.py sets up sys.path; import directly.
from constants import (
    ALPHA_S,
    ETA_OVER_S,
    FLOW_KAPPA,
    HBARC,
    KAPPA2,
    KAPPA2_ERR,
    KAPPA2_MESONIC,
    KAPPA4,
    NE20_BETA2,
    O16_A,
    O16_R0,
    QHAT_0,
    SIGMA_NN_FM2,
    T_C0_GEV,
    T_C0_MEV,
    T_C0_MEV_ERR,
)

# ---------------------------------------------------------------------------
# Temperature: MeV/GeV consistency
# ---------------------------------------------------------------------------

class TestTemperatureConstants:
    """Verify crossover temperature values and units."""

    def test_T_c0_mev_gev_consistent(self):
        """T_C0_MEV and T_C0_GEV must agree within 0.01 MeV."""
        assert abs(T_C0_MEV - T_C0_GEV * 1000) < 0.01, (
            f"T_C0_MEV={T_C0_MEV} != T_C0_GEV*1000={T_C0_GEV * 1000}"
        )

    def test_T_c0_in_literature_range(self):
        """T_c0 should be in [154, 159] MeV (HotQCD PLB 795 range +2 sigma)."""
        assert 154.0 <= T_C0_MEV <= 159.0, (
            f"T_C0_MEV={T_C0_MEV} outside [154, 159] MeV"
        )

    def test_T_c0_error_positive(self):
        assert T_C0_MEV_ERR > 0

    def test_T_c0_error_reasonable(self):
        """Uncertainty should be less than 3 MeV (current lattice precision)."""
        assert T_C0_MEV_ERR < 3.0


# ---------------------------------------------------------------------------
# Crossover curvature kappa2
# ---------------------------------------------------------------------------

class TestKappa2:
    """Validate kappa2 crossover curvature against literature bounds."""

    def test_kappa2_in_literature_range(self):
        """kappa2 must be in [0.008, 0.020] covering all lattice results."""
        assert 0.008 <= KAPPA2 <= 0.020, f"KAPPA2={KAPPA2} outside [0.008, 0.020]"

    def test_kappa2_primary_value(self):
        """Primary value should be near HotQCD central value 0.012."""
        assert abs(KAPPA2 - 0.012) < 1e-10, f"KAPPA2={KAPPA2}, expected 0.012"

    def test_kappa2_err_positive(self):
        assert KAPPA2_ERR > 0

    def test_kappa2_mesonic_in_range(self):
        """Mesonic correlator extraction should be consistent with combined range."""
        assert 0.008 <= KAPPA2_MESONIC <= 0.020

    def test_kappa4_small_and_positive(self):
        """kappa4 should be small and non-negative."""
        assert KAPPA4 >= 0
        assert KAPPA4 < 0.001, f"KAPPA4={KAPPA4} seems too large"


# ---------------------------------------------------------------------------
# KSS bound: eta/s >= 1/(4*pi)
# ---------------------------------------------------------------------------

class TestTransportCoefficients:
    """QGP transport coefficients must respect theoretical bounds."""

    KSS_BOUND = 1.0 / (4 * math.pi)

    def test_eta_over_s_above_kss_bound(self):
        assert ETA_OVER_S >= self.KSS_BOUND, (
            f"ETA_OVER_S={ETA_OVER_S:.4f} < KSS bound {self.KSS_BOUND:.4f}"
        )

    def test_eta_over_s_not_too_large(self):
        """QGP is a strongly coupled fluid; eta/s < 3x KSS is typical."""
        assert ETA_OVER_S <= 3 * self.KSS_BOUND, (
            f"ETA_OVER_S={ETA_OVER_S:.4f} > 3x KSS (weakly coupled regime)"
        )

    def test_qhat_positive(self):
        assert QHAT_0 > 0

    def test_alpha_s_in_range(self):
        """Strong coupling at typical scale should be in (0.1, 0.5)."""
        assert 0.1 < ALPHA_S < 0.5, f"ALPHA_S={ALPHA_S} outside (0.1, 0.5)"


# ---------------------------------------------------------------------------
# Flow response coefficients
# ---------------------------------------------------------------------------

class TestFlowKappa:
    """Validate flow response coefficient dictionary."""

    def test_flow_kappa_has_required_harmonics(self):
        assert set(FLOW_KAPPA.keys()) == {2, 3, 4, 5}, (
            f"FLOW_KAPPA keys: {set(FLOW_KAPPA.keys())}, expected {{2, 3, 4, 5}}"
        )

    def test_flow_kappa_values_positive(self):
        for n, kappa in FLOW_KAPPA.items():
            assert kappa > 0, f"FLOW_KAPPA[{n}]={kappa} is not positive"

    def test_flow_kappa_hierarchy(self):
        """Response decreases with harmonic order: kappa_2 > kappa_3 > kappa_4."""
        assert FLOW_KAPPA[2] > FLOW_KAPPA[3] > FLOW_KAPPA[4], (
            f"Expected kappa_2 > kappa_3 > kappa_4, got "
            f"{FLOW_KAPPA[2]} > {FLOW_KAPPA[3]} > {FLOW_KAPPA[4]}"
        )


# ---------------------------------------------------------------------------
# Nuclear geometry
# ---------------------------------------------------------------------------

class TestNuclearConstants:
    """Validate nuclear radius and deformation parameters."""

    def test_o16_radius_in_range(self):
        """O-16 radius should be near 2.608 fm (arXiv:2507.05853)."""
        assert 2.4 <= O16_R0 <= 2.8, f"O16_R0={O16_R0} fm outside [2.4, 2.8]"

    def test_o16_skin_positive(self):
        assert O16_A > 0

    def test_ne20_deformation_positive(self):
        """Ne-20 is prolate; beta2 should be positive and < 1."""
        assert 0 < NE20_BETA2 < 1.0, f"NE20_BETA2={NE20_BETA2} not in (0, 1)"

    def test_ne20_deformation_consistent_with_atlas(self):
        """ATLAS measurement gives beta2 ~ 0.45 (arXiv:2509.05171)."""
        assert 0.35 <= NE20_BETA2 <= 0.55, (
            f"NE20_BETA2={NE20_BETA2} outside ATLAS-consistent range [0.35, 0.55]"
        )


# ---------------------------------------------------------------------------
# Cross-section and units
# ---------------------------------------------------------------------------

class TestUnits:
    """Verify unit conversion constants."""

    def test_hbarc_value(self):
        """hbar*c ~ 0.197 GeV*fm (standard value used in the codebase)."""
        assert abs(HBARC - 0.197) < 0.001, f"HBARC={HBARC}"

    def test_sigma_nn_in_fm2_range(self):
        """inelastic NN cross-section at LHC energies ~ 7 fm^2."""
        assert 5.0 <= SIGMA_NN_FM2 <= 10.0, (
            f"SIGMA_NN_FM2={SIGMA_NN_FM2} fm^2 outside [5, 10]"
        )

    def test_o16_skin_depth_in_range(self):
        """O16_A is the Woods-Saxon skin depth 'a' parameter; expected ~0.5 fm."""
        assert 0.3 <= O16_A <= 0.8, f"O16_A (skin depth)={O16_A} fm outside [0.3, 0.8]"
