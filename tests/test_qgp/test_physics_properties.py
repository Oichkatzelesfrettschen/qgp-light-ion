"""Property-based tests for QGP physics functions using Hypothesis.

These tests verify that physics functions satisfy mathematical and physical
invariants across their entire valid input domain. They complement unit tests
by discovering edge cases and boundary conditions that hand-picked test cases
might miss.

All tests use @pytest.mark.fuzz to allow selective execution:
  pytest -m fuzz tests/test_qgp/test_physics_properties.py -v
  pytest -m "not fuzz" tests/ # Skip property tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

# Ensure src/ is on the path
SRC_DIR = Path(__file__).parent.parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cosmology.reionization_fronts import ionized_fraction_evolution
from qgp.physics import NUCLEI, bdmps_energy_loss, raa_model, woods_saxon

# =============================================================================
# Tier 1: QGP Physics Properties
# =============================================================================


@pytest.mark.fuzz
class TestWoodsSaxonProperties:
    """Property tests for Woods-Saxon nuclear density profile."""

    @given(r=st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False))
    def test_output_bounded_0_1(self, r: float) -> None:
        """Woods-Saxon density rho(r) must be in [0, 1] for all r >= 0.

        Justification: rho(r) = 1 / (1 + exp((r - R0) / a)), a logistic function
        that is analytically bounded in [0, 1].
        """
        rho = woods_saxon(np.array([r]), NUCLEI["O"])[0]
        assert 0.0 <= rho <= 1.0, f"rho out of bounds at r={r}: {rho}"

    @given(r1=st.floats(0.0, 4.9), r2=st.floats(5.1, 20.0))
    def test_monotonically_decreasing(self, r1: float, r2: float) -> None:
        """Woods-Saxon density must be monotonically non-increasing with r.

        Justification: Nuclear density decreases with distance from center.
        rtol=1e-7 tolerance (MeV scale) for floating-point rounding.
        """
        rho1 = woods_saxon(np.array([r1]), NUCLEI["O"])[0]
        rho2 = woods_saxon(np.array([r2]), NUCLEI["O"])[0]
        assert rho1 >= rho2 or np.isclose(rho1, rho2, rtol=1e-7), (
            f"Not monotonic at r1={r1}, r2={r2}: rho1={rho1}, rho2={rho2}"
        )

    @given(r=st.floats(0.0, 1.0))
    def test_core_density_near_unity(self, r: float) -> None:
        """Near nuclear center (r < 1 fm), density should be close to 1.

        Justification: Nuclear center has bulk saturation density rho0 ≈ 0.16/fm^3,
        normalized to 1 in the Woods-Saxon convention.
        Tolerance: rho > 0.8 at r < 1 fm.
        """
        rho = woods_saxon(np.array([r]), NUCLEI["O"])[0]
        assert rho > 0.8, f"Core density too low at r={r}: {rho}"


@pytest.mark.fuzz
class TestBdmpsProperties:
    """Property tests for BDMPS-Z radiative energy loss."""

    @given(E=st.floats(1.0, 1000.0), L=st.floats(0.1, 20.0), qhat=st.floats(0.1, 10.0))
    def test_energy_loss_non_negative(self, E: float, L: float, qhat: float) -> None:
        """Energy loss dE >= 0 always.

        Justification: ΔE = min(q̂L²/4, 0.9E) is non-negative by construction
        (both terms are non-negative).
        rtol=0, atol=0: strict non-negativity, no tolerance needed.
        """
        dE = bdmps_energy_loss(E=E, L=L, qhat=qhat)
        assert dE >= 0.0, f"Negative energy loss at E={E}, L={L}, qhat={qhat}: {dE}"

    @given(E=st.floats(1.0, 1000.0), L=st.floats(0.1, 20.0), qhat=st.floats(0.1, 10.0))
    def test_energy_loss_bounded_by_E(self, E: float, L: float, qhat: float) -> None:
        """Energy loss dE <= E (cannot lose more energy than the particle has).

        Justification: ΔE = min(dE_raw, 0.9E) is saturated at 0.9E by definition.
        Allow tiny floating-point overshoot: dE <= E within rtol=1e-7.
        """
        dE = bdmps_energy_loss(E=E, L=L, qhat=qhat)
        assert dE <= E or np.isclose(dE, E, rtol=1e-7), (
            f"Energy loss exceeds particle energy: E={E}, dE={dE}"
        )

    @given(
        L1=st.floats(0.5, 5.0),
        L2=st.floats(5.01, 15.0),
        qhat=st.floats(0.1, 3.0),
    )
    def test_quadratic_L_scaling_unsaturated(
        self, L1: float, L2: float, qhat: float
    ) -> None:
        """In unsaturated regime (E >> dE), energy loss scales as dE ~ q̂L².

        Justification: BDMPS-Z formula gives ΔE = (αs q̂ L²) / (4π) before
        saturation at 0.9E. At large E, saturation is inactive.
        Test with E=50 TeV so dE < 0.9E always.
        rtol=5%: floating-point accumulation in (L2/L1)² and saturation branch
        can contribute small relative error.
        """
        E_large = 50000.0
        dE1 = bdmps_energy_loss(E=E_large, L=L1, qhat=qhat)
        dE2 = bdmps_energy_loss(E=E_large, L=L2, qhat=qhat)

        expected_ratio = (L2 / L1) ** 2
        actual_ratio = dE2 / (dE1 + 1e-30)  # Avoid division by zero
        assert np.isclose(actual_ratio, expected_ratio, rtol=0.05), (
            f"L² scaling violated: L1={L1}, L2={L2}, ratio={actual_ratio}, "
            f"expected={expected_ratio}"
        )

    @given(
        qhat1=st.floats(0.1, 2.0),
        qhat2=st.floats(2.01, 10.0),
        E=st.floats(100.0, 500.0),
        L=st.floats(1.0, 5.0),
    )
    def test_linear_qhat_scaling_unsaturated(
        self, qhat1: float, qhat2: float, E: float, L: float
    ) -> None:
        """In unsaturated regime, energy loss scales linearly with q̂.

        Justification: ΔE = (αs / (4π)) * q̂ * L² (linear in q̂ before saturation).
        rtol=5%: same justification as L² scaling test.
        """
        dE1 = bdmps_energy_loss(E=E, L=L, qhat=qhat1)
        dE2 = bdmps_energy_loss(E=E, L=L, qhat=qhat2)

        expected_ratio = qhat2 / qhat1
        actual_ratio = dE2 / (dE1 + 1e-30)
        assert np.isclose(actual_ratio, expected_ratio, rtol=0.05), (
            f"q̂ scaling violated: qhat1={qhat1}, qhat2={qhat2}, ratio={actual_ratio}, "
            f"expected={expected_ratio}"
        )


@pytest.mark.fuzz
class TestRaaProperties:
    """Property tests for nuclear modification factor R_AA."""

    @given(pT=st.floats(5.0, 100.0), L=st.floats(1.0, 10.0), qhat=st.floats(0.5, 5.0))
    def test_raa_in_0_1(self, pT: float, L: float, qhat: float) -> None:
        """R_AA = (pT - ΔE) / pT must be in (0, 1].

        Justification: 0 < ΔE < 0.9*pT by construction, so 0.1 < R_AA < 1.
        Allow rtol=1e-7 overshoot at upper bound from floating-point rounding.
        """
        raa = raa_model(np.array([pT]), L_eff=L, qhat=qhat)[0]
        assert raa > 0.0, f"R_AA not positive: pT={pT}, L={L}, qhat={qhat}, R_AA={raa}"
        assert raa <= 1.0 or np.isclose(raa, 1.0, rtol=1e-7), (
            f"R_AA exceeds 1: pT={pT}, L={L}, qhat={qhat}, R_AA={raa}"
        )

    @given(
        L1=st.floats(1.0, 4.9),
        L2=st.floats(5.0, 10.0),
        pT=st.floats(10.0, 50.0),
        qhat=st.floats(1.0, 3.0),
    )
    def test_raa_decreases_with_path_length(
        self, L1: float, L2: float, pT: float, qhat: float
    ) -> None:
        """More traversed medium → more suppression → lower R_AA.

        Justification: Longer path L → larger ΔE → smaller (pT - ΔE) → smaller R_AA.
        rtol=1e-7 tolerance for floating-point rounding near saturation.
        """
        raa1 = raa_model(np.array([pT]), L_eff=L1, qhat=qhat)[0]
        raa2 = raa_model(np.array([pT]), L_eff=L2, qhat=qhat)[0]
        assert raa1 >= raa2 or np.isclose(raa1, raa2, rtol=1e-7), (
            f"R_AA not decreasing with L: L1={L1}, L2={L2}, raa1={raa1}, raa2={raa2}"
        )

@pytest.mark.fuzz
class TestRaaScalingProperties:
    """Additional property tests for R_AA scaling laws."""

    @given(
        pT_base=st.floats(10.0, 50.0),
        L1=st.floats(1.0, 5.0),
        L2=st.floats(5.1, 10.0),
        qhat=st.floats(0.5, 3.0),
    )
    def test_raa_monotonic_in_path_length(
        self, pT_base: float, L1: float, L2: float, qhat: float
    ) -> None:
        """R_AA should decrease with increasing path length L.

        Justification: Longer path → larger energy loss → smaller R_AA.
        This is a monotonicity property across the domain.
        """
        raa1 = raa_model(np.array([pT_base]), L_eff=L1, qhat=qhat)[0]
        raa2 = raa_model(np.array([pT_base]), L_eff=L2, qhat=qhat)[0]
        assert raa1 >= raa2 or np.isclose(raa1, raa2, rtol=1e-7), (
            f"R_AA not monotonic in L: raa1={raa1} (L1={L1}), "
            f"raa2={raa2} (L2={L2})"
        )


# =============================================================================
# Tier 2: Cosmology Properties
# =============================================================================


@pytest.mark.fuzz
class TestIonizedFractionProperties:
    """Property tests for cosmic reionization ionized fraction evolution."""

    @given(z=st.floats(0.0, 30.0, allow_nan=False))
    def test_x_e_bounded_0_1(self, z: float) -> None:
        """Ionized fraction x_e(z) must be in [0, 1] for all z.

        Justification: x_e = 0.5 * (1 - erf(...)) is analytically in [0, 1]
        (erf is bounded in [-1, 1]).
        atol=1e-12: absolute tolerance for erf floating-point rounding.
        """
        x_e = ionized_fraction_evolution(np.array([z]))[0]
        assert x_e >= 0.0 or np.isclose(x_e, 0.0, atol=1e-12), (
            f"x_e < 0 at z={z}: {x_e}"
        )
        assert x_e <= 1.0 or np.isclose(x_e, 1.0, atol=1e-12), (
            f"x_e > 1 at z={z}: {x_e}"
        )

    @given(z1=st.floats(8.0, 20.0), z2=st.floats(4.0, 7.9))
    def test_higher_z_less_ionized(self, z1: float, z2: float) -> None:
        """Ionized fraction decreases with increasing z (earlier universe less ionized).

        Justification: Early universe (high z) has low ionization; reionization
        proceeds as universe cools and z decreases. x_e(z) is non-increasing.
        rtol=1e-9: erf-based model with floating-point accumulation.
        """
        x_e_high_z = ionized_fraction_evolution(np.array([z1]))[0]
        x_e_low_z = ionized_fraction_evolution(np.array([z2]))[0]
        assert x_e_high_z <= x_e_low_z or np.isclose(x_e_high_z, x_e_low_z, rtol=1e-9), (
            f"x_e not decreasing with z: z1={z1}, z2={z2}, "
            f"x_e(z1)={x_e_high_z}, x_e(z2)={x_e_low_z}"
        )
