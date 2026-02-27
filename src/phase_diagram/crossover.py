"""Crossover line from lattice QCD (HotQCD 2019)."""

from __future__ import annotations

import numpy as np

from phase_diagram.params import PhaseTransitionParams


def crossover_temperature(mu_B: np.ndarray, params: PhaseTransitionParams) -> np.ndarray:
    """
    Calculate crossover temperature T_c(mu_B) from lattice QCD parametrization.

    T_c(mu_B)/T_c(0) = 1 - kappa2*(mu_B/T_c(0))^2 - kappa4*(mu_B/T_c(0))^4

    Valid for mu_B < 300 MeV (lattice constraint).
    Source: HotQCD Collaboration, Phys. Lett. B 795 (2019) 15.
    """
    ratio = mu_B / params.T_c0
    return params.T_c0 * (1 - params.kappa2 * ratio**2 - params.kappa4 * ratio**4)


def crossover_uncertainty_band(
    mu_B: np.ndarray, params: PhaseTransitionParams
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate uncertainty band for crossover using error propagation.

    delta_T^2 = (dT/dT_c0)^2 * delta_T_c0^2 + (dT/dkappa2)^2 * delta_kappa2^2
    """
    ratio = mu_B / params.T_c0

    T_central = crossover_temperature(mu_B, params)

    dT_dTc0 = 1 - params.kappa2 * ratio**2 + params.kappa2 * 2 * ratio**2
    dT_dkappa2 = -params.T_c0 * ratio**2

    delta_T = np.sqrt(
        (dT_dTc0 * params.T_c0_err) ** 2 + (dT_dkappa2 * params.kappa2_err) ** 2
    )

    # Systematic ~5 MeV at higher mu_B for model uncertainty
    systematic = 5.0 * (mu_B / 300) ** 2
    total_err = np.sqrt(delta_T**2 + systematic**2)

    T_upper = T_central + total_err
    T_lower = T_central - total_err

    return T_upper, T_lower
