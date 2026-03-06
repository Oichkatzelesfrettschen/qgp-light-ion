"""Chemical freeze-out parametrizations and uncertainty bands."""

from __future__ import annotations

import numpy as np


def freeze_out_parametrization(mu_B: np.ndarray) -> np.ndarray:
    """
    Chemical freeze-out curve from constant epsilon/n = 0.951 GeV criterion.

    T(mu_B) = a - b*mu_B^2 - c*mu_B^4

    Fitted to match experimental freeze-out points from AGS to LHC energies.
    Source: Andronic et al., Nature 561, 321 (2018).
    """
    a = 166.0      # MeV (limiting temperature at high energy)
    b = 0.139e-3   # MeV^-1 (curvature)
    c = 0.053e-9   # MeV^-3 (higher order)

    T = a - b * mu_B**2 - c * mu_B**4
    return np.maximum(T, 30)  # Physical floor


def freeze_out_from_sqrt_s(sqrt_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate freeze-out (T, mu_B) from collision energy using phenomenological fit.

    mu_B(sqrt_s) = c / (1 + d*sqrt_s)  with c=1477 MeV, d=0.343 GeV^-1.

    Source: arXiv:2408.06473.
    """
    c = 1477.0   # MeV
    d = 0.343    # GeV^-1

    mu_B = c / (1 + d * sqrt_s)
    T = freeze_out_parametrization(mu_B)

    return T, mu_B


def freeze_out_uncertainty_band(
    mu_B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate freeze-out systematic uncertainty band.

    Systematic uncertainties from Andronic et al., Nature 561 (2018) 321:
    - Model-dependent uncertainties in hadron resonance gas
    - Feed-down correction uncertainties (~5 MeV at low mu_B, larger at high mu_B)
    """
    T_central = freeze_out_parametrization(mu_B)

    T_sys = 5.0 + 3.0 * (mu_B / 500) ** 2
    model_unc = 8.0 * np.tanh(mu_B / 400)
    total_sys = np.sqrt(T_sys**2 + model_unc**2)

    T_upper = T_central + total_sys
    T_lower = T_central - total_sys

    return T_upper, T_lower
