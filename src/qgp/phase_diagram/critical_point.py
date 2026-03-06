"""Critical point exclusion region and FRG consensus estimates."""

from __future__ import annotations

import numpy as np

from .params import PhaseTransitionParams


def critical_point_exclusion_region(n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the EXCLUDED region for the critical point (December 2025 lattice).

    Borsanyi et al., PRD 112, L111505 (arXiv:2502.10267):
    Yang-Lee edge singularity analysis excludes CP at mu_B < 450 MeV (2sigma).

    Returns coordinates for a shaded exclusion-region polygon.
    """
    params = PhaseTransitionParams()
    mu_B_boundary = params.mu_B_excluded_2sigma  # 450 MeV
    T_max = 200.0

    mu_B = np.array([0, 0, mu_B_boundary, mu_B_boundary, 0])
    T = np.array([0, T_max, T_max, 0, 0])

    return mu_B, T


def critical_point_exclusion_boundary(n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the exclusion boundary line (mu_B = 450 MeV at 2sigma).
    """
    params = PhaseTransitionParams()
    T = np.linspace(0, 200, n_points)
    mu_B = np.full_like(T, params.mu_B_excluded_2sigma)
    return mu_B, T


def critical_point_frg_ellipse(n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate ellipse representing FRG consensus CP uncertainty region.

    QM2025 consensus: CEP at (T=110, mu_B=630) MeV with ~10% uncertainty.
    Source: Fu, Pawlowski, Rennecke, arXiv:2510.11270.
    """
    params = PhaseTransitionParams()
    theta = np.linspace(0, 2 * np.pi, n_points)

    sigma_T = params.T_cp_frg_err    # ~11 MeV
    sigma_mu = params.mu_B_cp_frg_err  # ~63 MeV

    T = params.T_cp_frg + sigma_T * np.sin(theta)
    mu_B = params.mu_B_cp_frg + sigma_mu * np.cos(theta)

    return mu_B, T


def critical_point_frg_box() -> tuple[float, float, float, float]:
    """
    Generate rectangular uncertainty region for FRG consensus CP.

    Returns corners (mu_min, mu_max, T_min, T_max) of the ~10% uncertainty box.
    """
    params = PhaseTransitionParams()

    mu_min = params.mu_B_cp_frg - params.mu_B_cp_frg_err
    mu_max = params.mu_B_cp_frg + params.mu_B_cp_frg_err
    T_min = params.T_cp_frg - params.T_cp_frg_err
    T_max = params.T_cp_frg + params.T_cp_frg_err

    return (mu_min, mu_max, T_min, T_max)


def critical_point_ellipse_excluded(n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    DEPRECATED: Generate ellipse for Clarke et al. 2024 estimate.

    WARNING: This CP location is NOW EXCLUDED by Borsanyi et al. (Dec 2025).
    Kept for reference to show what has been ruled out.
    """
    params = PhaseTransitionParams()
    theta = np.linspace(0, 2 * np.pi, n_points)

    sigma_T = (params.T_cp_clarke_err_up + params.T_cp_clarke_err_down) / 2
    sigma_mu = (params.mu_B_cp_clarke_err_up + params.mu_B_cp_clarke_err_down) / 2

    T = params.T_cp_clarke + sigma_T * np.sin(theta)
    mu_B = params.mu_B_cp_clarke + sigma_mu * np.cos(theta)

    return mu_B, T


def critical_point_box_excluded() -> tuple[float, float, float, float]:
    """
    DEPRECATED: Rectangular region for Clarke et al. estimate -- NOW EXCLUDED.
    """
    params = PhaseTransitionParams()

    mu_min = params.mu_B_cp_clarke - params.mu_B_cp_clarke_err_down
    mu_max = params.mu_B_cp_clarke + params.mu_B_cp_clarke_err_up
    T_min = params.T_cp_clarke - params.T_cp_clarke_err_down
    T_max = params.T_cp_clarke + params.T_cp_clarke_err_up

    return (mu_min, mu_max, T_min, T_max)
