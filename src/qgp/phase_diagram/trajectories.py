"""Trajectories and astrophysical regions on the QCD phase diagram."""

from __future__ import annotations

import numpy as np


def isentropic_trajectory(s_over_nB: float, n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate isentropic trajectory for given entropy/baryon ratio s/n_B.

    Trajectories of constant entropy per baryon characterize the hydrodynamic
    evolution at different collision energies.

    Representative s/n_B values for RHIC BES:
      sqrt_s = 200 GeV -> s/n_B ~ 420
      sqrt_s = 62.4 GeV -> s/n_B ~ 144
      sqrt_s = 27 GeV  -> s/n_B ~ 70
      sqrt_s = 19.6 GeV -> s/n_B ~ 51
      sqrt_s = 7.7 GeV  -> s/n_B ~ 25

    Source: arXiv:1506.07350 (RHIC BES review).
    """
    T = np.linspace(400, 50, n_points)
    mu_B = np.zeros_like(T)

    for i, Ti in enumerate(T):
        if Ti > 170:  # QGP phase
            mu_B[i] = 3 * Ti / s_over_nB * (170 / Ti) ** 0.5
        else:  # Hadron phase -- steeper increase
            mu_B_170 = 3 * 170 / s_over_nB * (170 / 170) ** 0.5
            mu_B[i] = mu_B_170 + (170 - Ti) * (300 / s_over_nB)

    mask = mu_B < 600
    return mu_B[mask], T[mask]


def early_universe_trajectory(n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate early universe cooling trajectory.

    The universe evolved along mu_B ~ 0 from T ~ 10^12 K (QGP epoch)
    down to T ~ 10^10 K (hadronization at t ~ 10^-5 s).
    Baryon asymmetry eta_B ~ 6e-10 makes mu_B effectively negligible.
    """
    T = np.linspace(500, 50, n_points)
    mu_B = np.zeros_like(T) + 0.1  # Tiny offset for plot visibility
    return mu_B, T


def neutron_star_trajectory(
    n_points: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate neutron star core region boundary.

    Neutron stars probe T ~ 0 (cold), high mu_B ~ 1000-1500 MeV.
    Core densities reach 5-10 x nuclear saturation density.
    mu_B at center: ~1200-1500 MeV (model dependent).
    """
    mu_B = np.linspace(900, 1500, n_points)
    T_center = np.zeros_like(mu_B) + 10   # ~10 MeV for plot visibility
    T_upper = T_center + 30
    T_lower = T_center - 5

    return mu_B, T_center, T_upper, T_lower


def color_superconductivity_region() -> tuple[np.ndarray, np.ndarray]:
    """
    Return approximate boundary of color superconducting phase.

    At very high mu_B (> 400-500 MeV) and low T, quarks can form Cooper pairs
    leading to color superconductivity (2SC, CFL, and crystalline phases).
    Gap scale: Delta ~ 10-100 MeV depending on density.
    Boundary is highly uncertain; this is schematic.
    """
    mu_B = np.array([500, 600, 800, 1000, 1200, 1400, 1600])
    T_boundary = np.array([80, 60, 40, 30, 25, 20, 15])  # Approximate
    return mu_B, T_boundary


def cooling_trajectory(n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate cooling trajectory for a central Pb-Pb collision at the LHC.

    Starts at T_0 ~ 400 MeV, mu_B ~ 0; evolves through QGP and crosses T_c.
    Ends at kinetic freeze-out (~100 MeV).
    """
    t = np.linspace(0, 1, n_points)

    T_0 = 400.0  # Initial temperature (MeV)
    T_f = 100.0  # Kinetic freeze-out (MeV)
    T = T_0 - (T_0 - T_f) * (1 - np.exp(-3 * t)) / (1 - np.exp(-3))

    # Very small mu_B throughout (LHC is nearly baryon-free)
    mu_B_0 = 0.5  # MeV
    mu_B_f = 2.0  # MeV
    mu_B = mu_B_0 + (mu_B_f - mu_B_0) * t**0.5

    return mu_B, T
