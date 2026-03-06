"""
Reionization bubble dynamics and QGP analogy.

This module models ionized bubble growth during cosmic reionization and draws
parallels with QGP bubble nucleation in heavy-ion collisions.

Physical basis:
- Reionization: UV photons from galaxies ionize neutral hydrogen (z~6)
- Bubble growth: Individual ionized bubbles expand and overlap (percolation)
- QGP analogy: Similar to deconfinement bubbles nucleating in cooling QCD matter
- Overlap transition: Ionization fraction changes sharply (~0.5) at overlap
- JWST observations: Ionization fronts at z~7-8 with measured sizes

References:
- Gnedin, ApJ 535 (2000) L75 (percolation theory application)
- Gnedin & Ostriker, ApJ 486 (1997) 581 (bubble dynamics)
- Furlanetto et al., Phys.Reps. 433 (2006) 181 (comprehensive review)
- Planck 2018 XIII (arXiv:1807.06209) ionization history constraints
- JWST ERG (arXiv:2310.02468) direct z>7 reionization observations
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import erf
from numpy.typing import NDArray

__all__ = [
    "ReionizationBubble",
    "bubble_growth_rate",
    "overlap_probability",
]


class ReionizationBubble:
    """Ionized bubble dynamics during cosmic reionization."""

    def __init__(
        self,
        redshift: float,
        expansion_rate_H0: float = 67.4,  # Planck 2018 (km/s/Mpc)
        ionization_fraction: float = 0.5,
    ) -> None:
        """
        Initialize reionization bubble at given redshift.

        Parameters
        ----------
        redshift : float
            Redshift z (typical range 5-8 during reionization)
        expansion_rate_H0 : float, optional
            Hubble constant H0 (km/s/Mpc), default 67.4 (Planck 2018)
        ionization_fraction : float, optional
            Global ionization fraction (0 to 1), default 0.5
        """
        self.z = redshift
        self.H0_kmsMpc = expansion_rate_H0
        self.x_e = ionization_fraction

        # Age of universe at redshift z (approximate formula)
        # Age ~ 2/(3*H0) * (1+z)^{-3/2} for Λ CDM
        # More precise: numerical integration (here use approximation)
        self.H0_inv_sec = 3.086e17 / expansion_rate_H0  # 1/H0 in seconds
        self.age_universe_sec = (2.0 / 3.0) * self.H0_inv_sec / (1.0 + redshift) ** 1.5

    def growth_timescale(self, bubble_size_cMpc: float) -> float:
        """
        Dynamical growth timescale for ionized bubble.

        Uses the recombination timescale argument: ionization front slows as
        it advances through decreasing density, and bubble size relates to
        the ionized volume fraction via overlap.

        Parameters
        ----------
        bubble_size_cMpc : float
            Comoving bubble radius (cMpc)

        Returns
        -------
        float
            Growth timescale (Myr)
        """
        # Recombination timescale: t_rec ~ 1 / (alpha * n_e)
        # n_e ~ 10^{-3} to 10^{-4} cm^{-3} at z=6
        # alpha ~ 2e-13 cm^3 s^{-1} at T=10^4 K
        # Typical: t_rec ~ 10-50 Myr at z=6

        # Growth rate suppressed by recombination: dR/dt ~ c / (1 + recombination drag)
        # Normalized: shorter bubbles grow faster (less recombination loss)
        t_rec_Myr = 20.0 * (1.0 + self.z) / 7.0  # Scales with (1+z)

        # Growth timescale: geometric mean of dynamical and recombination times
        t_dyn_cMpc = bubble_size_cMpc / 3e4  # Speed of light in cMpc/Myr
        t_growth = np.sqrt(t_dyn_cMpc * t_rec_Myr)

        return t_growth

    def critical_bubble_size(self) -> float:
        """
        Critical bubble size for percolation overlap (void fraction ~ 0.5).

        In percolation theory, the overlap transition occurs when the average
        separation between bubble centers equals their average size. This gives
        a characteristic scale set by the ionization fraction.

        Returns
        -------
        float
            Critical radius (cMpc) for 50% ionization fraction
        """
        # Percolation threshold: V_bubble / V_total ~ 0.17 to 0.5 depending on geometry
        # For continuous medium with Gaussian overlap: critical radius where x_e ~ 0.5
        # Estimate: R_crit ~ (3 / (4 * pi * n_sources * sigma))^{1/3}
        # where n_sources ~ number density of ionizing sources, sigma ~ cross-section

        # Simplified model: R_crit inversely related to ionization fraction
        # At x_e = 0.5: R_crit ~ 30-40 cMpc (percolation scale)
        # Typical values: 10-100 cMpc at z~6

        if self.x_e < 0.01:
            return 100.0  # Fully neutral: largest bubbles needed
        elif self.x_e > 0.99:
            return 5.0  # Fully ionized: small residual bubbles
        else:
            # Power law: R_crit ∝ (1 - x_e)^α
            # At x_e=0.5: (1-0.5)^0.6 = 0.5^0.6 ≈ 0.66, so R_crit ≈ 66 cMpc
            # Matches percolation scale
            R_crit = 100.0 * (1.0 - self.x_e) ** 0.6
            return max(R_crit, 1.0)

    def ionization_probability(self, distance_cMpc: float) -> float:
        """
        Probability of ionization at given distance from bubble center.

        Models the ionization profile around an expanding bubble:
        - Sharp core: fully ionized within critical radius
        - Diffuse edge: ionization front with finite width
        - Tail: rapid decay beyond mean free path of ionizing photons

        Parameters
        ----------
        distance_cMpc : float
            Distance from bubble center (cMpc)

        Returns
        -------
        float
            Ionization probability (0 to 1)
        """
        R_crit = self.critical_bubble_size()
        # Ionization profile: error function (diffusion model)
        # Standard deviation represents ionization front width ~ 0.1-0.2 * R_crit
        sigma = 0.15 * R_crit

        # Complementary error function: ionized if inside bubble + diffuse edge
        if distance_cMpc < R_crit:
            return 1.0 - 0.5 * (1.0 + erf((distance_cMpc - R_crit) / (sigma * np.sqrt(2))))
        else:
            return 0.5 * (1.0 + erf((R_crit - distance_cMpc) / (sigma * np.sqrt(2))))


def bubble_growth_rate(
    redshift: NDArray[np.floating[Any]],
    bubble_size_cMpc: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """
    Growth rate of ionized bubbles as function of redshift and size.

    Parameters
    ----------
    redshift : NDArray
        Redshift grid (z=5 to z=8 during reionization)
    bubble_size_cMpc : NDArray
        Bubble radius grid (cMpc)

    Returns
    -------
    NDArray
        Growth rate dR/dt (cMpc/Myr) on (redshift, size) grid
    """
    if redshift.ndim == 1 and bubble_size_cMpc.ndim == 1:
        Z, R = np.meshgrid(redshift, bubble_size_cMpc, indexing="ij")
    elif redshift.ndim == 2 and bubble_size_cMpc.ndim == 2:
        Z, R = redshift, bubble_size_cMpc
    else:
        raise ValueError("Input grids must be 1D or 2D")

    dRdt = np.zeros_like(Z, dtype=np.float64)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            bubble = ReionizationBubble(redshift=float(Z[i, j]))
            t_growth = bubble.growth_timescale(float(R[i, j]))
            # dR/dt ~ R / t_growth (exponential growth in early stage)
            dRdt[i, j] = float(R[i, j]) / t_growth if t_growth > 0 else 0.0

    return dRdt


def overlap_probability(
    ionization_fraction: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """
    Probability of bubble overlap (adjacent bubbles touching) vs ionization.

    In percolation theory, the overlap probability transitions from 0 to 1
    around a critical ionization fraction (typically x_e ~ 0.2 to 0.5).
    This drives the rapid transition in ionization during reionization.

    Parameters
    ----------
    ionization_fraction : NDArray
        Global ionization fraction grid (0 to 1)

    Returns
    -------
    NDArray
        Overlap probability (0 to 1)
    """
    # Percolation model: smooth transition from disconnected to connected phase
    # Use Fermi function (smooth step) centered at critical ionization x_e_crit ~ 0.35
    x_e_crit = 0.35
    transition_width = 0.15  # Transition occurs over ~0.15 in x_e

    # Fermi function: 1 / (1 + exp(-(x - x_crit) / width))
    exponent = -(ionization_fraction - x_e_crit) / transition_width
    # Clip to avoid overflow
    exponent = np.clip(exponent, -100, 100)
    overlap = 1.0 / (1.0 + np.exp(exponent))

    return overlap
