"""
Strangeness threshold and canonical suppression in QGP.

This module implements the canonical strangeness suppression framework:
- Strangeness threshold effects (discontinuity at strange quark creation)
- Canonical ensemble formalism (fixed strangeness number S)
- Grid-based suppression factors for K, Lambda, Xi particles

Physical basis:
- In heavy-ion collisions, strange quarks are produced in a conserved way
- Canonical ensemble: P(S) ∝ exp(-μ_S * S / T) with fixed total strangeness
- Suppression factors: γ_s^(3/2) for baryons, γ_s for mesons (Bessel function formalism)
- Threshold: suppression becomes zero below critical strangeness density

Reference: Andronic et al. NPA 834 (2010) 237c; Becattini et al. PRC 73 (2006) 064905
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CanonicalEnsemble",
    "compute_suppression_factor",
    "strangeness_threshold",
]


class CanonicalEnsemble:
    """Canonical ensemble for fixed strangeness number."""

    def __init__(
        self,
        T: float,
        mu_B: float,
        V: float = 1.0,
        s_density: float = 0.1,
    ) -> None:
        """
        Initialize canonical ensemble with thermodynamic parameters.

        Parameters
        ----------
        T : float
            Temperature (MeV)
        mu_B : float
            Baryon chemical potential (MeV)
        V : float, optional
            Volume (fm³), default 1.0 (normalized)
        s_density : float, optional
            Strangeness density (fm⁻³), default 0.1
        """
        self.T = T
        self.mu_B = mu_B
        self.V = V
        self.s_density = s_density

        # Chemical potentials (approximate)
        # mu_S ~ mu_B/3 (from flavor SU(3) symmetry in baryon sector)
        # mu_Q ~ (mu_B - 3*mu_S)/2 ~ mu_B/2 (electric charge conservation)
        self.mu_S = mu_B / 3.0
        self.mu_Q = mu_B / 2.0

    def partition_function_ratio(self, n_strange: int) -> float:
        """
        Partition function ratio for n_strange quarks.

        Uses Poisson approximation with cumulant expansion:
        Z(S) / Z(0) ∝ exp(-S * μ_S / T + ...)

        Parameters
        ----------
        n_strange : int
            Number of strange quarks

        Returns
        -------
        float
            Relative weight in canonical ensemble
        """
        exponent = -n_strange * self.mu_S / self.T
        return float(np.exp(exponent))

    def threshold_density(self) -> float:
        """
        Critical strangeness density below which suppression is maximal.

        At threshold, the number of strange quarks changes discontinuously
        as a function of temperature and density.

        Returns
        -------
        float
            Threshold strangeness density (fm⁻³)
        """
        # Approximate: threshold at T_c when chemical potentials balance
        T_c = 156.5  # MeV, crossover temperature
        if T_c > self.T:
            # Below crossover: suppression increases rapidly
            return 0.05 * (1.0 - self.T / T_c)
        else:
            # Above crossover: suppression decreases, threshold rises
            return 0.05 * (self.T - T_c) / 50.0

    def suppression_factor(self, particle_type: str) -> float:
        """
        Strangeness suppression factor γ_s for particle type.

        Different particles have different effective powers of γ_s:
        - Kaons (K): γ_s (one strange quark)
        - Lambdas (Λ): γ_s^(3/2) (one strange quark but in baryon sector)
        - Xi (Ξ): γ_s^3 (two strange quarks)

        Parameters
        ----------
        particle_type : str
            One of "kaon", "lambda", "xi"

        Returns
        -------
        float
            Suppression factor (typically 0.4-0.8)
        """
        # Base factor: depends on relative chemical potential
        # γ_s = exp(-μ_S / T) for conservation law
        base = np.exp(-self.mu_S / self.T)

        # Apply threshold: below threshold density, suppression is stronger
        threshold = self.threshold_density()
        if self.s_density < threshold:
            threshold_factor = self.s_density / threshold
            base *= threshold_factor

        # Particle-specific exponent
        if particle_type.lower() == "kaon":
            return float(base)
        elif particle_type.lower() == "lambda":
            return float(base ** 1.5)
        elif particle_type.lower() == "xi":
            return float(base ** 3.0)
        else:
            raise ValueError(f"Unknown particle type: {particle_type}")


def compute_suppression_factor(
    T: NDArray[np.floating[Any]],
    mu_B: NDArray[np.floating[Any]],
    particle_type: str = "kaon",
    s_density: float = 0.1,
) -> NDArray[np.floating[Any]]:
    """
    Compute strangeness suppression factor on a grid.

    Parameters
    ----------
    T : NDArray
        Temperature grid (MeV)
    mu_B : NDArray
        Baryon chemical potential grid (MeV)
    particle_type : str, optional
        Particle type: "kaon", "lambda", "xi"
    s_density : float, optional
        Strangeness density (fm⁻³)

    Returns
    -------
    NDArray
        Suppression factor γ_s on (T, mu_B) grid
    """
    if T.ndim == 1 and mu_B.ndim == 1:
        # Create 2D grid
        T_2d, muB_2d = np.meshgrid(T, mu_B, indexing="ij")
    elif T.ndim == 2 and mu_B.ndim == 2:
        T_2d, muB_2d = T, mu_B
    else:
        raise ValueError("T and mu_B must be 1D or 2D arrays")

    suppression = np.zeros_like(T_2d)
    for i in range(T_2d.shape[0]):
        for j in range(T_2d.shape[1]):
            ensemble = CanonicalEnsemble(
                T=float(T_2d[i, j]),
                mu_B=float(muB_2d[i, j]),
                s_density=s_density,
            )
            suppression[i, j] = ensemble.suppression_factor(particle_type)

    return np.asarray(suppression, dtype=np.float64)


def strangeness_threshold(T: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """
    Strangeness threshold density as function of temperature.

    At the threshold, strange quark production transitions from suppressed
    to normal (unsuppressed).

    Parameters
    ----------
    T : NDArray
        Temperature grid (MeV)

    Returns
    -------
    NDArray
        Threshold strangeness density (fm⁻³)
    """
    T_c = 156.5  # Crossover temperature (MeV)

    # Piecewise function:
    # Below T_c: suppression increases (threshold decreases)
    # Above T_c: suppression decreases (threshold increases)
    threshold = np.zeros_like(T)

    below_Tc = T_c > T
    above_Tc = T_c <= T

    threshold[below_Tc] = 0.05 * (1.0 - T[below_Tc] / T_c)
    threshold[above_Tc] = 0.05 * (T[above_Tc] - T_c) / 50.0

    return np.maximum(threshold, 0.001)  # Keep above numerical floor
