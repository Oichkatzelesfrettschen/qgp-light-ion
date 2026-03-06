"""
Dark energy equation of state and cosmological distance measures.

Infers dark energy properties (w, Ω_Λ) from baryon acoustic oscillation (BAO)
measurements and type Ia supernovae (SNe) distance modulus data.

Physical basis:
- Accelerated expansion driven by dark energy with equation of state w = p/ρ
- BAO: sound horizon at drag epoch sets standard ruler (DESI 2025)
- Comoving distance: integral of proper time over redshift
- Distance modulus: μ = 5 log₁₀(d_L) + 25 (luminosity distance)

References:
- DESI collaboration (2025) BAO measurements at z=0.5, 0.75, 1.0
- Perlmutter et al., ApJ 517 (1999) 565 (SNe cosmology)
- Planck 2018 XIII (arXiv:1807.06209) concordance model (Ω_Λ ~ 0.68)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import quad
from numpy.typing import NDArray

__all__ = [
    "DarkEnergyModel",
    "comoving_distance",
    "distance_modulus",
    "bao_measurement",
]


class DarkEnergyModel:
    """Dark energy equation of state and cosmological parameters."""

    def __init__(
        self,
        w: float = -1.0,
        Omega_Lambda: float = 0.68,
        Omega_m: float = 0.32,
        H0: float = 67.4,
    ) -> None:
        """
        Initialize dark energy model with equation of state.

        Parameters
        ----------
        w : float, optional
            Dark energy equation of state, default -1.0 (cosmological constant)
        Omega_Lambda : float, optional
            Dark energy density parameter, default 0.68 (Planck 2018)
        Omega_m : float, optional
            Matter density parameter, default 0.32
        H0 : float, optional
            Hubble constant (km/s/Mpc), default 67.4 (Planck 2018)
        """
        self.w = w
        self.Omega_Lambda = Omega_Lambda
        self.Omega_m = Omega_m
        self.Omega_k = 1.0 - Omega_Lambda - Omega_m  # Spatial curvature
        self.H0 = H0
        self.H0_inv_Gyr = 307.75 / H0  # 1/H0 in Gyears

    def Hubble_z(self, z: float) -> float:
        """
        Hubble parameter H(z) normalized to H0.

        Parameters
        ----------
        z : float
            Redshift

        Returns
        -------
        float
            H(z) / H0
        """
        a = 1.0 / (1.0 + z)

        # H(z)/H0 = sqrt(Ω_m * (1+z)³ + Ω_k * (1+z)² + Ω_Λ * a^(3(1+w)))
        return np.sqrt(
            self.Omega_m * (1.0 + z) ** 3
            + self.Omega_k * (1.0 + z) ** 2
            + self.Omega_Lambda * a ** (3.0 * (1.0 + self.w))
        )

    def comoving_distance_Mpc(self, z: float) -> float:
        """
        Comoving distance to redshift z.

        Parameters
        ----------
        z : float
            Redshift

        Returns
        -------
        float
            Comoving distance (Mpc)
        """

        def integrand(z_prime: float) -> float:
            return 1.0 / self.Hubble_z(z_prime)

        # Numerical integration
        result, _ = quad(integrand, 0, z, limit=100)

        # Multiply by c/H0 to get distance
        c_over_H0 = 2998.0 / self.H0  # Speed of light / H0 in Mpc

        return float(c_over_H0 * result)

    def luminosity_distance_Mpc(self, z: float) -> float:
        """
        Luminosity distance to redshift z.

        d_L = (1+z) * d_c for flat universe (Ω_k ≈ 0)

        Parameters
        ----------
        z : float
            Redshift

        Returns
        -------
        float
            Luminosity distance (Mpc)
        """
        d_c = self.comoving_distance_Mpc(z)

        if abs(self.Omega_k) < 0.001:
            # Flat universe
            return (1.0 + z) * d_c
        else:
            # Curved universe (more complex formula)
            sqrt_K = np.sqrt(abs(self.Omega_k))
            K_term = np.sinh(sqrt_K * self.Hubble_z(z) * d_c / self.H0)
            return float((1.0 + z) * K_term / sqrt_K)

    def distance_modulus(self, z: float) -> float:
        """
        Distance modulus μ(z) = 5 log₁₀(d_L / 10 pc) + 25.

        Parameters
        ----------
        z : float
            Redshift

        Returns
        -------
        float
            Distance modulus (mag)
        """
        d_L_Mpc = self.luminosity_distance_Mpc(z)
        d_L_pc = d_L_Mpc * 1e6

        # Distance modulus
        return float(5.0 * np.log10(d_L_pc / 10.0) + 25.0)

    def age_of_universe_Gyr(self) -> float:
        """
        Age of the universe in Gyr.

        Uses numerical integration: age = ∫₀¹ da / (a * H(a) / H₀)

        Returns
        -------
        float
            Age (Gyr)
        """

        def integrand(a: float) -> float:
            if a < 1e-6:
                return 0.0  # Early universe singularity
            z = 1.0 / a - 1.0
            return 1.0 / (a * self.Hubble_z(z))

        # Integrate from small a (early) to 1 (today)
        result, _ = quad(integrand, 1e-6, 1.0, limit=100)

        return float(self.H0_inv_Gyr * result)


def comoving_distance(
    z: NDArray[np.floating[Any]],
    w: float = -1.0,
    Omega_Lambda: float = 0.68,
) -> NDArray[np.floating[Any]]:
    """
    Comoving distance on redshift grid.

    Parameters
    ----------
    z : NDArray
        Redshift array
    w : float, optional
        Dark energy equation of state
    Omega_Lambda : float, optional
        Dark energy density

    Returns
    -------
    NDArray
        Comoving distances (Mpc)
    """
    model = DarkEnergyModel(w=w, Omega_Lambda=Omega_Lambda)

    distances = np.zeros_like(z, dtype=np.float64)
    for i, z_val in enumerate(z):
        distances[i] = model.comoving_distance_Mpc(float(z_val))

    return distances


def distance_modulus(
    z: NDArray[np.floating[Any]],
    w: float = -1.0,
) -> NDArray[np.floating[Any]]:
    """
    Distance modulus grid for SNe fitting.

    Parameters
    ----------
    z : NDArray
        Redshift array (z < 2 typical for SNe)
    w : float, optional
        Dark energy equation of state

    Returns
    -------
    NDArray
        Distance modulus μ(z) (mag)
    """
    model = DarkEnergyModel(w=w)

    mu = np.zeros_like(z, dtype=np.float64)
    for i, z_val in enumerate(z):
        mu[i] = model.distance_modulus(float(z_val))

    return mu


def bao_measurement(
    z_BAO: float = 0.5,
    sound_horizon_Mpc: float = 149.3,
) -> dict[str, float]:
    """
    BAO measurement at given redshift (DESI 2025 model).

    Parameters
    ----------
    z_BAO : float, optional
        BAO measurement redshift, default 0.5
    sound_horizon_Mpc : float, optional
        Sound horizon at drag epoch (Mpc), default 149.3 (Planck 2018)

    Returns
    -------
    dict
        BAO measurements: 'd_M' (comoving distance), 'D_V' (volume distance)
    """
    model = DarkEnergyModel()

    # Comoving distance at z_BAO
    d_M = model.comoving_distance_Mpc(z_BAO)

    # Hubble parameter at z_BAO
    H_z = model.H0 * model.Hubble_z(z_BAO)

    # Volume-averaged distance D_V = [z * d_M² * c / H(z)]^(1/3)
    D_V = (z_BAO * d_M**2 * 2998.0 / H_z) ** (1.0 / 3.0)

    # BAO scale: r_s / D_V (dimensionless)
    BAO_scale = sound_horizon_Mpc / D_V

    return {
        "z_BAO": z_BAO,
        "d_M_Mpc": d_M,
        "H_z_kms": H_z,
        "D_V_Mpc": D_V,
        "r_s_over_D_V": BAO_scale,
    }
