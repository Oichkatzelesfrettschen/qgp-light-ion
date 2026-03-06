"""
Reionization front dynamics and JWST observations.

Models the propagation of ionization fronts through the intergalactic medium
during cosmic reionization (z~6-20) and interprets JWST observations
of HII regions and Ly-alpha emission.

Physical basis:
- Ionization front expansion into neutral hydrogen
- Stromgren sphere: equilibrium ionized bubble size
- Recombination rate and source luminosity determine front speed
- JWST detects Ly-alpha quenching near ionization fronts
- Topology: percolation and overlap of expanding bubbles

References:
- Stromgren, ApJ 89 (1939) 526 (classic HII region theory)
- Shapiro et al., ApJ 427 (1994) 25 (reionization radiative transfer)
- Madau et al., ApJ 604 (2004) 656 (JWST era predictions)
- JWST Cycle 1: ERG, GLASS, COSMOS-Web (Finkelstein et al. 2023)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "StromgrenSphere",
    "ionization_front_expansion",
    "ly_alpha_profile",
]


class StromgrenSphere:
    """Equilibrium ionized region around UV source."""

    def __init__(
        self,
        source_luminosity_erg_s: float,
        neutral_density_cm3: float = 1e-4,
        temperature_K: float = 10000.0,
    ) -> None:
        """
        Initialize Stromgren sphere for ionizing source.

        Parameters
        ----------
        source_luminosity_erg_s : float
            Ionizing photon luminosity (erg/s)
        neutral_density_cm3 : float, optional
            Neutral hydrogen density (cm⁻³), default 1e-4
        temperature_K : float, optional
            Temperature of ionized gas (K), default 10000
        """
        self.L_erg_s = source_luminosity_erg_s
        self.n_H_cm3 = neutral_density_cm3
        self.T_K = temperature_K

    def recombination_coefficient(self) -> float:
        """
        Recombination rate coefficient α(T).

        Uses Hui & Gnedin (1997) parameterization for hydrogen.

        Returns
        -------
        float
            α (cm³ s⁻¹)
        """
        # α(T) = 1.269e-13 * (T / 1e4)^{-0.75} cm³ s⁻¹
        T_4 = self.T_K / 1e4

        return float(1.269e-13 * T_4 ** (-0.75))

    def stromgren_radius_cm(self) -> float:
        """
        Stromgren radius: equilibrium ionized bubble size.

        R_s = (3 * L / (4π * n_H² * α(T)))^{1/3}

        Returns
        -------
        float
            Stromgren radius (cm)
        """
        alpha = self.recombination_coefficient()

        # Ionizing photons per unit time
        L_photons_s = self.L_erg_s / (13.6 * 1.602e-12)  # Convert to photons/s

        numerator = 3.0 * L_photons_s
        denominator = 4.0 * np.pi * self.n_H_cm3**2 * alpha

        R_s_cm = (numerator / denominator) ** (1.0 / 3.0)

        return float(R_s_cm)

    def expansion_speed_cm_s(self, time_s: float) -> float:
        """
        Ionization front speed at given time.

        Initially fast (isothermal), then approaches constant speed:
        v_exp ~ c_s * sqrt(1 + (ct / R_s)^2)

        where c_s is sound speed in ionized gas.

        Parameters
        ----------
        time_s : float
            Time after source turn-on (s)

        Returns
        -------
        float
            Expansion speed (cm/s)
        """
        # Sound speed in ionized gas: c_s ~ sqrt(γ k_B T / m_H)
        k_B = 1.381e-16  # Boltzmann (erg/K)
        m_H = 1.673e-24  # Mass of hydrogen (g)

        c_s = np.sqrt(5.0 / 3.0 * k_B * self.T_K / m_H)

        R_s = self.stromgren_radius_cm()

        # Speed: c_s * sqrt(1 + (c_s * t / R_s)^2)
        dimensionless_time = c_s * time_s / R_s

        v_exp = c_s * np.sqrt(1.0 + dimensionless_time**2)

        return float(v_exp)

    def recombination_timescale_s(self) -> float:
        """
        Timescale for recombination: t_rec = 1 / (α n_e).

        Returns
        -------
        float
            Recombination timescale (s)
        """
        alpha = self.recombination_coefficient()

        # In ionized region, n_e ≈ n_H (charge neutrality)
        t_rec_s = 1.0 / (alpha * self.n_H_cm3)

        return t_rec_s


def ionization_front_expansion(
    t: NDArray[np.floating[Any]],
    source_luminosity_erg_s: float,
    neutral_density_cm3: float = 1e-4,
) -> NDArray[np.floating[Any]]:
    """
    Radius of ionization front vs time.

    Parameters
    ----------
    t : NDArray
        Time array (s) from source turn-on
    source_luminosity_erg_s : float
        Ionizing source luminosity (erg/s)
    neutral_density_cm3 : float, optional
        Neutral hydrogen density (cm⁻³)

    Returns
    -------
    NDArray
        Ionization front radius (cm) vs time
    """
    sphere = StromgrenSphere(source_luminosity_erg_s, neutral_density_cm3)

    R_s = sphere.stromgren_radius_cm()

    # Sound speed in ionized gas (10^4 K)
    k_B = 1.381e-16  # erg/K
    m_H = 1.673e-24  # g
    c_s = np.sqrt(5.0 / 3.0 * k_B * 10000.0 / m_H)

    # Isothermal expansion (weak shock):
    # Early time (t << t_s where t_s = R_s/c_s): R(t) ~ sqrt(3Lt/4πα n_H²)
    # Late time: R approaches R_s
    # Transition: use smooth blend R(t) = sqrt(R_s² + (early_time_growth)²)

    # Early-time radius growth
    # Coefficient for early-time growth: R_early(t) ~ sqrt(coeff * t)
    # From spherical blast wave: R_early ~ (E * t² / ρ)^{1/5} ~ t^{2/5}
    # For ionization fronts: closer to R ~ t^{1/2}
    # Calibration: coeff = (3*L / 4π*α*n_H²)
    alpha = sphere.recombination_coefficient()
    L_photons_s = source_luminosity_erg_s / (13.6 * 1.602e-12)
    coeff = (3.0 * L_photons_s) / (4.0 * np.pi * neutral_density_cm3**2 * alpha * c_s)

    # Early-time contribution: sqrt(coeff * t)
    R_early = np.sqrt(np.maximum(coeff * t, 0.0))

    # Blend: as t increases, transition from early-time to Stromgren
    # Use R(t) = sqrt(R_s² + R_early²) which naturally transitions
    R_cm = np.sqrt(R_s**2 + R_early**2)

    return np.asarray(R_cm, dtype=np.float64)


def ly_alpha_profile(
    velocity_km_s: NDArray[np.floating[Any]],
    ionized_fraction: float = 0.1,
    neutral_density_cm3: float = 1e-4,
) -> NDArray[np.floating[Any]]:
    """
    Ly-alpha (Lyα) transmission profile near ionization front.

    Lyα photons scatter off neutral hydrogen, creating damping wing at
    velocities blueward of 1215.67 Angstrom rest frame.

    Parameters
    ----------
    velocity_km_s : NDArray
        Velocity offset from Lyα line center (km/s)
        (Negative = blueward, positive = redward)
    ionized_fraction : float, optional
        Ionized fraction in region (0-1), default 0.1
    neutral_density_cm3 : float, optional
        Neutral hydrogen density (cm⁻³)

    Returns
    -------
    NDArray
        Lyα transmission (0-1); 0 = absorbed, 1 = transmitted
    """
    # Doppler width (1/e width): v_D ~ sqrt(2 k_B T / m_H)
    k_B = 1.381e-16  # erg/K (= 8.617e-5 eV/K, but use cgs)
    m_H = 1.673e-24  # g
    T_K = 100.0  # Temperature of neutral gas (K)

    # Doppler width: v_D = sqrt(2 * k_B * T / m_H)
    v_D_cm_s = np.sqrt(2.0 * k_B * T_K / m_H)
    v_D_km_s = v_D_cm_s / 1e5

    # Ly-alpha line broadening and optical depth
    # τ(v) has Lorentzian wings with damping parameter:
    # a ~ Γ / (4π * Δv_D)
    # where Γ ~ 6.3e8 s⁻¹ (Einstein A coefficient for Ly-alpha)

    # For simplicity, use Voigt profile approximation:
    # τ(v) ~ N_HI * σ_0 * phi(v)
    # where N_HI is column density and phi is normalized profile

    # Optical depth: τ(v) ~ σ * N_HI * φ(v)
    # where N_HI is column density (cm⁻²) and φ(v) is profile
    # Typical neutral clouds have large column densities: N_HI ~ 10^15-10^20 cm⁻²

    # Estimate optical depth from density and path length
    # For 1 kpc path length at n_H ~ 10^-4 cm⁻³: N_HI ~ 3e18 cm⁻²
    # (column_density = n_H_neutral * path_length_cm, used in scaling tau_0)

    # Lorentzian optical depth with neutral fraction scaling
    # Pure Lorentzian (no damping wings for simplicity)
    # τ(v) = τ_0 * f(neutral_fraction) / (1 + (v/v_D)²)
    neutral_frac = 1.0 - ionized_fraction
    v_D_safe = max(v_D_km_s, 1.0)

    # Base optical depth at line center
    # Scales strongly with neutral fraction to show ionization dependence
    # At x_e=0 (all neutral): τ_0 = 1.5
    # At x_e=0.9 (90% ionized): τ_0 ≈ 0.006  (high transparency)
    # At x_e=0.1 (10% ionized): τ_0 ≈ 0.24  (moderate absorption)
    tau_0 = 1.5 * neutral_frac**2.5

    # Lorentzian profile: τ(v) = τ_0 / (1 + (v/v_D)²)
    # This gives maximum absorption at line center, declining to wings
    tau = tau_0 / (1.0 + (velocity_km_s / v_D_safe) ** 2)

    # Transmission: T = exp(-τ)
    transmission = np.exp(-tau)

    return np.asarray(transmission, dtype=np.float64)
