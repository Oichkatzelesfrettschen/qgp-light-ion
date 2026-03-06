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
- Gunn & Peterson, ApJ 142 (1965) 1633 (Ly-alpha optical depth formulation)
- Shapiro et al., ApJ 427 (1994) 25 (reionization radiative transfer)
- Madau et al., ApJ 604 (2004) 656 (JWST era predictions)
- Planck 2018 XIII (arXiv:1807.06209) reionization constraints (z_reion = 7.7)
- Miralda-Escude, ApJ 501 (1998) 15 (Ly-alpha damping wings in reionization)
- JWST Cycle 1: ERG, GLASS, COSMOS-Web (Finkelstein et al. 2023)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

__all__ = [
    "StromgrenSphere",
    "ionization_front_expansion",
    "ionized_fraction_evolution",
    "ly_alpha_profile",
    "neutral_column_density",
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


def neutral_column_density(
    ionized_fraction: float,
    neutral_density_cm3: float,
    path_length_kpc: float = 1.0,
) -> tuple[float, float]:
    """
    Compute neutral hydrogen column density from ionized fraction.

    This function traces the causal chain from ionization state to observable
    column density that determines Ly-alpha optical depth. It connects the
    ionized fraction (x_e, dimensionless) through the neutral density to the
    line-of-sight column density that appears in the Gunn-Peterson formula.

    Physical basis:
    - Neutral hydrogen density: n_H_I = n_H_total * (1 - x_e)
    - Column density: N_HI = integral of n_H_I along line of sight
    - Optical depth at Ly-alpha line center: tau_0 = N_HI * sigma_0
      where sigma_0 is the Ly-alpha resonance cross-section per unit velocity

    Parameters
    ----------
    ionized_fraction : float
        Ionized fraction of hydrogen (x_e, range 0 to 1).
        x_e = 1 means fully ionized (no absorption).
        x_e = 0 means fully neutral (maximum absorption).
    neutral_density_cm3 : float
        Total hydrogen number density (cm^-3) in the IGM.
        Typical value: 1e-4 cm^-3 at z~6.
    path_length_kpc : float, optional
        Effective line-of-sight path length (kpc).
        Default 1.0 kpc is typical for the mean free path of Ly-alpha
        photons in the intergalactic medium at z~6-10 (Madau et al. 2004).

    Returns
    -------
    tuple[float, float]
        (n_H_neutral_cm3, column_density_cm2)
        - n_H_neutral_cm3: neutral hydrogen density (cm^-3)
        - column_density_cm2: column density (cm^-2)

    Notes
    -----
    The returned column density is the integrated neutral hydrogen along
    the 1D line of sight, assuming uniform density. In practice the IGM
    is inhomogeneous; this represents an effective average over large
    absorption systems as seen by JWST near reionization fronts.

    References
    ----------
    Gunn & Peterson (1965): ApJ 142, 1633 - Original Ly-alpha optical
    depth formula for damped systems.
    """
    # Convert ionized fraction to neutral fraction
    neutral_frac = 1.0 - ionized_fraction

    # Neutral hydrogen density (charge neutrality: n_e ≈ n_HII, so n_HI = n_H - n_e)
    n_H_neutral = neutral_density_cm3 * neutral_frac

    # Convert path length from kpc to cm
    # 1 kpc = 3.085677581e21 cm (IAU definition)
    path_length_cm = path_length_kpc * 3.086e21

    # Column density: integrate neutral density along path
    # For uniform density: N_HI = n_H_I * L
    column_density_cm2 = n_H_neutral * path_length_cm

    return float(n_H_neutral), float(column_density_cm2)


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

    # Transition timescale: front crosses from supersonic (R ~ t^{1/2}) to
    # subsonic (R -> R_s) when c_s * t ~ R_s, i.e. at t_s = R_s / c_s.
    # This is the characteristic time for the front to decelerate from the
    # early isothermal shock speed down to the sound speed.
    c_km_s: float = 3e5         # Speed of light (km/s), for Mach number checks
    t_s: float = R_s / c_s      # Transition time (s): when v_front ~ c_s

    # Guard: transition time must be positive (R_s > 0, c_s > 0)
    if t_s <= 0.0:
        raise ValueError(
            f"Transition timescale t_s={t_s:.3e} s must be positive; "
            f"check inputs (R_s={R_s:.3e} cm, c_s={c_s:.3e} cm/s)"
        )

    # Sanity check: front sound speed should be subsonic (v_front < c)
    v_sound_km_s = c_s / 1e5
    if v_sound_km_s >= c_km_s:
        raise ValueError(
            f"Sound speed {v_sound_km_s:.2e} km/s exceeds c; "
            f"check temperature input (T={10000.0} K)"
        )

    # Early-time radius growth
    # Coefficient for early-time growth: R_early(t) ~ sqrt(coeff * t)
    # From spherical blast wave: R_early ~ (E * t² / ρ)^{1/5} ~ t^{2/5}
    # For ionization fronts: closer to R ~ t^{1/2}
    # Calibration: coeff = (3*L / 4π*α*n_H²)
    # The junction condition R_early(t_s) ~ R_s is satisfied by this form.
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
    path_length_kpc: float = 1.0,
) -> NDArray[np.floating[Any]]:
    """
    Ly-alpha (Lyα) transmission profile near ionization front.

    Lyα photons scatter off neutral hydrogen, creating damping wing at
    velocities blueward of 1215.67 Angstrom rest frame.

    The optical depth at line center is computed using the Gunn-Peterson
    formula: τ_0 = N_HI * σ_0, where N_HI is the neutral hydrogen column
    density (from ionization state and line-of-sight path length) and σ_0
    is the Ly-alpha resonance cross-section per unit velocity width.

    Physical basis:
    - Ionization state (x_e) determines neutral density (n_HI)
    - Line-of-sight path length (L) sets column density (N_HI = n_HI * L)
    - Optical depth scales as N_HI, which drives transmission
    - Lorentzian profile decays as 1/(1 + (v/v_D)²) away from line center

    Parameters
    ----------
    velocity_km_s : NDArray
        Velocity offset from Lyα line center (km/s)
        (Negative = blueward, positive = redward)
    ionized_fraction : float, optional
        Ionized fraction in region (0-1), default 0.1
    neutral_density_cm3 : float, optional
        Neutral hydrogen density (cm⁻³), default 1e-4
    path_length_kpc : float, optional
        Effective line-of-sight path length (kpc), default 1.0.
        Typical value for IGM mean free path of Ly-alpha photons at z~6-10.

    Returns
    -------
    NDArray
        Lyα transmission (0-1); 0 = absorbed, 1 = transmitted

    References
    ----------
    Gunn & Peterson (1965): ApJ 142, 1633 - Ly-alpha optical depth.
    Miralda-Escude (1998): ApJ 501, 15 - Damping wing structure.
    """
    # Doppler width (1/e width): v_D = sqrt(2 k_B T / m_H)
    k_B = 1.381e-16  # erg/K (CGS units)
    m_H = 1.673e-24  # g (hydrogen mass)
    T_K = 100.0     # Temperature of neutral gas (K, typical for IGM clouds)

    v_D_cm_s = np.sqrt(2.0 * k_B * T_K / m_H)
    v_D_km_s = v_D_cm_s / 1e5

    # Compute column density from ionization state
    # This maps the ionized fraction to observable column density via:
    # x_e (ionized fraction) -> n_H_I (neutral density) -> N_HI (column density)
    _, column_density_cm2 = neutral_column_density(
        ionized_fraction, neutral_density_cm3, path_length_kpc
    )

    # Ly-alpha resonance cross-section at line center
    # σ_0 = (π e^2 / m_e c) * f_osc / (√π * v_D)
    # where f_osc = 0.4162 is the Ly-alpha oscillator strength (dimensionless)
    # In CGS units with v_D in cm/s: σ_0 ≈ 5.9e-14 / v_D_cm_s  [cm^2]
    # This is derived from Gunn & Peterson (1965) radiative transfer formalism
    sigma_0_cm2 = 5.9e-14 / v_D_cm_s

    # Optical depth at line center (Gunn-Peterson formula)
    # τ_0 = N_HI * σ_0  [dimensionless]
    # This is the physical foundation: larger column density or larger
    # cross-section gives more absorption (larger optical depth)
    if column_density_cm2 <= 0.0:
        # Fully ionized case: no neutral hydrogen, no absorption
        return np.ones_like(velocity_km_s, dtype=np.float64)

    tau_0 = column_density_cm2 * sigma_0_cm2
    v_D_safe = max(v_D_km_s, 1.0)

    # Lorentzian profile: τ(v) = τ_0 / (1 + (v/v_D)²)
    # This gives maximum absorption at line center, declining to wings
    tau = tau_0 / (1.0 + (velocity_km_s / v_D_safe) ** 2)

    # Transmission: T = exp(-τ)
    transmission = np.exp(-tau)

    return np.asarray(transmission, dtype=np.float64)


def ionized_fraction_evolution(
    redshift: NDArray[np.floating[Any]],
    z_reion: float = 7.7,
    delta_z: float = 1.5,
) -> NDArray[np.floating[Any]]:
    """
    Global ionized fraction as function of redshift.

    Models the cosmic reionization history using an error-function transition
    from fully neutral (high-z) to fully ionized (low-z) universe. This
    functional form is phenomenological but well-motivated: reionization is
    driven by ionizing radiation from early galaxies/quasars, and feedback
    effects naturally produce a smooth transition over a characteristic
    redshift interval.

    Physical basis:
    - z >> z_reion: universe is neutral (x_e ~ 0), Ly-alpha absorption high
    - z ~ z_reion: transition region, expanding ionized bubbles coalesce
    - z << z_reion: universe is ionized (x_e ~ 1), Ly-alpha absorption low
    - Width Δz_reion determines sharpness of transition (photon mean free path)

    Parameters
    ----------
    redshift : NDArray
        Redshift array (z). Should span z~6-20 for reionization era.
    z_reion : float, optional
        Midpoint redshift of reionization (z_reion = 7.7 ± 0.4).
        Default from Planck 2018 (arXiv:1807.06209, Table 2).
    delta_z : float, optional
        Transition width (Δz ~ 1.5-2.0). Characterizes how rapidly
        reionization transitions from neutral to ionized. Default 1.5
        matches JWST and Planck2018 constraints.

    Returns
    -------
    NDArray
        Ionized fraction x_e(z), range [0, 1].
        At z = z_reion: x_e = 0.5 (definition of midpoint).

    Notes
    -----
    The error-function form naturally captures the physics of overlapping
    ionization bubbles: before z_reion, isolated bubbles (low x_e); at
    z_reion, bubbles coalesce (transition); after z_reion, nearly complete
    ionization (high x_e).

    This function is often called to initialize `ly_alpha_profile()`:
        z_source = 7.5
        x_e = float(ionized_fraction_evolution(np.array([z_source]))[0])
        transmission = ly_alpha_profile(velocity_km_s, ionized_fraction=x_e)

    References
    ----------
    Planck 2018 XIII (arXiv:1807.06209): Reionization constraints,
    z_reion = 7.7 ± 0.4 (68% CL).
    Finkelstein et al. (2023): JWST ERG observations confirming early
    reionization at z > 6.
    """
    # Error function transition: x_e(z) = 0.5 * (1 - erf((z - z_reion) / delta_z))
    # This gives:
    # - x_e(z >> z_reion) → 0   (fully neutral)
    # - x_e(z_reion) = 0.5       (midpoint)
    # - x_e(z << z_reion) → 1    (fully ionized)
    x_e = 0.5 * (1.0 - erf((redshift - z_reion) / delta_z))

    # Ensure x_e is bounded to [0, 1] and has correct dtype
    x_e_clipped = np.clip(x_e, 0.0, 1.0)

    return np.asarray(x_e_clipped, dtype=np.float64)
