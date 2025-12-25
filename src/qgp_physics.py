"""
qgp_physics.py - Physics-based models for QGP visualization

This module implements the actual physics formulas and models discussed
in the document, providing realistic data for sophisticated visualizations.

Physical constants and formulas from:
- Bjorken energy density estimation
- Woods-Saxon nuclear density profiles
- Glauber model geometry
- BDMPS-Z energy loss formalism
- Statistical hadronization with canonical suppression
- Relativistic hydrodynamics flow response
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.special import iv as bessel_i  # Modified Bessel function I_n(x)

warnings.filterwarnings("ignore")

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# QCD constants
T_C = 0.155  # Critical/crossover temperature [GeV] ≈ 155 MeV
EPSILON_C = 1.0  # Critical energy density for deconfinement [GeV/fm³]
HBARC = 0.197  # ℏc [GeV·fm]
FM_TO_GEV_INV = 1 / HBARC  # Conversion factor

# Transport coefficients (typical values)
ETA_OVER_S = 0.12  # Shear viscosity to entropy ratio (near KSS bound 1/4π ≈ 0.08)
QHAT_0 = 2.0  # Jet transport coefficient at T = T_c [GeV²/fm]


# Nuclear parameters
@dataclass
class Nucleus:
    """Nuclear properties for Glauber model calculations."""

    name: str
    A: int  # Mass number
    Z: int  # Atomic number
    R0: float  # Nuclear radius parameter [fm]
    a: float  # Skin thickness [fm]
    beta2: float = 0.0  # Quadrupole deformation
    beta3: float = 0.0  # Octupole deformation


# Standard nuclei used in LHC collisions
NUCLEI = {
    "O": Nucleus("Oxygen-16", A=16, Z=8, R0=2.608, a=0.513, beta2=0.0),
    "Ne": Nucleus("Neon-20", A=20, Z=10, R0=2.791, a=0.535, beta2=0.45),  # Prolate!
    "Ar": Nucleus("Argon-40", A=40, Z=18, R0=3.427, a=0.569, beta2=0.0),
    "Xe": Nucleus("Xenon-129", A=129, Z=54, R0=5.36, a=0.59, beta2=0.18),
    "Pb": Nucleus("Lead-208", A=208, Z=82, R0=6.62, a=0.546, beta2=0.0),
}

# =============================================================================
# WOODS-SAXON DENSITY PROFILE
# =============================================================================


def woods_saxon(r: np.ndarray, nucleus: Nucleus, theta: float = 0) -> np.ndarray:
    """
    Woods-Saxon nuclear density distribution.

    ρ(r) = ρ₀ / (1 + exp((r - R(θ))/a))

    For deformed nuclei:
    R(θ) = R₀(1 + β₂Y₂₀(θ) + β₃Y₃₀(θ))

    Parameters
    ----------
    r : array
        Radial distance [fm]
    nucleus : Nucleus
        Nuclear parameters
    theta : float
        Polar angle for deformed nuclei [rad]

    Returns
    -------
    density : array
        Nuclear density (normalized to peak = 1)
    """
    # Spherical harmonics Y_l0 at theta
    Y20 = 0.25 * np.sqrt(5 / np.pi) * (3 * np.cos(theta) ** 2 - 1)
    Y30 = 0.25 * np.sqrt(7 / np.pi) * (5 * np.cos(theta) ** 3 - 3 * np.cos(theta))

    # Deformed radius
    R = nucleus.R0 * (1 + nucleus.beta2 * Y20 + nucleus.beta3 * Y30)

    # Woods-Saxon form
    rho = 1.0 / (1.0 + np.exp((r - R) / nucleus.a))
    return rho


def get_nuclear_profile_2d(
    nucleus: Nucleus, grid_size: int = 100, extent: float = 10.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 2D nuclear density profile in the transverse plane.

    Returns x, y coordinates and density ρ(x,y).
    """
    x = np.linspace(-extent, extent, grid_size)
    y = np.linspace(-extent, extent, grid_size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # For deformed nuclei, average over orientations or use specific angle
    # Here we show the maximum deformation (θ = π/2 for prolate)
    if nucleus.beta2 > 0:
        # Show prolate shape - elongated along one axis
        # Compute per-point angles for deformed profile
        rho = np.zeros_like(R)
        for i in range(grid_size):
            for j in range(grid_size):
                r_eff = np.sqrt(X[i, j] ** 2 + Y[i, j] ** 2)
                theta_eff = np.arctan2(abs(Y[i, j]), abs(X[i, j]))
                rho[i, j] = woods_saxon(np.array([r_eff]), nucleus, theta_eff)[0]
    else:
        rho = woods_saxon(R, nucleus)

    return X, Y, rho


# =============================================================================
# GLAUBER MODEL - MONTE CARLO NUCLEON POSITIONS
# =============================================================================


def sample_nucleon_positions(nucleus: Nucleus, n_events: int = 1) -> np.ndarray:
    """
    Sample nucleon positions from Woods-Saxon distribution using rejection sampling.

    Returns array of shape (n_events, A, 3) with (x, y, z) positions in fm.
    """
    positions = np.zeros((n_events, nucleus.A, 3))

    for event in range(n_events):
        for i in range(nucleus.A):
            accepted = False
            while not accepted:
                # Sample in a box
                r_max = nucleus.R0 + 5 * nucleus.a
                x = np.random.uniform(-r_max, r_max)
                y = np.random.uniform(-r_max, r_max)
                z = np.random.uniform(-r_max, r_max)
                r = np.sqrt(x**2 + y**2 + z**2)

                # Rejection sampling
                theta = np.arccos(z / (r + 1e-10))
                rho = woods_saxon(np.array([r]), nucleus, theta)[0]
                if np.random.random() < rho:
                    positions[event, i] = [x, y, z]
                    accepted = True

    return positions


def calculate_participants(
    pos_A: np.ndarray, pos_B: np.ndarray, b: float, sigma_nn: float = 7.0
) -> tuple[int, int]:
    """
    Calculate N_part and N_coll for a collision at impact parameter b.

    Parameters
    ----------
    pos_A, pos_B : array (A, 3)
        Nucleon positions for each nucleus
    b : float
        Impact parameter [fm]
    sigma_nn : float
        Nucleon-nucleon cross section [fm²] (≈70 mb at LHC)

    Returns
    -------
    N_part : int
        Number of participating nucleons
    N_coll : int
        Number of binary collisions
    """
    # Shift nucleus B by impact parameter in x-direction
    pos_B_shifted = pos_B.copy()
    pos_B_shifted[:, 0] += b

    # Interaction radius from cross section
    r_int = np.sqrt(sigma_nn / np.pi)

    participants_A = np.zeros(len(pos_A), dtype=bool)
    participants_B = np.zeros(len(pos_B), dtype=bool)
    n_coll = 0

    for i, pA in enumerate(pos_A):
        for j, pB in enumerate(pos_B_shifted):
            # Transverse distance
            d_T = np.sqrt((pA[0] - pB[0]) ** 2 + (pA[1] - pB[1]) ** 2)
            if d_T < r_int:
                participants_A[i] = True
                participants_B[j] = True
                n_coll += 1

    return int(np.sum(participants_A) + np.sum(participants_B)), n_coll


def glauber_centrality_scan(
    nucleus_A: Nucleus, nucleus_B: Nucleus, n_b_points: int = 20, n_events: int = 100
) -> dict:
    """
    Perform Glauber model scan over impact parameter.

    Returns dictionary with b, <N_part>, <N_coll>, centrality estimates.
    """
    b_max = nucleus_A.R0 + nucleus_B.R0 + 2  # fm
    b_values = np.linspace(0, b_max, n_b_points)

    npart_mean = []
    ncoll_mean = []

    for b in b_values:
        npart_list = []
        ncoll_list = []
        for _ in range(n_events):
            pos_A = sample_nucleon_positions(nucleus_A, 1)[0]
            pos_B = sample_nucleon_positions(nucleus_B, 1)[0]
            np_, nc_ = calculate_participants(pos_A, pos_B, b)
            npart_list.append(np_)
            ncoll_list.append(nc_)
        npart_mean.append(np.mean(npart_list))
        ncoll_mean.append(np.mean(ncoll_list))

    return {
        "b": b_values,
        "N_part": np.array(npart_mean),
        "N_coll": np.array(ncoll_mean),
    }


# =============================================================================
# ECCENTRICITY - INITIAL STATE GEOMETRY
# =============================================================================


def calculate_eccentricities(positions: np.ndarray) -> dict[str, float]:
    """
    Calculate spatial eccentricities ε_n from nucleon positions.

    ε_n = |<r² e^{inφ}>| / <r²>

    These drive the flow harmonics v_n.
    """
    x = positions[:, 0]
    y = positions[:, 1]

    # Center of mass
    x_cm = np.mean(x)
    y_cm = np.mean(y)
    x = x - x_cm
    y = y - y_cm

    r2 = x**2 + y**2
    phi = np.arctan2(y, x)

    eccentricities = {}
    for n in [2, 3, 4, 5]:
        numerator = np.abs(np.mean(r2 * np.exp(1j * n * phi)))
        denominator = np.mean(r2)
        eccentricities[f"epsilon_{n}"] = numerator / (denominator + 1e-10)

    # Also return participant plane angle
    eccentricities["Psi_2"] = 0.5 * np.angle(np.mean(r2 * np.exp(2j * phi)))

    return eccentricities


# =============================================================================
# BJORKEN ENERGY DENSITY
# =============================================================================


def bjorken_energy_density(dET_dy: float, A_perp: float, tau_0: float = 1.0) -> float:
    """
    Bjorken initial energy density estimate.

    ε_Bj = (1/τ₀A_⊥) dE_T/dy |_{y=0}

    Parameters
    ----------
    dET_dy : float
        Transverse energy per unit rapidity [GeV]
    A_perp : float
        Transverse overlap area [fm²]
    tau_0 : float
        Formation time [fm/c], typically ~1 fm/c

    Returns
    -------
    epsilon : float
        Energy density [GeV/fm³]
    """
    return dET_dy / (tau_0 * A_perp)


def estimate_system_parameters(nucleus: Nucleus, centrality: float = 0.05) -> dict:
    """
    Estimate collision parameters for a given system and centrality.

    Based on experimental data and Glauber model scaling.
    """
    # Approximate scaling relations (fitted to data)
    # Central collisions

    if nucleus.name.startswith("Lead"):
        N_part = 383 * (1 - centrality)
        dNch_deta = 1940 * (1 - 0.8 * centrality)
        A_perp = 160  # fm² for central
    elif nucleus.name.startswith("Xenon"):
        N_part = 236 * (1 - centrality)
        dNch_deta = 1200 * (1 - 0.8 * centrality)
        A_perp = 100
    elif nucleus.name.startswith("Oxygen"):
        N_part = 32 * (1 - centrality)
        dNch_deta = 135 * (1 - 0.8 * centrality)
        A_perp = 15  # fm²
    elif nucleus.name.startswith("Neon"):
        N_part = 40 * (1 - centrality)
        dNch_deta = 170 * (1 - 0.8 * centrality)
        A_perp = 20
    else:
        # Generic scaling
        N_part = nucleus.A * (1 - centrality)
        dNch_deta = 5 * N_part
        A_perp = np.pi * nucleus.R0**2 * (1 - centrality)

    # Estimate dE_T/dy from dN_ch/dη (roughly 0.5-0.7 GeV per particle)
    dET_dy = 0.6 * dNch_deta

    epsilon_Bj = bjorken_energy_density(dET_dy, A_perp)

    return {
        "N_part": N_part,
        "dNch_deta": dNch_deta,
        "A_perp": A_perp,
        "dET_dy": dET_dy,
        "epsilon_Bj": epsilon_Bj,
    }


# =============================================================================
# QCD PHASE DIAGRAM
# =============================================================================


def qcd_crossover_line(mu_B: np.ndarray) -> np.ndarray:
    """
    QCD crossover temperature as function of baryon chemical potential.

    Parameterization from lattice QCD:
    T_c(μ_B) = T_c(0) * (1 - κ(μ_B/T_c)²)

    with κ ≈ 0.013
    """
    T_c0 = 0.156  # GeV at μ_B = 0
    kappa = 0.013
    return T_c0 * (1 - kappa * (mu_B / T_c0) ** 2)


def qcd_phase_boundaries() -> dict[str, np.ndarray]:
    """
    Generate QCD phase diagram boundaries for visualization.
    """
    mu_B = np.linspace(0, 0.8, 100)  # GeV

    # Crossover line (lattice QCD, valid for small μ_B)
    T_crossover = qcd_crossover_line(mu_B)

    # Hypothetical first-order line (beyond critical point)
    mu_B_crit = 0.4  # Estimated critical point location
    T_crit = qcd_crossover_line(np.array([mu_B_crit]))[0]

    # First-order transition (schematic, steeper than crossover)
    mu_B_fo = np.linspace(mu_B_crit, 0.8, 50)
    T_fo = T_crit * (1 - 0.5 * ((mu_B_fo - mu_B_crit) / (0.8 - mu_B_crit)) ** 1.5)

    return {
        "mu_B_crossover": mu_B[mu_B < mu_B_crit],
        "T_crossover": T_crossover[mu_B < mu_B_crit],
        "mu_B_critical": mu_B_crit,
        "T_critical": T_crit,
        "mu_B_first_order": mu_B_fo,
        "T_first_order": T_fo,
    }


# =============================================================================
# HYDRODYNAMIC FLOW RESPONSE
# =============================================================================


def flow_from_eccentricity(
    epsilon_n: float, n: int, eta_over_s: float = 0.12, system_size: float = 3.0
) -> float:
    """
    Estimate v_n from initial eccentricity using hydrodynamic response.

    v_n ≈ κ_n × ε_n × (response function)

    The response depends on viscosity and system size.

    Updated 2025-12: Calibrated to CMS O-O/Ne-Ne flow measurements.
    """
    # Response coefficients (calibrated to 2025 LHC data)
    # Increased from original to match CMS v2 ~ 0.06 in ultracentral O-O
    kappa = {2: 0.35, 3: 0.25, 4: 0.15, 5: 0.08}

    # Viscous damping factor
    # Reduced damping in small systems - 2025 data shows strong flow
    knudsen = eta_over_s / system_size  # Proxy for Knudsen number
    damping = np.exp(-n * knudsen * 3)  # Reduced from 5 to 3

    return kappa.get(n, 0.1) * epsilon_n * damping


def generate_flow_vs_centrality(
    nucleus: Nucleus, centrality_bins: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Generate realistic v_n vs centrality data.

    Updated 2025-12: Incorporates nuclear structure effects that create
    intrinsic eccentricity even in head-on collisions:
    - O-16: Possible tetrahedral alpha clustering (ε₂_base ~ 0.15)
    - Ne-20: Prolate deformation β₂ ~ 0.45 (ε₂_base ~ 0.25)

    Calibrated to match ALICE arXiv:2509.06428 and CMS arXiv:2510.02580.
    """
    v2 = []
    v3 = []
    v4 = []

    # System size proxy
    R_sys = nucleus.R0

    # Nuclear structure baseline eccentricity (survives to central collisions)
    # This is the key physics: nuclear structure creates ε₂ > 0 even at b=0
    #
    # Calibrated to ALICE arXiv:2509.06428 and CMS arXiv:2510.02580:
    # - O-O ultracentral v2{2} ~ 0.061
    # - Ne-Ne ultracentral v2{2} ~ 0.066
    # - Ne/O ratio ~ 1.08 (ultracentral) to ~1.05 (10% centrality)
    if nucleus.name.startswith("Oxygen"):
        # Alpha clustering creates eccentricity in central collisions
        epsilon_2_base = 0.22
        epsilon_3_base = 0.12  # Fluctuation-driven
    elif nucleus.name.startswith("Neon"):
        # Prolate deformation creates ~8% more eccentricity than O-16
        # ALICE arXiv:2509.06428 shows Ne/O ratio ~ 1.08 ultracentral
        epsilon_2_base = 0.235  # ~7% more than O-16
        epsilon_3_base = 0.10
    else:
        # Spherical nuclei: no intrinsic eccentricity
        epsilon_2_base = 0.0
        epsilon_3_base = 0.10

    for cent in centrality_bins:
        # Geometric eccentricity from impact parameter
        # Peaks around 30-40% centrality for collision geometry
        epsilon_2_geom = 0.4 * np.sin(np.pi * cent / 100) * (1 - 0.3 * cent / 100)

        # Nuclear structure contribution: strongest in central, decreases toward peripheral
        # (In peripheral collisions, geometry dominates over nuclear structure)
        structure_weight = np.exp(-cent / 50)  # Decays with centrality
        epsilon_2_struct = epsilon_2_base * structure_weight

        # Total eccentricity (structure + geometry, not simple sum due to fluctuations)
        epsilon_2 = np.sqrt(epsilon_2_struct**2 + epsilon_2_geom**2)

        # ε₃ is fluctuation-driven, roughly constant or slight increase with centrality
        epsilon_3 = epsilon_3_base * (1 + 0.5 * cent / 100)

        # ε₄ from fluctuations and nonlinear coupling
        epsilon_4 = 0.08 * (1 + 0.3 * cent / 100)

        # Add modest deformation boost for Ne-20 (prolate shape enhances v₂)
        # Effect is small: ALICE shows Ne/O ratio only ~1.08
        if nucleus.beta2 > 0:
            # Deformation effect strongest in central collisions
            deformation_boost = nucleus.beta2 * 0.08 * structure_weight  # Reduced from 0.5
            epsilon_2 = np.sqrt(epsilon_2**2 + deformation_boost**2)

        # Convert to flow
        v2.append(flow_from_eccentricity(epsilon_2, 2, ETA_OVER_S, R_sys))
        v3.append(flow_from_eccentricity(epsilon_3, 3, ETA_OVER_S, R_sys))
        v4.append(flow_from_eccentricity(epsilon_4, 4, ETA_OVER_S, R_sys))

    return {
        "centrality": centrality_bins,
        "v2": np.array(v2),
        "v3": np.array(v3),
        "v4": np.array(v4),
    }


def azimuthal_distribution(
    phi: np.ndarray, v2: float, v3: float, Psi_2: float = 0, Psi_3: float = 0
) -> np.ndarray:
    """
    Generate azimuthal particle distribution.

    dN/dφ ∝ 1 + 2v₂cos(2(φ-Ψ₂)) + 2v₃cos(3(φ-Ψ₃)) + ...
    """
    return 1 + 2 * v2 * np.cos(2 * (phi - Psi_2)) + 2 * v3 * np.cos(3 * (phi - Psi_3))


# =============================================================================
# JET QUENCHING AND ENERGY LOSS
# =============================================================================


def bdmps_energy_loss(E: float, L: float, qhat: float) -> float:
    """
    BDMPS-Z radiative energy loss.

    ⟨ΔE⟩ ∝ αs × q̂ × L²

    Parameters
    ----------
    E : float
        Initial parton energy [GeV]
    L : float
        Path length through medium [fm]
    qhat : float
        Transport coefficient [GeV²/fm]

    Returns
    -------
    Delta_E : float
        Energy loss [GeV]
    """
    alpha_s = 0.3  # Strong coupling
    # omega_c = 0.5 * qhat * L^2 is characteristic gluon energy (for reference)

    # Simplified BDMPS formula: Delta_E = alpha_s * qhat * L^2 / 4
    Delta_E = alpha_s * qhat * L**2 / 4

    # Can't lose more than you have
    return min(Delta_E, 0.9 * E)


def raa_model(pT: np.ndarray, L_eff: float, qhat: float, n_spectrum: float = 6.0) -> np.ndarray:
    """
    Calculate R_AA based on energy loss.

    R_AA ≈ (p_T / (p_T + ⟨ΔE⟩))^n

    where n is the power-law index of the spectrum.
    """
    Delta_E = np.array([bdmps_energy_loss(p, L_eff, qhat) for p in pT])

    # Shift in spectrum due to energy loss
    # This is simplified; full calculation involves fragmentation
    R_AA = (pT / (pT + Delta_E)) ** n_spectrum

    # High pT limit: R_AA → 1
    # Low pT: different physics (Cronin, shadowing)
    # Apply smooth transition
    R_AA = R_AA * (1 - np.exp(-pT / 2))  # Suppress very low pT modification
    R_AA = R_AA + (1 - R_AA) * np.exp(-pT / 50)  # Approach 1 at very high pT

    return np.clip(R_AA, 0.1, 1.2)


def generate_raa_data(
    system: str, pT_range: tuple[float, float] = (0.5, 100), n_points: int = 50
) -> dict[str, np.ndarray]:
    """
    Generate R_AA vs p_T data for a collision system.

    Calibrated to match 2025 LHC measurements:
    - CMS arXiv:2510.09864: O–O R_AA min = 0.69 ± 0.04 at pT ≈ 6 GeV
    - CMS: Ne–Ne R_AA min ≈ 0.65 at pT ≈ 6 GeV
    - ALICE: Pb–Pb R_AA min ≈ 0.15–0.20 at pT ≈ 7 GeV (0–10% central)
    """
    pT = np.logspace(np.log10(pT_range[0]), np.log10(pT_range[1]), n_points)

    # Phenomenological R_AA parameterization calibrated to data
    # R_AA(pT) = 1 - suppression_depth * f(pT) where f peaks around pT_peak
    params = {
        "pp": {"suppression_max": 0.0, "pT_peak": 6.0, "width": 3.0, "high_pT_recovery": 50},
        "pPb": {"suppression_max": 0.05, "pT_peak": 4.0, "width": 2.5, "high_pT_recovery": 30},
        "OO": {
            "suppression_max": 0.31,
            "pT_peak": 6.0,
            "width": 4.0,
            "high_pT_recovery": 40,
        },  # CMS: 0.69 min
        "NeNe": {
            "suppression_max": 0.38,
            "pT_peak": 6.0,
            "width": 4.0,
            "high_pT_recovery": 45,
        },  # CMS: ~0.65 min at 6 GeV
        "XeXe": {"suppression_max": 0.65, "pT_peak": 7.0, "width": 5.0, "high_pT_recovery": 60},
        "PbPb": {
            "suppression_max": 0.82,
            "pT_peak": 7.0,
            "width": 5.0,
            "high_pT_recovery": 80,
        },  # ALICE: 0.15-0.20 min
    }

    p = params.get(system, params["OO"])

    # Suppression profile: peaks around pT_peak, rises at low and high pT
    # Low-pT rise (Cronin-like): exp(-(pT - pT_peak)^2 / width^2) for pT < pT_peak
    # High-pT recovery: (1 - exp(-pT / recovery_scale))

    suppression_profile = np.exp(-(((pT - p["pT_peak"]) / p["width"]) ** 2))
    # Suppress at very low pT (soft physics dominates, no suppression)
    low_pT_cutoff = 1.0 / (1.0 + np.exp(-(pT - 1.5) / 0.5))
    # High pT recovery toward unity
    high_pT_recovery = 1.0 - np.exp(-pT / p["high_pT_recovery"])

    suppression = (
        p["suppression_max"] * suppression_profile * low_pT_cutoff * (1.0 - 0.5 * high_pT_recovery)
    )
    R_AA = 1.0 - suppression

    # Ensure physical bounds
    R_AA = np.clip(R_AA, 0.1, 1.1)

    # Add realistic errors (CMS systematic: 3-6%)
    rel_err = 0.04 + 0.02 * np.exp(-pT / 10)
    err = np.maximum(R_AA * rel_err, 0.02)

    return {
        "pT": pT,
        "R_AA": R_AA,
        "err": err,
    }


# =============================================================================
# STRANGENESS ENHANCEMENT - CANONICAL SUPPRESSION
# =============================================================================


def canonical_suppression_factor(strangeness: int, x: float) -> float:
    """
    Canonical suppression factor for strangeness.

    γ_S = I_{|S|}(x) / I_0(x)

    where x depends on the volume and strange quark density.
    In the grand-canonical limit (x → ∞), γ_S → 1.
    """
    S = abs(strangeness)
    return bessel_i(S, x) / bessel_i(0, x)


def strangeness_enhancement_curve(dNch_deta: np.ndarray) -> dict[str, np.ndarray]:
    """
    Generate strangeness enhancement vs multiplicity.

    The enhancement factor increases from pp to Pb-Pb as
    canonical suppression is lifted.
    """
    # x parameter scales with volume ∝ multiplicity
    # Calibrated to match experimental data
    x = 0.1 * dNch_deta**0.6

    # Single strange (K, Λ)
    enhancement_S1 = canonical_suppression_factor(1, x)

    # Double strange (Ξ)
    enhancement_S2 = canonical_suppression_factor(2, x)

    # Triple strange (Ω)
    enhancement_S3 = canonical_suppression_factor(3, x)

    # Normalize to pp baseline (low multiplicity)
    x_pp = 0.1 * 10**0.6
    norm_S1 = canonical_suppression_factor(1, x_pp)
    norm_S2 = canonical_suppression_factor(2, x_pp)
    norm_S3 = canonical_suppression_factor(3, x_pp)

    return {
        "dNch_deta": dNch_deta,
        "enhancement_K": enhancement_S1 / norm_S1,
        "enhancement_Lambda": enhancement_S1 / norm_S1,
        "enhancement_Xi": enhancement_S2 / norm_S2,
        "enhancement_Omega": enhancement_S3 / norm_S3,
    }


# =============================================================================
# SPACETIME EVOLUTION
# =============================================================================


def temperature_evolution(tau: np.ndarray, T_0: float, tau_0: float = 0.5) -> np.ndarray:
    """
    Temperature evolution in Bjorken expansion.

    T(τ) = T₀ × (τ₀/τ)^{1/3}

    For ideal gas (cs² = 1/3).
    """
    return T_0 * (tau_0 / tau) ** (1 / 3)


def energy_density_profile_2d(
    x: np.ndarray, y: np.ndarray, nucleus_A: Nucleus, nucleus_B: Nucleus, b: float
) -> np.ndarray:
    """
    Initial energy density profile in transverse plane.

    ε(x,y) ∝ T_A(x,y) × T_B(x-b,y)

    where T_A,B are nuclear thickness functions.
    """
    X, Y = np.meshgrid(x, y)

    # Thickness function: integral of Woods-Saxon along z
    def thickness(xx, yy, nuc, x_shift=0):
        r_perp = np.sqrt((xx - x_shift) ** 2 + yy**2)
        # Approximate integral
        return woods_saxon(r_perp, nuc) * 2 * nuc.R0

    T_A = thickness(X, Y, nucleus_A)
    T_B = thickness(X, Y, nucleus_B, x_shift=b)

    # Binary collision density
    epsilon = T_A * T_B

    # Normalize
    epsilon = epsilon / np.max(epsilon)

    return epsilon


# =============================================================================
# ALPHA CLUSTERING IN OXYGEN-16
# =============================================================================


def oxygen_alpha_cluster_positions() -> np.ndarray:
    """
    Generate O-16 alpha cluster configuration (tetrahedral).

    Four alpha particles at vertices of a tetrahedron.
    """
    # Tetrahedron vertices (edge length ~ 3.5 fm from nuclear physics)
    a = 1.8  # Scale factor [fm]

    vertices = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) * a

    # Each alpha has 4 nucleons in a small cluster
    nucleon_positions = []
    alpha_size = 0.5  # fm, internal alpha size

    for v in vertices:
        # 4 nucleons per alpha, randomly distributed
        for _ in range(4):
            offset = np.random.normal(0, alpha_size, 3)
            nucleon_positions.append(v + offset)

    return np.array(nucleon_positions)


# =============================================================================
# EXPORT FUNCTIONS FOR DATA FILES
# =============================================================================


def export_for_pgfplots(data: dict, filename: str, header: str = ""):
    """Export data dictionary to space-separated file for pgfplots."""

    # Get arrays and ensure same length
    keys = list(data.keys())
    arrays = [np.atleast_1d(data[k]) for k in keys]

    with open(filename, "w") as f:
        f.write(f"# {header}\n")
        f.write("# " + " ".join(keys) + "\n")
        for i in range(len(arrays[0])):
            row = [str(arr[i] if i < len(arr) else arr[-1]) for arr in arrays]
            f.write(" ".join(row) + "\n")


if __name__ == "__main__":
    # Quick test of the physics module
    print("QGP Physics Module - Self-test")
    print("=" * 50)

    # Test Woods-Saxon
    r = np.linspace(0, 10, 50)
    for name, nuc in [("O-16", NUCLEI["O"]), ("Pb-208", NUCLEI["Pb"])]:
        rho = woods_saxon(r, nuc)
        print(f"{name}: R_0 = {nuc.R0:.2f} fm, ρ(0) = {rho[0]:.3f}")

    # Test Bjorken energy density
    params = estimate_system_parameters(NUCLEI["O"], 0.05)
    print(f"\nO-O central: ε_Bj = {params['epsilon_Bj']:.1f} GeV/fm³")

    params = estimate_system_parameters(NUCLEI["Pb"], 0.05)
    print(f"Pb-Pb central: ε_Bj = {params['epsilon_Bj']:.1f} GeV/fm³")

    # Test R_AA
    raa_OO = generate_raa_data("OO")
    print(f"\nO-O R_AA at 6 GeV: {np.interp(6, raa_OO['pT'], raa_OO['R_AA']):.2f}")

    raa_PbPb = generate_raa_data("PbPb")
    print(f"Pb-Pb R_AA at 6 GeV: {np.interp(6, raa_PbPb['pT'], raa_PbPb['R_AA']):.2f}")

    print("\nPhysics module loaded successfully!")
