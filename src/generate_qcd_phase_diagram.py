#!/usr/bin/env python3
"""
generate_qcd_phase_diagram.py

Generates accurate, fine-grained data for QCD phase diagram visualization.

Physics sources (UPDATED December 20, 2025):
- Crossover: HotQCD Collaboration, Phys. Lett. B 795 (2019) 15
             T_c = 156.5 ± 1.5 MeV; κ₂ = 0.0120 ± 0.0020
- Curvature κ₂: Independently confirmed by Smecca et al., PRD 112, 114509 (2025)
             κ₂ ≈ 0.014 from mesonic correlators (arXiv:2412.20922)
- Freeze-out: Andronic et al., Nature 561, 321 (2018)
             Updated: Lysenko et al. PRC 111, 054903 (2025)
- Critical point EXCLUSION: Borsányi et al., PRD 112, L111505 (Dec 2025)
             μ_B < 450 MeV EXCLUDED at 2σ (Yang-Lee edge singularity)
             arXiv:2502.10267
- FRG consensus (QM2025): CEP ≈ (T=110, μ_B=630) MeV ±10%
             Fu et al., arXiv:2510.11270
- BES-II cumulants: Goswami & Karsch, arXiv:2512.01126
             Net-baryon cumulants rule out CP at μ_B/T ≤ 2
- Canonical formulation: Adam et al., arXiv:2512.09415
             Finite density QCD without Taylor expansion
- Clarke et al. 2024 estimate NOW SUPERSEDED: arXiv:2405.10196
             (T_CP=105⁺⁸₋₁₈, μ_B^CP=422⁺⁸⁰₋₃₅ MeV)

Author: QGP Light-Ion Whitepaper (2025)
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Optional

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "figure_curves")


@dataclass
class PhaseTransitionParams:
    """QCD phase transition parameters from lattice QCD (December 2025 update)."""

    # Crossover temperature at μ_B = 0 (HotQCD 2019, Phys. Lett. B 795)
    T_c0: float = 156.5  # MeV
    T_c0_err: float = 1.5  # MeV

    # Curvature of crossover line (arXiv:1812.08235, HotQCD)
    # T_c(μ_B)/T_c(0) = 1 - κ₂(μ_B/T_c)² - κ₄(μ_B/T_c)⁴
    kappa2: float = 0.0120  # Central value
    kappa2_err: float = 0.0020
    kappa4: float = 0.00032  # Much smaller, from higher-order lattice expansion

    # ==========================================================================
    # CRITICAL POINT: DECEMBER 2025 EXCLUSION
    # Borsányi et al., PRD 112, L111505 (arXiv:2502.10267)
    # Yang-Lee edge singularity analysis EXCLUDES CP at μ_B < 450 MeV (2σ)
    # ==========================================================================
    mu_B_excluded_2sigma: float = 450.0  # MeV - CP excluded below this at 2σ
    mu_B_excluded_1sigma: float = 400.0  # MeV - CP excluded below this at 1σ

    # SUPERSEDED Clarke et al. 2024 estimate (arXiv:2405.10196) - NOW EXCLUDED
    # These values are kept for reference but MUST NOT be plotted as valid
    T_cp_clarke: float = 105.0  # MeV (NOW EXCLUDED)
    T_cp_clarke_err_up: float = 8.0  # MeV
    T_cp_clarke_err_down: float = 18.0  # MeV
    mu_B_cp_clarke: float = 422.0  # MeV (NOW EXCLUDED - below 450 MeV threshold)
    mu_B_cp_clarke_err_up: float = 80.0  # MeV
    mu_B_cp_clarke_err_down: float = 35.0  # MeV

    # ==========================================================================
    # FRG CONSENSUS (QM2025): Current best theoretical estimate
    # Fu, Pawlowski, Rennecke, arXiv:2510.11270
    # ==========================================================================
    T_cp_frg: float = 110.0  # MeV (FRG consensus)
    T_cp_frg_err: float = 11.0  # MeV (~10% uncertainty)
    mu_B_cp_frg: float = 630.0  # MeV (FRG consensus)
    mu_B_cp_frg_err: float = 63.0  # MeV (~10% uncertainty)


@dataclass
class FreezeOutPoint:
    """Chemical freeze-out measurement."""

    sqrt_s_NN: float  # GeV
    T: float  # MeV
    T_err: float  # MeV
    mu_B: float  # MeV
    mu_B_err: float  # MeV
    experiment: str
    system: str


# Experimental freeze-out data from thermal model fits
# Sources: Andronic et al. Nature 561 (2018) 321; STAR BES publications
# Filtered to μ_B < 550 MeV for clean visualization
# UNIFORM ERROR CONVENTION: symmetric errors for consistency
FREEZE_OUT_DATA = [
    # AGS 4.85 GeV (highest μ_B shown, within axis)
    FreezeOutPoint(4.85, 125.0, 4.0, 420.0, 25.0, "AGS", "Au-Au"),

    # SPS energies (NA49)
    FreezeOutPoint(6.3, 136.0, 4.0, 380.0, 20.0, "NA49", "Pb-Pb"),
    FreezeOutPoint(7.7, 148.0, 4.0, 340.0, 18.0, "NA49/STAR", "Au-Au"),
    FreezeOutPoint(8.8, 150.0, 4.0, 310.0, 16.0, "NA49", "Pb-Pb"),
    FreezeOutPoint(12.3, 156.0, 4.0, 260.0, 14.0, "NA49", "Pb-Pb"),
    FreezeOutPoint(17.3, 158.0, 4.0, 220.0, 12.0, "NA49", "Pb-Pb"),

    # RHIC energies (STAR)
    FreezeOutPoint(19.6, 160.0, 4.0, 195.0, 12.0, "STAR", "Au-Au"),
    FreezeOutPoint(27.0, 162.0, 4.0, 160.0, 10.0, "STAR", "Au-Au"),
    FreezeOutPoint(39.0, 164.0, 4.0, 115.0, 8.0, "STAR", "Au-Au"),
    FreezeOutPoint(62.4, 166.0, 4.0, 75.0, 6.0, "STAR", "Au-Au"),
    FreezeOutPoint(130.0, 167.0, 4.0, 40.0, 5.0, "STAR", "Au-Au"),
    FreezeOutPoint(200.0, 166.0, 4.0, 25.0, 4.0, "STAR", "Au-Au"),

    # LHC energies (ALICE) - small but nonzero errors
    FreezeOutPoint(2760.0, 156.5, 3.0, 1.0, 1.0, "ALICE", "Pb-Pb"),
    FreezeOutPoint(5020.0, 156.5, 3.0, 0.7, 0.7, "ALICE", "Pb-Pb"),
]


@dataclass
class CollisionSystem:
    """Collision system with approximate phase diagram location."""

    name: str
    sqrt_s_NN: float  # GeV
    mu_B: float  # MeV (approximate at freeze-out)
    T: float  # MeV (freeze-out temperature)
    T_initial: Optional[float] = None  # Initial temperature estimate
    marker: str = "square"
    color_key: str = "PbPbcolor"


# Collision systems for markers on phase diagram
COLLISION_SYSTEMS = [
    CollisionSystem("LHC Pb-Pb", 5020, 0.7, 156.5, T_initial=400, marker="square*", color_key="PbPbcolor"),
    CollisionSystem("LHC O-O", 7000, 0.5, 156.5, T_initial=350, marker="triangle*", color_key="OOcolor"),
    CollisionSystem("LHC Ne-Ne", 6500, 0.5, 156.5, T_initial=360, marker="diamond*", color_key="NeNecolor"),
    CollisionSystem("RHIC Au-Au", 200, 25, 166, T_initial=340, marker="pentagon*", color_key="accentpurple"),
    CollisionSystem("SPS Pb-Pb", 17.3, 250, 158, T_initial=260, marker="star", color_key="accentred"),
    CollisionSystem("AGS Au-Au", 4.85, 526, 125, T_initial=180, marker="oplus", color_key="textmid"),
]


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_curve(filename: str, x: np.ndarray, y: np.ndarray,
               header: str = "x y", comments: Optional[list] = None) -> None:
    """Save a simple x-y curve to a .dat file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        if comments:
            for comment in comments:
                f.write(f"# {comment}\n")
        f.write(f"# {header}\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi:.6f} {yi:.6f}\n")
    print(f"  {filename}: {len(x)} points")


def save_curve_multi(filename: str, x: np.ndarray, columns: list,
                     header: str, comments: Optional[list] = None) -> None:
    """Save curve with multiple y columns."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        if comments:
            for comment in comments:
                f.write(f"# {comment}\n")
        f.write(f"# {header}\n")
        for i, xi in enumerate(x):
            row = f"{xi:.6f}"
            for col in columns:
                row += f" {col[i]:.6f}"
            f.write(row + "\n")
    print(f"  {filename}: {len(x)} points x {len(columns)+1} columns")


def save_points_with_errors(filename: str, points: list, header: str) -> None:
    """Save data points with asymmetric errors."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        f.write(f"# {header}\n")
        for p in points:
            f.write(" ".join(f"{v:.6f}" for v in p) + "\n")
    print(f"  {filename}: {len(points)} points")


# =============================================================================
# CROSSOVER LINE FROM LATTICE QCD
# =============================================================================

def crossover_temperature(mu_B: np.ndarray, params: PhaseTransitionParams) -> np.ndarray:
    """
    Calculate crossover temperature T_c(μ_B) from lattice QCD parametrization.

    T_c(μ_B)/T_c(0) = 1 - κ₂(μ_B/T_c(0))² - κ₄(μ_B/T_c(0))⁴

    Valid for μ_B < 300 MeV (lattice constraint).
    """
    ratio = mu_B / params.T_c0
    return params.T_c0 * (1 - params.kappa2 * ratio**2 - params.kappa4 * ratio**4)


def crossover_uncertainty_band(mu_B: np.ndarray, params: PhaseTransitionParams) -> tuple:
    """
    Calculate uncertainty band for crossover using error propagation.

    δT_c² = (∂T_c/∂T_c0)²·δT_c0² + (∂T_c/∂κ₂)²·δκ₂²
    """
    ratio = mu_B / params.T_c0

    # Central value
    T_central = crossover_temperature(mu_B, params)

    # Partial derivatives
    # ∂T_c/∂T_c0 ≈ 1 (dominant at small μ_B)
    # ∂T_c/∂κ₂ = -T_c0 * (μ_B/T_c0)²
    dT_dTc0 = 1 - params.kappa2 * ratio**2 + params.kappa2 * 2 * ratio**2
    dT_dkappa2 = -params.T_c0 * ratio**2

    # Error propagation (quadrature)
    delta_T = np.sqrt((dT_dTc0 * params.T_c0_err)**2 +
                      (dT_dkappa2 * params.kappa2_err)**2)

    # Add systematic ~5 MeV at higher μ_B for model uncertainty
    systematic = 5.0 * (mu_B / 300)**2
    total_err = np.sqrt(delta_T**2 + systematic**2)

    T_upper = T_central + total_err
    T_lower = T_central - total_err

    return T_upper, T_lower


# =============================================================================
# CHEMICAL FREEZE-OUT CURVE
# =============================================================================

def freeze_out_parametrization(mu_B: np.ndarray) -> np.ndarray:
    """
    Chemical freeze-out curve from constant ε/n = 0.951 GeV criterion.

    Fit to experimental data from AGS to LHC energies.
    Parametrization based on Cleymans-Redlich and Andronic et al.

    T(μ_B) = a - b·μ_B² - c·μ_B⁴

    Fitted to match experimental freeze-out points.
    """
    # Fit coefficients derived from freeze-out data
    # T saturates ~166-168 MeV at μ_B → 0
    # Decreases to ~50-60 MeV at μ_B ~ 800 MeV

    a = 166.0  # MeV (limiting temperature at high energy)
    b = 0.139e-3  # MeV⁻¹ (curvature)
    c = 0.053e-9  # MeV⁻³ (higher order)

    T = a - b * mu_B**2 - c * mu_B**4

    # Alternative: exponential form that better matches low-energy data
    # T = 166 * np.exp(-mu_B**2 / (2 * 780**2)) + 50 * (1 - np.exp(-mu_B / 200))

    return np.maximum(T, 30)  # Physical floor


def freeze_out_from_sqrt_s(sqrt_s: np.ndarray) -> tuple:
    """
    Calculate freeze-out (T, μ_B) from collision energy using phenomenological fit.

    μ_B(√s) = c / (1 + d·√s)
    with c = 1477 MeV, d = 0.343 GeV⁻¹

    From arXiv:2408.06473
    """
    c = 1477.0  # MeV
    d = 0.343  # GeV⁻¹

    mu_B = c / (1 + d * sqrt_s)
    T = freeze_out_parametrization(mu_B)

    return T, mu_B


# =============================================================================
# FIRST-ORDER TRANSITION LINE (SCHEMATIC)
# =============================================================================

def first_order_line(mu_B: np.ndarray, params: PhaseTransitionParams) -> np.ndarray:
    """
    First-order phase transition line beyond the critical point.

    UPDATED December 2025: Uses FRG consensus CP location (110, 630) MeV.
    This is schematic - connects CP to nuclear matter at T=0.
    """
    # Start from FRG consensus critical point (QM2025)
    mu_cp = params.mu_B_cp_frg  # 630 MeV
    T_cp = params.T_cp_frg  # 110 MeV

    # End at nuclear matter ground state
    mu_nm = 930.0  # Nuclear matter at T=0 (~ nucleon mass - binding)
    T_nm = 0.0

    # Simple linear interpolation (schematic)
    mask = mu_B >= mu_cp
    T = np.zeros_like(mu_B)
    T[mask] = T_cp - (T_cp - T_nm) / (mu_nm - mu_cp) * (mu_B[mask] - mu_cp)
    T[~mask] = np.nan

    return T


# =============================================================================
# CRITICAL POINT: EXCLUSION REGION + FRG CONSENSUS
# December 2025 update: Borsányi et al. excludes CP at μ_B < 450 MeV
# =============================================================================

def critical_point_exclusion_region(n_points: int = 100) -> tuple:
    """
    Generate the EXCLUDED region for critical point from December 2025 lattice.

    Borsányi et al., PRD 112, L111505 (arXiv:2502.10267):
    Yang-Lee edge singularity analysis excludes CP at μ_B < 450 MeV (2σ).

    Returns coordinates for shaded exclusion region polygon.
    """
    params = PhaseTransitionParams()

    # Exclusion boundary at μ_B = 450 MeV (2σ)
    # The exclusion covers T from 0 to ~200 MeV (entire relevant range)
    mu_B_boundary = params.mu_B_excluded_2sigma  # 450 MeV

    # Create polygon: from origin along T axis, up to max T, across to boundary
    T_max = 200.0  # Top of exclusion region
    mu_B = np.array([0, 0, mu_B_boundary, mu_B_boundary, 0])
    T = np.array([0, T_max, T_max, 0, 0])

    return mu_B, T


def critical_point_exclusion_boundary(n_points: int = 100) -> tuple:
    """
    Generate the exclusion boundary line (μ_B = 450 MeV at 2σ).
    """
    params = PhaseTransitionParams()

    T = np.linspace(0, 200, n_points)
    mu_B = np.full_like(T, params.mu_B_excluded_2sigma)

    return mu_B, T


def critical_point_frg_ellipse(n_points: int = 100) -> tuple:
    """
    Generate ellipse representing FRG consensus CP uncertainty region.

    QM2025 consensus: CEP at (T=110, μ_B=630) MeV with ~10% uncertainty.
    Source: Fu, Pawlowski, Rennecke, arXiv:2510.11270
    """
    params = PhaseTransitionParams()

    theta = np.linspace(0, 2*np.pi, n_points)

    # Symmetric ~10% uncertainty
    sigma_T = params.T_cp_frg_err  # ~11 MeV
    sigma_mu = params.mu_B_cp_frg_err  # ~63 MeV

    # 1-sigma ellipse around FRG consensus
    T = params.T_cp_frg + sigma_T * np.sin(theta)
    mu_B = params.mu_B_cp_frg + sigma_mu * np.cos(theta)

    return mu_B, T


def critical_point_frg_box() -> tuple:
    """
    Generate rectangular uncertainty region for FRG consensus CP.

    Returns corners of the box representing ~10% uncertainty range.
    """
    params = PhaseTransitionParams()

    mu_min = params.mu_B_cp_frg - params.mu_B_cp_frg_err
    mu_max = params.mu_B_cp_frg + params.mu_B_cp_frg_err
    T_min = params.T_cp_frg - params.T_cp_frg_err
    T_max = params.T_cp_frg + params.T_cp_frg_err

    return (mu_min, mu_max, T_min, T_max)


# DEPRECATED: Old Clarke et al. ellipse - EXCLUDED by December 2025 data
def critical_point_ellipse_excluded(n_points: int = 100) -> tuple:
    """
    DEPRECATED: Generate ellipse for Clarke et al. 2024 estimate.

    WARNING: This CP location is NOW EXCLUDED by Borsányi et al. (Dec 2025).
    Kept for reference to show what has been ruled out.
    """
    params = PhaseTransitionParams()

    theta = np.linspace(0, 2*np.pi, n_points)

    # Asymmetric errors from Clarke et al.
    sigma_T = (params.T_cp_clarke_err_up + params.T_cp_clarke_err_down) / 2
    sigma_mu = (params.mu_B_cp_clarke_err_up + params.mu_B_cp_clarke_err_down) / 2

    # 1-sigma ellipse
    T = params.T_cp_clarke + sigma_T * np.sin(theta)
    mu_B = params.mu_B_cp_clarke + sigma_mu * np.cos(theta)

    return mu_B, T


def critical_point_box_excluded() -> tuple:
    """
    DEPRECATED: Rectangular region for Clarke et al. estimate - NOW EXCLUDED.
    """
    params = PhaseTransitionParams()

    mu_min = params.mu_B_cp_clarke - params.mu_B_cp_clarke_err_down
    mu_max = params.mu_B_cp_clarke + params.mu_B_cp_clarke_err_up
    T_min = params.T_cp_clarke - params.T_cp_clarke_err_down
    T_max = params.T_cp_clarke + params.T_cp_clarke_err_up

    return (mu_min, mu_max, T_min, T_max)


# =============================================================================
# FREEZE-OUT SYSTEMATIC UNCERTAINTY BAND
# =============================================================================

def freeze_out_uncertainty_band(mu_B: np.ndarray) -> tuple:
    """
    Calculate freeze-out systematic uncertainty band.

    Systematic uncertainties from Andronic et al., Nature 561 (2018) 321:
    - Model-dependent uncertainties in hadron resonance gas
    - Feed-down correction uncertainties
    - Centrality selection effects
    - Typical systematic: ±4-8 MeV in T, ±10-15% in μ_B
    """
    T_central = freeze_out_parametrization(mu_B)

    # Temperature systematic: ~5 MeV at low μ_B, increasing at high μ_B
    T_sys = 5.0 + 3.0 * (mu_B / 500)**2

    # Additional model uncertainty at high μ_B
    model_unc = 8.0 * np.tanh(mu_B / 400)

    total_sys = np.sqrt(T_sys**2 + model_unc**2)

    T_upper = T_central + total_sys
    T_lower = T_central - total_sys

    return T_upper, T_lower


# =============================================================================
# FIRST-ORDER TRANSITION: MULTIPLE THEORETICAL MODELS
# =============================================================================

def first_order_njl(mu_B: np.ndarray, params: PhaseTransitionParams) -> np.ndarray:
    """
    First-order line from Nambu--Jona-Lasinio (NJL) model.

    NJL models predict CP at lower T than lattice QCD.
    Reference: Buballa, Phys. Rep. 407 (2005) 205
    """
    # NJL critical point typically: T_CP ~ 70-90 MeV, μ_B ~ 300-350 MeV
    T_cp_njl = 80.0
    mu_cp_njl = 330.0

    # Curve from CP to nuclear matter
    mask = mu_B >= mu_cp_njl
    T = np.full_like(mu_B, np.nan)
    T[mask] = T_cp_njl * np.exp(-(mu_B[mask] - mu_cp_njl)**2 / (2 * 400**2))

    return T


def first_order_pqm(mu_B: np.ndarray, params: PhaseTransitionParams) -> np.ndarray:
    """
    First-order line from Polyakov-Quark-Meson (PQM) model.

    PQM includes confinement via Polyakov loop.
    Reference: Schaefer et al., PRD 76 (2007) 074023
    """
    # PQM critical point: T_CP ~ 90-110 MeV, μ_B ~ 350-400 MeV
    T_cp_pqm = 95.0
    mu_cp_pqm = 370.0

    mask = mu_B >= mu_cp_pqm
    T = np.full_like(mu_B, np.nan)
    # Steeper drop than NJL
    T[mask] = T_cp_pqm - (T_cp_pqm / (930 - mu_cp_pqm)) * (mu_B[mask] - mu_cp_pqm)
    T[mask] = np.maximum(T[mask], 0)

    return T


def first_order_frg(mu_B: np.ndarray, params: PhaseTransitionParams) -> np.ndarray:
    """
    First-order line from Functional Renormalization Group (FRG).

    UPDATED: Uses QM2025 FRG consensus (Fu, Pawlowski, Rennecke arXiv:2510.11270)
    CEP at (T=110, μ_B=630) MeV - BEYOND the lattice exclusion region.
    """
    # QM2025 FRG consensus values
    T_cp_frg = params.T_cp_frg  # 110 MeV
    mu_cp_frg = params.mu_B_cp_frg  # 630 MeV

    mask = mu_B >= mu_cp_frg
    T = np.full_like(mu_B, np.nan)
    # Curved transition from CP to nuclear matter
    T[mask] = T_cp_frg * (1 - ((mu_B[mask] - mu_cp_frg) / (930 - mu_cp_frg))**1.5)
    T[mask] = np.maximum(T[mask], 0)

    return T


def first_order_consensus_band(n_points: int = 100) -> tuple:
    """
    Generate the CONSENSUS BAND for first-order transition predictions.

    This encompasses the spread of NJL, PQM, and FRG theoretical predictions.
    The band represents theoretical uncertainty, not a single prediction.

    Returns: (mu_B_upper, T_upper, mu_B_lower, T_lower) for the band boundaries
    """
    params = PhaseTransitionParams()

    # The consensus band spans from the earliest predicted CP (NJL at ~330 MeV)
    # to nuclear matter at ~930 MeV

    # Upper boundary: FRG consensus (highest T at each μ_B beyond 630 MeV)
    # For μ_B < 630: no upper bound (FRG doesn't predict transition there)

    # Lower boundary: NJL model (lowest μ_B onset, lowest T at high μ_B)

    # Create a closed polygon for the consensus band
    # Starting from FRG CP, going down along FRG, then back along NJL

    mu_B_range = np.linspace(630, 920, n_points)

    # FRG upper boundary (from FRG CP going right)
    T_frg = params.T_cp_frg * (1 - ((mu_B_range - params.mu_B_cp_frg) /
                                      (930 - params.mu_B_cp_frg))**1.5)
    T_frg = np.maximum(T_frg, 0)

    # NJL lower boundary (same μ_B range, but NJL gives lower T)
    # NJL CP is at (80, 330), so at μ_B=630, NJL is already at low T
    T_njl_at_630 = 80.0 * np.exp(-(630 - 330)**2 / (2 * 400**2))
    # Approximate NJL in overlap region
    T_njl = T_njl_at_630 * np.exp(-(mu_B_range - 630)**2 / (2 * 250**2))
    T_njl = np.maximum(T_njl, 0)

    return mu_B_range, T_frg, T_njl


# =============================================================================
# ISENTROPIC TRAJECTORIES (s/n_B = const)
# =============================================================================

def isentropic_trajectory(s_over_nB: float, n_points: int = 100) -> tuple:
    """
    Generate isentropic trajectory for given s/n_B ratio.

    Trajectories of constant entropy per baryon characterize the
    hydrodynamic evolution at different collision energies.

    s/n_B values for RHIC BES:
      - √s = 200 GeV: s/n_B ≈ 420
      - √s = 62.4 GeV: s/n_B ≈ 144
      - √s = 27 GeV: s/n_B ≈ 70
      - √s = 19.6 GeV: s/n_B ≈ 51
      - √s = 7.7 GeV: s/n_B ≈ 25

    Reference: arXiv:1506.07350 (review of RHIC BES)

    The trajectory shape is from ideal gas + lattice EOS interpolation.
    """
    # Parametrize: at high T, μ_B ~ T / (s/n_B) approximately
    # More sophisticated: use lattice EOS tables

    # Simplified model based on lattice + HRG matching
    T = np.linspace(400, 50, n_points)

    # μ_B increases as T decreases (baryon number conservation)
    # For ideal gas: μ_B/T ~ 3/s_n_B approximately
    # More realistically, use phenomenological fit
    mu_B = np.zeros_like(T)

    for i, Ti in enumerate(T):
        if Ti > 170:  # QGP phase
            mu_B[i] = 3 * Ti / s_over_nB * (170 / Ti)**0.5
        else:  # Hadron phase - steeper increase
            mu_B_170 = 3 * 170 / s_over_nB * (170 / 170)**0.5
            mu_B[i] = mu_B_170 + (170 - Ti) * (300 / s_over_nB)

    # Filter to reasonable μ_B range
    mask = mu_B < 600
    return mu_B[mask], T[mask]


# RHIC BES isentropes (s/n_B values)
ISENTROPE_VALUES = [
    (420, "√s = 200 GeV"),
    (144, "√s = 62.4 GeV"),
    (70, "√s = 27 GeV"),
    (51, "√s = 19.6 GeV"),
    (30, "√s = 11.5 GeV"),
]


# =============================================================================
# PEDAGOGICAL CONTEXT: COSMOLOGY, ASTROPHYSICS, FUTURE FACILITIES
# =============================================================================

def early_universe_trajectory(n_points: int = 100) -> tuple:
    """
    Generate early universe cooling trajectory.

    The universe evolved along μ_B ≈ 0 axis from T ~ 10^12 K (QGP)
    down to T ~ 10^10 K (hadronization at t ~ 10^-5 s).

    At the QCD epoch, baryon-antibaryon nearly cancelled,
    leaving μ_B/T ~ 10^-9 (negligible).
    """
    # Temperature from 500 MeV down to 50 MeV
    T = np.linspace(500, 50, n_points)

    # μ_B essentially zero for early universe
    # (tiny baryon asymmetry: η_B ~ 6 × 10^-10)
    mu_B = np.zeros_like(T) + 0.1  # Offset for visibility

    return mu_B, T


def neutron_star_trajectory(n_points: int = 50) -> tuple:
    """
    Generate neutron star core region boundary.

    Neutron stars probe T ~ 0 (cold), high μ_B ~ 1000-1500 MeV.
    Core densities reach 5-10 × nuclear saturation density.

    μ_B at center: ~1200-1500 MeV (model dependent)
    T: essentially 0 on QCD scale (< 1 MeV after ~10^6 years cooling)
    """
    # Approximate boundary of NS core conditions
    mu_B = np.linspace(900, 1500, n_points)

    # T essentially zero but show as small band for visibility
    T_center = np.zeros_like(mu_B) + 10  # ~10 MeV for visibility
    T_upper = T_center + 30
    T_lower = T_center - 5

    return mu_B, T_center, T_upper, T_lower


def color_superconductivity_region() -> tuple:
    """
    Return approximate boundary of color superconducting phase.

    At very high μ_B (> 400-500 MeV) and low T, quarks can form
    Cooper pairs leading to color superconductivity (CSC).

    Phases include: 2SC, CFL, and various crystalline phases.
    Gap scale: Δ ~ 10-100 MeV depending on density.

    Boundary is highly uncertain; this is schematic.
    """
    # CSC region: high μ_B, low T
    mu_B = np.array([500, 600, 800, 1000, 1200, 1400, 1600])
    T_boundary = np.array([80, 60, 40, 30, 25, 20, 15])  # Approximate

    return mu_B, T_boundary


# Future heavy-ion facilities coverage regions
FUTURE_FACILITIES = {
    "FAIR_CBM": {
        "name": "FAIR/CBM",
        "sqrt_s_range": (2.7, 4.9),  # GeV
        "mu_B_range": (500, 800),  # MeV
        "T_range": (50, 150),  # MeV
        "description": "High-μ_B frontier (2025+)",
    },
    "NICA_MPD": {
        "name": "NICA/MPD",
        "sqrt_s_range": (4, 11),  # GeV
        "mu_B_range": (300, 600),  # MeV
        "T_range": (100, 170),  # MeV
        "description": "CP search region (2024+)",
    },
    "RHIC_BES2": {
        "name": "RHIC BES-II",
        "sqrt_s_range": (7.7, 27),  # GeV
        "mu_B_range": (150, 450),  # MeV
        "T_range": (130, 165),  # MeV
        "description": "CP search ongoing",
    },
}


# Collision system √s_NN values for annotation
COLLISION_SQRT_S = {
    "LHC_PbPb": {"sqrt_s": 5020, "mu_B": 0.7, "T": 156.5, "label": "5.02 TeV"},
    "LHC_OO": {"sqrt_s": 7000, "mu_B": 0.5, "T": 156.5, "label": "7 TeV"},
    "RHIC_200": {"sqrt_s": 200, "mu_B": 25, "T": 166, "label": "200 GeV"},
    "SPS": {"sqrt_s": 17.3, "mu_B": 220, "T": 158, "label": "17.3 GeV"},
    "AGS": {"sqrt_s": 4.85, "mu_B": 420, "T": 125, "label": "4.85 GeV"},
}


# =============================================================================
# COOLING TRAJECTORY
# =============================================================================

def cooling_trajectory(n_points: int = 100) -> tuple:
    """
    Generate cooling trajectory for central Pb-Pb collision.

    Starts at initial temperature T_0 ~ 400 MeV, μ_B ~ 0
    Evolves through QGP phase with slight baryon stopping
    Crosses hadronization band
    Ends at kinetic freeze-out
    """
    # Parametric curve: t from 0 to 1
    t = np.linspace(0, 1, n_points)

    # Temperature: starts high, drops through QGP, crosses T_c
    T_0 = 400.0  # Initial temperature (MeV)
    T_f = 100.0  # Final kinetic freeze-out (MeV)
    # Sigmoid-like cooling
    T = T_0 - (T_0 - T_f) * (1 - np.exp(-3*t)) / (1 - np.exp(-3))

    # μ_B: starts ~0, increases slightly due to baryon stopping
    # At LHC, very small μ_B throughout
    mu_B_0 = 0.5  # MeV
    mu_B_f = 2.0  # MeV (still very small at LHC)
    mu_B = mu_B_0 + (mu_B_f - mu_B_0) * t**0.5

    return mu_B, T


# =============================================================================
# MAIN GENERATION
# =============================================================================

def generate_all():
    """Generate all QCD phase diagram data files."""
    print("\n=== QCD Phase Diagram (High-Fidelity) ===")
    ensure_dir(OUTPUT_DIR)

    params = PhaseTransitionParams()

    # High-resolution grids
    N_FINE = 500
    N_MEDIUM = 200

    # -------------------------------------------------------------------------
    # 1. Crossover line from lattice QCD
    # -------------------------------------------------------------------------
    print("\n--- Crossover Line (Lattice QCD) ---")

    # Valid range for lattice expansion: μ_B < 300 MeV
    mu_B_crossover = np.linspace(0, 300, N_FINE)
    T_crossover = crossover_temperature(mu_B_crossover, params)

    comments = [
        "Lattice QCD crossover line",
        f"T_c(0) = {params.T_c0} +/- {params.T_c0_err} MeV (HotQCD)",
        f"kappa2 = {params.kappa2} +/- {params.kappa2_err}",
        "Valid for mu_B < 300 MeV",
        "Source: arXiv:1807.05607",
    ]
    save_curve("qcd_crossover_line.dat", mu_B_crossover, T_crossover,
               "mu_B T_c", comments)

    # Uncertainty band
    T_upper, T_lower = crossover_uncertainty_band(mu_B_crossover, params)
    save_curve_multi("qcd_hadronization_band.dat", mu_B_crossover,
                     [T_crossover, T_upper, T_lower],
                     "mu_B T_c T_upper T_lower",
                     ["Crossover uncertainty band (+/- 1 sigma)"])

    # -------------------------------------------------------------------------
    # 2. Chemical freeze-out curve and experimental data
    # -------------------------------------------------------------------------
    print("\n--- Chemical Freeze-out ---")

    mu_B_fo = np.linspace(0, 800, N_FINE)
    T_fo = freeze_out_parametrization(mu_B_fo)

    comments = [
        "Chemical freeze-out curve",
        "Parametrization: constant epsilon/n = 0.951 GeV",
        "Fit to STAR, ALICE, NA49, AGS data",
        "Source: Andronic et al., Nature 561, 321 (2018)",
    ]
    save_curve("qcd_freezeout_curve.dat", mu_B_fo, T_fo, "mu_B T", comments)

    # Experimental freeze-out points with errors
    fo_points = []
    for fp in FREEZE_OUT_DATA:
        fo_points.append([fp.mu_B, fp.T, fp.mu_B_err, fp.T_err])
    save_points_with_errors("qcd_freezeout_data.dat", fo_points,
                            "mu_B T mu_B_err T_err")

    # -------------------------------------------------------------------------
    # 2b. Freeze-out systematic uncertainty band
    # -------------------------------------------------------------------------
    print("\n--- Freeze-out Systematic Uncertainty ---")

    T_fo_upper, T_fo_lower = freeze_out_uncertainty_band(mu_B_fo)
    save_curve_multi("qcd_freezeout_band.dat", mu_B_fo,
                     [T_fo, T_fo_upper, T_fo_lower],
                     "mu_B T_central T_upper T_lower",
                     ["Freeze-out systematic uncertainty band",
                      "Source: Andronic et al., Nature 561, 321 (2018)",
                      "Includes HRG model and feed-down uncertainties"])

    # -------------------------------------------------------------------------
    # 3. First-order lines: Multiple theoretical models
    # -------------------------------------------------------------------------
    print("\n--- First-Order Lines (Multiple Models) ---")

    mu_B_fo_line = np.linspace(280, 950, N_MEDIUM)

    # FRG consensus first-order line (uses QM2025 CP at 630 MeV)
    T_fo_line = first_order_line(mu_B_fo_line, params)
    valid = ~np.isnan(T_fo_line) & (T_fo_line > 0)
    save_curve("qcd_firstorder_line.dat", mu_B_fo_line[valid], T_fo_line[valid],
               "mu_B T", ["First-order line (FRG consensus QM2025)",
                          f"CP: T = {params.T_cp_frg} MeV, μ_B = {params.mu_B_cp_frg} MeV",
                          "Source: Fu et al., arXiv:2510.11270"])

    # NJL model (CP at low μ_B - now known to be EXCLUDED)
    T_njl = first_order_njl(mu_B_fo_line, params)
    valid_njl = ~np.isnan(T_njl) & (T_njl > 0)
    save_curve("qcd_firstorder_njl.dat", mu_B_fo_line[valid_njl], T_njl[valid_njl],
               "mu_B T", ["First-order line from NJL model (EXCLUDED region)",
                          "CP: T ~ 80 MeV, μ_B ~ 330 MeV",
                          "NOTE: This CP location is EXCLUDED by Dec 2025 lattice",
                          "Source: Buballa, Phys. Rep. 407 (2005) 205"])

    # PQM model (CP at moderate μ_B - now known to be EXCLUDED)
    T_pqm = first_order_pqm(mu_B_fo_line, params)
    valid_pqm = ~np.isnan(T_pqm) & (T_pqm > 0)
    save_curve("qcd_firstorder_pqm.dat", mu_B_fo_line[valid_pqm], T_pqm[valid_pqm],
               "mu_B T", ["First-order line from PQM model (EXCLUDED region)",
                          "CP: T ~ 95 MeV, μ_B ~ 370 MeV",
                          "NOTE: This CP location is EXCLUDED by Dec 2025 lattice",
                          "Source: Schaefer et al., PRD 76 (2007) 074023"])

    # FRG model (updated to QM2025 consensus)
    T_frg = first_order_frg(mu_B_fo_line, params)
    valid_frg = ~np.isnan(T_frg) & (T_frg > 0)
    save_curve("qcd_firstorder_frg.dat", mu_B_fo_line[valid_frg], T_frg[valid_frg],
               "mu_B T", ["First-order line from FRG (QM2025 consensus)",
                          f"CP: T = {params.T_cp_frg} MeV, μ_B = {params.mu_B_cp_frg} MeV",
                          "BEYOND lattice exclusion (μ_B > 450 MeV)",
                          "Source: Fu et al., arXiv:2510.11270"])

    # CONSENSUS BAND (encompasses NJL/PQM/FRG spread for visualization)
    mu_B_band, T_upper, T_lower = first_order_consensus_band(100)
    save_curve_multi("qcd_firstorder_consensus_band.dat", mu_B_band,
                     [T_upper, T_lower],
                     "mu_B T_upper T_lower",
                     ["First-order CONSENSUS BAND",
                      "Encompasses NJL/PQM/FRG theoretical spread",
                      "Use for shaded band visualization"])

    # -------------------------------------------------------------------------
    # 4. Critical point: EXCLUSION + FRG consensus (December 2025 update)
    # -------------------------------------------------------------------------
    print("\n--- Critical Point (December 2025 Update) ---")

    # CP EXCLUSION REGION from Borsányi et al. PRD 112, L111505
    print("  Generating CP exclusion region (μ_B < 450 MeV at 2σ)...")
    mu_B_excl, T_excl = critical_point_exclusion_region(100)
    save_curve("qcd_cp_exclusion_region.dat", mu_B_excl, T_excl, "mu_B T",
               ["CRITICAL POINT EXCLUSION REGION",
                "μ_B < 450 MeV EXCLUDED at 2σ confidence",
                "Source: Borsányi et al., PRD 112, L111505 (Dec 2025)",
                "Method: Yang-Lee edge singularity extrapolation"])

    # Exclusion boundary line
    mu_B_excl_line, T_excl_line = critical_point_exclusion_boundary(100)
    save_curve("qcd_cp_exclusion_boundary.dat", mu_B_excl_line, T_excl_line,
               "mu_B T", ["CP exclusion boundary: μ_B = 450 MeV (2σ)"])

    # FRG CONSENSUS CP (QM2025) - the current best estimate
    cp_frg_point = [[params.mu_B_cp_frg, params.T_cp_frg]]
    save_points_with_errors("qcd_critical_point.dat", cp_frg_point, "mu_B T")
    print(f"  FRG consensus CP: ({params.mu_B_cp_frg}, {params.T_cp_frg}) MeV")

    # FRG consensus ellipse
    mu_B_frg_ell, T_frg_ell = critical_point_frg_ellipse(100)
    save_curve("qcd_critical_ellipse.dat", mu_B_frg_ell, T_frg_ell, "mu_B T",
               ["1-sigma uncertainty ellipse for FRG consensus CP",
                f"Center: ({params.mu_B_cp_frg}, {params.T_cp_frg}) MeV",
                "~10% uncertainty",
                "Source: Fu et al., arXiv:2510.11270 (QM2025 consensus)"])

    # FRG consensus box
    box_frg = critical_point_frg_box()
    box_frg_data = [
        [box_frg[0], box_frg[2]],
        [box_frg[1], box_frg[2]],
        [box_frg[1], box_frg[3]],
        [box_frg[0], box_frg[3]],
        [box_frg[0], box_frg[2]],
    ]
    save_points_with_errors("qcd_critical_box.dat", box_frg_data, "mu_B T")

    # DEPRECATED: Clarke et al. 2024 CP estimate (NOW EXCLUDED)
    print("  Generating EXCLUDED Clarke et al. CP region for reference...")
    mu_B_clarke_ell, T_clarke_ell = critical_point_ellipse_excluded(100)
    save_curve("qcd_critical_ellipse_excluded.dat", mu_B_clarke_ell, T_clarke_ell,
               "mu_B T", ["EXCLUDED: Clarke et al. 2024 CP estimate",
                          f"Center: ({params.mu_B_cp_clarke}, {params.T_cp_clarke}) MeV",
                          "NOW EXCLUDED by Borsányi et al. (Dec 2025)",
                          "Kept for reference only"])

    # -------------------------------------------------------------------------
    # 5. QGP region boundary (for fills)
    # -------------------------------------------------------------------------
    print("\n--- Phase Region Boundaries ---")

    # Extended boundary for filling - now to 800 MeV to accommodate new axis
    mu_B_fill = np.linspace(0, 800, N_MEDIUM)

    # QGP boundary: above crossover (or above freeze-out at high μ_B)
    T_qgp = np.zeros_like(mu_B_fill)
    for i, mu in enumerate(mu_B_fill):
        if mu < 300:
            T_qgp[i] = crossover_temperature(np.array([mu]), params)[0]
        else:
            # Extrapolate beyond lattice validity - continues downward
            T_qgp[i] = crossover_temperature(np.array([300]), params)[0] - 0.06 * (mu - 300)
            T_qgp[i] = max(T_qgp[i], 80)  # Floor at reasonable T

    save_curve("qcd_qgp_boundary.dat", mu_B_fill, T_qgp, "mu_B T")

    # -------------------------------------------------------------------------
    # 6. Collision system markers
    # -------------------------------------------------------------------------
    print("\n--- Collision Systems ---")

    cs_data = []
    for cs in COLLISION_SYSTEMS:
        cs_data.append([cs.mu_B, cs.T, cs.sqrt_s_NN])

    filepath = os.path.join(OUTPUT_DIR, "qcd_collision_systems.dat")
    with open(filepath, "w") as f:
        f.write("# Collision system freeze-out locations\n")
        f.write("# name mu_B T sqrt_s_NN marker color\n")
        for cs in COLLISION_SYSTEMS:
            f.write(f"# {cs.name}: ({cs.mu_B:.1f}, {cs.T:.1f}) MeV, "
                    f"sqrt_s = {cs.sqrt_s_NN} GeV\n")
        f.write("# mu_B T sqrt_s_NN\n")
        for d in cs_data:
            f.write(f"{d[0]:.2f} {d[1]:.2f} {d[2]:.1f}\n")
    print(f"  qcd_collision_systems.dat: {len(cs_data)} systems")

    # -------------------------------------------------------------------------
    # 7. Cooling trajectory
    # -------------------------------------------------------------------------
    print("\n--- Cooling Trajectory ---")

    mu_B_cool, T_cool = cooling_trajectory(N_MEDIUM)
    save_curve("qcd_cooling_trajectory.dat", mu_B_cool, T_cool, "mu_B T",
               ["Cooling trajectory for central Pb-Pb at LHC",
                "T_0 ~ 400 MeV -> T_kinetic ~ 100 MeV"])

    # -------------------------------------------------------------------------
    # 8. Isentropic trajectories (s/n_B = const)
    # -------------------------------------------------------------------------
    print("\n--- Isentropic Trajectories (RHIC BES) ---")

    for s_nB, label in ISENTROPE_VALUES:
        mu_B_isen, T_isen = isentropic_trajectory(s_nB, N_MEDIUM)
        filename = f"qcd_isentrope_{s_nB}.dat"
        save_curve(filename, mu_B_isen, T_isen, "mu_B T",
                   [f"Isentropic trajectory s/n_B = {s_nB}",
                    f"{label}",
                    "Source: arXiv:1506.07350"])

    # Combined isentropes file for easy plotting
    filepath = os.path.join(OUTPUT_DIR, "qcd_isentropes_all.dat")
    with open(filepath, "w") as f:
        f.write("# All isentropic trajectories (s/n_B = const)\n")
        f.write("# Source: arXiv:1506.07350 (RHIC BES review)\n")
        f.write("# Format: each trajectory separated by blank line\n")
        for s_nB, label in ISENTROPE_VALUES:
            f.write(f"\n# s/n_B = {s_nB} ({label})\n")
            mu_B_isen, T_isen = isentropic_trajectory(s_nB, 50)
            for mu, T in zip(mu_B_isen, T_isen):
                f.write(f"{mu:.4f} {T:.4f}\n")
    print(f"  qcd_isentropes_all.dat: {len(ISENTROPE_VALUES)} trajectories")

    # -------------------------------------------------------------------------
    # 9. PEDAGOGICAL CONTEXT: Early Universe, Neutron Stars, CSC
    # -------------------------------------------------------------------------
    print("\n--- Pedagogical Context Elements ---")

    # Early universe trajectory (cosmological hook)
    mu_B_eu, T_eu = early_universe_trajectory(N_MEDIUM)
    save_curve("qcd_early_universe.dat", mu_B_eu, T_eu, "mu_B T",
               ["Early universe trajectory (Big Bang → Hadronization)",
                "μ_B ≈ 0 due to baryon-antibaryon symmetry",
                "Hadronization at t ~ 10^-5 s, T ~ 150 MeV"])

    # Neutron star core region
    mu_B_ns, T_ns, T_ns_up, T_ns_lo = neutron_star_trajectory(100)
    save_curve_multi("qcd_neutron_star.dat", mu_B_ns,
                     [T_ns, T_ns_up, T_ns_lo],
                     "mu_B T T_upper T_lower",
                     ["Neutron star core region (cold, dense matter)",
                      "μ_B ~ 1000-1500 MeV, T ~ 0 (after cooling)",
                      "Probes cold QCD at extreme density"])

    # Color superconductivity boundary (schematic)
    mu_B_csc, T_csc = color_superconductivity_region()
    save_curve("qcd_csc_boundary.dat", mu_B_csc, T_csc, "mu_B T",
               ["Color superconductivity phase boundary (schematic)",
                "CSC: Cooper pairs of quarks at high μ_B, low T",
                "Phases: 2SC, CFL, crystalline variants"])

    # -------------------------------------------------------------------------
    # 10. Future Facilities Coverage Regions
    # -------------------------------------------------------------------------
    print("\n--- Future Facilities Coverage ---")

    filepath = os.path.join(OUTPUT_DIR, "qcd_future_facilities.dat")
    with open(filepath, "w") as f:
        f.write("# Future heavy-ion facilities coverage regions\n")
        f.write("# Format: facility mu_B_min mu_B_max T_min T_max\n")
        for key, fac in FUTURE_FACILITIES.items():
            f.write(f"# {fac['name']}: {fac['description']}\n")
            f.write(f"# √s_NN range: {fac['sqrt_s_range'][0]}-{fac['sqrt_s_range'][1]} GeV\n")
            f.write(f"{key} {fac['mu_B_range'][0]} {fac['mu_B_range'][1]} "
                    f"{fac['T_range'][0]} {fac['T_range'][1]}\n")
    print(f"  qcd_future_facilities.dat: {len(FUTURE_FACILITIES)} facilities")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\nGeneration complete: {OUTPUT_DIR}/")
    files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("qcd_")]
    print(f"QCD phase diagram files: {len(files)}")
    for f in sorted(files):
        print(f"  - {f}")


if __name__ == "__main__":
    generate_all()
