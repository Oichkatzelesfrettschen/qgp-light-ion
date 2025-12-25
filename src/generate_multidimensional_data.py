#!/usr/bin/env python3
"""
generate_multidimensional_data.py

Comprehensive multi-dimensional data generation for QGP visualization.
Generates data across 1D through 8D analysis frameworks with proper
physical models and precomputed grids for high-quality TikZ/pgfplots rendering.

Dimensional Framework:
- 1D: Spectra, distributions, evolution curves
- 2D: Phase diagrams, correlation functions, density projections
- 3D: Spacetime evolution, surfaces, isosurfaces (sliced)
- 4D: Spacetime + observable, animated sequences, parameter scans
- 5D+: Correlation matrices, parallel coordinates, dimensionality reduction

Physics Models:
- Relativistic hydrodynamics (viscous corrections)
- Statistical hadronization (grand canonical)
- Jet energy loss (BDMPS-Z + collisional)
- HBT correlations (Gaussian source)
- Thermal photon emission
"""

import os
import sys
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import iv as bessel_i

warnings.filterwarnings("ignore")

# Add src to path for qgp_physics module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qgp_physics import (
    ETA_OVER_S,
)

# =============================================================================
# CONSTANTS AND PARAMETERS
# =============================================================================

# Physical constants
ALPHA_S = 0.3  # Strong coupling
SIGMA_NN = 7.0  # fm^2, nucleon-nucleon cross section at LHC
C_LIGHT = 1.0  # Natural units

# Thermal parameters
T_CHEM = 0.156  # Chemical freeze-out temperature [GeV]
T_KIN = 0.100  # Kinetic freeze-out temperature [GeV]

# Medium parameters
QHAT_PbPb = 2.5  # GeV^2/fm for central Pb-Pb
TAU_0 = 0.6  # Formation time [fm/c]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_1d(
    filename: str,
    x: np.ndarray,
    y: np.ndarray,
    x_label: str = "x",
    y_label: str = "y",
    header: str = "",
):
    """Save 1D data for pgfplots."""
    with open(filename, "w") as f:
        if header:
            f.write(f"# {header}\n")
        f.write(f"# {x_label} {y_label}\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi:.8e} {yi:.8e}\n")


def save_1d_multi(filename: str, data: dict[str, np.ndarray], header: str = ""):
    """Save multiple 1D columns."""
    keys = list(data.keys())
    n_points = len(data[keys[0]])
    with open(filename, "w") as f:
        if header:
            f.write(f"# {header}\n")
        f.write("# " + " ".join(keys) + "\n")
        for i in range(n_points):
            row = [f"{data[k][i]:.8e}" for k in keys]
            f.write(" ".join(row) + "\n")


def save_2d_grid(filename: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, header: str = ""):
    """Save 2D grid data with proper pgfplots formatting."""
    with open(filename, "w") as f:
        if header:
            f.write(f"# {header}\n")
        f.write("# x y z\n")
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                f.write(f"{X[i, j]:.6e} {Y[i, j]:.6e} {Z[i, j]:.6e}\n")
            f.write("\n")  # Blank line between rows


def save_3d_slices(
    base_filename: str,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    V: np.ndarray,
    z_values: np.ndarray,
    header: str = "",
):
    """Save 3D data as multiple 2D slices along z-axis."""
    for k, z_val in enumerate(z_values):
        filename = f"{base_filename}_z{k:03d}.dat"
        with open(filename, "w") as f:
            if header:
                f.write(f"# {header} at z={z_val:.3f}\n")
            f.write("# x y value\n")
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    f.write(f"{X[i, j, k]:.6e} {Y[i, j, k]:.6e} {V[i, j, k]:.6e}\n")
                f.write("\n")


# =============================================================================
# 1D VISUALIZATIONS
# =============================================================================


def generate_1d_spectra(output_dir: str):
    """
    Generate 1D particle spectra and distributions.

    Physics:
    - pT spectra from Blast-Wave model
    - Rapidity distributions
    - Temperature evolution
    - dE/dx energy loss distributions
    """
    print("  Generating 1D spectra...")
    spec_dir = os.path.join(output_dir, "1d_spectra")
    ensure_dir(spec_dir)

    # 1. Transverse momentum spectra - Blast-Wave model
    # dN/dpT ∝ pT × mT × I_0(pT sinh(ρ)/T) × K_1(mT cosh(ρ)/T)
    pT = np.linspace(0.1, 20, 200)  # GeV

    for particle, mass in [("pion", 0.140), ("kaon", 0.494), ("proton", 0.938)]:
        mT = np.sqrt(pT**2 + mass**2)
        T_kin = 0.100  # GeV
        # Full Blast-Wave uses beta_s = 0.65, rho = arctanh(beta_s); simplified here

        # Simplified Blast-Wave (thermal + flow)
        spectrum = pT * mT * np.exp(-mT / T_kin)
        # Flow boost
        flow_factor = 1 + 0.3 * pT / (1 + pT)
        spectrum *= flow_factor
        # Normalize
        spectrum /= np.max(spectrum)

        save_1d(
            os.path.join(spec_dir, f"pt_spectrum_{particle}.dat"),
            pT,
            spectrum,
            "pT_GeV",
            "dN_dpT",
            f"Transverse momentum spectrum for {particle}",
        )

    # 2. Rapidity distribution - Gaussian + plateau
    y = np.linspace(-6, 6, 200)

    for system, sigma_y, dNdy_mid in [("PbPb", 3.5, 1940), ("OO", 2.5, 135)]:
        # Gaussian in rapidity with saturation at midrapidity
        dNdy = dNdy_mid * np.exp(-(y**2) / (2 * sigma_y**2))
        # Add slight plateau at midrapidity
        plateau = 0.1 * dNdy_mid * (1 - np.tanh(np.abs(y) - 1) ** 2)
        dNdy = np.maximum(dNdy, plateau)

        save_1d(
            os.path.join(spec_dir, f"rapidity_dist_{system}.dat"),
            y,
            dNdy,
            "y",
            "dN_dy",
            f"Rapidity distribution for {system}",
        )

    # 3. Temperature evolution - Bjorken expansion with viscous corrections
    tau = np.linspace(0.5, 15, 200)  # fm/c

    for system, T0, tau0 in [("PbPb", 0.350, 0.6), ("OO", 0.280, 0.8)]:
        # Ideal Bjorken: T ∝ τ^(-1/3)
        T_ideal = T0 * (tau0 / tau) ** (1.0 / 3)

        # Viscous correction: slower cooling due to viscous heating
        # dT/dτ includes viscous correction ~ η/s × (T/τ)
        delta_T = 0.02 * (ETA_OVER_S / 0.08) * T_ideal * (1 - np.exp(-tau / 2))
        T_viscous = T_ideal + delta_T

        save_1d_multi(
            os.path.join(spec_dir, f"temperature_evolution_{system}.dat"),
            {"tau": tau, "T_ideal": T_ideal * 1000, "T_viscous": T_viscous * 1000},
            f"Temperature evolution in MeV for {system}",
        )

    # 4. Energy loss probability distribution
    # P(ΔE) from BDMPS-Z
    dE = np.linspace(0, 50, 200)  # GeV

    for L, qhat, label in [(6.0, 2.5, "PbPb"), (2.5, 1.5, "OO")]:
        omega_c = 0.5 * qhat * L**2
        # BDMPS-Z: P(ΔE) ~ (ΔE)^(-3/2) for ΔE < ω_c, exponential cutoff above
        P_dE = np.where(dE > 0.1, (dE + 0.1) ** (-1.5) * np.exp(-dE / omega_c), 0)
        P_dE /= np.trapz(P_dE, dE)  # Normalize

        save_1d(
            os.path.join(spec_dir, f"energy_loss_dist_{label}.dat"),
            dE,
            P_dE,
            "dE_GeV",
            "P_dE",
            f"Energy loss distribution for {label}, L={L} fm",
        )

    # 5. Multiplicity distributions - Negative Binomial
    n = np.arange(0, 500)

    for system, n_mean, k in [("PbPb_central", 1940, 50), ("OO_central", 135, 20)]:
        # Negative binomial: P(n) = C(n+k-1, n) × p^k × (1-p)^n
        p = k / (n_mean + k)
        from scipy.special import comb

        P_n = comb(n + k - 1, n) * p**k * (1 - p) ** n
        P_n /= np.sum(P_n)

        save_1d(
            os.path.join(spec_dir, f"mult_dist_{system}.dat"),
            n.astype(float),
            P_n,
            "n",
            "P_n",
            f"Multiplicity distribution for {system}",
        )


# =============================================================================
# 2D VISUALIZATIONS
# =============================================================================


def generate_2d_correlations(output_dir: str):
    """
    Generate 2D correlation functions and phase space data.

    Physics:
    - Two-particle correlations C(Δη, Δφ)
    - Ridge structure from collective flow
    - HBT correlations in q-space
    - Flow coefficient correlations
    """
    print("  Generating 2D correlations...")
    corr_dir = os.path.join(output_dir, "2d_correlations")
    ensure_dir(corr_dir)

    # 1. Two-particle correlation C(Δη, Δφ)
    # Note: 50x50 grid to stay within pgfplots memory limits
    delta_eta = np.linspace(-5, 5, 50)
    delta_phi = np.linspace(-np.pi, np.pi, 50)
    DETA, DPHI = np.meshgrid(delta_eta, delta_phi)

    # Baseline from jet fragmentation + underlying event
    for system, v2, v3, jet_sigma_eta, ridge_amp in [
        ("PbPb", 0.08, 0.03, 0.5, 0.15),
        ("OO", 0.05, 0.025, 0.6, 0.08),
        ("pp", 0.01, 0.005, 0.4, 0.02),
    ]:
        # Near-side jet peak
        jet_peak = (
            2.0 * np.exp(-(DETA**2) / (2 * jet_sigma_eta**2)) * np.exp(-(DPHI**2) / (2 * 0.3**2))
        )

        # Away-side jet (back-to-back)
        away_jet = (
            0.8
            * np.exp(-(DETA**2) / (2 * 1.0**2))
            * np.exp(-((np.abs(DPHI) - np.pi) ** 2) / (2 * 0.5**2))
        )

        # Flow modulation (long-range ridge)
        ridge = (
            ridge_amp
            * (1 - np.exp(-np.abs(DETA) / 1.5))
            * (1 + 2 * v2 * np.cos(2 * DPHI) + 2 * v3 * np.cos(3 * DPHI))
        )

        # Combinatorial background
        background = 1.0

        C2 = background + jet_peak + away_jet + ridge

        save_2d_grid(
            os.path.join(corr_dir, f"C2_deta_dphi_{system}.dat"),
            DETA,
            DPHI,
            C2,
            f"Two-particle correlation C(Δη,Δφ) for {system}",
        )

    # 2. HBT correlation in (q_out, q_side) space
    q_out = np.linspace(-0.3, 0.3, 80)  # GeV
    q_side = np.linspace(-0.3, 0.3, 80)  # GeV
    Q_OUT, Q_SIDE = np.meshgrid(q_out, q_side)

    for system, R_out, R_side in [("PbPb", 6.0, 5.5), ("OO", 2.5, 2.3), ("pp", 1.2, 1.1)]:
        # Gaussian HBT: C(q) = 1 + λ exp(-R_out²q_out² - R_side²q_side²)
        lambda_param = 0.5
        q2 = (R_out * Q_OUT) ** 2 + (R_side * Q_SIDE) ** 2
        C_HBT = 1 + lambda_param * np.exp(-q2)

        save_2d_grid(
            os.path.join(corr_dir, f"HBT_qout_qside_{system}.dat"),
            Q_OUT,
            Q_SIDE,
            C_HBT,
            f"HBT correlation in q-space for {system}",
        )

    # 3. v_n correlations: v2 vs v3 event-by-event
    n_events = 5000
    np.random.seed(42)

    for system, v2_mean, v3_mean, corr in [("PbPb", 0.08, 0.03, -0.1), ("OO", 0.05, 0.025, 0.05)]:
        # Generate correlated fluctuations
        sigma_v2 = v2_mean * 0.3
        sigma_v3 = v3_mean * 0.4

        cov = [[sigma_v2**2, corr * sigma_v2 * sigma_v3], [corr * sigma_v2 * sigma_v3, sigma_v3**2]]
        v2_v3 = np.random.multivariate_normal([v2_mean, v3_mean], cov, n_events)
        v2_events = np.clip(v2_v3[:, 0], 0, 0.3)
        v3_events = np.clip(v2_v3[:, 1], 0, 0.15)

        # Save scatter data
        save_1d_multi(
            os.path.join(corr_dir, f"v2_v3_scatter_{system}.dat"),
            {"v2": v2_events, "v3": v3_events},
            f"Event-by-event v2-v3 correlations for {system}",
        )

        # Also save 2D histogram
        v2_bins = np.linspace(0, 0.2, 50)
        v3_bins = np.linspace(0, 0.1, 50)
        hist, _, _ = np.histogram2d(v2_events, v3_events, bins=[v2_bins, v3_bins], density=True)
        V2, V3 = np.meshgrid(v2_bins[:-1], v3_bins[:-1])

        save_2d_grid(
            os.path.join(corr_dir, f"v2_v3_hist_{system}.dat"),
            V2.T,
            V3.T,
            hist,
            f"v2-v3 joint distribution for {system}",
        )

    # 4. R_AA vs centrality vs pT (2D surface)
    centrality = np.linspace(0, 80, 50)  # %
    pT = np.linspace(1, 100, 50)  # GeV
    CENT, PT = np.meshgrid(centrality, pT)

    # Model: R_AA decreases with centrality (more participants),
    # increases with pT (less relative energy loss)
    L_eff = 6 * (1 - CENT / 100) ** 0.5  # Path length decreases toward peripheral
    qhat_eff = 2.5 * (1 - CENT / 100) ** 0.7
    dE = 0.3 * qhat_eff * L_eff**2
    R_AA = (PT / (PT + dE)) ** 6
    R_AA = R_AA * (1 - np.exp(-PT / 2))  # Low pT modification
    R_AA = np.clip(R_AA, 0.05, 1.0)

    save_2d_grid(
        os.path.join(corr_dir, "RAA_cent_pT_PbPb.dat"),
        CENT,
        PT,
        R_AA,
        "R_AA(centrality, pT) for Pb-Pb",
    )


# =============================================================================
# 3D VISUALIZATIONS
# =============================================================================


def generate_3d_spacetime(output_dir: str):
    """
    Generate 3D spacetime evolution data.

    Physics:
    - Energy density ε(x, y, τ) evolution
    - Temperature field T(x, y, τ)
    - Flow velocity field u^μ(x, y, τ)
    - Freeze-out hypersurface
    """
    print("  Generating 3D spacetime...")
    st_dir = os.path.join(output_dir, "3d_spacetime")
    ensure_dir(st_dir)

    # Grid parameters
    n_xy = 50

    x = np.linspace(-8, 8, n_xy)
    y = np.linspace(-8, 8, n_xy)
    tau_values = np.array([0.6, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0])

    X, Y = np.meshgrid(x, y)

    # Initial energy density profile (almond-shaped for b=7 fm)
    b = 7.0  # Impact parameter
    R_Pb = 6.62

    def initial_epsilon(x, y, b):
        """Initial energy density from optical Glauber."""

        # Nuclear thickness functions
        def T_A(x, y):
            r = np.sqrt(x**2 + y**2)
            return np.exp(-(r**2) / (2 * R_Pb**2 / 3))

        epsilon = T_A(x + b / 2, y) * T_A(x - b / 2, y)
        return epsilon

    epsilon_0 = initial_epsilon(X, Y, b)
    epsilon_0 = epsilon_0 / np.max(epsilon_0) * 50  # Peak ~ 50 GeV/fm^3

    # Add fluctuations (hot spots)
    np.random.seed(123)
    n_hotspots = 15
    for _ in range(n_hotspots):
        x0 = np.random.uniform(-5, 5)
        y0 = np.random.uniform(-3, 3)
        amp = np.random.uniform(0.5, 2.0)
        sigma = np.random.uniform(0.5, 1.0)
        epsilon_0 += amp * 10 * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2))

    # Time evolution: ε(τ) ∝ τ^(-4/3) in ideal Bjorken
    # Plus transverse expansion
    for _i, tau in enumerate(tau_values):
        # Bjorken cooling
        epsilon_tau = epsilon_0 * (0.6 / tau) ** (4.0 / 3)

        # Transverse expansion (diffusion-like spreading)
        sigma_expand = 0.5 * np.sqrt(tau)
        epsilon_tau = gaussian_filter(epsilon_tau, sigma=sigma_expand)

        # Temperature from ε ∝ T^4
        T_tau = (epsilon_tau / 15) ** (0.25) * 1000  # MeV, Stefan-Boltzmann
        T_tau = np.clip(T_tau, 50, 500)

        # Save slices
        save_2d_grid(
            os.path.join(st_dir, f"epsilon_xy_tau{tau:.1f}.dat"),
            X,
            Y,
            epsilon_tau,
            f"Energy density at τ={tau:.1f} fm/c",
        )

        save_2d_grid(
            os.path.join(st_dir, f"temperature_xy_tau{tau:.1f}.dat"),
            X,
            Y,
            T_tau,
            f"Temperature in MeV at τ={tau:.1f} fm/c",
        )

    # Save tau metadata
    save_1d(
        os.path.join(st_dir, "tau_values.dat"),
        tau_values,
        np.arange(len(tau_values)),
        "tau_fm_c",
        "index",
        "Proper time values for slices",
    )

    # 2. Flow velocity field at fixed τ
    tau_fixed = 3.0
    epsilon_fixed = epsilon_0 * (0.6 / tau_fixed) ** (4.0 / 3)

    # Flow from pressure gradients: u ∝ -∇P ∝ -∇ε
    grad_x = np.gradient(epsilon_fixed, x, axis=1)
    grad_y = np.gradient(epsilon_fixed, y, axis=0)

    # Normalize to get direction, scale by flow velocity
    grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-10
    u_x = -grad_x / grad_mag * 0.5 * (1 - np.exp(-tau_fixed))
    u_y = -grad_y / grad_mag * 0.5 * (1 - np.exp(-tau_fixed))

    # Subsample for quiver plot
    skip = 5
    save_1d_multi(
        os.path.join(st_dir, f"flow_velocity_tau{tau_fixed:.1f}.dat"),
        {
            "x": X[::skip, ::skip].flatten(),
            "y": Y[::skip, ::skip].flatten(),
            "ux": u_x[::skip, ::skip].flatten(),
            "uy": u_y[::skip, ::skip].flatten(),
        },
        f"Flow velocity field at τ={tau_fixed:.1f} fm/c",
    )

    # 3. Freeze-out hypersurface (T = T_fo isosurface)
    T_fo = 156.5  # MeV

    # For each τ, find contour at T = T_fo
    freeze_out_points = []
    for i, tau in enumerate(tau_values):
        epsilon_tau = epsilon_0 * (0.6 / tau) ** (4.0 / 3)
        epsilon_tau = gaussian_filter(epsilon_tau, sigma=0.5 * np.sqrt(tau))
        T_tau = (epsilon_tau / 15) ** (0.25) * 1000

        # Find contour using matplotlib (more commonly available)
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            cs = ax.contour(x, y, T_tau, levels=[T_fo])
            plt.close(fig)
            for path in cs.collections[0].get_paths():
                vertices = path.vertices[::5]  # Subsample
                for xi, yi in vertices:
                    freeze_out_points.append([xi, yi, tau])
        except Exception:
            # Fallback: simple threshold-based contour
            mask = np.abs(T_tau - T_fo) < 10  # Within 10 MeV
            for i in range(len(y)):
                for j in range(len(x)):
                    if mask[i, j]:
                        freeze_out_points.append([x[j], y[i], tau])

    if freeze_out_points:
        fo_arr = np.array(freeze_out_points)
        save_1d_multi(
            os.path.join(st_dir, "freeze_out_surface.dat"),
            {"x": fo_arr[:, 0], "y": fo_arr[:, 1], "tau": fo_arr[:, 2]},
            "Freeze-out hypersurface points (T=156.5 MeV)",
        )


# =============================================================================
# 4D+ VISUALIZATIONS
# =============================================================================


def generate_4d_parameter_scans(output_dir: str):
    """
    Generate higher-dimensional data representations.

    Physics:
    - Parameter space exploration (η/s, q̂, T_0, ...)
    - Observable correlations (multi-parameter fits)
    - Dimensionality reduction of high-D physics
    """
    print("  Generating 4D+ parameter scans...")
    hd_dir = os.path.join(output_dir, "4d_parameters")
    ensure_dir(hd_dir)

    # 1. Parameter scan: v2 dependence on (η/s, centrality, system_size)
    eta_s_values = np.linspace(0.04, 0.24, 10)
    centrality_values = np.linspace(0, 60, 15)

    for R_system, system_name in [(6.62, "Pb"), (2.6, "O")]:
        results = []
        for eta_s in eta_s_values:
            for cent in centrality_values:
                # Eccentricity increases with centrality
                epsilon_2 = 0.5 * np.sin(np.pi * cent / 100) * (1 - 0.3 * cent / 100)

                # v2 response depends on η/s and system size
                # v2/ε2 ≈ const × (1 - a × η/s / R_T)
                knudsen = eta_s / R_system
                response = 0.25 * (1 - 5 * knudsen)
                v2 = response * epsilon_2
                v2 = np.clip(v2, 0, 0.2)

                results.append([eta_s, cent, v2])

        results = np.array(results)
        save_1d_multi(
            os.path.join(hd_dir, f"v2_etaS_cent_{system_name}.dat"),
            {"eta_s": results[:, 0], "centrality": results[:, 1], "v2": results[:, 2]},
            f"v2(η/s, centrality) for {system_name} system",
        )

    # 2. Multi-observable correlation matrix
    # Simulated "events" with correlated observables
    n_events = 10000
    np.random.seed(42)

    # Define correlation structure
    # Observables: dNch/dη, <pT>, v2, v3, R_AA(10), HBT_R
    n_obs = 6
    obs_names = ["dNch_deta", "mean_pT", "v2", "v3", "RAA_10GeV", "HBT_R"]
    obs_means = [1500, 0.85, 0.07, 0.03, 0.4, 5.0]
    obs_sigmas = [300, 0.1, 0.02, 0.01, 0.08, 1.0]

    # Correlation matrix (physics-motivated)
    corr_matrix = np.array(
        [
            [1.0, 0.3, 0.4, 0.2, -0.5, 0.7],  # dNch correlates with everything
            [0.3, 1.0, 0.1, 0.05, -0.3, 0.2],  # <pT>
            [0.4, 0.1, 1.0, 0.3, -0.2, 0.3],  # v2
            [0.2, 0.05, 0.3, 1.0, -0.1, 0.15],  # v3
            [-0.5, -0.3, -0.2, -0.1, 1.0, -0.4],  # R_AA anti-correlates with mult
            [0.7, 0.2, 0.3, 0.15, -0.4, 1.0],  # HBT size scales with system
        ]
    )

    # Generate correlated samples
    L = np.linalg.cholesky(corr_matrix)
    z = np.random.normal(0, 1, (n_events, n_obs))
    samples_corr = z @ L.T

    # Scale to physical values
    samples = np.zeros_like(samples_corr)
    for i in range(n_obs):
        samples[:, i] = obs_means[i] + samples_corr[:, i] * obs_sigmas[i]

    # Clip to physical ranges
    samples[:, 0] = np.clip(samples[:, 0], 100, 3000)  # dNch
    samples[:, 1] = np.clip(samples[:, 1], 0.5, 1.5)  # <pT>
    samples[:, 2] = np.clip(samples[:, 2], 0, 0.2)  # v2
    samples[:, 3] = np.clip(samples[:, 3], 0, 0.1)  # v3
    samples[:, 4] = np.clip(samples[:, 4], 0.1, 1.0)  # R_AA
    samples[:, 5] = np.clip(samples[:, 5], 1, 10)  # HBT_R

    # Save full dataset
    data_dict = {name: samples[:, i] for i, name in enumerate(obs_names)}
    save_1d_multi(
        os.path.join(hd_dir, "multiobs_events.dat"), data_dict, "Multi-observable event data"
    )

    # Save correlation matrix
    with open(os.path.join(hd_dir, "correlation_matrix.dat"), "w") as f:
        f.write("# Observable correlation matrix\n")
        f.write("# " + " ".join(obs_names) + "\n")
        for i, name in enumerate(obs_names):
            row = [f"{corr_matrix[i, j]:.4f}" for j in range(n_obs)]
            f.write(f"{name} " + " ".join(row) + "\n")

    # 3. PCA reduction for visualization
    # Standardize
    samples_std = (samples - np.mean(samples, axis=0)) / np.std(samples, axis=0)

    # Compute PCA
    cov = np.cov(samples_std.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Project to first 2 PCs
    pc_scores = samples_std @ eigenvectors[:, :2]

    save_1d_multi(
        os.path.join(hd_dir, "pca_projection.dat"),
        {"PC1": pc_scores[:, 0], "PC2": pc_scores[:, 1]},
        "PCA projection to 2D",
    )

    # Save PC loadings
    save_1d_multi(
        os.path.join(hd_dir, "pca_loadings.dat"),
        {
            "PC1_loading": eigenvectors[:, 0],
            "PC2_loading": eigenvectors[:, 1],
            "obs_index": np.arange(n_obs),
        },
        "PCA loadings (feature importance)",
    )

    # Save explained variance
    var_explained = eigenvalues / np.sum(eigenvalues)
    save_1d(
        os.path.join(hd_dir, "pca_variance.dat"),
        np.arange(1, n_obs + 1),
        var_explained,
        "PC",
        "variance_fraction",
        "PCA explained variance",
    )

    # 4. Parallel coordinates data (for high-D visualization)
    # Select subset of events for clarity
    subset_idx = np.random.choice(n_events, 500, replace=False)
    subset = samples[subset_idx]

    # Normalize each observable to [0, 1] for parallel coordinates
    subset_norm = (subset - np.min(subset, axis=0)) / (
        np.max(subset, axis=0) - np.min(subset, axis=0) + 1e-10
    )

    data_dict = {name: subset_norm[:, i] for i, name in enumerate(obs_names)}
    data_dict["event_id"] = np.arange(len(subset_idx))
    save_1d_multi(
        os.path.join(hd_dir, "parallel_coords.dat"),
        data_dict,
        "Normalized data for parallel coordinates",
    )


# =============================================================================
# PHYSICS CONNECTIONS AND NOVEL ELUCIDATIONS
# =============================================================================


def generate_physics_connections(output_dir: str):
    """
    Generate data illustrating novel physics connections.

    Key insights:
    1. Knudsen number scaling: why small systems behave differently
    2. Eccentricity-to-flow connection across system sizes
    3. Energy loss path length dependence
    4. Strangeness canonical vs grand-canonical transition
    """
    print("  Generating physics connections...")
    conn_dir = os.path.join(output_dir, "physics_connections")
    ensure_dir(conn_dir)

    # 1. Knudsen number scaling
    # Kn = λ_mfp / L_system = (η/s) × (T/ε)^(1/2) / R_T
    R_systems = np.linspace(1, 8, 50)  # fm
    eta_s = 0.12
    T = 0.200  # GeV

    # Knudsen number
    Kn = eta_s * np.sqrt(T) / R_systems

    # Response: v2/ε2 decreases with Kn
    response = 0.3 / (1 + 2 * Kn)

    save_1d_multi(
        os.path.join(conn_dir, "knudsen_scaling.dat"),
        {"R_fm": R_systems, "Knudsen": Kn, "v2_over_eps2": response},
        "Knudsen number and hydrodynamic response vs system size",
    )

    # Mark specific systems
    systems = [("pp", 1.0), ("pPb", 1.5), ("OO", 2.6), ("PbPb", 6.6)]
    system_data = {"name_idx": [], "R_fm": [], "Knudsen": [], "response": []}
    for i, (_name, R) in enumerate(systems):
        kn = eta_s * np.sqrt(T) / R
        resp = 0.3 / (1 + 2 * kn)
        system_data["name_idx"].append(i)
        system_data["R_fm"].append(R)
        system_data["Knudsen"].append(kn)
        system_data["response"].append(resp)

    save_1d_multi(
        os.path.join(conn_dir, "knudsen_systems.dat"),
        {k: np.array(v) for k, v in system_data.items()},
        "Knudsen number for specific systems",
    )

    # 2. Universal v2/ε2 scaling
    # Data compilation across systems
    epsilon_2 = np.linspace(0.05, 0.6, 50)

    for system, kappa in [("PbPb", 0.25), ("OO", 0.18), ("pp", 0.10)]:
        v2 = kappa * epsilon_2
        save_1d(
            os.path.join(conn_dir, f"v2_vs_eps2_{system}.dat"),
            epsilon_2,
            v2,
            "epsilon_2",
            "v2",
            f"v2 vs ε2 linear response for {system}",
        )

    # 3. Energy loss vs path length
    L = np.linspace(0, 10, 100)  # fm
    qhat = 2.0  # GeV^2/fm

    # BDMPS-Z: ΔE ∝ L^2 (radiative)
    dE_rad = 0.3 * qhat * L**2

    # Collisional: ΔE ∝ L (elastic)
    dE_coll = 0.2 * L

    # Total
    dE_total = dE_rad + dE_coll

    save_1d_multi(
        os.path.join(conn_dir, "energy_loss_vs_L.dat"),
        {"L_fm": L, "dE_rad": dE_rad, "dE_coll": dE_coll, "dE_total": dE_total},
        "Energy loss components vs path length",
    )

    # 4. Canonical suppression as function of volume
    V = np.logspace(0, 4, 100)  # fm^3

    # x parameter ∝ V^(1/3) for strangeness
    x = 0.5 * V ** (1.0 / 3)

    gamma_S1 = bessel_i(1, x) / bessel_i(0, x)
    gamma_S2 = bessel_i(2, x) / bessel_i(0, x)
    gamma_S3 = bessel_i(3, x) / bessel_i(0, x)

    save_1d_multi(
        os.path.join(conn_dir, "canonical_suppression_vs_V.dat"),
        {"V_fm3": V, "gamma_S1": gamma_S1, "gamma_S2": gamma_S2, "gamma_S3": gamma_S3},
        "Canonical suppression factors vs volume",
    )

    # 5. QGP signatures threshold: where do signatures turn on?
    dNch = np.logspace(0, 3, 100)

    # Each signature has a different "turn-on" threshold
    # Based on Knudsen number and finite-size effects

    # Flow: requires Kn < 1
    flow_strength = 1 / (1 + 100 / dNch)

    # Strangeness: canonical → grand canonical
    strange_enh = 1 + 4 * (1 - np.exp(-dNch / 50))

    # Jet quenching: requires sufficient path length
    # R_AA suppression
    L_eff = 2 * (dNch / 100) ** (1.0 / 3)
    R_AA_signature = 1 - 0.8 * (1 - np.exp(-L_eff / 2))

    # HBT: always measurable but interpretation changes
    HBT_size = 1 + 3 * (dNch / 1000) ** (1.0 / 3)

    save_1d_multi(
        os.path.join(conn_dir, "qgp_signature_onset.dat"),
        {
            "dNch_deta": dNch,
            "flow_v2_norm": flow_strength,
            "strange_enhancement": strange_enh,
            "jet_suppression": 1 - R_AA_signature,
            "HBT_size": HBT_size / 5,
        },
        "QGP signature strength vs multiplicity",
    )


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate multi-dimensional QGP data")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    args = parser.parse_args()

    output_dir = args.output_dir
    ensure_dir(output_dir)

    print(f"Generating multi-dimensional QGP data in '{output_dir}'...")

    # Generate all datasets
    generate_1d_spectra(output_dir)
    generate_2d_correlations(output_dir)
    generate_3d_spacetime(output_dir)
    generate_4d_parameter_scans(output_dir)
    generate_physics_connections(output_dir)

    print("\nMulti-dimensional data generation complete!")
    print(f"Output directory: {output_dir}")

    # List generated subdirectories
    subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    print(f"\nGenerated subdirectories: {', '.join(sorted(subdirs))}")


if __name__ == "__main__":
    main()
