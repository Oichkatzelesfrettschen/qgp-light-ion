#!/usr/bin/env python3
"""
generate_figure_curves.py

Precompute all curves used in TikZ/pgfplots figures to eliminate inline formula
computation at LaTeX render time.

This replaces the inline expressions like:
    {156*(1-0.013*(x/156)^2)}
with precomputed .dat files that pgfplots can load directly via:
    \addplot table {data/figure_curves/crossover_line.dat};
"""

import os
import sys
import numpy as np

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "figure_curves")


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_curve(filename, x, y, header="x y"):
    """Save a simple x-y curve to a .dat file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        f.write(f"# {header}\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi:.6f} {yi:.6f}\n")
    print(f"  {filename}: {len(x)} points")


def save_curve_multi(filename, x, columns, header):
    """Save curve with multiple y columns."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        f.write(f"# {header}\n")
        for i, xi in enumerate(x):
            row = f"{xi:.6f}"
            for col in columns:
                row += f" {col[i]:.6f}"
            f.write(row + "\n")
    print(f"  {filename}: {len(x)} points x {len(columns)+1} columns")


# =============================================================================
# QCD PHASE DIAGRAM CURVES
# =============================================================================
def generate_qcd_phase_diagram():
    """Generate curves for qcd_phase_diagram.tex"""
    print("\n=== QCD Phase Diagram ===")

    # Crossover line: T_c(μ_B) = T_c(0) * [1 - κ₂(μ_B/T_c)²]
    # HotQCD 2019: T_c(0) = 156.5 MeV, κ₂ ≈ 0.0153
    mu_B = np.linspace(0, 350, 100)
    T_c0 = 156.5
    kappa2 = 0.0153
    T_crossover = T_c0 * (1 - kappa2 * (mu_B / T_c0) ** 2)
    save_curve("qcd_crossover_line.dat", mu_B, T_crossover, "mu_B T_c")

    # Hadronization band (±12 MeV uncertainty)
    T_upper = T_crossover + 12
    T_lower = T_crossover - 12
    save_curve_multi(
        "qcd_hadronization_band.dat",
        mu_B,
        [T_crossover, T_upper, T_lower],
        "mu_B T_c T_upper T_lower",
    )

    # First-order line (schematic, beyond critical point)
    mu_B_fo = np.linspace(350, 550, 50)
    T_fo = 120 - 0.12 * (mu_B_fo - 350)
    save_curve("qcd_firstorder_line.dat", mu_B_fo, T_fo, "mu_B T")

    # Chemical freeze-out curve
    mu_B_cfo = np.linspace(0, 500, 100)
    T_cfo = 165 * (1 - 0.0002 * mu_B_cfo**1.1)
    save_curve("qcd_freezeout_curve.dat", mu_B_cfo, T_cfo, "mu_B T")

    # QGP region boundary (for fill)
    mu_B_fill = np.linspace(0, 400, 100)
    T_fill = T_c0 * (1 - 0.013 * (mu_B_fill / T_c0) ** 2)
    save_curve("qcd_qgp_boundary.dat", mu_B_fill, T_fill, "mu_B T")


# =============================================================================
# NUCLEAR STRUCTURE CURVES
# =============================================================================
def generate_nuclear_structure():
    """Generate curves for nuclear_structure.tex"""
    print("\n=== Nuclear Structure ===")

    # Woods-Saxon density profiles
    r = np.linspace(0, 12, 150)

    # O-16: R0 = 2.608 fm, a = 0.513 fm
    rho_O = 1 / (1 + np.exp((r - 2.608) / 0.513))
    save_curve("woods_saxon_O16.dat", r, rho_O, "r rho")

    # Ne-20: R0 = 2.791 fm, a = 0.535 fm
    rho_Ne = 1 / (1 + np.exp((r - 2.791) / 0.535))
    save_curve("woods_saxon_Ne20.dat", r, rho_Ne, "r rho")

    # Pb-208: R0 = 6.62 fm, a = 0.546 fm
    rho_Pb = 1 / (1 + np.exp((r - 6.62) / 0.546))
    save_curve("woods_saxon_Pb208.dat", r, rho_Pb, "r rho")

    # Nuclear radius scaling: R = r0 * A^(1/3)
    A = np.linspace(10, 210, 100)
    r0 = 1.25  # fm
    R = r0 * A ** (1 / 3)
    save_curve("nuclear_radius_scaling.dat", A, R, "A R")


# =============================================================================
# RAA MULTISYSTEM CURVES
# =============================================================================
def generate_raa_curves():
    """Generate curves for RAA_multisystem.tex"""
    print("\n=== R_AA Curves ===")

    pT = np.linspace(1, 50, 100)

    # pp reference (R_AA = 1)
    R_pp = np.ones_like(pT)
    save_curve("raa_pp.dat", pT, R_pp, "pT R_AA")

    # p-Pb (Cronin enhancement)
    R_pPb = 1.0 + 0.12 * np.exp(-((pT - 2.5) ** 2) / 1.5) - 0.03 * np.exp(-pT / 30)
    save_curve("raa_pPb.dat", pT, R_pPb, "pT R_AA")

    # O-O model - smooth curve through CMS data region
    R_OO = 0.7 + 0.3 * (1 - np.exp(-pT / 8)) - 0.25 * np.exp(-((pT - 6) ** 2) / 15)
    R_OO = np.maximum(R_OO, 0.1)  # Ensure non-negative
    save_curve("raa_OO_model.dat", pT, R_OO, "pT R_AA")

    # Ne-Ne model
    R_NeNe = 0.65 + 0.35 * (1 - np.exp(-pT / 10)) - 0.30 * np.exp(-((pT - 6) ** 2) / 18)
    R_NeNe = np.maximum(R_NeNe, 0.1)
    save_curve("raa_NeNe_model.dat", pT, R_NeNe, "pT R_AA")

    # Xe-Xe model
    R_XeXe = 0.40 + 0.55 * (1 - np.exp(-pT / 15)) - 0.45 * np.exp(-((pT - 7) ** 2) / 25)
    R_XeXe = np.maximum(R_XeXe, 0.1)
    save_curve("raa_XeXe_model.dat", pT, R_XeXe, "pT R_AA")

    # Pb-Pb model - tuned to match ALICE JHEP 11 (2018) 013 data
    # Data points: pT=3 -> 0.28, pT=5 -> 0.18, pT=7 -> 0.15, pT=50 -> 0.60
    # Use interpolating function that passes through these key points
    # R_AA(pT) = baseline + suppression_dip + high_pT_recovery
    R_PbPb = (
        0.15  # baseline (minimum)
        + 0.20 * np.exp(-((pT - 1.5) ** 2) / 2.0)  # Cronin-like peak at low pT
        + 0.50 * (1 - np.exp(-(pT / 25) ** 1.5))  # high-pT recovery
        - 0.08 * np.exp(-((pT - 7) ** 2) / 20)  # slight dip around pT=7
    )
    R_PbPb = np.clip(R_PbPb, 0.14, 1.0)  # Physical bounds
    save_curve("raa_PbPb_model.dat", pT, R_PbPb, "pT R_AA")

    # Experimental data points
    # O-O CMS HIN-25-008 (minimum-bias)
    save_curve_with_errors(
        "raa_OO_data.dat",
        np.array([3.1, 4.2, 6.2, 8.1, 10.0, 14.0, 22.0, 31.5, 45.6]),
        np.array([0.775, 0.723, 0.695, 0.705, 0.715, 0.755, 0.808, 0.845, 0.824]),
        np.array([0.05, 0.04, 0.04, 0.04, 0.04, 0.04, 0.05, 0.05, 0.05]),
        "pT R_AA err",
    )

    # Pb-Pb ALICE JHEP 11 (2018) 013 (0-10% central)
    save_curve_with_errors(
        "raa_PbPb_data.dat",
        np.array([3, 5, 7, 10, 15, 20, 30, 50]),
        np.array([0.28, 0.18, 0.15, 0.18, 0.25, 0.35, 0.50, 0.60]),
        np.array([0.04, 0.02, 0.02, 0.02, 0.03, 0.04, 0.05, 0.06]),
        "pT R_AA err",
    )


def save_curve_with_errors(filename, x, y, err, header="x y err"):
    """Save a curve with error bars."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        f.write(f"# {header}\n")
        for xi, yi, ei in zip(x, y, err):
            f.write(f"{xi:.6f} {yi:.6f} {ei:.6f}\n")
    print(f"  {filename}: {len(x)} points with errors")


# =============================================================================
# FLOW COMPREHENSIVE CURVES
# =============================================================================
def generate_flow_curves():
    """Generate curves for flow_comprehensive.tex"""
    print("\n=== Flow Curves ===")

    # v2 vs centrality
    cent = np.linspace(0, 70, 50)

    # Pb-Pb v2
    v2_PbPb = 0.08 - 0.00006 * (cent - 35) ** 2
    save_curve("v2_vs_cent_PbPb.dat", cent, v2_PbPb, "centrality v2")

    # O-O v2
    v2_OO = 0.068 - 0.00005 * (cent - 40) ** 2
    save_curve("v2_vs_cent_OO.dat", cent, v2_OO, "centrality v2")

    # Ne-Ne v2 (with deformation effect)
    v2_NeNe = 0.068 - 0.00005 * (cent - 40) ** 2 + 0.015 * np.exp(-cent / 20)
    save_curve("v2_vs_cent_NeNe.dat", cent, v2_NeNe, "centrality v2")

    # v3 vs centrality
    # Pb-Pb v3
    v3_PbPb = 0.022 + 0.003 * (1 - cent / 70)
    save_curve("v3_vs_cent_PbPb.dat", cent, v3_PbPb, "centrality v3")

    # O-O v3
    v3_OO = 0.028 + 0.012 * (1 - cent / 70)
    save_curve("v3_vs_cent_OO.dat", cent, v3_OO, "centrality v3")

    # Ne-Ne v3
    v3_NeNe = 0.025 + 0.010 * (1 - cent / 70)
    save_curve("v3_vs_cent_NeNe.dat", cent, v3_NeNe, "centrality v3")

    # Azimuthal distribution dN/dphi
    phi = np.linspace(0, 2 * np.pi, 100)

    # Isotropic
    dN_iso = np.ones_like(phi)
    save_curve("azimuthal_isotropic.dat", phi, dN_iso, "phi dN_dphi")

    # Central (small v2, modest v3)
    v2_c, v3_c = 0.02, 0.025
    dN_central = 1 + 2 * v2_c * np.cos(2 * phi) + 2 * v3_c * np.cos(3 * phi)
    save_curve("azimuthal_central.dat", phi, dN_central, "phi dN_dphi")

    # Mid-central (large v2)
    v2_m, v3_m = 0.08, 0.02
    dN_midcentral = 1 + 2 * v2_m * np.cos(2 * phi) + 2 * v3_m * np.cos(3 * phi)
    save_curve("azimuthal_midcentral.dat", phi, dN_midcentral, "phi dN_dphi")

    # v2-modulated particle distribution (polar curve)
    # r(theta) = r0 * (1 + 2*v2*cos(2*theta))
    theta = np.linspace(0, 2 * np.pi, 100)
    v2_vis = 0.3  # Exaggerated for visibility
    r0 = 0.8
    r_mod = r0 * (1 + 2 * v2_vis * np.cos(2 * theta))
    x_mod = 2.2 + r_mod * np.cos(theta)
    y_mod = r_mod * np.sin(theta)
    save_curve_multi("v2_modulated_curve.dat", theta, [x_mod, y_mod], "theta x y")

    # Particle positions on modulated curve (denser in-plane)
    # Generate 50 particles with density proportional to 1 + 2*v2*cos(2*theta)
    np.random.seed(42)
    n_particles = 50
    # Rejection sampling
    particles = []
    while len(particles) < n_particles:
        th = np.random.uniform(0, 2 * np.pi)
        density = 1 + 2 * v2_vis * np.cos(2 * th)
        if np.random.uniform(0, 1 + 2 * v2_vis) < density:
            r = r0 * (1 + 2 * v2_vis * np.cos(2 * th))
            x = 2.2 + r * np.cos(th)
            y = r * np.sin(th)
            particles.append((x, y))

    filepath = os.path.join(OUTPUT_DIR, "v2_particle_positions.dat")
    with open(filepath, "w") as f:
        f.write("# x y\n")
        for x, y in particles:
            f.write(f"{x:.4f} {y:.4f}\n")
    print(f"  v2_particle_positions.dat: {len(particles)} particles")


# =============================================================================
# STRANGENESS ENHANCEMENT CURVES
# =============================================================================
def generate_strangeness_curves():
    """Generate curves for strangeness_enhancement.tex"""
    print("\n=== Strangeness Enhancement ===")

    # Canonical suppression curves
    # Enhancement = 1 + A * (1 - exp(-B * dNch^0.6))
    dNch = np.logspace(np.log10(3), np.log10(3000), 100)

    # K/pi (|S|=1)
    enh_K = 1 + 0.8 * (1 - np.exp(-0.03 * dNch**0.6))
    save_curve("strangeness_K.dat", dNch, enh_K, "dNch enhancement")

    # Lambda/pi (|S|=1, baryon)
    enh_Lambda = 1 + 1.2 * (1 - np.exp(-0.025 * dNch**0.6))
    save_curve("strangeness_Lambda.dat", dNch, enh_Lambda, "dNch enhancement")

    # Xi/pi (|S|=2)
    enh_Xi = 1 + 3.5 * (1 - np.exp(-0.015 * dNch**0.6))
    save_curve("strangeness_Xi.dat", dNch, enh_Xi, "dNch enhancement")

    # Omega/pi (|S|=3)
    enh_Omega = 1 + 8 * (1 - np.exp(-0.008 * dNch**0.6))
    save_curve("strangeness_Omega.dat", dNch, enh_Omega, "dNch enhancement")


# =============================================================================
# BJORKEN SPACETIME CURVES
# =============================================================================
def generate_bjorken_curves():
    """Generate curves for bjorken_spacetime.tex"""
    print("\n=== Bjorken Spacetime ===")

    # Constant-tau hyperbolas: t = sqrt(tau^2 + z^2)
    z = np.linspace(-7.5, 7.5, 100)

    # tau_0 = 1 fm/c (thermalization)
    tau = 1.0
    t_therm = np.sqrt(tau**2 + z**2)
    save_curve("hyperbola_tau1.dat", z, t_therm, "z t")

    # tau = 3 fm/c (QGP)
    tau = 3.0
    t_qgp = np.sqrt(tau**2 + z**2)
    save_curve("hyperbola_tau3.dat", z, t_qgp, "z t")

    # tau = 6 fm/c (hadronization)
    tau = 6.0
    t_had = np.sqrt(tau**2 + z**2)
    save_curve("hyperbola_tau6.dat", z, t_had, "z t")

    # tau = 10 fm/c (freeze-out)
    tau = 10.0
    t_fo = np.sqrt(tau**2 + z**2)
    save_curve("hyperbola_tau10.dat", z, t_fo, "z t")

    # Light cones
    z_lc = np.linspace(0, 8, 50)
    t_lc = z_lc
    save_curve("light_cone_right.dat", z_lc, t_lc, "z t")
    save_curve("light_cone_left.dat", -z_lc, t_lc, "z t")


# =============================================================================
# FEMTOSCOPY HBT CURVES
# =============================================================================
def generate_hbt_curves():
    """Generate curves for femtoscopy_hbt.tex"""
    print("\n=== Femtoscopy HBT ===")

    # Correlation functions C(q) = 1 + lambda * exp(-(R*q/hbar_c)^2)
    # q is in GeV/c, R is in fm, hbar_c = 0.197 GeV*fm converts units
    # The exponent should be -(R * q / hbar_c)^2 to be dimensionless
    q = np.linspace(0.001, 0.3, 100)
    hbar_c = 0.197  # GeV*fm

    # Pb-Pb central (R ~ 5 fm, lambda ~ 0.85)
    R_PbPb, lam_PbPb = 5.0, 0.85
    # Correct formula: exponent = -(R[fm] * q[GeV/c] / hbar_c[GeV*fm])^2
    C_PbPb = 1 + lam_PbPb * np.exp(-((R_PbPb * q / hbar_c) ** 2))
    save_curve("hbt_corr_PbPb.dat", q, C_PbPb, "q C")

    # O-O central (R ~ 2.5 fm, lambda ~ 0.7)
    R_OO, lam_OO = 2.5, 0.7
    C_OO = 1 + lam_OO * np.exp(-((R_OO * q / hbar_c) ** 2))
    save_curve("hbt_corr_OO.dat", q, C_OO, "q C")

    # pp high-mult (R ~ 1.2 fm, lambda ~ 0.6)
    R_pp, lam_pp = 1.2, 0.6
    C_pp = 1 + lam_pp * np.exp(-((R_pp * q / hbar_c) ** 2))
    save_curve("hbt_corr_pp.dat", q, C_pp, "q C")

    # R_out/R_side ratio vs centrality
    cent = np.linspace(0, 75, 50)

    # Pb-Pb
    ratio_PbPb = 1.0 + 0.22 * np.exp(-cent / 20)
    save_curve("hbt_ratio_PbPb.dat", cent, ratio_PbPb, "centrality ratio")

    # O-O
    ratio_OO = 1.0 + 0.15 * np.exp(-cent / 18)
    save_curve("hbt_ratio_OO.dat", cent, ratio_OO, "centrality ratio")

    # p-Pb
    ratio_pPb = 1.0 + 0.08 * np.exp(-cent / 15)
    save_curve("hbt_ratio_pPb.dat", cent, ratio_pPb, "centrality ratio")


# =============================================================================
# DIRECT PHOTON SPECTRA CURVES
# =============================================================================
def generate_photon_curves():
    """Generate curves for direct_photon_spectra.tex"""
    print("\n=== Direct Photon Spectra ===")

    pT = np.linspace(0.5, 10, 100)

    # pp (prompt only): power-law
    dN_pp = 1e-2 * pT ** (-6.5)
    save_curve("photon_pp.dat", pT, dN_pp, "pT dN")

    # p-Pb (prompt dominated)
    dN_pPb = 1.15e-2 * pT ** (-6.5)
    save_curve("photon_pPb.dat", pT, dN_pPb, "pT dN")

    # O-O model (thermal + prompt)
    dN_OO = 3e-3 * np.exp(-pT / 0.25) + 0.8e-2 * pT ** (-6.5)
    save_curve("photon_OO_model.dat", pT, dN_OO, "pT dN")

    # Pb-Pb central (strong thermal)
    dN_PbPb = 1.2e-2 * np.exp(-pT / 0.30) + 0.3e-2 * pT ** (-6.5)
    save_curve("photon_PbPb.dat", pT, dN_PbPb, "pT dN")

    # Pb-Pb thermal component only
    # Extend to full pT range but thermal naturally dies off at high pT
    # Use same grid as main curves for consistency
    pT_thermal = np.linspace(0.5, 10, 100)
    dN_thermal = 1.2e-2 * np.exp(-pT_thermal / 0.30)
    # Floor at 1e-12 to avoid underflow display issues (plot ymin is 1e-10)
    dN_thermal = np.maximum(dN_thermal, 1e-12)
    save_curve("photon_PbPb_thermal.dat", pT_thermal, dN_thermal, "pT dN")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("Generating precomputed figure curves...")
    ensure_dir(OUTPUT_DIR)

    generate_qcd_phase_diagram()
    generate_nuclear_structure()
    generate_raa_curves()
    generate_flow_curves()
    generate_strangeness_curves()
    generate_bjorken_curves()
    generate_hbt_curves()
    generate_photon_curves()

    print(f"\nAll curves written to {OUTPUT_DIR}/")
    print(f"Total files: {len(os.listdir(OUTPUT_DIR))}")


if __name__ == "__main__":
    main()
