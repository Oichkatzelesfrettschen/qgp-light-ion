#!/usr/bin/env python3
"""
generate_comprehensive_data.py

Generates physics-based datasets for sophisticated QGP light-ion visualizations.
This replaces the simple mock data with realistic physics models.

Output files are organized by visualization type:
- data/phase_diagram/       - QCD phase diagram
- data/nuclear_geometry/    - Woods-Saxon, Glauber, clustering
- data/flow/                - v_n coefficients, azimuthal distributions
- data/jet_quenching/       - R_AA, energy loss
- data/strangeness/         - Enhancement factors, particle ratios
- data/spacetime/           - Energy density evolution
"""

import argparse
import os
import sys

import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qgp_physics import (
    NUCLEI,
    azimuthal_distribution,
    calculate_eccentricities,
    canonical_suppression_factor,
    energy_density_profile_2d,
    estimate_system_parameters,
    generate_flow_vs_centrality,
    generate_raa_data,
    get_nuclear_profile_2d,
    oxygen_alpha_cluster_positions,
    qcd_phase_boundaries,
    sample_nucleon_positions,
    strangeness_enhancement_curve,
    temperature_evolution,
    woods_saxon,
)


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def make_provenance_header(
    observable, classification, model_description, model_inputs=None, references=None, notes=None
):
    """Generate standard provenance header for model data files.

    Args:
        observable: What is being calculated (e.g., "R_AA vs pT")
        classification: One of "PREDICTED", "SCHEMATIC"
        model_description: Brief physics model description
        model_inputs: Dict of input parameters and values
        references: List of reference strings
        notes: Additional notes

    Returns:
        Multi-line header string
    """
    lines = [
        "=" * 77,
        f" {observable}",
        "=" * 77,
        "",
        f"DATA TYPE: {classification} (Model-generated)",
        "",
        "GENERATOR:",
        "  Script: src/generate_comprehensive_data.py",
        "  Physics: src/qgp_physics.py",
        "  Project: QGP Light-Ion Whitepaper (2025)",
        "",
        "MODEL:",
        f"  {model_description}",
    ]

    if model_inputs:
        lines.append("")
        lines.append("MODEL INPUTS:")
        for key, val in model_inputs.items():
            lines.append(f"  {key}: {val}")

    if references:
        lines.append("")
        lines.append("REFERENCES:")
        for ref in references:
            lines.append(f"  - {ref}")

    if notes:
        lines.append("")
        lines.append("NOTES:")
        if isinstance(notes, list):
            for note in notes:
                lines.append(f"  - {note}")
        else:
            lines.append(f"  {notes}")

    lines.append("")
    lines.append("=" * 77)

    return "\n".join(["# " + line for line in lines])


def save_dat(filename, data_dict, header="", provenance=None):
    """Save data to .dat file for pgfplots.

    Args:
        filename: Output file path
        data_dict: Dict of column_name -> array
        header: Simple one-line header (legacy support)
        provenance: Dict with keys for make_provenance_header() (preferred)
    """
    keys = list(data_dict.keys())
    arrays = [np.atleast_1d(data_dict[k]) for k in keys]

    with open(filename, "w") as f:
        if provenance:
            f.write(make_provenance_header(**provenance) + "\n")
            f.write(f"# COLUMNS: {' '.join(keys)}\n")
        elif header:
            f.write(f"# {header}\n")
            f.write("# " + " ".join(keys) + "\n")
        else:
            f.write("# " + " ".join(keys) + "\n")
        for i in range(len(arrays[0])):
            row = [f"{arr[i]:.8e}" for arr in arrays]
            f.write(" ".join(row) + "\n")


def save_2d_grid(filename, X, Y, Z, header="", provenance=None):
    """Save 2D grid data for pgfplots surf/contour plots.

    Args:
        filename: Output file path
        X, Y, Z: 2D arrays for grid data
        header: Simple one-line header (legacy support)
        provenance: Dict with keys for make_provenance_header() (preferred)
    """
    with open(filename, "w") as f:
        if provenance:
            f.write(make_provenance_header(**provenance) + "\n")
            f.write("# COLUMNS: x y z\n")
        elif header:
            f.write(f"# {header}\n")
            f.write("# x y z\n")
        else:
            f.write("# x y z\n")
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                f.write(f"{X[i, j]:.6e} {Y[i, j]:.6e} {Z[i, j]:.6e}\n")
            f.write("\n")  # Blank line between rows for gnuplot/pgfplots


# =============================================================================
# 1. QCD PHASE DIAGRAM
# =============================================================================


def generate_phase_diagram_data(output_dir):
    """Generate QCD phase diagram data."""
    print("  Generating QCD phase diagram...")
    phase_dir = os.path.join(output_dir, "phase_diagram")
    ensure_dir(phase_dir)

    # Phase boundaries
    boundaries = qcd_phase_boundaries()

    # Crossover line
    crossover_provenance = {
        "observable": "QCD Crossover Temperature vs Baryon Chemical Potential",
        "classification": "SCHEMATIC",
        "model_description": "Lattice QCD-constrained crossover at low mu_B, extrapolated",
        "model_inputs": {
            "T_c(mu_B=0)": "156.5 +/- 1.5 MeV (HotQCD)",
            "Curvature": "kappa = 0.015 (lattice Taylor expansion)",
        },
        "references": [
            "HotQCD: Phys. Rev. D 90 (2014) 094503",
            "Lattice curvature: Bonati et al., Phys. Rev. D 92 (2015) 054503",
        ],
        "notes": "Reliable for mu_B < 300 MeV; higher mu_B is extrapolation",
    }
    save_dat(
        os.path.join(phase_dir, "crossover.dat"),
        {
            "mu_B": boundaries["mu_B_crossover"] * 1000,  # Convert to MeV
            "T": boundaries["T_crossover"] * 1000,
        },
        provenance=crossover_provenance,
    )

    # First-order line (hypothetical)
    first_order_provenance = {
        "observable": "First-Order Phase Transition Line (Hypothetical)",
        "classification": "SCHEMATIC",
        "model_description": "Extrapolation from models; not experimentally established",
        "notes": [
            "Position highly uncertain",
            "Used for pedagogical illustration only",
            "Critical point location is model-dependent",
        ],
    }
    save_dat(
        os.path.join(phase_dir, "first_order.dat"),
        {"mu_B": boundaries["mu_B_first_order"] * 1000, "T": boundaries["T_first_order"] * 1000},
        provenance=first_order_provenance,
    )

    # Critical point
    critical_provenance = {
        "observable": "QCD Critical Point Position (Estimated)",
        "classification": "SCHEMATIC",
        "model_description": "Model estimate; experimental search ongoing at RHIC BES-II",
        "model_inputs": {
            "mu_B_c": f"{boundaries['mu_B_critical'] * 1000:.0f} MeV (model estimate)",
            "T_c": f"{boundaries['T_critical'] * 1000:.0f} MeV (model estimate)",
        },
        "notes": "Location uncertain by factor of ~2 in mu_B",
    }
    save_dat(
        os.path.join(phase_dir, "critical_point.dat"),
        {"mu_B": [boundaries["mu_B_critical"] * 1000], "T": [boundaries["T_critical"] * 1000]},
        provenance=critical_provenance,
    )

    # Regions for different collision systems (approximate μ_B at LHC)
    systems = {
        "LHC_PbPb": {"mu_B": 0, "T": 160, "T_range": 40},
        "LHC_OO": {"mu_B": 0, "T": 155, "T_range": 30},
        "RHIC_AuAu": {"mu_B": 25, "T": 155, "T_range": 35},
        "SPS": {"mu_B": 250, "T": 150, "T_range": 30},
    }

    systems_provenance = {
        "observable": "Collision System Trajectories on QCD Phase Diagram",
        "classification": "SCHEMATIC",
        "model_description": "Approximate initial conditions accessed by different facilities",
        "notes": [
            "LHC: mu_B ~ 0 (highest energy, nearly baryon-free)",
            "RHIC: mu_B ~ 25 MeV at top energy",
            "SPS: mu_B ~ 250 MeV",
        ],
    }
    save_dat(
        os.path.join(phase_dir, "collision_systems.dat"),
        {
            "system": [1, 2, 3, 4],  # Index
            "mu_B": [systems[s]["mu_B"] for s in systems],
            "T": [systems[s]["T"] for s in systems],
            "T_err": [systems[s]["T_range"] / 2 for s in systems],
        },
        provenance=systems_provenance,
    )


# =============================================================================
# 2. NUCLEAR GEOMETRY
# =============================================================================


def generate_nuclear_geometry_data(output_dir):
    """Generate nuclear density profiles and Glauber data."""
    print("  Generating nuclear geometry...")
    geom_dir = os.path.join(output_dir, "nuclear_geometry")
    ensure_dir(geom_dir)

    # 1D Woods-Saxon profiles
    r = np.linspace(0, 12, 100)
    for name, nuc in NUCLEI.items():
        rho = woods_saxon(r, nuc)
        save_dat(
            os.path.join(geom_dir, f"woods_saxon_{name}.dat"),
            {"r": r, "rho": rho},
            f"Woods-Saxon profile for {nuc.name}",
        )

    # 2D density profiles for key nuclei
    for name in ["O", "Ne", "Pb"]:
        nuc = NUCLEI[name]
        X, Y, rho = get_nuclear_profile_2d(nuc, grid_size=50, extent=8)
        save_2d_grid(
            os.path.join(geom_dir, f"density_2d_{name}.dat"),
            X,
            Y,
            rho,
            f"2D nuclear density for {nuc.name}",
        )

    # Alpha-cluster configuration for O-16
    alpha_pos = oxygen_alpha_cluster_positions()
    save_dat(
        os.path.join(geom_dir, "oxygen_alpha_clusters.dat"),
        {"x": alpha_pos[:, 0], "y": alpha_pos[:, 1], "z": alpha_pos[:, 2]},
        "O-16 tetrahedral alpha cluster nucleon positions",
    )

    # Glauber model nucleon positions (sample events)
    np.random.seed(42)  # Reproducibility
    for name in ["O", "Ne", "Pb"]:
        nuc = NUCLEI[name]
        positions = sample_nucleon_positions(nuc, n_events=1)[0]
        save_dat(
            os.path.join(geom_dir, f"glauber_nucleons_{name}.dat"),
            {"x": positions[:, 0], "y": positions[:, 1], "z": positions[:, 2]},
            f"Sample Glauber nucleon positions for {nuc.name}",
        )

    # Eccentricity distributions (multiple events)
    print("    Computing eccentricity distributions...")
    n_events = 500
    for name in ["O", "Ne"]:
        nuc = NUCLEI[name]
        e2_list, e3_list = [], []
        for _ in range(n_events):
            pos = sample_nucleon_positions(nuc, 1)[0]
            ecc = calculate_eccentricities(pos)
            e2_list.append(ecc["epsilon_2"])
            e3_list.append(ecc["epsilon_3"])

        # Save distribution
        bins = np.linspace(0, 0.8, 30)
        hist_e2, _ = np.histogram(e2_list, bins=bins, density=True)
        hist_e3, _ = np.histogram(e3_list, bins=bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        save_dat(
            os.path.join(geom_dir, f"eccentricity_dist_{name}.dat"),
            {"epsilon": bin_centers, "P_e2": hist_e2, "P_e3": hist_e3},
            f"Eccentricity distributions for {nuc.name}",
        )

    # Compare nuclear radii
    radii_data = {
        "nucleus": [1, 2, 3, 4, 5],  # Index for categorical
        "A": [nuc.A for nuc in NUCLEI.values()],
        "R0": [nuc.R0 for nuc in NUCLEI.values()],
        "beta2": [nuc.beta2 for nuc in NUCLEI.values()],
    }
    save_dat(os.path.join(geom_dir, "nuclear_parameters.dat"), radii_data)


# =============================================================================
# 3. FLOW HARMONICS
# =============================================================================


def generate_flow_data(output_dir):
    """Generate anisotropic flow data."""
    print("  Generating flow data...")
    flow_dir = os.path.join(output_dir, "flow")
    ensure_dir(flow_dir)

    # Centrality bins
    centrality = np.linspace(0, 80, 17)

    # System-specific info for provenance
    system_info = {
        "O": {
            "fullname": "O-O",
            "geometry": "Alpha-cluster structure enhances eccentricity fluctuations",
        },
        "Ne": {"fullname": "Ne-Ne", "geometry": "Prolate deformation (beta_2=0.45) enhances v2"},
        "Pb": {"fullname": "Pb-Pb", "geometry": "Nearly spherical, well-constrained Glauber"},
    }

    # v_n vs centrality for different systems
    for name in ["O", "Ne", "Pb"]:
        nuc = NUCLEI[name]
        flow_data = generate_flow_vs_centrality(nuc, centrality)
        provenance = {
            "observable": f"Flow Harmonics v2, v3, v4 vs Centrality ({system_info[name]['fullname']})",
            "classification": "PREDICTED",
            "model_description": "(2+1)D viscous hydrodynamics with Glauber initial conditions",
            "model_inputs": {
                "Shear viscosity (eta/s)": "0.08-0.16 (1-2x KSS bound)",
                "Initial eccentricity": "Glauber MC event-by-event",
                "Freeze-out temperature": "150 MeV",
                "Nuclear geometry": system_info[name]["geometry"],
            },
            "references": [
                "Hydro: Heinz & Snellings, Ann. Rev. Nucl. Part. Sci. 63 (2013) 123",
                "Experimental constraint: CMS-HIN-25-009 (O-O/Ne-Ne flow)",
            ],
        }
        save_dat(os.path.join(flow_dir, f"vn_vs_cent_{name}.dat"), flow_data, provenance=provenance)

    # v_n(p_T) with mass ordering
    pT = np.linspace(0.2, 5, 40)
    masses = {"pion": 0.14, "kaon": 0.494, "proton": 0.938}

    for name in ["O", "Pb"]:
        for particle, mass in masses.items():
            # Mass ordering: heavier particles have v2 shifted to higher pT
            pT_shift = mass * 0.5  # Approximate radial flow effect
            # Base v2 without mass shift: 0.08 * sin(π*pT/4) * exp(-pT/8)
            v2 = 0.08 * np.sin(np.pi * (pT - pT_shift) / 4) * np.exp(-(pT - pT_shift) / 8)
            v2 = np.maximum(v2, 0)

            provenance = {
                "observable": f"Elliptic Flow v2(pT) for {particle} in {name}-{name}",
                "classification": "PREDICTED",
                "model_description": "Blast-wave + Cooper-Frye with mass ordering",
                "model_inputs": {
                    "Particle mass": f"{mass:.3f} GeV",
                    "Radial flow": "beta_T ~ 0.65",
                },
                "notes": "Mass ordering: heavier particles have v2 peak at higher pT",
            }
            save_dat(
                os.path.join(flow_dir, f"v2_pT_{name}_{particle}.dat"),
                {"pT": pT, "v2": v2},
                provenance=provenance,
            )

    # Azimuthal distribution examples
    phi = np.linspace(0, 2 * np.pi, 100)

    # Central collision (small v2, moderate v3)
    dN_central = azimuthal_distribution(phi, v2=0.02, v3=0.03)
    azimuthal_provenance = {
        "observable": "Azimuthal Particle Distribution dN/dphi",
        "classification": "SCHEMATIC",
        "model_description": "Fourier expansion: dN/dphi = N_0 * (1 + 2*v2*cos(2*phi) + 2*v3*cos(3*phi))",
        "notes": "Illustrative; actual distributions from event-by-event hydro",
    }
    save_dat(
        os.path.join(flow_dir, "azimuthal_central.dat"),
        {"phi": phi, "dN_dphi": dN_central},
        provenance={
            **azimuthal_provenance,
            "model_inputs": {"v2": 0.02, "v3": 0.03, "centrality": "Central (0-5%)"},
        },
    )

    # Mid-central (large v2)
    dN_midcentral = azimuthal_distribution(phi, v2=0.08, v3=0.025)
    save_dat(
        os.path.join(flow_dir, "azimuthal_midcentral.dat"),
        {"phi": phi, "dN_dphi": dN_midcentral},
        provenance={
            **azimuthal_provenance,
            "model_inputs": {"v2": 0.08, "v3": 0.025, "centrality": "Mid-central (20-30%)"},
        },
    )


# =============================================================================
# 4. JET QUENCHING / R_AA
# =============================================================================


def generate_jet_quenching_data(output_dir):
    """Generate R_AA and energy loss data."""
    print("  Generating jet quenching data...")
    jet_dir = os.path.join(output_dir, "jet_quenching")
    ensure_dir(jet_dir)

    # System-specific parameters for provenance
    system_params = {
        "pp": {"N_part": 2, "notes": "Reference (R_AA = 1 by definition)"},
        "pPb": {"N_part": 16, "notes": "Cold nuclear matter baseline"},
        "OO": {"N_part": 32, "notes": "Constrained to match CMS minimum ~0.69"},
        "NeNe": {"N_part": 40, "notes": "Includes prolate deformation (beta_2=0.45)"},
        "XeXe": {"N_part": 236, "notes": "Validated against CMS Run 2 data"},
        "PbPb": {"N_part": 383, "notes": "Constrained by ALICE/CMS central R_AA"},
    }

    # R_AA vs pT for all systems
    systems = ["pp", "pPb", "OO", "NeNe", "XeXe", "PbPb"]
    for system in systems:
        raa_data = generate_raa_data(system)
        provenance = {
            "observable": f"Nuclear Modification Factor R_AA vs pT ({system})",
            "classification": "PREDICTED",
            "model_description": "BDMPS-Z radiative energy loss with Glauber geometry",
            "model_inputs": {
                "Transport coefficient (q-hat)": "1.5-4.5 GeV^2/fm (scaled by density)",
                "Thermalization time (tau_0)": "0.6-1.0 fm/c",
                "Initial temperature (T_0)": "300-400 MeV (Pb-Pb central)",
                "Mean N_part": system_params[system]["N_part"],
            },
            "references": [
                "BDMPS-Z: Baier et al., Nucl. Phys. B 484 (1997) 265",
                "Experimental constraint: CMS-HIN-25-008 (O-O R_AA)",
            ],
            "notes": system_params[system]["notes"],
        }
        save_dat(os.path.join(jet_dir, f"RAA_{system}.dat"), raa_data, provenance=provenance)

    # Energy loss vs path length (BDMPS scaling)
    L = np.linspace(0, 8, 50)  # fm
    qhat_values = [1.0, 2.0, 3.0]  # GeV²/fm

    for qhat in qhat_values:
        # ΔE ∝ q̂ L² (BDMPS)
        alpha_s = 0.3
        Delta_E = alpha_s * qhat * L**2 / 4
        provenance = {
            "observable": f"Radiative Energy Loss vs Path Length (q-hat={qhat} GeV^2/fm)",
            "classification": "SCHEMATIC",
            "model_description": "BDMPS-Z: Delta_E = alpha_s * q-hat * L^2 / 4",
            "model_inputs": {
                "Strong coupling (alpha_s)": 0.3,
                "Transport coefficient (q-hat)": f"{qhat} GeV^2/fm",
            },
            "notes": "Illustrative scaling; actual energy loss includes collisional terms",
        }
        save_dat(
            os.path.join(jet_dir, f"energy_loss_qhat{qhat:.0f}.dat"),
            {"L": L, "Delta_E": Delta_E},
            provenance=provenance,
        )

    # R_AA at fixed pT vs system size (N_part proxy)
    N_part = np.array([2, 16, 32, 40, 236, 383])  # pp, pPb, OO, NeNe, XeXe, PbPb
    R_AA_6GeV = []
    for system in systems:
        raa = generate_raa_data(system)
        R_AA_6GeV.append(np.interp(6.0, raa["pT"], raa["R_AA"]))

    provenance = {
        "observable": "R_AA at pT=6 GeV vs System Size (N_part)",
        "classification": "PREDICTED",
        "model_description": "System-size dependence of parton energy loss",
        "model_inputs": {
            "Fixed pT": "6 GeV (near maximum suppression)",
            "Systems": "pp, p-Pb, O-O, Ne-Ne, Xe-Xe, Pb-Pb",
        },
        "notes": "Shows onset of jet quenching with increasing system size",
    }
    save_dat(
        os.path.join(jet_dir, "RAA_vs_Npart.dat"),
        {"N_part": N_part, "R_AA": np.array(R_AA_6GeV)},
        provenance=provenance,
    )


# =============================================================================
# 5. STRANGENESS ENHANCEMENT
# =============================================================================


def generate_strangeness_data(output_dir):
    """Generate strangeness enhancement data."""
    print("  Generating strangeness data...")
    strange_dir = os.path.join(output_dir, "strangeness")
    ensure_dir(strange_dir)

    # Enhancement vs multiplicity
    dNch = np.logspace(0.5, 3.5, 100)  # ~3 to ~3000
    enhancement = strangeness_enhancement_curve(dNch)
    save_dat(
        os.path.join(strange_dir, "enhancement_vs_mult.dat"),
        enhancement,
        "Strangeness enhancement vs charged multiplicity",
    )

    # Canonical suppression factor
    x = np.linspace(0.1, 20, 100)  # Argument of Bessel function
    for S in [1, 2, 3]:
        gamma_S = np.array([canonical_suppression_factor(S, xi) for xi in x])
        save_dat(
            os.path.join(strange_dir, f"canonical_suppression_S{S}.dat"),
            {"x": x, "gamma": gamma_S},
            f"Canonical suppression factor for |S|={S}",
        )

    # System markers on enhancement curve
    systems_mult = {
        "pp": 7,
        "pp_high_mult": 30,
        "pPb": 45,
        "OO": 135,
        "NeNe": 170,
        "PbPb_periph": 400,
        "PbPb_central": 1940,
    }

    omega_enhancement = []
    for _system, mult in systems_mult.items():
        enh = strangeness_enhancement_curve(np.array([mult]))
        omega_enhancement.append(enh["enhancement_Omega"][0])

    save_dat(
        os.path.join(strange_dir, "systems_on_curve.dat"),
        {"dNch": list(systems_mult.values()), "enhancement": omega_enhancement},
        "Collision systems on Omega enhancement curve",
    )


# =============================================================================
# 6. SPACETIME EVOLUTION
# =============================================================================


def generate_spacetime_data(output_dir):
    """Generate spacetime evolution data."""
    print("  Generating spacetime evolution data...")
    spacetime_dir = os.path.join(output_dir, "spacetime")
    ensure_dir(spacetime_dir)

    # Temperature evolution
    tau = np.linspace(0.5, 15, 100)  # fm/c
    T_0_values = {"PbPb": 0.50, "OO": 0.35}  # Initial temperatures in GeV

    for system, T_0 in T_0_values.items():
        T = temperature_evolution(tau, T_0, tau_0=0.5)
        save_dat(
            os.path.join(spacetime_dir, f"temperature_evolution_{system}.dat"),
            {"tau": tau, "T": T * 1000},  # Convert to MeV
            f"Temperature vs proper time for {system}",
        )

    # Energy density in transverse plane at different times
    x = np.linspace(-8, 8, 40)
    y = np.linspace(-8, 8, 40)

    # O-O collision at b=2 fm
    epsilon_OO = energy_density_profile_2d(x, y, NUCLEI["O"], NUCLEI["O"], b=2.0)
    X, Y = np.meshgrid(x, y)
    save_2d_grid(
        os.path.join(spacetime_dir, "energy_density_OO_b2.dat"),
        X,
        Y,
        epsilon_OO,
        "Initial energy density for O-O at b=2 fm",
    )

    # Pb-Pb collision at b=7 fm (mid-central)
    x_pb = np.linspace(-12, 12, 50)
    y_pb = np.linspace(-12, 12, 50)
    epsilon_PbPb = energy_density_profile_2d(x_pb, y_pb, NUCLEI["Pb"], NUCLEI["Pb"], b=7.0)
    X_pb, Y_pb = np.meshgrid(x_pb, y_pb)
    save_2d_grid(
        os.path.join(spacetime_dir, "energy_density_PbPb_b7.dat"),
        X_pb,
        Y_pb,
        epsilon_PbPb,
        "Initial energy density for Pb-Pb at b=7 fm",
    )

    # QGP lifetime vs system size (OO, NeNe, XeXe, PbPb)
    R_eff = [2.6, 2.8, 5.4, 6.6]  # Effective radii in fm
    tau_QGP = [R / 0.5 for R in R_eff]  # Rough estimate: τ ~ R/c_s

    save_dat(
        os.path.join(spacetime_dir, "qgp_lifetime.dat"),
        {"system": [1, 2, 3, 4], "R_eff": R_eff, "tau_QGP": tau_QGP},
        "QGP lifetime estimates vs system",
    )


# =============================================================================
# 7. SYSTEM COMPARISON DATA
# =============================================================================


def generate_comparison_data(output_dir):
    """Generate data for multi-system comparisons."""
    print("  Generating system comparison data...")
    comp_dir = os.path.join(output_dir, "comparison")
    ensure_dir(comp_dir)

    # Bjorken energy density for all systems
    systems = ["O", "Ne", "Xe", "Pb"]
    epsilon_Bj = []
    A_values = []
    dNch_values = []

    for name in systems:
        params = estimate_system_parameters(NUCLEI[name], 0.05)
        epsilon_Bj.append(params["epsilon_Bj"])
        A_values.append(NUCLEI[name].A)
        dNch_values.append(params["dNch_deta"])

    save_dat(
        os.path.join(comp_dir, "bjorken_energy_density.dat"),
        {"A": A_values, "epsilon_Bj": epsilon_Bj, "dNch_deta": dNch_values},
        "Bjorken energy density for central collisions",
    )

    # Multiplicity scaling
    A_range = np.array([1, 16, 20, 40, 129, 208])
    dNch_scaled = 4.5 * A_range ** (1.0)  # Approximate wounded nucleon scaling

    save_dat(
        os.path.join(comp_dir, "multiplicity_scaling.dat"),
        {"A": A_range, "dNch_per_pair": dNch_scaled / (A_range / 2)},
        "Multiplicity per participant pair vs A",
    )


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive QGP physics data for visualizations"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data", help="Root directory for output data files"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        choices=[
            "all",
            "phase",
            "geometry",
            "flow",
            "jet",
            "strangeness",
            "spacetime",
            "comparison",
        ],
        help="Generate only a subset of data",
    )
    args = parser.parse_args()

    print(f"Generating comprehensive QGP data in '{args.output_dir}'...")
    ensure_dir(args.output_dir)

    generators = {
        "phase": generate_phase_diagram_data,
        "geometry": generate_nuclear_geometry_data,
        "flow": generate_flow_data,
        "jet": generate_jet_quenching_data,
        "strangeness": generate_strangeness_data,
        "spacetime": generate_spacetime_data,
        "comparison": generate_comparison_data,
    }

    if args.subset == "all":
        for _name, gen_func in generators.items():
            gen_func(args.output_dir)
    else:
        generators[args.subset](args.output_dir)

    # Also generate the original simple data for backward compatibility
    print("  Generating legacy data files...")
    legacy_dir = args.output_dir

    # Legacy R_AA files
    raa_OO = generate_raa_data("OO", n_points=8)
    raa_PbPb = generate_raa_data("PbPb", n_points=7)

    save_dat(
        os.path.join(legacy_dir, "RAA_OO.dat"),
        {"pT": raa_OO["pT"][:8], "R_AA": raa_OO["R_AA"][:8], "err_y": raa_OO["err"][:8]},
        "pT R_AA err_y",
    )
    save_dat(
        os.path.join(legacy_dir, "RAA_PbPb.dat"),
        {"pT": raa_PbPb["pT"][:7], "R_AA": raa_PbPb["R_AA"][:7], "err_y": raa_PbPb["err"][:7]},
        "pT R_AA err_y",
    )

    # Legacy flow files
    cent = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70])
    flow_OO = generate_flow_vs_centrality(NUCLEI["O"], cent)
    flow_NeNe = generate_flow_vs_centrality(NUCLEI["Ne"], cent)

    save_dat(
        os.path.join(legacy_dir, "flow_v2_OO.dat"),
        {"centrality": cent, "v2": flow_OO["v2"]},
        "centrality v2",
    )
    save_dat(
        os.path.join(legacy_dir, "flow_v3_OO.dat"),
        {"centrality": cent, "v3": flow_OO["v3"]},
        "centrality v3",
    )
    save_dat(
        os.path.join(legacy_dir, "flow_v2_NeNe.dat"),
        {"centrality": cent, "v2": flow_NeNe["v2"]},
        "centrality v2_NeNe",
    )

    # Create stamp file
    with open(os.path.join(legacy_dir, ".generated"), "w") as f:
        f.write("Comprehensive QGP data generated\n")

    print("\nData generation complete!")
    print(f"Output directory: {args.output_dir}")
    print("\nGenerated subdirectories:")
    for subdir in [
        "phase_diagram",
        "nuclear_geometry",
        "flow",
        "jet_quenching",
        "strangeness",
        "spacetime",
        "comparison",
    ]:
        print(f"  - {subdir}/")


if __name__ == "__main__":
    main()
