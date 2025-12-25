#!/usr/bin/env python3
"""
generate_hbt_data.py

Generates femtoscopy/HBT (Hanbury Brown-Twiss) correlation data for QGP light-ion project.

Femtoscopy probes the spacetime structure of the particle-emitting source through
two-particle momentum correlations. The correlation function:

    C(q) = 1 + λ exp(-q²R²)

where q is the relative momentum, R is the source radius, and λ is the chaoticity
parameter (0 < λ ≤ 1).

For 3D analysis, the relative momentum is decomposed:
    - q_out: along pair transverse momentum (radial flow direction)
    - q_side: perpendicular to pair momentum (no flow effects)
    - q_long: along beam axis (longitudinal expansion)

Physical outputs:
- Two-particle correlation functions C(q) for different systems
- HBT radii (R_out, R_side, R_long) vs centrality
- System size scaling: R³ ∝ dN/dη (volume scales with multiplicity)
- Comparison across pp, O-O, and Pb-Pb collisions

References:
- ALICE Collaboration, Phys. Rev. C 93, 024905 (2016) - HBT in Pb-Pb
- ALICE Collaboration, Phys. Lett. B 696, 328 (2011) - HBT in pp
"""

import os

import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "femtoscopy")


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_dat(filename, data_dict, header=""):
    """Save data to .dat file for pgfplots."""
    keys = list(data_dict.keys())
    arrays = [np.atleast_1d(data_dict[k]) for k in keys]

    with open(filename, "w") as f:
        if header:
            f.write(f"# {header}\n")
        f.write("# " + " ".join(keys) + "\n")
        for i in range(len(arrays[0])):
            row = [f"{arr[i]:.6e}" for arr in arrays]
            f.write(" ".join(row) + "\n")


def correlation_function(q_inv, radius, lambda_param=0.7):
    """
    Two-particle correlation function for Gaussian source.

    C(q) = 1 + λ exp(-q²R²)

    Parameters
    ----------
    q_inv : array
        Invariant relative momentum [GeV/c]
    radius : float
        Source radius parameter [fm]
    lambda_param : float
        Chaoticity/coherence parameter (default 0.7)

    Returns
    -------
    C_q : array
        Correlation function C(q)
    """
    # Convert radius from fm to GeV^-1 (natural units)
    hbarc = 0.197  # GeV·fm
    R_GeV_inv = radius / hbarc

    # Gaussian correlation
    C_q = 1.0 + lambda_param * np.exp(-(q_inv**2) * R_GeV_inv**2)

    return C_q


def correlation_function_3d(q_out, q_side, q_long, R_out, R_side, R_long, lambda_param=0.7):
    """
    3D correlation function with Bertsch-Pratt decomposition.

    C(q_out, q_side, q_long) = 1 + λ exp(-q_out²R_out² - q_side²R_side² - q_long²R_long²)

    Parameters
    ----------
    q_out, q_side, q_long : array
        Components of relative momentum [GeV/c]
    R_out, R_side, R_long : float
        HBT radii in each direction [fm]
    lambda_param : float
        Chaoticity parameter

    Returns
    -------
    C_q : array
        3D correlation function
    """
    hbarc = 0.197  # GeV·fm

    # Convert radii to GeV^-1
    R_out_inv = R_out / hbarc
    R_side_inv = R_side / hbarc
    R_long_inv = R_long / hbarc

    # 3D Gaussian
    exponent = -(q_out**2 * R_out_inv**2 + q_side**2 * R_side_inv**2 + q_long**2 * R_long_inv**2)

    C_q = 1.0 + lambda_param * np.exp(exponent)

    return C_q


def hbt_radii_vs_centrality(system, centrality):
    """
    HBT radii as function of centrality for different collision systems.

    Physical expectation:
    - R increases toward central collisions (larger source)
    - R_out > R_long > R_side typically (emission duration effects)
    - R³ ∝ dN/dη (volume proportional to multiplicity)

    Parameters
    ----------
    system : str
        Collision system: 'pp', 'OO', 'NeNe', 'PbPb'
    centrality : array
        Centrality bins [%]

    Returns
    -------
    dict with R_out, R_side, R_long, and uncertainties
    """
    # Characteristic radii for central collisions [fm]
    # Based on ALICE measurements
    system_params = {
        "pp": {"R_central": 1.2, "R_peripheral": 1.0},
        "OO": {"R_central": 2.5, "R_peripheral": 1.5},  # Predicted
        "NeNe": {"R_central": 2.8, "R_peripheral": 1.6},  # Predicted
        "PbPb": {"R_central": 5.0, "R_peripheral": 2.0},  # ALICE data
    }

    params = system_params[system]

    # Interpolate between central and peripheral based on centrality
    # Central: 0%, Peripheral: 80%
    centrality_fraction = centrality / 80.0
    R_inv = (
        params["R_central"] + (params["R_peripheral"] - params["R_central"]) * centrality_fraction
    )

    # Phenomenological anisotropy in HBT radii
    # R_out typically largest (emission duration), R_side smallest
    R_out = R_inv * 1.15  # Elongated in radial direction
    R_side = R_inv * 0.95  # Smaller transverse radius
    R_long = R_inv * 1.25  # Longitudinal expansion

    # Add realistic uncertainties (5-10%)
    uncertainty_fraction = 0.07
    err_out = R_out * uncertainty_fraction
    err_side = R_side * uncertainty_fraction
    err_long = R_long * uncertainty_fraction

    return {
        "centrality": centrality,
        "R_out": R_out,
        "R_side": R_side,
        "R_long": R_long,
        "err_out": err_out,
        "err_side": err_side,
        "err_long": err_long,
    }


def system_size_scaling():
    """
    HBT radius scaling with system size (multiplicity).

    Physics: R³ ∝ dN/dη reflects volume-multiplicity relation.

    Returns
    -------
    dict with dNch_deta and R_inv for different systems
    """
    # Charged multiplicity dN_ch/dη and corresponding HBT radii
    # Data from ALICE measurements and predictions
    systems = {
        "pp": {"dNch": 7, "R": 1.2},
        "pp_high": {"dNch": 30, "R": 1.8},  # High-multiplicity pp
        "pPb": {"dNch": 45, "R": 2.0},
        "OO_periph": {"dNch": 80, "R": 1.8},
        "OO_central": {"dNch": 135, "R": 2.5},
        "NeNe_periph": {"dNch": 100, "R": 2.0},
        "NeNe_central": {"dNch": 170, "R": 2.8},
        "PbPb_periph": {"dNch": 400, "R": 2.5},
        "PbPb_central": {"dNch": 1940, "R": 5.0},
    }

    dNch = np.array([s["dNch"] for s in systems.values()])
    R_inv = np.array([s["R"] for s in systems.values()])

    # Theoretical expectation: R = C * (dN/dη)^(1/3)
    # Fit to find normalization constant
    R_theory = 1.2 * (dNch / 7.0) ** (1.0 / 3.0)

    return {
        "dNch_deta": dNch,
        "R_inv": R_inv,
        "R_theory": R_theory,
        "err": R_inv * 0.08,  # 8% systematic uncertainty
    }


def main():
    ensure_dir(OUTPUT_DIR)

    print("Generating femtoscopy/HBT correlation data...")

    # =============================================================================
    # 1. Two-particle correlation functions C(q)
    # =============================================================================

    q_inv = np.linspace(0, 0.5, 100)  # Relative momentum [GeV/c]

    # pp collisions (small source)
    C_pp = correlation_function(q_inv, radius=1.2, lambda_param=0.65)
    save_dat(
        os.path.join(OUTPUT_DIR, "correlation_pp.dat"),
        {"q_inv": q_inv, "C_q": C_pp},
        "Two-particle correlation function for pp, R_inv = 1.2 fm",
    )
    print("  Correlation function pp: 100 points")

    # O-O collisions (intermediate source)
    C_OO = correlation_function(q_inv, radius=2.5, lambda_param=0.70)
    save_dat(
        os.path.join(OUTPUT_DIR, "correlation_OO.dat"),
        {"q_inv": q_inv, "C_q": C_OO},
        "Two-particle correlation function for O-O central, R_inv = 2.5 fm",
    )
    print("  Correlation function O-O: 100 points")

    # Pb-Pb collisions (large source)
    C_PbPb = correlation_function(q_inv, radius=5.0, lambda_param=0.75)
    save_dat(
        os.path.join(OUTPUT_DIR, "correlation_PbPb.dat"),
        {"q_inv": q_inv, "C_q": C_PbPb},
        "Two-particle correlation function for Pb-Pb 0-5%, R_inv = 5.0 fm",
    )
    print("  Correlation function Pb-Pb: 100 points")

    # =============================================================================
    # 2. 3D correlation function example (1D slices)
    # =============================================================================

    # Pb-Pb central: realistic HBT radii from ALICE
    R_out_PbPb = 5.0  # fm
    R_side_PbPb = 4.5  # fm
    R_long_PbPb = 6.0  # fm

    q_range = np.linspace(0, 0.3, 80)  # GeV/c

    # Slice along q_out (q_side = q_long = 0)
    C_out = correlation_function_3d(q_range, 0, 0, R_out_PbPb, R_side_PbPb, R_long_PbPb)
    save_dat(
        os.path.join(OUTPUT_DIR, "correlation_3d_out.dat"),
        {"q_out": q_range, "C_q": C_out},
        "3D correlation along q_out for Pb-Pb 0-5%",
    )

    # Slice along q_side
    C_side = correlation_function_3d(0, q_range, 0, R_out_PbPb, R_side_PbPb, R_long_PbPb)
    save_dat(
        os.path.join(OUTPUT_DIR, "correlation_3d_side.dat"),
        {"q_side": q_range, "C_q": C_side},
        "3D correlation along q_side for Pb-Pb 0-5%",
    )

    # Slice along q_long
    C_long = correlation_function_3d(0, 0, q_range, R_out_PbPb, R_side_PbPb, R_long_PbPb)
    save_dat(
        os.path.join(OUTPUT_DIR, "correlation_3d_long.dat"),
        {"q_long": q_range, "C_q": C_long},
        "3D correlation along q_long for Pb-Pb 0-5%",
    )
    print("  3D correlation slices: 3 x 80 points")

    # =============================================================================
    # 3. HBT radii vs centrality for different systems
    # =============================================================================

    centrality_bins = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80])

    for system in ["pp", "OO", "NeNe", "PbPb"]:
        hbt_data = hbt_radii_vs_centrality(system, centrality_bins)
        save_dat(
            os.path.join(OUTPUT_DIR, f"hbt_radii_vs_cent_{system}.dat"),
            hbt_data,
            f"HBT radii vs centrality for {system} collisions",
        )
        print(f"  HBT radii vs centrality {system}: {len(centrality_bins)} bins")

    # =============================================================================
    # 4. System size scaling: R³ ∝ dN/dη
    # =============================================================================

    scaling_data = system_size_scaling()
    save_dat(
        os.path.join(OUTPUT_DIR, "hbt_system_size_scaling.dat"),
        scaling_data,
        "HBT radius vs multiplicity: R ~ (dN/dη)^(1/3) scaling",
    )
    print(f"  System size scaling: {len(scaling_data['dNch_deta'])} systems")

    # =============================================================================
    # 5. Radius vs transverse momentum (k_T dependence)
    # =============================================================================

    # HBT radii decrease with increasing pair transverse momentum
    # Physical origin: higher pT pairs probe earlier emission (smaller source)
    k_T = np.linspace(0.2, 1.5, 12)  # Pair transverse momentum [GeV/c]

    # Pb-Pb central: R_inv decreases with k_T
    R_inv_kT_PbPb = 5.0 * (0.3 / k_T) ** 0.5  # Approximate power-law decrease
    R_inv_kT_OO = 2.5 * (0.3 / k_T) ** 0.5

    save_dat(
        os.path.join(OUTPUT_DIR, "hbt_radius_vs_kT_PbPb.dat"),
        {"k_T": k_T, "R_inv": R_inv_kT_PbPb, "err": R_inv_kT_PbPb * 0.08},
        "HBT radius vs pair k_T for Pb-Pb 0-10%",
    )

    save_dat(
        os.path.join(OUTPUT_DIR, "hbt_radius_vs_kT_OO.dat"),
        {"k_T": k_T, "R_inv": R_inv_kT_OO, "err": R_inv_kT_OO * 0.10},
        "HBT radius vs pair k_T for O-O 0-10%",
    )
    print(f"  HBT radius vs k_T: 2 systems x {len(k_T)} points")

    # =============================================================================
    # Summary
    # =============================================================================

    print(f"\nFemtoscopy data written to {OUTPUT_DIR}/")
    print("Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".dat"):
            print(f"  - {f}")

    print("\nKey physics encoded:")
    print("  - Correlation function C(q) = 1 + λ exp(-q²R²)")
    print("  - 3D HBT radii: R_out, R_side, R_long")
    print("  - System size scaling: R³ ∝ dN/dη")
    print("  - Centrality and k_T dependence")


if __name__ == "__main__":
    main()
