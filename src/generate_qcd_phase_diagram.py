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

from __future__ import annotations

import os
import sys

import numpy as np

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils import (
    ensure_dir,
    save_curve,
    save_curve_multi,
    save_points_with_errors,
)
from phase_diagram.critical_point import (
    critical_point_ellipse_excluded,
    critical_point_exclusion_boundary,
    critical_point_exclusion_region,
    critical_point_frg_box,
    critical_point_frg_ellipse,
)
from phase_diagram.crossover import crossover_temperature, crossover_uncertainty_band
from phase_diagram.first_order import (
    first_order_consensus_band,
    first_order_frg,
    first_order_line,
    first_order_njl,
    first_order_pqm,
)
from phase_diagram.freeze_out import (
    freeze_out_parametrization,
    freeze_out_uncertainty_band,
)
from phase_diagram.params import (
    COLLISION_SYSTEMS,
    FREEZE_OUT_DATA,
    FUTURE_FACILITIES,
    ISENTROPE_VALUES,
    PhaseTransitionParams,
)
from phase_diagram.trajectories import (
    color_superconductivity_region,
    cooling_trajectory,
    early_universe_trajectory,
    isentropic_trajectory,
    neutron_star_trajectory,
)

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "figure_curves")

# =============================================================================
# MAIN GENERATION
# =============================================================================


def generate_all() -> None:
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
    save_curve("qcd_crossover_line.dat", mu_B_crossover, T_crossover, "mu_B T_c", comments, output_dir=OUTPUT_DIR)

    # Uncertainty band
    T_upper, T_lower = crossover_uncertainty_band(mu_B_crossover, params)
    save_curve_multi(
        "qcd_hadronization_band.dat",
        mu_B_crossover,
        [T_crossover, T_upper, T_lower],
        "mu_B T_c T_upper T_lower",
        ["Crossover uncertainty band (+/- 1 sigma)"],
        output_dir=OUTPUT_DIR,
    )

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
    save_curve("qcd_freezeout_curve.dat", mu_B_fo, T_fo, "mu_B T", comments, output_dir=OUTPUT_DIR)

    # Experimental freeze-out points with errors
    fo_points = []
    for fp in FREEZE_OUT_DATA:
        fo_points.append([fp.mu_B, fp.T, fp.mu_B_err, fp.T_err])
    save_points_with_errors("qcd_freezeout_data.dat", fo_points, "mu_B T mu_B_err T_err", output_dir=OUTPUT_DIR)

    # -------------------------------------------------------------------------
    # 2b. Freeze-out systematic uncertainty band
    # -------------------------------------------------------------------------
    print("\n--- Freeze-out Systematic Uncertainty ---")

    T_fo_upper, T_fo_lower = freeze_out_uncertainty_band(mu_B_fo)
    save_curve_multi(
        "qcd_freezeout_band.dat",
        mu_B_fo,
        [T_fo, T_fo_upper, T_fo_lower],
        "mu_B T_central T_upper T_lower",
        [
            "Freeze-out systematic uncertainty band",
            "Source: Andronic et al., Nature 561, 321 (2018)",
            "Includes HRG model and feed-down uncertainties",
        ],
        output_dir=OUTPUT_DIR,
    )

    # -------------------------------------------------------------------------
    # 3. First-order lines: Multiple theoretical models
    # -------------------------------------------------------------------------
    print("\n--- First-Order Lines (Multiple Models) ---")

    mu_B_fo_line = np.linspace(280, 950, N_MEDIUM)

    # FRG consensus first-order line (uses QM2025 CP at 630 MeV)
    T_fo_line = first_order_line(mu_B_fo_line, params)
    valid = ~np.isnan(T_fo_line) & (T_fo_line > 0)
    save_curve(
        "qcd_firstorder_line.dat",
        mu_B_fo_line[valid],
        T_fo_line[valid],
        "mu_B T",
        [
            "First-order line (FRG consensus QM2025)",
            f"CP: T = {params.T_cp_frg} MeV, μ_B = {params.mu_B_cp_frg} MeV",
            "Source: Fu et al., arXiv:2510.11270",
        ],
        output_dir=OUTPUT_DIR,
    )

    # NJL model (CP at low μ_B - now known to be EXCLUDED)
    T_njl = first_order_njl(mu_B_fo_line, params)
    valid_njl = ~np.isnan(T_njl) & (T_njl > 0)
    save_curve(
        "qcd_firstorder_njl.dat",
        mu_B_fo_line[valid_njl],
        T_njl[valid_njl],
        "mu_B T",
        [
            "First-order line from NJL model (EXCLUDED region)",
            "CP: T ~ 80 MeV, μ_B ~ 330 MeV",
            "NOTE: This CP location is EXCLUDED by Dec 2025 lattice",
            "Source: Buballa, Phys. Rep. 407 (2005) 205",
        ],
        output_dir=OUTPUT_DIR,
    )

    # PQM model (CP at moderate μ_B - now known to be EXCLUDED)
    T_pqm = first_order_pqm(mu_B_fo_line, params)
    valid_pqm = ~np.isnan(T_pqm) & (T_pqm > 0)
    save_curve(
        "qcd_firstorder_pqm.dat",
        mu_B_fo_line[valid_pqm],
        T_pqm[valid_pqm],
        "mu_B T",
        [
            "First-order line from PQM model (EXCLUDED region)",
            "CP: T ~ 95 MeV, μ_B ~ 370 MeV",
            "NOTE: This CP location is EXCLUDED by Dec 2025 lattice",
            "Source: Schaefer et al., PRD 76 (2007) 074023",
        ],
        output_dir=OUTPUT_DIR,
    )

    # FRG model (updated to QM2025 consensus)
    T_frg = first_order_frg(mu_B_fo_line, params)
    valid_frg = ~np.isnan(T_frg) & (T_frg > 0)
    save_curve(
        "qcd_firstorder_frg.dat",
        mu_B_fo_line[valid_frg],
        T_frg[valid_frg],
        "mu_B T",
        [
            "First-order line from FRG (QM2025 consensus)",
            f"CP: T = {params.T_cp_frg} MeV, μ_B = {params.mu_B_cp_frg} MeV",
            "BEYOND lattice exclusion (μ_B > 450 MeV)",
            "Source: Fu et al., arXiv:2510.11270",
        ],
        output_dir=OUTPUT_DIR,
    )

    # CONSENSUS BAND (encompasses NJL/PQM/FRG spread for visualization)
    mu_B_band, T_upper, T_lower = first_order_consensus_band(100)
    save_curve_multi(
        "qcd_firstorder_consensus_band.dat",
        mu_B_band,
        [T_upper, T_lower],
        "mu_B T_upper T_lower",
        [
            "First-order CONSENSUS BAND",
            "Encompasses NJL/PQM/FRG theoretical spread",
            "Use for shaded band visualization",
        ],
        output_dir=OUTPUT_DIR,
    )

    # -------------------------------------------------------------------------
    # 4. Critical point: EXCLUSION + FRG consensus (December 2025 update)
    # -------------------------------------------------------------------------
    print("\n--- Critical Point (December 2025 Update) ---")

    # CP EXCLUSION REGION from Borsányi et al. PRD 112, L111505
    print("  Generating CP exclusion region (μ_B < 450 MeV at 2σ)...")
    mu_B_excl, T_excl = critical_point_exclusion_region(100)
    save_curve(
        "qcd_cp_exclusion_region.dat",
        mu_B_excl,
        T_excl,
        "mu_B T",
        [
            "CRITICAL POINT EXCLUSION REGION",
            "μ_B < 450 MeV EXCLUDED at 2σ confidence",
            "Source: Borsányi et al., PRD 112, L111505 (Dec 2025)",
            "Method: Yang-Lee edge singularity extrapolation",
        ],
        output_dir=OUTPUT_DIR,
    )

    # Exclusion boundary line
    mu_B_excl_line, T_excl_line = critical_point_exclusion_boundary(100)
    save_curve(
        "qcd_cp_exclusion_boundary.dat",
        mu_B_excl_line,
        T_excl_line,
        "mu_B T",
        ["CP exclusion boundary: μ_B = 450 MeV (2σ)"],
        output_dir=OUTPUT_DIR,
    )

    # FRG CONSENSUS CP (QM2025) - the current best estimate
    cp_frg_point = [[params.mu_B_cp_frg, params.T_cp_frg]]
    save_points_with_errors("qcd_critical_point.dat", cp_frg_point, "mu_B T", output_dir=OUTPUT_DIR)
    print(f"  FRG consensus CP: ({params.mu_B_cp_frg}, {params.T_cp_frg}) MeV")

    # FRG consensus ellipse
    mu_B_frg_ell, T_frg_ell = critical_point_frg_ellipse(100)
    save_curve(
        "qcd_critical_ellipse.dat",
        mu_B_frg_ell,
        T_frg_ell,
        "mu_B T",
        [
            "1-sigma uncertainty ellipse for FRG consensus CP",
            f"Center: ({params.mu_B_cp_frg}, {params.T_cp_frg}) MeV",
            "~10% uncertainty",
            "Source: Fu et al., arXiv:2510.11270 (QM2025 consensus)",
        ],
        output_dir=OUTPUT_DIR,
    )

    # FRG consensus box
    box_frg = critical_point_frg_box()
    box_frg_data = [
        [box_frg[0], box_frg[2]],
        [box_frg[1], box_frg[2]],
        [box_frg[1], box_frg[3]],
        [box_frg[0], box_frg[3]],
        [box_frg[0], box_frg[2]],
    ]
    save_points_with_errors("qcd_critical_box.dat", box_frg_data, "mu_B T", output_dir=OUTPUT_DIR)

    # DEPRECATED: Clarke et al. 2024 CP estimate (NOW EXCLUDED)
    print("  Generating EXCLUDED Clarke et al. CP region for reference...")
    mu_B_clarke_ell, T_clarke_ell = critical_point_ellipse_excluded(100)
    save_curve(
        "qcd_critical_ellipse_excluded.dat",
        mu_B_clarke_ell,
        T_clarke_ell,
        "mu_B T",
        [
            "EXCLUDED: Clarke et al. 2024 CP estimate",
            f"Center: ({params.mu_B_cp_clarke}, {params.T_cp_clarke}) MeV",
            "NOW EXCLUDED by Borsányi et al. (Dec 2025)",
            "Kept for reference only",
        ],
        output_dir=OUTPUT_DIR,
    )

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

    save_curve("qcd_qgp_boundary.dat", mu_B_fill, T_qgp, "mu_B T", output_dir=OUTPUT_DIR)

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
            f.write(f"# {cs.name}: ({cs.mu_B:.1f}, {cs.T:.1f}) MeV, sqrt_s = {cs.sqrt_s_NN} GeV\n")
        f.write("# mu_B T sqrt_s_NN\n")
        for d in cs_data:
            f.write(f"{d[0]:.2f} {d[1]:.2f} {d[2]:.1f}\n")
    print(f"  qcd_collision_systems.dat: {len(cs_data)} systems")

    # -------------------------------------------------------------------------
    # 7. Cooling trajectory
    # -------------------------------------------------------------------------
    print("\n--- Cooling Trajectory ---")

    mu_B_cool, T_cool = cooling_trajectory(N_MEDIUM)
    save_curve(
        "qcd_cooling_trajectory.dat",
        mu_B_cool,
        T_cool,
        "mu_B T",
        ["Cooling trajectory for central Pb-Pb at LHC", "T_0 ~ 400 MeV -> T_kinetic ~ 100 MeV"],
        output_dir=OUTPUT_DIR,
    )

    # -------------------------------------------------------------------------
    # 8. Isentropic trajectories (s/n_B = const)
    # -------------------------------------------------------------------------
    print("\n--- Isentropic Trajectories (RHIC BES) ---")

    for s_nB, label in ISENTROPE_VALUES:
        mu_B_isen, T_isen = isentropic_trajectory(s_nB, N_MEDIUM)
        filename = f"qcd_isentrope_{s_nB}.dat"
        save_curve(
            filename,
            mu_B_isen,
            T_isen,
            "mu_B T",
            [f"Isentropic trajectory s/n_B = {s_nB}", f"{label}", "Source: arXiv:1506.07350"],
            output_dir=OUTPUT_DIR,
        )

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
    save_curve(
        "qcd_early_universe.dat",
        mu_B_eu,
        T_eu,
        "mu_B T",
        [
            "Early universe trajectory (Big Bang → Hadronization)",
            "μ_B ≈ 0 due to baryon-antibaryon symmetry",
            "Hadronization at t ~ 10^-5 s, T ~ 150 MeV",
        ],
        output_dir=OUTPUT_DIR,
    )

    # Neutron star core region
    mu_B_ns, T_ns, T_ns_up, T_ns_lo = neutron_star_trajectory(100)
    save_curve_multi(
        "qcd_neutron_star.dat",
        mu_B_ns,
        [T_ns, T_ns_up, T_ns_lo],
        "mu_B T T_upper T_lower",
        [
            "Neutron star core region (cold, dense matter)",
            "μ_B ~ 1000-1500 MeV, T ~ 0 (after cooling)",
            "Probes cold QCD at extreme density",
        ],
        output_dir=OUTPUT_DIR,
    )

    # Color superconductivity boundary (schematic)
    mu_B_csc, T_csc = color_superconductivity_region()
    save_curve(
        "qcd_csc_boundary.dat",
        mu_B_csc,
        T_csc,
        "mu_B T",
        [
            "Color superconductivity phase boundary (schematic)",
            "CSC: Cooper pairs of quarks at high μ_B, low T",
            "Phases: 2SC, CFL, crystalline variants",
        ],
        output_dir=OUTPUT_DIR,
    )

    # -------------------------------------------------------------------------
    # 10. Future Facilities Coverage Regions
    # -------------------------------------------------------------------------
    print("\n--- Future Facilities Coverage ---")

    filepath = os.path.join(OUTPUT_DIR, "qcd_future_facilities.dat")
    with open(filepath, "w") as f:
        f.write("# Future heavy-ion facilities coverage regions\n")
        f.write("# Format: facility mu_B_min mu_B_max T_min T_max\n")
        for key, fac in FUTURE_FACILITIES.items():
            f.write(f"# {fac['name']}: {fac['description']}\n")  # type: ignore[index]
            f.write(f"# sqrt_s_NN range: {fac['sqrt_s_range'][0]}-{fac['sqrt_s_range'][1]} GeV\n")  # type: ignore[index]
            f.write(
                f"{key} {fac['mu_B_range'][0]} {fac['mu_B_range'][1]} "  # type: ignore[index]
                f"{fac['T_range'][0]} {fac['T_range'][1]}\n"  # type: ignore[index]
            )
    print(f"  qcd_future_facilities.dat: {len(FUTURE_FACILITIES)} facilities")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\nGeneration complete: {OUTPUT_DIR}/")
    files = [fname for fname in os.listdir(OUTPUT_DIR) if fname.startswith("qcd_")]
    print(f"QCD phase diagram files: {len(files)}")
    for fname in sorted(files):
        print(f"  - {fname}")


if __name__ == "__main__":
    generate_all()
