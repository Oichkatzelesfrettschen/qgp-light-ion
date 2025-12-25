#!/usr/bin/env python3
"""
generate_photon_data.py

Generates direct photon spectrum data for QGP light-ion visualizations.

Direct photons are electromagnetic probes of the QGP that escape without
strong interaction. The spectrum contains two main components:

1. Prompt photons: from initial hard scattering (pQCD calculable)
   - Power-law spectrum: dN/dp_T ∝ p_T^(-n), n ≈ 5-6
   - Dominant at high p_T (> 3 GeV/c)

2. Thermal photons: from hot QGP and hadron gas
   - Exponential spectrum: dN/dp_T ∝ exp(-p_T/T_eff)
   - Enhanced at low-intermediate p_T (1-3 GeV/c)
   - T_eff > T_kinetic (blue-shifted from radial flow)

The thermal photon excess in A-A vs pp collisions is a QGP signature.

Physical outputs:
- Direct photon spectra for pp, pPb, O-O, Pb-Pb
- Prompt vs thermal component separation
- Effective temperature extraction
- R_γ = (direct photons)/(decay photons) ratio

References:
- ALICE Collaboration, Phys. Lett. B 754, 235 (2016) - Direct photons in Pb-Pb
- PHENIX Collaboration, Phys. Rev. Lett. 104, 132301 (2010) - Thermal photon discovery
"""

import os

import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "photons")


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
            row = [f"{arr[i]:.8e}" for arr in arrays]
            f.write(" ".join(row) + "\n")


def prompt_photon_spectrum(pT, system="pp", sqrts=5020):
    """
    Prompt photon spectrum from initial hard scattering (NLO pQCD).

    dN/dp_T ∝ p_T^(-n) where n ≈ 5-6 for central rapidity

    Parameters
    ----------
    pT : array
        Transverse momentum [GeV/c]
    system : str
        Collision system (affects normalization)
    sqrts : float
        Center-of-mass energy per nucleon pair [GeV]

    Returns
    -------
    dN_dpT : array
        Prompt photon spectrum [arbitrary units, normalized at high pT]
    """
    # Power-law index (slightly system/energy dependent)
    power_index = 5.5 + 0.1 * (sqrts / 5020 - 1)  # Mild energy dependence

    # Normalization depends on system size
    system_norm = {
        "pp": 1.0,
        "pPb": 8.0,  # ~N_coll scaling
        "OO": 16.0,
        "NeNe": 20.0,
        "PbPb": 350.0,  # Central Pb-Pb N_coll ≈ 1700
    }

    norm = system_norm.get(system, 1.0)

    # Power-law with smooth cutoff at low pT to avoid divergence
    pT_min = 0.5  # GeV/c
    dN_dpT = norm * pT * (pT**2 + pT_min**2) ** (-power_index / 2)

    return dN_dpT


def thermal_photon_spectrum(pT, T_eff, system="PbPb"):
    """
    Thermal photon spectrum from QGP and hadron gas.

    dN/dp_T ∝ p_T exp(-p_T/T_eff)

    The effective temperature T_eff > T_kinetic due to radial flow.
    T_eff reflects average temperature weighted by emission rates.

    Parameters
    ----------
    pT : array
        Transverse momentum [GeV/c]
    T_eff : float
        Effective temperature [GeV]
    system : str
        Collision system (affects normalization)

    Returns
    -------
    dN_dpT : array
        Thermal photon spectrum [arbitrary units]
    """
    # Normalization by system size (proportional to volume × lifetime)
    system_norm = {
        "pp": 0.0,  # No thermal component in pp
        "pPb": 0.02,  # Minimal thermal component
        "OO": 0.15,
        "NeNe": 0.25,
        "PbPb": 1.0,  # Reference
    }

    norm = system_norm.get(system, 0.0)

    # Exponential thermal spectrum with pT prefactor (phase space)
    dN_dpT = norm * pT * np.exp(-pT / T_eff)

    return dN_dpT


def direct_photon_spectrum(pT, system="PbPb", T_eff=0.250):
    """
    Total direct photon spectrum = prompt + thermal.

    Parameters
    ----------
    pT : array
        Transverse momentum [GeV/c]
    system : str
        Collision system
    T_eff : float
        Effective temperature for thermal component [GeV]

    Returns
    -------
    dict with pT, prompt, thermal, total, and uncertainties
    """
    prompt = prompt_photon_spectrum(pT, system)
    thermal = thermal_photon_spectrum(pT, T_eff, system)
    total = prompt + thermal

    # Uncertainties (larger at low pT where thermal dominates)
    # Statistical + systematic combined
    rel_uncertainty = 0.15 + 0.20 * np.exp(-pT / 2.0)  # Higher at low pT
    uncertainty = total * rel_uncertainty

    return {
        "pT": pT,
        "prompt": prompt,
        "thermal": thermal,
        "total": total,
        "uncertainty": uncertainty,
    }


def photon_ratio_R_gamma(pT, system="PbPb"):
    """
    Ratio of direct photons to decay photons (from hadrons).

    R_γ = N_direct / N_decay

    R_γ > 1 indicates thermal photon excess.
    pp baseline: R_γ ≈ 1 (dominated by prompt photons and hadronic decays)

    Parameters
    ----------
    pT : array
        Transverse momentum [GeV/c]
    system : str
        Collision system

    Returns
    -------
    R_gamma : array
        Direct-to-decay photon ratio
    """
    # Base ratio from prompt photons (roughly constant ≈ 1.0 at high pT)
    R_prompt = 1.0

    # Thermal enhancement (system dependent, peaked at low-intermediate pT)
    if system == "pp":
        thermal_enhancement = np.zeros_like(pT)
    elif system == "pPb":
        thermal_enhancement = 0.05 * np.exp(-((pT - 1.5) ** 2) / 2.0)
    elif system == "OO":
        thermal_enhancement = 0.15 * np.exp(-((pT - 1.8) ** 2) / 1.5)
    elif system == "NeNe":
        thermal_enhancement = 0.25 * np.exp(-((pT - 1.8) ** 2) / 1.5)
    elif system == "PbPb":
        thermal_enhancement = 0.4 * np.exp(-((pT - 2.0) ** 2) / 2.0)
    else:
        thermal_enhancement = np.zeros_like(pT)

    R_gamma = R_prompt + thermal_enhancement

    return R_gamma


def extract_effective_temperature(system_params):
    """
    Effective temperature from inverse slope of thermal photon spectrum.

    T_eff is extracted from exponential fit in low-intermediate pT range.
    Physical interpretation: temperature-weighted average over spacetime.

    Parameters
    ----------
    system_params : dict
        System parameters with 'dNch' (multiplicity) and 'tau_QGP' (lifetime)

    Returns
    -------
    T_eff : float
        Effective temperature [MeV]
    """
    # Empirical scaling with multiplicity and system lifetime
    # T_eff increases with system size up to saturation
    dNch = system_params["dNch"]
    # tau_QGP available: system_params.get("tau_QGP", 5.0) fm/c

    # Base temperature from QGP (typical T ≈ 200-300 MeV)
    T_QGP = 200 + 50 * np.log(dNch / 100)  # MeV, saturates at ~300 MeV

    # Blue-shift from radial flow: T_eff = T + m⟨β_T⟩²/2
    # For photons (m=0), mainly reflects temperature profile
    beta_T_avg = 0.3 * (1 - np.exp(-dNch / 500))  # Saturation with multiplicity
    flow_boost = 100 * beta_T_avg  # MeV

    T_eff = T_QGP + flow_boost

    return T_eff


def main():
    ensure_dir(OUTPUT_DIR)

    print("Generating direct photon spectrum data...")

    # Transverse momentum range [GeV/c]
    pT = np.linspace(0.5, 10, 100)

    # =============================================================================
    # 1. Direct photon spectra for different systems
    # =============================================================================

    # Effective temperatures for thermal component (system dependent)
    T_eff_params = {
        "pp": 0.180,  # GeV (no thermal, but set for completeness)
        "pPb": 0.200,
        "OO": 0.240,
        "NeNe": 0.260,
        "PbPb": 0.280,
    }

    for system in ["pp", "pPb", "OO", "NeNe", "PbPb"]:
        T_eff = T_eff_params[system]
        spectrum = direct_photon_spectrum(pT, system, T_eff)

        save_dat(
            os.path.join(OUTPUT_DIR, f"direct_photon_spectrum_{system}.dat"),
            spectrum,
            f"Direct photon spectrum for {system}, T_eff = {T_eff * 1000:.0f} MeV",
        )
        print(f"  Direct photon spectrum {system:6s}: 100 points (T_eff = {T_eff * 1000:.0f} MeV)")

    # =============================================================================
    # 2. Prompt vs thermal decomposition (Pb-Pb example)
    # =============================================================================

    pT_detailed = np.linspace(0.5, 12, 150)
    spectrum_PbPb = direct_photon_spectrum(pT_detailed, "PbPb", T_eff=0.280)

    save_dat(
        os.path.join(OUTPUT_DIR, "photon_decomposition_PbPb.dat"),
        {
            "pT": spectrum_PbPb["pT"],
            "prompt": spectrum_PbPb["prompt"],
            "thermal": spectrum_PbPb["thermal"],
            "total": spectrum_PbPb["total"],
        },
        "Prompt + thermal decomposition for Pb-Pb 0-20%",
    )
    print("  Prompt/thermal decomposition Pb-Pb: 150 points")

    # O-O decomposition for comparison
    spectrum_OO = direct_photon_spectrum(pT_detailed, "OO", T_eff=0.240)
    save_dat(
        os.path.join(OUTPUT_DIR, "photon_decomposition_OO.dat"),
        {
            "pT": spectrum_OO["pT"],
            "prompt": spectrum_OO["prompt"],
            "thermal": spectrum_OO["thermal"],
            "total": spectrum_OO["total"],
        },
        "Prompt + thermal decomposition for O-O 0-10%",
    )
    print("  Prompt/thermal decomposition O-O: 150 points")

    # =============================================================================
    # 3. Direct-to-decay photon ratio R_γ
    # =============================================================================

    pT_ratio = np.linspace(0.8, 6, 80)

    for system in ["pp", "pPb", "OO", "NeNe", "PbPb"]:
        R_gamma = photon_ratio_R_gamma(pT_ratio, system)
        uncertainty = 0.15 * np.ones_like(R_gamma)  # Typical 15% uncertainty

        save_dat(
            os.path.join(OUTPUT_DIR, f"photon_ratio_R_gamma_{system}.dat"),
            {"pT": pT_ratio, "R_gamma": R_gamma, "uncertainty": uncertainty},
            f"Direct-to-decay photon ratio for {system}",
        )
        print(f"  R_γ ratio {system:6s}: 80 points")

    # =============================================================================
    # 4. Effective temperature vs system size
    # =============================================================================

    systems_T_eff = [
        {"name": "pp", "dNch": 7, "tau_QGP": 0},
        {"name": "pPb", "dNch": 45, "tau_QGP": 1.5},
        {"name": "OO", "dNch": 135, "tau_QGP": 3.5},
        {"name": "NeNe", "dNch": 170, "tau_QGP": 4.0},
        {"name": "XeXe", "dNch": 850, "tau_QGP": 8.0},
        {"name": "PbPb", "dNch": 1940, "tau_QGP": 11.0},
    ]

    T_eff_list = []
    dNch_list = []
    T_eff_err = []

    for sys_params in systems_T_eff:
        T_eff_calc = extract_effective_temperature(sys_params)
        T_eff_list.append(T_eff_calc)
        dNch_list.append(sys_params["dNch"])
        T_eff_err.append(25)  # Typical 25 MeV uncertainty

    save_dat(
        os.path.join(OUTPUT_DIR, "effective_temperature_vs_system.dat"),
        {
            "dNch": np.array(dNch_list),
            "T_eff": np.array(T_eff_list),
            "uncertainty": np.array(T_eff_err),
        },
        "Effective temperature vs charged multiplicity",
    )
    print(f"  T_eff vs system size: {len(systems_T_eff)} systems")

    # =============================================================================
    # 5. Inverse slope fit demonstration (exponential extraction)
    # =============================================================================

    # Demonstrate how T_eff is extracted from exponential fit
    # Use low-intermediate pT range (1-3 GeV/c) for thermal photons

    pT_fit_range = np.linspace(1.0, 3.5, 50)

    # Pb-Pb thermal spectrum
    thermal_PbPb = thermal_photon_spectrum(pT_fit_range, T_eff=0.280, system="PbPb")

    # For plotting: ln(dN/dp_T / p_T) vs p_T should be linear with slope -1/T_eff
    # dN/dp_T = A * p_T * exp(-p_T/T_eff)
    # ln(dN/dp_T / p_T) = ln(A) - p_T/T_eff
    ln_yield = np.log(thermal_PbPb / pT_fit_range)

    save_dat(
        os.path.join(OUTPUT_DIR, "thermal_photon_inverse_slope_fit.dat"),
        {
            "pT": pT_fit_range,
            "ln_yield": ln_yield,
        },
        "Inverse slope fit for T_eff extraction: ln(dN/dpT/pT) vs pT",
    )
    print("  Inverse slope fit data: 50 points")

    # =============================================================================
    # 6. Photon v2 (elliptic flow of photons) - small but measurable
    # =============================================================================

    # Direct photons also show elliptic flow (small, v2 ~ 0.01-0.03)
    # Reflects early-time anisotropic emission from QGP

    pT_v2 = np.linspace(1.0, 5.0, 30)

    # Pb-Pb photon v2 (increases with pT, smaller than hadron v2)
    v2_photon_PbPb = 0.015 + 0.015 * (1 - np.exp(-pT_v2 / 3.0))
    v2_err = 0.005 * np.ones_like(v2_photon_PbPb)

    save_dat(
        os.path.join(OUTPUT_DIR, "photon_v2_PbPb.dat"),
        {
            "pT": pT_v2,
            "v2": v2_photon_PbPb,
            "uncertainty": v2_err,
        },
        "Direct photon elliptic flow v2 for Pb-Pb 20-40%",
    )
    print("  Photon v2: 30 points")

    # =============================================================================
    # Summary
    # =============================================================================

    print(f"\nDirect photon data written to {OUTPUT_DIR}/")
    print("Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".dat"):
            print(f"  - {f}")

    print("\nKey physics encoded:")
    print("  - Prompt photons: power-law dN/dpT ∝ pT^(-n)")
    print("  - Thermal photons: exponential dN/dpT ∝ pT exp(-pT/T_eff)")
    print("  - T_eff ≈ 220-280 MeV (system dependent)")
    print("  - R_γ ratio shows thermal excess at low-intermediate pT")
    print("  - Photon v2 reflects early-time anisotropy")


if __name__ == "__main__":
    main()
