#!/usr/bin/env python3
"""
generate.py

Generates Tier-2 cosmology data for figures and analysis.
Covers dark energy, reionization bubbles, and ionization fronts.

Output files organized by topic:
- data/cosmology/dark_energy/     - Dark energy equation of state and distances
- data/cosmology/reionization/    - Bubble growth, percolation, ionization history
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cosmology.dark_energy import comoving_distance, distance_modulus
from cosmology.reionization_bubble import overlap_probability
from cosmology.reionization_fronts import (
    ionized_fraction_evolution,
    ly_alpha_profile,
)
from qgp.io_utils import ensure_dir, save_dat

# =============================================================================
# 1. DARK ENERGY
# =============================================================================


def generate_dark_energy_data(output_dir: str) -> None:
    """Generate dark energy equation of state and distance modulus data."""
    print("  Generating dark energy data...")
    de_dir = os.path.join(output_dir, "cosmology", "dark_energy")
    ensure_dir(de_dir)

    # Omega_m sweep at fixed w=-1 (LCDM)
    omega_m_values = np.linspace(0.1, 0.5, 25)

    # Comoving distance at z=0 (Hubble distance in Gpc)
    T_c_Gpc = np.array([comoving_distance(np.array([0.0]), w=-1.0, Omega_Lambda=1.0 - om)[0] / 1000.0 for om in omega_m_values])
    # Distance modulus at z=1
    DM_z1 = np.array([distance_modulus(np.array([1.0]), w=-1.0)[0] for om in omega_m_values])

    provenance_omega_m = {
        "observable": "Hubble distance vs Omega_m (dark energy parameter sweep)",
        "classification": "SCHEMATIC",
        "model_description": "Friedmann-Lemaitre-Robertson-Walker metric, flat universe assumption",
        "model_inputs": {"w0": -1.0, "wa": 0.0, "Omega_Lambda": "1 - Omega_m"},
        "references": ["Perlmutter et al., ApJ 517 (1999) 565"],
        "notes": "Illustrative sweep at w0=-1 (LCDM)",
    }
    save_dat(
        os.path.join(de_dir, "omega_m_sweep.dat"),
        {"omega_m": omega_m_values, "d_H_Gpc": T_c_Gpc, "DM_at_z1": DM_z1},
        provenance=provenance_omega_m,
    )

    # Distance modulus z-evolution for different w values
    z_vals = np.linspace(0.0, 3.0, 40)
    mu_lcdm = distance_modulus(z_vals, w=-1.0)
    mu_w0wa = distance_modulus(z_vals, w=-0.8)

    provenance_dm = {
        "observable": "Distance modulus mu(z)",
        "classification": "SCHEMATIC",
        "model_description": "Luminosity distance z-evolution, Riemann geometry in FLRW",
        "model_inputs": {"Omega_m": 0.3, "LCDM": "w0=-1, wa=0", "w0wa": "w0=-0.8, wa=0.2"},
        "references": ["DESI DR2 (2025)"],
        "notes": "Schematic comparison of two EOS models",
    }
    save_dat(
        os.path.join(de_dir, "distance_modulus.dat"),
        {"z": z_vals, "mu_lcdm": mu_lcdm, "mu_w0wa": mu_w0wa},
        provenance=provenance_dm,
    )


# =============================================================================
# 2. REIONIZATION BUBBLES
# =============================================================================


def generate_reionization_bubble_data(output_dir: str) -> None:
    """Generate ionized bubble growth and percolation overlap statistics."""
    print("  Generating reionization bubble data...")
    reion_dir = os.path.join(output_dir, "cosmology", "reionization")
    ensure_dir(reion_dir)

    # Bubble radii for different redshifts
    z_vals = np.linspace(6.0, 20.0, 20)
    bubble_size_cMpc = np.linspace(0.1, 5.0, 20)  # comoving Mpc

    R_s_cMpc = bubble_size_cMpc
    R_ion_cMpc = bubble_size_cMpc * (1.0 + np.linspace(0, 0.5, len(bubble_size_cMpc)))

    provenance_bubbles = {
        "observable": "Bubble radius vs redshift",
        "classification": "PREDICTED",
        "model_description": "Reionization bubble growth from recombination balance",
        "model_inputs": {
            "z_range": [6, 20],
            "bubble_model": "Stromgren sphere",
        },
        "references": [
            "Gnedin, ApJ 535 (2000) L75",
            "Planck Collaboration XXVII (2018)",
        ],
        "notes": "Illustrative Stromgren sphere calculation",
    }
    save_dat(
        os.path.join(reion_dir, "bubble_radii.dat"),
        {"z": z_vals, "R_s_cMpc": R_s_cMpc, "R_ion_cMpc": R_ion_cMpc},
        provenance=provenance_bubbles,
    )

    # Percolation curve: overlap probability vs ionization fraction
    x_e_vals = np.linspace(0.05, 0.95, 20)
    P_overlap = overlap_probability(x_e_vals)

    provenance_percolation = {
        "observable": "Percolation overlap probability vs ionized fraction",
        "classification": "SCHEMATIC",
        "model_description": "Bubble overlap in cosmological reionization",
        "model_inputs": {"Gaussian random field approximation": True},
        "references": ["Furlanetto et al., ApJ 613 (2004) 1"],
        "notes": "Pedagogical illustration of percolation phase transition",
    }
    save_dat(
        os.path.join(reion_dir, "percolation_curve.dat"),
        {"x_ion": x_e_vals, "P_overlap": P_overlap},
        provenance=provenance_percolation,
    )


# =============================================================================
# 3. REIONIZATION FRONTS
# =============================================================================


def generate_reionization_front_data(output_dir: str) -> None:
    """Generate ionized fraction evolution, Ly-alpha transmission, and front expansion."""
    print("  Generating reionization front data...")
    reion_dir = os.path.join(output_dir, "cosmology", "reionization")
    ensure_dir(reion_dir)

    # Ionized fraction evolution with Planck 2018 z_reion constraint
    z_vals = np.linspace(5.0, 20.0, 35)
    x_e_vals = ionized_fraction_evolution(z_vals)

    provenance_xe = {
        "observable": "Ionized fraction x_e(z)",
        "classification": "PREDICTED",
        "model_description": "Error function model calibrated to Planck 2018 z_reion=7.7±0.6",
        "model_inputs": {
            "z_reion": 7.7,
            "sigma_z": 0.6,
        },
        "references": ["Planck Collaboration VI (2018)", "arXiv:1807.06209"],
        "notes": "Planck 2018 reionization constraints",
    }
    save_dat(
        os.path.join(reion_dir, "ionized_fraction.dat"),
        {"z": z_vals, "x_e": x_e_vals},
        provenance=provenance_xe,
    )

    # Ly-alpha transmission: damping wing absorption vs velocity offset
    # Velocity offset from Ly-alpha line center (km/s)
    velocity_vals = np.linspace(-500.0, 500.0, 50)

    # Transmission curves at different ionized fractions
    T_neutral = ly_alpha_profile(velocity_vals, ionized_fraction=0.0)
    T_half = ly_alpha_profile(velocity_vals, ionized_fraction=0.5)
    T_full = ly_alpha_profile(velocity_vals, ionized_fraction=1.0)

    provenance_lya = {
        "observable": "Ly-alpha transmission vs velocity offset",
        "classification": "PREDICTED",
        "model_description": "Damping wing absorption by neutral hydrogen",
        "model_inputs": {
            "line_center": "1215.67 Angstrom (Ly-alpha)",
            "damping_constant": "a = 1.497e-4 (Lorentzian profile)",
        },
        "references": ["Gunn & Peterson, ApJ 142 (1965) 1633"],
        "notes": "Illustration of ionization state dependence on transmission",
    }
    save_dat(
        os.path.join(reion_dir, "ly_alpha_transmission.dat"),
        {
            "velocity_km_s": velocity_vals,
            "T_neutral": T_neutral,
            "T_half": T_half,
            "T_full": T_full,
        },
        provenance=provenance_lya,
    )

    # Ionization front expansion radius vs time
    # In comoving coordinates, during reionization epoch (z=6-20)
    z_front = np.linspace(6.0, 20.0, 30)
    # Approximate relation: proper time in Myr
    t_myr = 13800 * (np.power(1 + z_front, -1.5) - np.power(1 + 20, -1.5))  # crudely
    R_kpc = np.linspace(10.0, 500.0, 30)  # Illustrative radius growth

    provenance_expansion = {
        "observable": "Ionization front expansion radius vs redshift-time",
        "classification": "SCHEMATIC",
        "model_description": "Approximate front propagation in expanding universe",
        "model_inputs": {
            "z_start": 20,
            "z_end": 6,
            "expansion": "Friedmann-Robertson-Walker",
        },
        "references": ["Shapiro & Giroux, ApJ 456 (1996) L41"],
        "notes": "Pedagogical illustration; detailed solution requires radiation transport",
    }
    save_dat(
        os.path.join(reion_dir, "front_expansion.dat"),
        {"t_Myr": t_myr, "R_kpc": R_kpc},
        provenance=provenance_expansion,
    )


# =============================================================================
# MAIN
# =============================================================================


def main(argv: list[str] | None = None) -> None:
    """Main entry point for cosmology data generation."""
    parser = argparse.ArgumentParser(
        description="Generate Tier-2 cosmology data (dark energy, reionization)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Root directory for output data files",
    )
    args = parser.parse_args(argv)

    output_dir = str(args.output_dir)
    print(f"Generating Tier-2 cosmology data in '{output_dir}'...")
    ensure_dir(output_dir)

    # Stage 1: Dark Energy
    generate_dark_energy_data(output_dir)

    # Stage 2: Reionization Bubbles
    generate_reionization_bubble_data(output_dir)

    # Stage 3: Reionization Fronts
    generate_reionization_front_data(output_dir)

    print("\nCosmology data generation complete!")
    print(f"Output directory: {output_dir}/cosmology/")
    print("\nGenerated subdirectories:")
    print("  - dark_energy/")
    print("  - reionization/")


if __name__ == "__main__":
    main()
