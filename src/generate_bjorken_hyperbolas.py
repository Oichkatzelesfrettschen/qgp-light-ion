#!/usr/bin/env python3
"""
generate_bjorken_hyperbolas.py

Generates pre-computed data for Bjorken spacetime diagram:
- Constant proper-time hyperbolas: t = sqrt(tau^2 + z^2)
- Particle emission vectors from freeze-out surface
- Temperature/energy density at each proper time
"""

import os

import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "spacetime")


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def generate_hyperbola(tau, z_max, n_points=100):
    """
    Generate (z, t) points for constant proper-time hyperbola.

    Parameters
    ----------
    tau : float
        Proper time in fm/c
    z_max : float
        Maximum z extent (symmetric about 0)
    n_points : int
        Number of points

    Returns
    -------
    z : ndarray
        Longitudinal position
    t : ndarray
        Lab time: t = sqrt(tau^2 + z^2)
    """
    z = np.linspace(-z_max, z_max, n_points)
    t = np.sqrt(tau**2 + z**2)
    return z, t


def freeze_out_emission_vectors(tau_f, z_positions):
    """
    Calculate emission vectors perpendicular to freeze-out hyperbola.

    The 4-velocity normal to constant-tau surface is:
    u^μ = (t, 0, 0, z) / tau  (rapidity direction)

    Parameters
    ----------
    tau_f : float
        Freeze-out proper time
    z_positions : array
        z coordinates of emission points

    Returns
    -------
    dict with z, t, dz, dt for arrow coordinates
    """
    t = np.sqrt(tau_f**2 + z_positions**2)

    # Rapidity direction: perpendicular to hyperbola
    # dz/dt = z/t (tangent direction)
    # Normal: n_z = dt/ds, n_t = -dz/ds normalized

    # For constant tau: dt/dz = z/t
    # Normal vector (outward): (n_t, n_z) proportional to (tau/t, tau*z/t^2)
    # Simplifies to: (1, z/t) normalized, pointing outward in time

    n_z = z_positions / t  # Component along z
    n_t = tau_f / t  # Component along t

    # Normalize
    norm = np.sqrt(n_z**2 + n_t**2)
    n_z /= norm
    n_t /= norm

    # Arrow length
    arrow_len = 0.8

    return {"z": z_positions, "t": t, "dz": arrow_len * n_z, "dt": arrow_len * n_t}


def save_dat(filename, data_dict, header=""):
    """Save data to .dat file."""
    keys = list(data_dict.keys())
    arrays = [np.atleast_1d(data_dict[k]) for k in keys]

    with open(filename, "w") as f:
        if header:
            f.write(f"# {header}\n")
        f.write("# " + " ".join(keys) + "\n")
        for i in range(len(arrays[0])):
            row = [f"{arr[i]:.6f}" for arr in arrays]
            f.write(" ".join(row) + "\n")


def main():
    ensure_dir(OUTPUT_DIR)

    print("Generating Bjorken spacetime hyperbola data...")

    # Define key proper times (fm/c)
    tau_values = {"thermalization": 1.0, "qgp": 3.0, "hadronization": 6.0, "freezeout": 10.0}

    # Temperature at each stage (MeV) - Bjorken cooling T ∝ τ^(-1/3)
    T0 = 350  # Initial temperature at tau_0 = 1 fm/c
    tau_0 = 1.0
    temperatures = {name: T0 * (tau_0 / tau) ** (1 / 3) for name, tau in tau_values.items()}

    # Generate hyperbola data for each stage
    for name, tau in tau_values.items():
        z_max = min(tau * 0.8, 7.5)  # Limit z extent
        z, t = generate_hyperbola(tau, z_max, n_points=80)

        # Include temperature as metadata
        T = temperatures[name]

        save_dat(
            os.path.join(OUTPUT_DIR, f"hyperbola_{name}.dat"),
            {"z": z, "t": t},
            f"Constant tau={tau:.1f} fm/c hyperbola, T={T:.0f} MeV",
        )
        print(f"  tau = {tau:.1f} fm/c (T = {T:.0f} MeV): {len(z)} points")

    # Generate emission vectors for freeze-out
    z_emit = np.array([-6, -4, -2, 0, 2, 4, 6])
    emission = freeze_out_emission_vectors(tau_values["freezeout"], z_emit)

    save_dat(
        os.path.join(OUTPUT_DIR, "freezeout_emission.dat"),
        emission,
        "Particle emission vectors from freeze-out hyperbola",
    )
    print(f"  Freeze-out emission vectors: {len(z_emit)} arrows")

    # Generate light cone data
    z_lc = np.linspace(0, 8, 50)
    save_dat(
        os.path.join(OUTPUT_DIR, "light_cone_right.dat"),
        {"z": z_lc, "t": z_lc},
        "Right light cone: t = z",
    )
    save_dat(
        os.path.join(OUTPUT_DIR, "light_cone_left.dat"),
        {"z": -z_lc, "t": z_lc},
        "Left light cone: t = -z",
    )
    print("  Light cones: 2 x 50 points")

    # Summary of key physics parameters (for reference)
    _summary = {
        "tau": list(tau_values.values()),
        "T": [temperatures[name] for name in tau_values],
    }

    # Bjorken energy density evolution
    # ε(τ) = ε₀ (τ₀/τ)^(4/3) for ideal gas
    eps_0 = 15  # GeV/fm³ at tau_0
    tau_array = np.linspace(0.5, 12, 100)
    eps = eps_0 * (tau_0 / tau_array) ** (4 / 3)

    save_dat(
        os.path.join(OUTPUT_DIR, "bjorken_energy_density_evolution.dat"),
        {"tau": tau_array, "epsilon": eps},
        "Bjorken energy density evolution: eps(tau) = eps_0 (tau_0/tau)^(4/3)",
    )
    print("  Energy density evolution: 100 points")

    print(f"\nData written to {OUTPUT_DIR}/")
    print("Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".dat"):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
