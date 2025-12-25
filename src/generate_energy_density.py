#!/usr/bin/env python3
"""
Generate precomputed energy density heatmap data for TikZ/PGFPlots.

Creates a 2D matrix of energy density values for a non-central heavy-ion collision
with granular hot spots from individual nucleon-nucleon collisions.
"""

import os

import numpy as np


def energy_density(x, y, impact_param=3.0):
    """
    Compute energy density at (x, y) for a non-central collision.

    Combines smooth Woods-Saxon-like background with granular hot spots
    from individual nucleon-nucleon collisions.

    Args:
        x, y: Coordinates in fm
        impact_param: Impact parameter / 2 (nucleus offset from origin)

    Returns:
        Energy density in GeV/fm^3
    """
    b = impact_param  # Half impact parameter

    # Smooth background from two overlapping nuclei (Woods-Saxon-like)
    nucleus_a = np.exp(-((x + b) ** 2 + y**2) / 8)
    nucleus_b = np.exp(-((x - b) ** 2 + y**2) / 8)
    background = (nucleus_a + nucleus_b) * 50

    # Hot spots from individual nucleon-nucleon collisions
    # Creates the granular structure seen in Glauber MC initial conditions
    hot_spots = [
        (0.8, 0.3, 15, 0.30),  # (x_center, y_center, amplitude, sigma^2)
        (0.2, 0.5, 18, 0.25),
        (-0.5, 0.8, 12, 0.35),
        (1.2, 0.9, 16, 0.28),
        (-0.9, 0.2, 14, 0.32),
        (-1.5, 0.6, 13, 0.30),
        (0.5, 1.1, 11, 0.40),
        (-0.1, 1.0, 17, 0.27),
        (1.8, 0.4, 10, 0.35),
        (-1.1, 0.7, 15, 0.29),
        # Additional hot spots for more realistic texture
        (0.0, -0.5, 14, 0.33),
        (-0.7, -0.8, 12, 0.31),
        (1.0, -0.3, 16, 0.28),
        (-1.3, -0.5, 11, 0.36),
        (0.4, -1.0, 13, 0.34),
    ]

    granular = np.zeros_like(x)
    for xc, yc, amp, sigma2 in hot_spots:
        granular += amp * np.exp(-((x - xc) ** 2 + (y - yc) ** 2) / sigma2)

    return background + granular


def write_pgfplots_table(filename, x, y, z, samples):
    """
    Write data in PGFPlots table format for matrix plot.

    Format: x y z (tab-separated, row-by-row)
    """
    with open(filename, "w") as f:
        f.write(f"# Energy density heatmap data ({samples}x{samples} samples)\n")
        f.write("# x [fm]  y [fm]  epsilon [GeV/fm^3]\n")
        for j in range(samples):
            for i in range(samples):
                f.write(f"{x[i, j]:.4f}\t{y[i, j]:.4f}\t{z[i, j]:.4f}\n")
            f.write("\n")  # Blank line between y-rows (required for PGFPlots matrix format)


def main():
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    # Generate high-resolution grid
    samples = 150  # Start with 150 as requested
    x_range = (-5, 5)
    y_range = (-5, 5)

    x_1d = np.linspace(x_range[0], x_range[1], samples)
    y_1d = np.linspace(y_range[0], y_range[1], samples)
    x_grid, y_grid = np.meshgrid(x_1d, y_1d)

    # Compute energy density
    epsilon = energy_density(x_grid, y_grid, impact_param=1.5)

    # Clamp values to [0, 100] for colorbar
    epsilon = np.clip(epsilon, 0, 100)

    # Write output file
    output_file = os.path.join(data_dir, "energy_density_2d.dat")
    write_pgfplots_table(output_file, x_grid, y_grid, epsilon, samples)

    print(f"Generated {output_file}")
    print(f"  Grid: {samples} x {samples} = {samples**2} points")
    print(f"  x range: [{x_range[0]}, {x_range[1]}] fm")
    print(f"  y range: [{y_range[0]}, {y_range[1]}] fm")
    print(f"  epsilon range: [{epsilon.min():.2f}, {epsilon.max():.2f}] GeV/fm^3")

    # Also generate lower-resolution versions for testing
    for test_samples in [50, 75, 100]:
        x_1d = np.linspace(x_range[0], x_range[1], test_samples)
        y_1d = np.linspace(y_range[0], y_range[1], test_samples)
        x_grid, y_grid = np.meshgrid(x_1d, y_1d)
        epsilon = np.clip(energy_density(x_grid, y_grid, impact_param=1.5), 0, 100)
        output_file = os.path.join(data_dir, f"energy_density_2d_{test_samples}.dat")
        write_pgfplots_table(output_file, x_grid, y_grid, epsilon, test_samples)
        print(f"Generated {output_file} ({test_samples}x{test_samples})")


if __name__ == "__main__":
    main()
