"""Tests for src/cosmology/generate.py data generation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cosmology.generate import main


class TestCosmologyGenerate:
    """Test cosmology data generation script."""

    def test_main_runs_without_error(self, tmp_path: Path) -> None:
        """Verify main() runs without error."""
        main([f"--output-dir={tmp_path}"])
        # If we reach here, no exception was raised
        assert True

    def test_output_files_created(self, tmp_path: Path) -> None:
        """Verify all 7 expected .dat files are created."""
        main([f"--output-dir={tmp_path}"])

        expected_files = [
            tmp_path / "cosmology" / "dark_energy" / "omega_m_sweep.dat",
            tmp_path / "cosmology" / "dark_energy" / "distance_modulus.dat",
            tmp_path / "cosmology" / "reionization" / "bubble_radii.dat",
            tmp_path / "cosmology" / "reionization" / "percolation_curve.dat",
            tmp_path / "cosmology" / "reionization" / "ionized_fraction.dat",
            tmp_path / "cosmology" / "reionization" / "ly_alpha_transmission.dat",
            tmp_path / "cosmology" / "reionization" / "front_expansion.dat",
        ]

        for file_path in expected_files:
            assert file_path.exists(), f"Expected output file {file_path} not found"

    def test_dark_energy_file_has_provenance(self, tmp_path: Path) -> None:
        """Verify dark energy output includes provenance header."""
        main([f"--output-dir={tmp_path}"])
        omega_m_file = tmp_path / "cosmology" / "dark_energy" / "omega_m_sweep.dat"

        with open(omega_m_file) as f:
            content = f.read(500)
            assert "# DATA TYPE:" in content, (
                f"Expected provenance header with '# DATA TYPE:', got: {content[:200]}"
            )

    def test_ionized_fraction_monotonic(self, tmp_path: Path) -> None:
        """Verify ionized fraction is monotonically non-increasing with z."""
        main([f"--output-dir={tmp_path}"])
        xe_file = tmp_path / "cosmology" / "reionization" / "ionized_fraction.dat"

        # Read data (skip header lines)
        data = np.loadtxt(xe_file, skiprows=1)
        z_vals = data[:, 0]
        x_e_vals = data[:, 1]

        # Verify monotonicity: x_e should be non-increasing with z
        # (Higher z = earlier universe = lower ionized fraction)
        diffs = np.diff(x_e_vals)
        assert np.all(diffs <= 1e-10), (
            f"Ionized fraction not monotonically non-increasing. "
            f"Max positive diff: {np.max(diffs)}"
        )

    def test_ly_alpha_transmission_bounded(self, tmp_path: Path) -> None:
        """Verify Ly-alpha transmission values are in [0, 1]."""
        main([f"--output-dir={tmp_path}"])
        lya_file = tmp_path / "cosmology" / "reionization" / "ly_alpha_transmission.dat"

        # Read data (skip header lines)
        data = np.loadtxt(lya_file, skiprows=1)
        T_neutral = data[:, 1]
        T_half = data[:, 2]
        T_full = data[:, 3]

        # All transmission values should be in [0, 1]
        for col, name in [(T_neutral, "T_neutral"), (T_half, "T_half"), (T_full, "T_full")]:
            assert np.all(col >= -1e-10) and np.all(col <= 1.0 + 1e-10), (
                f"{name} values out of bounds [0,1]: min={np.min(col)}, max={np.max(col)}"
            )
