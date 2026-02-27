"""
test_data_generation.py - Integration tests for the QGP data generation pipeline.

These tests validate:
1. Required source files exist
2. Data format and physical constraints (when data/ is present)
3. Build artifact integrity (Makefile, LaTeX sources)

Tests that depend on generated data are skipped when data/ does not exist.
Run `make data` to generate data before running these tests.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dat_file(filepath: Path) -> tuple[np.ndarray, list[str]]:
    """Load a .dat file and return (data_array, column_names).

    Reads the last comment line before data rows as the column header.
    """
    col_names: list[str] = []
    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#"):
                col_names = stripped.lstrip("# ").split()
            elif stripped:
                break
    data = np.loadtxt(filepath)
    return data, col_names


def data_exists() -> bool:
    """Return True if the data/ directory has been generated."""
    return DATA_DIR.exists() and any(DATA_DIR.rglob("*.dat"))


skip_if_no_data = pytest.mark.skipif(
    not data_exists(),
    reason="data/ directory not present -- run `make data` first",
)


# ---------------------------------------------------------------------------
# Source file existence
# ---------------------------------------------------------------------------

class TestSourceFiles:
    """Verify that key source files are present."""

    def test_generate_comprehensive_data_exists(self):
        assert (SRC_DIR / "generate_comprehensive_data.py").exists(), (
            "generate_comprehensive_data.py not found in src/"
        )

    def test_qgp_physics_exists(self):
        assert (SRC_DIR / "qgp_physics.py").exists(), "qgp_physics.py not found in src/"

    def test_constants_exists(self):
        assert (SRC_DIR / "constants.py").exists(), "constants.py not found in src/"

    def test_io_utils_exists(self):
        assert (SRC_DIR / "io_utils.py").exists(), "io_utils.py not found in src/"

    def test_makefile_exists(self):
        makefile = PROJECT_ROOT / "Makefile"
        assert makefile.exists(), "Makefile not found"

    def test_makefile_required_targets(self):
        content = (PROJECT_ROOT / "Makefile").read_text()
        for target in ("all", "clean", "figures", "data"):
            assert f"{target}:" in content, f"Makefile missing target: {target}"

    def test_latex_main_exists(self):
        assert (PROJECT_ROOT / "qgp-light-ion.tex").exists()

    def test_bibliography_exists(self):
        assert (PROJECT_ROOT / "references.bib").exists()

    def test_required_figures_exist(self):
        figures_dir = PROJECT_ROOT / "figures"
        required = [
            "RAA_visualization.tex",
            "flow_visualization.tex",
            "geometry_visualization.tex",
        ]
        for name in required:
            assert (figures_dir / name).exists(), f"Figure {name} not found"


# ---------------------------------------------------------------------------
# Data file format (integration, requires make data)
# ---------------------------------------------------------------------------

class TestDataFormat:
    """Validate .dat file format when data/ is present."""

    @skip_if_no_data
    def test_dat_files_have_two_plus_columns(self):
        errors = []
        for filepath in DATA_DIR.rglob("*.dat"):
            try:
                data, _ = load_dat_file(filepath)
                if data.ndim == 1 or data.shape[1] < 2:
                    errors.append(f"{filepath.name}: only {data.ndim}D / {data.shape} shape")
            except Exception as exc:
                errors.append(f"{filepath.name}: parse error -- {exc}")
        assert not errors, "\n".join(errors)

    @skip_if_no_data
    def test_dat_files_have_enough_rows(self):
        errors = []
        for filepath in DATA_DIR.rglob("*.dat"):
            try:
                data, _ = load_dat_file(filepath)
                if data.ndim < 2 or data.shape[0] < 5:
                    errors.append(f"{filepath.name}: only {data.shape[0]} rows")
            except Exception as exc:
                errors.append(f"{filepath.name}: parse error -- {exc}")
        assert not errors, "\n".join(errors)

    @skip_if_no_data
    def test_dat_files_have_no_nan_or_inf(self):
        errors = []
        for filepath in DATA_DIR.rglob("*.dat"):
            try:
                data, _ = load_dat_file(filepath)
                if not np.all(np.isfinite(data)):
                    errors.append(f"{filepath.name}: contains NaN or Inf")
            except Exception as exc:
                errors.append(f"{filepath.name}: parse error -- {exc}")
        assert not errors, "\n".join(errors)


# ---------------------------------------------------------------------------
# Physics constraints on generated R_AA data
# ---------------------------------------------------------------------------

class TestRaaPhysics:
    """Validate physical constraints on R_AA data files."""

    @skip_if_no_data
    def test_raa_files_positive(self):
        """R_AA must be strictly positive."""
        for subdir in ("jet_quenching", "figure_curves"):
            for filepath in (DATA_DIR / subdir).glob("raa_*.dat"):
                data, _ = load_dat_file(filepath)
                raa = data[:, 1]
                assert np.all(raa > 0), f"{filepath.name}: R_AA has non-positive values"

    @skip_if_no_data
    def test_raa_bounded(self):
        """R_AA should not exceed 1.5 (no extreme Cronin)."""
        for subdir in ("jet_quenching", "figure_curves"):
            for filepath in (DATA_DIR / subdir).glob("raa_*.dat"):
                data, _ = load_dat_file(filepath)
                raa = data[:, 1]
                assert np.all(raa <= 1.5), f"{filepath.name}: R_AA exceeds 1.5"

    @skip_if_no_data
    def test_oo_raa_minimum_in_range(self):
        """O-O R_AA minimum should be in the range [0.5, 0.9]."""
        candidates = list((DATA_DIR / "jet_quenching").glob("*OO*.dat"))
        candidates += list((DATA_DIR / "figure_curves").glob("raa_OO_model.dat"))
        for filepath in candidates:
            data, _ = load_dat_file(filepath)
            raa = data[:, 1]
            assert 0.5 <= raa.min() <= 0.9, (
                f"{filepath.name}: R_AA min {raa.min():.3f} outside expected [0.5, 0.9]"
            )


# ---------------------------------------------------------------------------
# Physics constraints on flow data
# ---------------------------------------------------------------------------

class TestFlowPhysics:
    """Validate physical constraints on flow coefficient data."""

    @skip_if_no_data
    def test_vn_bounded(self):
        """v_n coefficients must satisfy |v_n| < 0.5."""
        for filepath in DATA_DIR.rglob("*v[2345]*.dat"):
            data, _ = load_dat_file(filepath)
            vn = data[:, 1]
            assert np.all(np.abs(vn) < 0.5), (
                f"{filepath.name}: |v_n| >= 0.5 (unphysical)"
            )

    @skip_if_no_data
    def test_vn_not_strongly_negative(self):
        """v_n should not be significantly negative."""
        for filepath in DATA_DIR.rglob("*v[2345]*.dat"):
            data, _ = load_dat_file(filepath)
            vn = data[:, 1]
            assert np.all(vn >= -0.05), (
                f"{filepath.name}: v_n < -0.05 (unphysical)"
            )


# ---------------------------------------------------------------------------
# Generator script smoke test
# ---------------------------------------------------------------------------

class TestGeneratorScript:
    """Smoke test: the data generator script can be imported."""

    def test_generator_imports(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0,'src'); import generate_comprehensive_data",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=30,
        )
        assert result.returncode == 0, (
            f"generate_comprehensive_data import failed:\n{result.stderr}"
        )
