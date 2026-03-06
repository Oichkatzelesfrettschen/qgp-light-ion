"""
test_io_utils.py - Unit tests for src/io_utils.py.

Covers all public functions including edge cases:
- save_dat / load_dat roundtrip
- save_2d_grid blank-line format
- save_curve with output_dir and comments
- load_dat error paths
- make_provenance_header format
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from qgp.io_utils import (
    ensure_dir,
    load_dat,
    make_provenance_header,
    save_2d_grid,
    save_curve,
    save_curve_multi,
    save_curve_with_errors,
    save_dat,
    save_points_with_errors,
)

# ---------------------------------------------------------------------------
# save_dat / load_dat roundtrip
# ---------------------------------------------------------------------------

class TestSaveDat:
    """save_dat must produce files readable by load_dat."""

    def test_roundtrip(self, tmp_path):
        """Save and reload must recover the same data."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        path = str(tmp_path / "test.dat")
        save_dat(path, {"x": x, "y": y}, header="test data")
        loaded = load_dat(path)
        np.testing.assert_allclose(loaded["x"], x, atol=1e-6)
        np.testing.assert_allclose(loaded["y"], y, atol=1e-6)

    def test_provenance_header_present(self, tmp_path):
        """When provenance is given, header must contain observable."""
        path = str(tmp_path / "prov.dat")
        save_dat(
            path,
            {"pT": np.array([1.0, 2.0]), "RAA": np.array([0.5, 0.6])},
            provenance={
                "observable": "R_AA vs pT",
                "classification": "PREDICTED",
                "model_description": "BDMPS-Z energy loss",
            },
        )
        with open(path) as f:
            content = f.read()
        assert "R_AA vs pT" in content
        assert "PREDICTED" in content

    def test_column_names_in_file(self, tmp_path):
        """Column names must appear in a comment line."""
        path = str(tmp_path / "cols.dat")
        save_dat(path, {"alpha": np.array([1.0]), "beta": np.array([2.0])})
        with open(path) as f:
            content = f.read()
        assert "alpha" in content
        assert "beta" in content


# ---------------------------------------------------------------------------
# save_2d_grid
# ---------------------------------------------------------------------------

class TestSave2dGrid:
    """save_2d_grid must produce blank lines between row-blocks (pgfplots requirement)."""

    def test_blank_line_between_rows(self, tmp_path):
        """Grid file must have blank lines separating row-blocks."""
        path = str(tmp_path / "grid.dat")
        X, Y = np.meshgrid(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        Z = X + Y
        save_2d_grid(path, X, Y, Z, header="2d grid")
        with open(path) as f:
            lines = f.readlines()
        # Find blank lines (excluding header)
        blank_count = sum(1 for line in lines if line.strip() == "" and not line.startswith("#"))
        assert blank_count >= 1, "No blank lines found between row-blocks"

    def test_with_provenance(self, tmp_path):
        """Grid file with provenance should contain observable."""
        path = str(tmp_path / "grid_prov.dat")
        X, Y = np.meshgrid(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        Z = np.ones_like(X)
        save_2d_grid(
            path, X, Y, Z,
            provenance={
                "observable": "energy density",
                "classification": "PREDICTED",
                "model_description": "test model",
            },
        )
        with open(path) as f:
            content = f.read()
        assert "energy density" in content


# ---------------------------------------------------------------------------
# save_curve
# ---------------------------------------------------------------------------

class TestSaveCurve:
    """save_curve must write x-y data with optional dir and comments."""

    def test_basic_curve(self, tmp_path):
        """Basic save_curve produces a readable file."""
        path = str(tmp_path / "curve.dat")
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        save_curve(path, x, y)
        with open(path) as f:
            lines = [ln for ln in f.readlines() if not ln.startswith("#")]
        assert len(lines) == 3

    def test_with_output_dir(self, tmp_path):
        """save_curve with output_dir creates file in that directory."""
        subdir = str(tmp_path / "subdir")
        os.makedirs(subdir, exist_ok=True)
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        save_curve("data.dat", x, y, output_dir=subdir)
        assert os.path.exists(os.path.join(subdir, "data.dat"))

    def test_with_comments(self, tmp_path):
        """save_curve with comments includes them in file."""
        path = str(tmp_path / "commented.dat")
        x = np.array([1.0])
        y = np.array([2.0])
        save_curve(path, x, y, comments=["Source: test", "Model: unit test"])
        with open(path) as f:
            content = f.read()
        assert "# Source: test" in content
        assert "# Model: unit test" in content


# ---------------------------------------------------------------------------
# save_curve_multi
# ---------------------------------------------------------------------------

class TestSaveCurveMulti:
    """save_curve_multi must write multi-column data."""

    def test_multi_column_output(self, tmp_path):
        """File must have correct number of columns."""
        path = str(tmp_path / "multi.dat")
        x = np.array([1.0, 2.0, 3.0])
        cols = [np.array([10.0, 20.0, 30.0]), np.array([100.0, 200.0, 300.0])]
        save_curve_multi(path, x, cols, header="x y1 y2")
        with open(path) as f:
            data_lines = [ln for ln in f.readlines() if not ln.startswith("#")]
        for line in data_lines:
            parts = line.strip().split()
            assert len(parts) == 3


# ---------------------------------------------------------------------------
# save_curve_with_errors
# ---------------------------------------------------------------------------

class TestSaveCurveWithErrors:
    """save_curve_with_errors must write x-y-err data."""

    def test_error_columns(self, tmp_path):
        """File must have 3 columns: x, y, err."""
        path = str(tmp_path / "errors.dat")
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        err = np.array([0.1, 0.2])
        save_curve_with_errors(path, x, y, err)
        with open(path) as f:
            data_lines = [ln for ln in f.readlines() if not ln.startswith("#")]
        for line in data_lines:
            assert len(line.strip().split()) == 3


# ---------------------------------------------------------------------------
# save_points_with_errors
# ---------------------------------------------------------------------------

class TestSavePointsWithErrors:
    """save_points_with_errors must handle asymmetric errors."""

    def test_writes_correct_rows(self, tmp_path):
        """Each point tuple becomes one row."""
        path = str(tmp_path / "points.dat")
        points = [(1.0, 2.0, 0.1, 0.2), (3.0, 4.0, 0.3, 0.4)]
        save_points_with_errors(path, points, header="x y err_lo err_hi")
        with open(path) as f:
            data_lines = [ln for ln in f.readlines() if not ln.startswith("#")]
        assert len(data_lines) == 2
        assert len(data_lines[0].strip().split()) == 4


# ---------------------------------------------------------------------------
# load_dat error paths
# ---------------------------------------------------------------------------

class TestLoadDat:
    """load_dat must raise appropriate errors for bad input."""

    def test_missing_file_raises_error(self):
        with pytest.raises(FileNotFoundError):
            load_dat("/nonexistent/path/data.dat")

    def test_no_header_raises_valueerror(self, tmp_path):
        """File with no comment lines should raise ValueError."""
        path = str(tmp_path / "noheader.dat")
        with open(path, "w") as f:
            f.write("1.0 2.0\n3.0 4.0\n")
        with pytest.raises(ValueError, match="No column header"):
            load_dat(path)

    def test_no_data_raises_valueerror(self, tmp_path):
        """File with only comments should raise ValueError."""
        path = str(tmp_path / "nodata.dat")
        with open(path, "w") as f:
            f.write("# x y\n")
        with pytest.raises(ValueError, match="No data rows"):
            load_dat(path)

    def test_column_mismatch_raises_valueerror(self, tmp_path):
        """Mismatched column count between header and data should raise ValueError."""
        path = str(tmp_path / "mismatch.dat")
        with open(path, "w") as f:
            f.write("# x y z\n")
            f.write("1.0 2.0\n")
        with pytest.raises(ValueError, match="data columns"):
            load_dat(path)


# ---------------------------------------------------------------------------
# make_provenance_header
# ---------------------------------------------------------------------------

class TestMakeProvenanceHeader:
    """make_provenance_header must produce well-formatted comment blocks."""

    def test_contains_observable(self):
        header = make_provenance_header(
            observable="v2 vs centrality",
            classification="PREDICTED",
            model_description="Hydrodynamic response model",
        )
        assert "v2 vs centrality" in header

    def test_all_lines_are_comments(self):
        """Every line must start with '# '."""
        header = make_provenance_header(
            observable="test",
            classification="SCHEMATIC",
            model_description="test model",
            model_inputs={"param1": 1.0, "param2": "value"},
            references=["Author et al., Journal (2025)"],
            notes=["Note 1", "Note 2"],
        )
        for line in header.split("\n"):
            assert line.startswith("# "), f"Line not a comment: {line!r}"

    def test_model_inputs_included(self):
        header = make_provenance_header(
            observable="test",
            classification="PREDICTED",
            model_description="test",
            model_inputs={"qhat": 2.0, "L_eff": 5.0},
        )
        assert "qhat" in header
        assert "L_eff" in header

    def test_references_included(self):
        header = make_provenance_header(
            observable="test",
            classification="PREDICTED",
            model_description="test",
            references=["CMS arXiv:2510.09864"],
        )
        assert "CMS arXiv:2510.09864" in header


# ---------------------------------------------------------------------------
# ensure_dir
# ---------------------------------------------------------------------------

class TestEnsureDir:
    """ensure_dir must create directories idempotently."""

    def test_creates_directory(self, tmp_path):
        target = str(tmp_path / "new" / "nested" / "dir")
        ensure_dir(target)
        assert os.path.isdir(target)

    def test_idempotent(self, tmp_path):
        target = str(tmp_path / "existing")
        os.makedirs(target)
        ensure_dir(target)  # Should not raise
        assert os.path.isdir(target)
