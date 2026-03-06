"""Tests for shared interpolation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from shared.interpolation import log_interp1d, monotone_cubic_spline


class TestMonotoneCubicSpline:
    """Tests for monotone cubic spline interpolation."""

    def test_linear_data_exact(self) -> None:
        """Cubic spline passes exactly through linear data points."""
        x = np.linspace(0, 10, 5)
        y = 2 * x + 3
        cs = monotone_cubic_spline(x, y)
        # Evaluate at interior point
        assert np.isclose(cs(5.0), 13.0, rtol=1e-10)

    def test_raises_on_unsorted_x(self) -> None:
        """Raises ValueError if x is not strictly increasing."""
        x = np.array([1.0, 3.0, 2.0])
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="strictly increasing"):
            monotone_cubic_spline(x, y)

    def test_raises_on_duplicate_x(self) -> None:
        """Raises ValueError if x has duplicates."""
        x = np.array([1.0, 2.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="strictly increasing"):
            monotone_cubic_spline(x, y)

    def test_smooth_interpolation(self) -> None:
        """Spline provides smooth interpolation between points."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 4.0, 9.0])  # y = x^2
        cs = monotone_cubic_spline(x, y)
        # Evaluate at midpoint between x[0] and x[1]
        y_mid = cs(0.5)
        # Should be closer to 0.25 (exact value for y=x^2) than linear interpolation 0.5
        assert 0.2 < y_mid < 0.4


class TestLogInterp1d:
    """Tests for log-log space interpolation (power-law curves)."""

    def test_power_law_exact(self) -> None:
        """Log-log interpolation is accurate for power-law data."""
        x = np.array([1.0, 2.0, 4.0, 8.0])
        y = x**2  # y = x^2 is power law with exponent 2
        x_new = np.array([3.0])
        result = log_interp1d(x, y, x_new)
        expected = 9.0  # 3^2
        # Log-log interpolation is exact for power laws
        assert np.isclose(result[0], expected, rtol=0.02)

    def test_vector_output_shape(self) -> None:
        """Output shape matches input x_new shape."""
        x = np.array([1.0, 2.0, 4.0])
        y = np.array([1.0, 4.0, 16.0])
        x_new = np.array([1.5, 2.5, 3.5])
        result = log_interp1d(x, y, x_new)
        assert result.shape == x_new.shape

    def test_single_point_interpolation(self) -> None:
        """Single interpolation point works correctly."""
        x = np.array([1.0, 10.0, 100.0])
        y = np.array([1.0, 100.0, 10000.0])  # y = x^2
        x_new = np.array([10.0])
        result = log_interp1d(x, y, x_new)
        # At grid point x=10, should recover y=100 exactly
        assert np.isclose(result[0], 100.0, rtol=1e-10)
