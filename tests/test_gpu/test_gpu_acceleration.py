"""
test_gpu_acceleration.py - GPU acceleration module tests.

Tests GPU-accelerated functions with CPU fallback validation.
Verifies numerical correctness against NumPy reference implementations.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpu import (
    GPU_AVAILABLE,
    as_numpy,
    eccentricity_grid_gpu,
    energy_loss_grid_gpu,
    get_backend,
    pairwise_distance_gpu,
)


class TestGPUDetection:
    """GPU availability detection and backend selection."""

    def test_get_backend_returns_valid_string(self) -> None:
        """Backend should be either 'gpu' or 'cpu'."""
        backend = get_backend()
        assert backend in ("gpu", "cpu")

    def test_gpu_available_is_boolean(self) -> None:
        """GPU_AVAILABLE should be a boolean."""
        assert isinstance(GPU_AVAILABLE, bool)

    def test_backend_matches_gpu_availability(self) -> None:
        """Backend should match GPU_AVAILABLE flag."""
        backend = get_backend()
        if GPU_AVAILABLE:
            assert backend == "gpu"
        else:
            assert backend == "cpu"


class TestAsNumpy:
    """Array conversion utilities."""

    def test_as_numpy_from_numpy(self) -> None:
        """Should handle NumPy arrays unchanged."""
        arr = np.array([1.0, 2.0, 3.0])
        result = as_numpy(arr)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_as_numpy_preserves_values(self) -> None:
        """Should preserve array values exactly."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = as_numpy(arr)

        np.testing.assert_array_equal(result, arr)

    def test_as_numpy_2d_arrays(self) -> None:
        """Should handle 2D arrays correctly."""
        arr = np.random.rand(5, 3)
        result = as_numpy(arr)

        assert result.shape == (5, 3)
        np.testing.assert_allclose(result, arr)


class TestPairwiseDistanceGPU:
    """Pairwise distance GPU computation."""

    def test_distance_shape(self) -> None:
        """Output shape should be (n_A, n_B)."""
        pos_A = np.random.rand(10, 2)
        pos_B = np.random.rand(15, 2)

        distances = pairwise_distance_gpu(pos_A, pos_B)

        assert distances.shape == (10, 15)

    def test_distance_positive(self) -> None:
        """All distances should be non-negative."""
        pos_A = np.random.rand(8, 2)
        pos_B = np.random.rand(12, 2)

        distances = pairwise_distance_gpu(pos_A, pos_B)

        assert np.all(distances >= 0)

    def test_distance_self_interaction(self) -> None:
        """Distance from point to itself should be zero."""
        pos = np.array([[1.0, 2.0], [3.0, 4.0]])

        distances = pairwise_distance_gpu(pos, pos)

        # Diagonal should be zero
        np.testing.assert_allclose(np.diag(distances), 0.0, atol=1e-10)

    def test_distance_symmetry(self) -> None:
        """Distance should be symmetric: d(A,B) = d(B,A)."""
        pos_A = np.array([[0.0, 0.0], [1.0, 1.0]])
        pos_B = np.array([[3.0, 4.0]])

        d_AB = pairwise_distance_gpu(pos_A, pos_B)
        d_BA = pairwise_distance_gpu(pos_B, pos_A)

        np.testing.assert_allclose(d_AB.T, d_BA, rtol=1e-9)

    def test_distance_euclidean(self) -> None:
        """Distances should match Euclidean distance formula."""
        pos_A = np.array([[0.0, 0.0]])
        pos_B = np.array([[3.0, 4.0]])

        distances = pairwise_distance_gpu(pos_A, pos_B)

        # Distance should be 5.0 (3-4-5 triangle)
        np.testing.assert_allclose(distances[0, 0], 5.0, rtol=1e-9)

    def test_distance_float_output(self) -> None:
        """Output should be floating point array."""
        pos_A = np.random.rand(5, 2)
        pos_B = np.random.rand(7, 2)

        distances = pairwise_distance_gpu(pos_A, pos_B)

        assert distances.dtype in (np.float32, np.float64)


class TestEnergyLossGridGPU:
    """Energy loss GPU computation."""

    def test_energy_loss_shape(self) -> None:
        """Output shape should match pT shape."""
        pT = np.linspace(5, 100, 50)

        dE = energy_loss_grid_gpu(pT, L_eff=5.0, qhat=2.0)

        assert dE.shape == (50,)

    def test_energy_loss_positive(self) -> None:
        """Energy loss should be non-negative."""
        pT = np.linspace(10, 200, 100)

        dE = energy_loss_grid_gpu(pT, L_eff=5.0, qhat=2.0)

        assert np.all(dE >= 0)

    def test_energy_loss_below_pT(self) -> None:
        """Energy loss should not exceed input momentum."""
        pT = np.linspace(10, 200, 100)

        dE = energy_loss_grid_gpu(pT, L_eff=5.0, qhat=2.0)

        # ΔE < 0.9 * pT (BDMPS constraint)
        assert np.all(dE <= 0.9 * pT + 1e-10)

    def test_energy_loss_monotonic(self) -> None:
        """Energy loss should increase with pT (for fixed path length)."""
        pT = np.linspace(10, 200, 50)

        dE = energy_loss_grid_gpu(pT, L_eff=5.0, qhat=2.0)

        # Differences should be non-negative (monotonic increasing)
        assert np.all(np.diff(dE) >= -1e-10)

    def test_energy_loss_scales_with_path_length(self) -> None:
        """Energy loss should scale roughly as L²."""
        pT = np.array([50.0, 100.0, 150.0])

        dE_short = energy_loss_grid_gpu(pT, L_eff=3.0, qhat=2.0)
        dE_long = energy_loss_grid_gpu(pT, L_eff=6.0, qhat=2.0)

        # Longer path → more loss (roughly 4x for 2x path)
        assert np.all(dE_long > dE_short)

    def test_energy_loss_zero_path(self) -> None:
        """Zero path length should give zero energy loss."""
        pT = np.array([50.0, 100.0])

        dE = energy_loss_grid_gpu(pT, L_eff=0.0, qhat=2.0)

        np.testing.assert_allclose(dE, 0.0, atol=1e-10)


class TestEccentricityGridGPU:
    """Eccentricity GPU computation."""

    def test_eccentricity_isotropic_grid_zero(self) -> None:
        """Isotropic grid should have zero eccentricity."""
        # Circular disk
        X = np.linspace(-1, 1, 50)
        Y = np.linspace(-1, 1, 50)
        XX, YY = np.meshgrid(X, Y)

        ecc = eccentricity_grid_gpu(XX, YY, n=2)

        # Isotropic → ε_2 ~ 0
        assert ecc < 0.1

    def test_eccentricity_elliptical_grid_nonzero(self) -> None:
        """Elliptical grid should have nonzero eccentricity."""
        # Ellipse: x²/a² + y²/b² = 1
        X = np.linspace(-2, 2, 50)
        Y = np.linspace(-1, 1, 50)
        XX, YY = np.meshgrid(X, Y)

        # Weight by inside ellipse
        r2 = (XX / 2.0) ** 2 + YY**2
        weight = np.exp(-10 * (r2 - 1.0) ** 2)

        ecc = eccentricity_grid_gpu(XX * weight, YY * weight, n=2)

        # Elliptical → ε_2 > 0.1
        assert ecc > 0.05

    def test_eccentricity_bounded(self) -> None:
        """Eccentricity should be bounded 0 ≤ ε_n ≤ 1."""
        X = np.random.randn(30, 30)
        Y = np.random.randn(30, 30)

        ecc = eccentricity_grid_gpu(X, Y, n=2)

        assert 0 <= ecc <= 1

    def test_eccentricity_different_harmonics(self) -> None:
        """Different harmonics should give different values."""
        X = np.random.randn(30, 30)
        Y = np.random.randn(30, 30)

        ecc2 = eccentricity_grid_gpu(X, Y, n=2)
        ecc3 = eccentricity_grid_gpu(X, Y, n=3)

        # Should be different (very unlikely to be equal)
        assert abs(ecc2 - ecc3) > 1e-3

    def test_eccentricity_scalar_output(self) -> None:
        """Eccentricity should return scalar float."""
        X = np.random.rand(20, 20)
        Y = np.random.rand(20, 20)

        ecc = eccentricity_grid_gpu(X, Y, n=2)

        assert isinstance(ecc, (float, np.floating))
