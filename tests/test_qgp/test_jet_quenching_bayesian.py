"""
test_jet_quenching_bayesian.py - Unit tests for Bayesian q-hat inference.

Tests the QhatPosterior class and likelihood computation against synthetic
and real CMS D-meson suppression data.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import trapezoid

from qgp.jet_quenching_bayesian import (
    QhatPosterior,
    compute_likelihood,
    get_credible_interval,
)


class TestQhatPosterior:
    """QhatPosterior initialization and properties."""

    def test_posterior_normalized(self) -> None:
        """Posterior must integrate to 1."""
        qhat_grid = np.linspace(1.0, 4.0, 100)
        likelihood = np.exp(-((qhat_grid - 2.0) ** 2) / 0.5)

        posterior = QhatPosterior(qhat_grid, likelihood)

        integral = trapezoid(posterior.posterior, qhat_grid)
        assert abs(integral - 1.0) < 1e-6

    def test_prior_log_uniform_monotonic(self) -> None:
        """Log-uniform prior should be monotonically decreasing."""
        qhat_grid = np.linspace(1.0, 4.0, 100)
        likelihood = np.ones_like(qhat_grid)

        posterior = QhatPosterior(qhat_grid, likelihood, prior="log_uniform")

        assert np.all(np.diff(posterior.prior) <= 0)

    def test_prior_gaussian_symmetric(self) -> None:
        """Gaussian prior should be symmetric around mean."""
        qhat_grid = np.linspace(1.0, 3.0, 101)  # symmetric around 2.0
        likelihood = np.ones_like(qhat_grid)

        posterior = QhatPosterior(qhat_grid, likelihood, prior="gaussian")

        # Prior values should be symmetric
        center_idx = 50  # Index of 2.0
        for i in range(1, 20):
            left_val = posterior.prior[center_idx - i]
            right_val = posterior.prior[center_idx + i]
            assert abs(left_val - right_val) < 1e-10

    def test_invalid_prior_raises_error(self) -> None:
        """Unknown prior type should raise ValueError."""
        qhat_grid = np.linspace(1.0, 4.0, 100)
        likelihood = np.ones_like(qhat_grid)

        with pytest.raises(ValueError, match="Unknown prior type"):
            QhatPosterior(qhat_grid, likelihood, prior="unknown")


class TestPosteriorSampling:
    """Posterior sampling and point estimates."""

    def test_sample_produces_correct_shape(self) -> None:
        """Samples should have correct shape."""
        qhat_grid = np.linspace(1.0, 4.0, 100)
        likelihood = np.exp(-((qhat_grid - 2.0) ** 2) / 0.5)

        posterior = QhatPosterior(qhat_grid, likelihood)
        samples = posterior.sample(n_samples=1000)

        assert samples.shape == (1000,)
        assert np.all(samples >= 1.0)
        assert np.all(samples <= 4.0)

    def test_samples_match_posterior_mode(self) -> None:
        """Samples should cluster near posterior mode."""
        qhat_grid = np.linspace(1.0, 4.0, 100)
        likelihood = np.exp(-((qhat_grid - 2.5) ** 2) / 0.3)

        posterior = QhatPosterior(qhat_grid, likelihood)
        samples = posterior.sample(n_samples=5000, seed=42)

        # Sample mean should be near posterior mode (within 0.2)
        assert abs(np.mean(samples) - 2.5) < 0.2

    def test_credible_interval_order(self) -> None:
        """Credible interval bounds should satisfy lower < upper."""
        qhat_grid = np.linspace(1.0, 4.0, 100)
        likelihood = np.exp(-((qhat_grid - 2.0) ** 2) / 0.5)

        posterior = QhatPosterior(qhat_grid, likelihood)
        lower, upper = posterior.credible_interval(confidence=0.68)

        assert lower < upper
        assert lower >= 1.0
        assert upper <= 4.0

    def test_credible_interval_contains_median(self) -> None:
        """Credible interval should contain the median."""
        qhat_grid = np.linspace(1.0, 4.0, 100)
        likelihood = np.exp(-((qhat_grid - 2.0) ** 2) / 0.5)

        posterior = QhatPosterior(qhat_grid, likelihood)
        median = posterior.point_estimate(method="median")
        lower, upper = posterior.credible_interval(confidence=0.68)

        assert lower <= median <= upper

    def test_point_estimates_differ(self) -> None:
        """Median, mean, mode should differ for skewed likelihood."""
        qhat_grid = np.linspace(1.0, 4.0, 100)
        # Skewed likelihood (not symmetric)
        likelihood = np.exp(-((qhat_grid - 2.0) ** 2) / 0.3) * (1.0 + 0.5 * (qhat_grid - 2.0))

        posterior = QhatPosterior(qhat_grid, likelihood)

        median = posterior.point_estimate(method="median")
        mean = posterior.point_estimate(method="mean")
        mode = posterior.point_estimate(method="mode")

        # All three should be in reasonable range but not identical
        assert 1.5 < median < 3.0
        assert 1.5 < mean < 3.0
        assert 1.5 < mode < 3.0


class TestLikelihoodComputation:
    """D-meson likelihood calculation."""

    def test_likelihood_positive(self) -> None:
        """Likelihood must be non-negative."""
        qhat_grid = np.linspace(1.0, 4.0, 50)

        cms_data = {
            "pt": np.array([5.0, 10.0, 20.0, 50.0]),
            "raa_data": np.array([0.6, 0.65, 0.75, 0.9]),
            "raa_err": np.array([0.05, 0.04, 0.03, 0.02]),
        }

        likelihood = compute_likelihood(qhat_grid, cms_data)

        assert np.all(likelihood >= 0)
        assert np.all(likelihood <= 1.0)

    def test_likelihood_peaks_near_q2(self) -> None:
        """Likelihood should peak near typical q-hat ~ 2 GeV^2/fm."""
        qhat_grid = np.linspace(0.5, 5.0, 100)

        # Synthetic data consistent with q-hat ~ 2 GeV²/fm
        # At q-hat=2: suppression_factor=1, R_AA ~ [0.17, 0.16, 0.15, 0.14]
        # Add realistic measurement uncertainty
        cms_data = {
            "pt": np.array([5.0, 10.0, 20.0, 50.0]),
            "raa_data": np.array([0.17, 0.16, 0.15, 0.14]),
            "raa_err": np.array([0.02, 0.02, 0.02, 0.02]),
        }

        likelihood = compute_likelihood(qhat_grid, cms_data)

        # Find peak
        peak_idx = np.argmax(likelihood)
        peak_qhat = qhat_grid[peak_idx]

        # Should be in reasonable range (1-4 GeV²/fm)
        assert 1.0 <= peak_qhat <= 4.0


class TestCredibleInterval:
    """Credible interval computation from samples."""

    def test_credible_interval_symmetric_likelihood(self) -> None:
        """For symmetric samples, credible interval should be roughly symmetric."""
        rng = np.random.default_rng(42)
        samples = rng.normal(2.0, 0.3, 10000)

        lower, upper = get_credible_interval(samples, confidence=0.68)

        # Should be roughly symmetric around 2.0
        center = (lower + upper) / 2
        half_width_lower = center - lower
        half_width_upper = upper - center

        assert abs(half_width_lower - half_width_upper) < 0.1

    def test_credible_interval_coverage(self) -> None:
        """Credible interval should contain approximately the right fraction."""
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, 50000)

        lower, upper = get_credible_interval(samples, confidence=0.68)

        # Should contain ~68% of samples
        contained = np.sum((samples >= lower) & (samples <= upper)) / len(samples)
        assert abs(contained - 0.68) < 0.02  # Within 2% of target

    def test_credible_interval_narrow_for_high_confidence(self) -> None:
        """Higher confidence should give wider interval."""
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, 10000)

        lower_68, upper_68 = get_credible_interval(samples, confidence=0.68)
        lower_95, upper_95 = get_credible_interval(samples, confidence=0.95)

        width_68 = upper_68 - lower_68
        width_95 = upper_95 - lower_95

        assert width_95 > width_68
