"""Tests for shared MCMC utilities."""

from __future__ import annotations

import numpy as np

from shared.mcmc import credible_interval, run_emcee_sampler


class TestRunEmceeSampler:
    """Tests for MCMC sampling with emcee."""

    def test_output_shape(self) -> None:
        """Chain shape matches expected (effective_samples, n_dim)."""
        log_prob = lambda theta: -0.5 * np.sum(theta**2)  # Standard normal
        chain = run_emcee_sampler(
            log_prob,
            n_walkers=8,
            n_dim=2,
            n_steps=100,
            initial_theta=np.zeros(2),
        )
        assert chain.ndim == 2
        assert chain.shape[1] == 2
        # effective_samples = n_walkers * (n_steps * 3 // 4)
        assert chain.shape[0] == 8 * (100 * 3 // 4)

    def test_reproducible_with_seed(self) -> None:
        """Same seed produces identical chains."""
        log_prob = lambda theta: -0.5 * np.sum(theta**2)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        chain1 = run_emcee_sampler(
            log_prob,
            n_walkers=4,
            n_dim=1,
            n_steps=50,
            initial_theta=np.zeros(1),
            rng=rng1,
        )
        chain2 = run_emcee_sampler(
            log_prob,
            n_walkers=4,
            n_dim=1,
            n_steps=50,
            initial_theta=np.zeros(1),
            rng=rng2,
        )
        assert np.allclose(chain1, chain2)

    def test_default_rng_seed(self) -> None:
        """Default RNG seed is 42."""
        log_prob = lambda theta: -0.5 * np.sum(theta**2)

        # Two calls with default rng=None should be identical
        chain1 = run_emcee_sampler(
            log_prob,
            n_walkers=4,
            n_dim=1,
            n_steps=50,
            initial_theta=np.zeros(1),
        )
        chain2 = run_emcee_sampler(
            log_prob,
            n_walkers=4,
            n_dim=1,
            n_steps=50,
            initial_theta=np.zeros(1),
        )
        assert np.allclose(chain1, chain2)


class TestCredibleInterval:
    """Tests for credible interval computation."""

    def test_credible_interval_gaussian(self) -> None:
        """68% credible interval of standard normal is approximately [-1, 1]."""
        rng = np.random.default_rng(0)
        samples = rng.normal(0, 1, 50000)
        lo, hi = credible_interval(samples, level=0.68)
        # 68% CI of N(0,1) is roughly [-1, 1]
        assert abs(lo - (-1.0)) < 0.05
        assert abs(hi - 1.0) < 0.05

    def test_credible_interval_width_increases_with_level(self) -> None:
        """Wider confidence level produces wider interval."""
        rng = np.random.default_rng(1)
        samples = rng.normal(0, 1, 10000)
        lo68, hi68 = credible_interval(samples, level=0.68)
        lo95, hi95 = credible_interval(samples, level=0.95)
        assert (hi68 - lo68) < (hi95 - lo95)

    def test_credible_interval_symmetric_around_median(self) -> None:
        """Interval is symmetric for symmetric distribution."""
        rng = np.random.default_rng(2)
        samples = rng.normal(5.0, 2.0, 10000)  # N(5, 2)
        lo, hi = credible_interval(samples, level=0.68)
        # Median of N(5, 2) is 5
        median = float(np.percentile(samples, 50))
        assert np.isclose(lo + hi, 2 * median, rtol=0.01)
