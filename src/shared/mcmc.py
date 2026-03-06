"""Shared MCMC sampling utilities for Bayesian inference across Tiers 1 and 2."""

from __future__ import annotations

from collections.abc import Callable

import emcee
import numpy as np
from numpy.typing import NDArray


def run_emcee_sampler(
    log_prob: Callable[[NDArray[np.float64]], float],
    n_walkers: int,
    n_dim: int,
    n_steps: int,
    initial_theta: NDArray[np.float64],
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Run emcee EnsembleSampler and return flat chain with burn-in discarded.

    Parameters
    ----------
    log_prob : callable
        Log-posterior function mapping parameter vector -> log probability.
    n_walkers : int
        Number of ensemble walkers (must be even and >= 2*n_dim).
    n_dim : int
        Dimensionality of the parameter space.
    n_steps : int
        Total MCMC steps per walker; first n_steps//4 are discarded as burn-in.
    initial_theta : NDArray[float64], shape (n_dim,)
        Starting parameter vector; walkers are initialized as Gaussian ball
        around this point with sigma=1e-3, using `rng` for reproducibility.
    rng : np.random.Generator, optional
        Seeded random generator for deterministic walker initialization.
        Default: np.random.default_rng(42) (project-wide CI seed).

    Returns
    -------
    NDArray[float64], shape (effective_samples, n_dim)
        Flat chain with burn-in discarded. effective_samples = n_walkers * (n_steps * 3//4).

    Notes
    -----
    RNG Seeding Contract:
    - If rng=None, uses np.random.default_rng(42) (project-wide CI seed)
    - If rng is provided, uses it for all walker perturbations
    - All walker initialization routed through Generator for full reproducibility
    - Different seeds produce different sequences (useful for Bayesian parallelization)
    """
    _rng = rng or np.random.default_rng(42)

    # Walker perturbation uses rng so initialization is fully reproducible
    p0 = initial_theta[None, :] + _rng.normal(0, 1e-3, (n_walkers, n_dim))

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob)
    sampler.run_mcmc(p0, n_steps, progress=False)

    # Discard burn-in (first 25%) and return flattened chain
    chain = sampler.get_chain(flat=True, discard=n_steps // 4)
    return np.asarray(chain, dtype=np.float64)


def credible_interval(
    samples: NDArray[np.float64],
    level: float = 0.68,
) -> tuple[float, float]:
    """Return symmetric credible interval at given probability level.

    Parameters
    ----------
    samples : NDArray[float64], shape (n_samples,)
        Posterior samples (e.g., from MCMC chain flattened)
    level : float, optional
        Credible level in (0, 1). Default 0.68 (1-sigma equivalent).

    Returns
    -------
    tuple[float, float]
        (lower, upper) bounds of symmetric credible interval.
        For level=0.68, returns 16th and 84th percentiles.
    """
    alpha = (1.0 - level) / 2.0
    lower = float(np.quantile(samples, alpha))
    upper = float(np.quantile(samples, 1.0 - alpha))
    return lower, upper
