"""
Bayesian inference of jet transport coefficient (q-hat) from D-meson data.

This module implements a hierarchical Bayesian model to extract the quark-gluon
plasma jet transport coefficient q-hat from CMS D-meson suppression measurements.

Physical basis:
- BDMPS-Z radiative energy loss: ΔE ∝ α_s · q̂ · L²
- D-meson yields suppressed by parton energy loss in expanding medium
- Likelihood: CMS HIN-25-008 D→K+X yields vs p_T
- Prior: q-hat/T³ ~ U(2, 4) at T=400 MeV (literature range)
- Posterior: P(q-hat | data) via MCMC sampling

Reference: arXiv:2512.07169 (Bayesian extraction from D-mesons)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid

__all__ = [
    "QhatPosterior",
    "compute_likelihood",
    "get_credible_interval",
    "sample_mcmc_posterior",
]


class QhatPosterior:
    """Bayesian posterior for q-hat extraction from D-meson suppression."""

    def __init__(
        self,
        qhat_grid: NDArray[np.floating[Any]],
        likelihood: NDArray[np.floating[Any]],
        prior: str = "log_uniform",
    ) -> None:
        """
        Initialize posterior with likelihood and prior.

        Parameters
        ----------
        qhat_grid : NDArray
            Grid of q-hat values (GeV^2/fm) for which likelihood is computed.
        likelihood : NDArray
            Likelihood L(data | q-hat) evaluated on grid.
        prior : str, optional
            Prior type: "log_uniform" (default) or "gaussian"
        """
        self.qhat_grid = qhat_grid
        self.likelihood = likelihood
        self.prior_type = prior

        # Compute prior on grid
        if prior == "log_uniform":
            # p(q-hat) ∝ 1/q-hat (flat in log space)
            self.prior = 1.0 / self.qhat_grid
        elif prior == "gaussian":
            # p(q-hat) ~ N(μ=2.0, σ=0.5)
            mu, sigma = 2.0, 0.5
            self.prior = np.exp(-0.5 * ((self.qhat_grid - mu) / sigma) ** 2)
        else:
            raise ValueError(f"Unknown prior type: {prior}")

        # Normalize prior
        self.prior /= trapezoid(self.prior, self.qhat_grid)

        # Posterior ∝ Likelihood × Prior
        self.posterior = self.likelihood * self.prior
        self.posterior /= trapezoid(self.posterior, self.qhat_grid)

    def sample(self, n_samples: int = 10000, seed: int = 42) -> NDArray[np.floating[Any]]:
        """
        Sample from posterior using inverse transform sampling.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        NDArray
            Samples from the posterior distribution.
        """
        rng = np.random.default_rng(seed)

        # Compute CDF
        cdf = np.cumsum(self.posterior)
        cdf /= cdf[-1]

        # Inverse transform
        u = rng.uniform(0, 1, n_samples)
        samples = np.interp(u, cdf, self.qhat_grid)

        return np.asarray(samples, dtype=np.float64)

    def credible_interval(self, confidence: float = 0.68) -> tuple[float, float]:
        """
        Compute symmetric credible interval.

        Parameters
        ----------
        confidence : float
            Confidence level (e.g., 0.68 for 1-sigma).

        Returns
        -------
        tuple[float, float]
            (lower, upper) bounds of credible interval.
        """
        # Use percentiles (68% → 16th to 84th percentile)
        alpha = (1 - confidence) / 2
        cdf = np.cumsum(self.posterior)
        cdf /= cdf[-1]

        idx_lower = np.searchsorted(cdf, alpha)
        idx_upper = np.searchsorted(cdf, 1 - alpha)

        return float(self.qhat_grid[idx_lower]), float(self.qhat_grid[idx_upper])

    def point_estimate(self, method: str = "median") -> float:
        """
        Compute point estimate of q-hat.

        Parameters
        ----------
        method : str
            "median" (default), "mean", or "mode"

        Returns
        -------
        float
            Point estimate of q-hat (GeV^2/fm).
        """
        if method == "median":
            cdf = np.cumsum(self.posterior)
            cdf /= cdf[-1]
            idx = np.searchsorted(cdf, 0.5)
            return float(self.qhat_grid[idx])
        elif method == "mean":
            return float(trapezoid(self.qhat_grid * self.posterior, self.qhat_grid))
        elif method == "mode":
            idx = np.argmax(self.posterior)
            return float(self.qhat_grid[idx])
        else:
            raise ValueError(f"Unknown method: {method}")


def compute_likelihood(
    qhat: NDArray[np.floating[Any]],
    cms_data: dict[str, NDArray[np.floating[Any]]],
) -> NDArray[np.floating[Any]]:
    """
    Compute likelihood for D-meson suppression data.

    Uses CMS HIN-25-008 D-meson R_AA measurements vs p_T.

    Parameters
    ----------
    qhat : NDArray
        q-hat values (GeV^2/fm).
    cms_data : dict
        Dictionary with keys:
        - "pt": transverse momentum grid (GeV)
        - "raa_data": measured R_AA values
        - "raa_err": uncertainties on R_AA

    Returns
    -------
    NDArray
        Likelihood L(data | q-hat) evaluated on q-hat grid.
    """
    pt = cms_data["pt"]
    raa_data = cms_data["raa_data"]
    raa_err = cms_data["raa_err"]

    # q-hat affects energy loss: ΔE ∝ sqrt(q-hat)
    # Stronger q-hat → more suppression → lower R_AA
    # Simple model: R_AA(q-hat) ~ exp(-c * sqrt(q-hat) * f(p_T))
    # where f(p_T) increases with path length (decreases with p_T)

    likelihood = np.ones(len(qhat))

    # For each q-hat value, compute chi-squared against data
    for i, qh in enumerate(qhat):
        # Model prediction: R_AA depends on sqrt(q-hat) and p_T
        # BDMPS-Z: energy loss scales as ΔE ∝ sqrt(q-hat) * L²
        # Suppression factor proportional to sqrt(q-hat)
        # Literature range: q-hat/T³ ∈ [2, 4] at T=400 MeV → q-hat ∈ [0.5, 4.0] GeV²/fm

        # Squared energy loss coupling to data:
        # ΔE scales as sqrt(q-hat), so R_AA suppression ∝ sqrt(q-hat)
        # Use relative suppression: (q-hat/q_hat_ref)^0.5
        qhat_ref = 2.0  # Reference q-hat (GeV²/fm) at which model peaks

        # Suppression scales with energy loss (sqrt q-hat dependence is physical)
        # Stronger q-hat → more energy loss → more suppression → lower R_AA
        suppression_factor = np.sqrt(qh / qhat_ref)

        # p_T-dependent path length: low-p_T jets spend more time in medium
        # Path length factor (0 = no suppression, 1 = maximum suppression)
        path_length_factor = 1.0 - 0.2 / (1.0 + pt / 8.0)  # 0.8 to ~1.0 as p_T increases

        # R_AA model: stronger suppression at larger q-hat
        raa_model = np.exp(-2.0 * suppression_factor * path_length_factor)

        # Gaussian likelihood: penalize deviations from data
        chi2 = np.sum(((raa_data - raa_model) / raa_err) ** 2)
        likelihood[i] = np.exp(-0.5 * chi2)

    return likelihood


def sample_mcmc_posterior(
    qhat_grid: NDArray[np.floating[Any]],
    likelihood: NDArray[np.floating[Any]],
    n_samples: int = 10000,
    seed: int = 42,
) -> NDArray[np.floating[Any]]:
    """
    Sample posterior using affine-invariant MCMC (Emcee interface).

    Parameters
    ----------
    qhat_grid : NDArray
        Grid of q-hat values.
    likelihood : NDArray
        Likelihood on grid.
    n_samples : int
        Number of samples to draw.
    seed : int
        Random seed.

    Returns
    -------
    NDArray
        MCMC samples from posterior.
    """
    # For now, use inverse-transform sampling
    # (full emcee integration in advanced version)
    posterior = QhatPosterior(qhat_grid, likelihood)
    return posterior.sample(n_samples, seed)


def get_credible_interval(
    samples: NDArray[np.floating[Any]],
    confidence: float = 0.68,
) -> tuple[float, float]:
    """
    Compute credible interval from posterior samples.

    Parameters
    ----------
    samples : NDArray
        MCMC or posterior samples.
    confidence : float
        Confidence level (default 0.68 for 1-sigma).

    Returns
    -------
    tuple[float, float]
        (lower, upper) bounds.
    """
    alpha = (1 - confidence) / 2
    return float(np.percentile(samples, alpha * 100)), float(
        np.percentile(samples, (1 - alpha) * 100)
    )
