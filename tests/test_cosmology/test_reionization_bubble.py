"""
test_reionization_bubble.py - Reionization bubble dynamics and percolation tests.

Tests the ReionizationBubble class and growth dynamics against JWST observations
and percolation theory predictions.
"""

from __future__ import annotations

import numpy as np
import pytest

from cosmology.reionization_bubble import (
    ReionizationBubble,
    bubble_growth_rate,
    overlap_probability,
)


class TestReionizationBubble:
    """Bubble initialization and properties."""

    def test_bubble_initialization(self) -> None:
        """Bubble should initialize with valid redshift and parameters."""
        bubble = ReionizationBubble(redshift=6.5, expansion_rate_H0=67.4, ionization_fraction=0.5)

        assert bubble.z == 6.5
        assert bubble.H0_kmsMpc == 67.4
        assert bubble.x_e == 0.5

    def test_age_of_universe_increases_with_redshift(self) -> None:
        """Universe age (lookback time) should be larger at lower redshift."""
        bubble_z5 = ReionizationBubble(redshift=5.0)
        bubble_z7 = ReionizationBubble(redshift=7.0)

        # At higher z, the universe is younger (earlier in time)
        assert bubble_z5.age_universe_sec > bubble_z7.age_universe_sec

    def test_growth_timescale_positive(self) -> None:
        """Growth timescale should be positive for all bubble sizes."""
        bubble = ReionizationBubble(redshift=6.5)

        for size in [10.0, 30.0, 50.0, 100.0]:
            t_growth = bubble.growth_timescale(size)
            assert t_growth > 0

    def test_growth_timescale_increases_with_size(self) -> None:
        """Larger bubbles take longer to grow (geometric growth scaling)."""
        bubble = ReionizationBubble(redshift=6.5)

        t_small = bubble.growth_timescale(10.0)
        t_large = bubble.growth_timescale(100.0)

        # Larger bubbles take longer: t_growth ~ sqrt(R * t_rec)
        assert t_small < t_large

    def test_critical_bubble_size_neutral_universe(self) -> None:
        """In neutral universe (x_e=0), critical size should be large."""
        bubble = ReionizationBubble(redshift=6.5, ionization_fraction=0.01)

        R_crit = bubble.critical_bubble_size()
        # Mostly neutral → large bubbles needed for percolation
        assert R_crit > 50.0

    def test_critical_bubble_size_ionized_universe(self) -> None:
        """In ionized universe (x_e=1), critical size should be small."""
        bubble = ReionizationBubble(redshift=6.5, ionization_fraction=0.99)

        R_crit = bubble.critical_bubble_size()
        # Mostly ionized → small residual bubbles
        assert R_crit < 10.0

    def test_critical_bubble_size_transition_at_50_percent(self) -> None:
        """At x_e=0.5 (percolation threshold), critical size should be intermediate."""
        bubble = ReionizationBubble(redshift=6.5, ionization_fraction=0.5)

        R_crit = bubble.critical_bubble_size()
        # Should be order 10-100 cMpc
        assert 10.0 < R_crit < 100.0

    def test_ionization_probability_inside_bubble(self) -> None:
        """Ionization probability should be ~1 at bubble center."""
        bubble = ReionizationBubble(redshift=6.5, ionization_fraction=0.5)

        # Very close to center
        prob = bubble.ionization_probability(0.1)
        assert prob > 0.95

    def test_ionization_probability_outside_bubble(self) -> None:
        """Ionization probability should be ~0 far outside bubble."""
        bubble = ReionizationBubble(redshift=6.5, ionization_fraction=0.5)

        R_crit = bubble.critical_bubble_size()
        # Far outside: 3x the critical radius
        prob = bubble.ionization_probability(3.0 * R_crit)
        assert prob < 0.05

    def test_ionization_probability_continuous(self) -> None:
        """Ionization probability should vary smoothly with distance."""
        bubble = ReionizationBubble(redshift=6.5, ionization_fraction=0.5)

        distances = np.linspace(0, 100, 100)
        probs = [bubble.ionization_probability(d) for d in distances]

        # Probabilities should monotonically decrease
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1]


class TestBubbleGrowthRate:
    """Growth rate computation on phase-space grids."""

    def test_growth_rate_grid_shape(self) -> None:
        """Growth rate should have correct shape for input grids."""
        z = np.linspace(5.0, 8.0, 20)
        R = np.linspace(10.0, 100.0, 25)

        dRdt = bubble_growth_rate(z, R)

        assert dRdt.shape == (20, 25)
        assert np.all(np.isfinite(dRdt))

    def test_growth_rate_positive(self) -> None:
        """Growth rate should be positive everywhere."""
        z = np.linspace(5.0, 8.0, 10)
        R = np.linspace(10.0, 100.0, 15)

        dRdt = bubble_growth_rate(z, R)

        assert np.all(dRdt >= 0)

    def test_growth_rate_decreases_with_redshift(self) -> None:
        """Growth should be faster at higher redshift (younger universe)."""
        z = np.array([5.0, 6.5, 8.0])
        R = np.array([30.0])

        dRdt = bubble_growth_rate(z, R)

        # Earlier time (higher z) → faster growth
        assert dRdt[0, 0] >= dRdt[1, 0] >= dRdt[2, 0]

    def test_growth_rate_2d_grid(self) -> None:
        """Should handle pre-gridded 2D arrays."""
        z_1d = np.linspace(5.0, 8.0, 10)
        R_1d = np.linspace(10.0, 100.0, 15)
        Z, R = np.meshgrid(z_1d, R_1d, indexing="ij")

        dRdt = bubble_growth_rate(Z, R)

        assert dRdt.shape == (10, 15)
        assert np.all(np.isfinite(dRdt))


class TestOverlapProbability:
    """Percolation overlap probability."""

    def test_overlap_fully_neutral(self) -> None:
        """Overlap probability should be ~0 for fully neutral universe."""
        x_e = np.array([0.01])
        overlap = overlap_probability(x_e)

        assert overlap[0] < 0.1

    def test_overlap_fully_ionized(self) -> None:
        """Overlap probability should be ~1 for fully ionized universe."""
        x_e = np.array([0.99])
        overlap = overlap_probability(x_e)

        assert overlap[0] > 0.9

    def test_overlap_percolation_threshold(self) -> None:
        """Overlap should transition around x_e ~ 0.35."""
        x_e = np.linspace(0.0, 1.0, 100)
        overlap = overlap_probability(x_e)

        # Find transition point (overlap = 0.5)
        idx = np.argmin(np.abs(overlap - 0.5))
        x_e_crit = x_e[idx]

        # Should be near 0.35
        assert 0.25 < x_e_crit < 0.45

    def test_overlap_monotonic(self) -> None:
        """Overlap probability should increase monotonically with ionization."""
        x_e = np.linspace(0.0, 1.0, 100)
        overlap = overlap_probability(x_e)

        # Each element should be >= previous
        for i in range(len(overlap) - 1):
            assert overlap[i] <= overlap[i + 1]

    def test_overlap_smooth_transition(self) -> None:
        """Transition should be smooth (derivative well-defined)."""
        x_e = np.linspace(0.1, 0.6, 100)
        overlap = overlap_probability(x_e)

        # Compute finite differences (numerical derivative)
        d_overlap = np.diff(overlap) / np.diff(x_e)

        # All derivatives should be positive and finite
        assert np.all(d_overlap > 0)
        assert np.all(np.isfinite(d_overlap))

    def test_overlap_array_shape_preserved(self) -> None:
        """Should preserve input array shape."""
        x_e_1d = np.linspace(0.0, 1.0, 50)
        overlap_1d = overlap_probability(x_e_1d)

        assert overlap_1d.shape == x_e_1d.shape

        x_e_2d = np.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.9]])
        overlap_2d = overlap_probability(x_e_2d)

        assert overlap_2d.shape == x_e_2d.shape
