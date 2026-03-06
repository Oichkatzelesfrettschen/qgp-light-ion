"""
conftest.py - Shared pytest fixtures for the QGP light-ion test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, settings

# Ensure src/ is on the path for all tests.
SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Hypothesis settings for property-based testing
# Register CI and dev profiles with different example counts for speed/thoroughness tradeoff
settings.register_profile("ci", max_examples=50, suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("dev", max_examples=200)

# Load CI profile by default (faster); override with HYPOTHESIS_PROFILE=dev in local dev
settings.load_profile("ci")

# ============================================================================
# Floating-Point Tolerance Policy for Property-Based Tests
# ============================================================================
#
# IEEE 754 arithmetic and numpy vectorization can produce results that differ
# from the mathematical ideal by more than a naive epsilon. ALL inequality
# assertions in Hypothesis tests MUST use one of these patterns:
#
# PATTERN 1 -- Relative tolerance (for physics ratios, energy loss):
#   assert np.isclose(a, b, rtol=1e-5, atol=0.0)
#   Use when: a/b should be close to 1 (e.g., R_AA ratios, cross sections)
#
# PATTERN 2 -- Absolute tolerance (for zero-floor checks, small differences):
#   assert np.isclose(a, b, rtol=0.0, atol=1e-10)
#   Use when: |a - b| should be close to zero (e.g., rho in [0,1])
#
# PATTERN 3 -- pytest.approx for scalar equality (at known values):
#   assert actual == pytest.approx(expected, rel=1e-5)
#   Use when: comparing against a hardcoded constant (T_c=156.5 MeV)
#
# PATTERN 4 -- Monotonicity with tolerance:
#   assert a >= b or np.isclose(a, b, rtol=1e-7)
#   Use when: monotonic inequality (x_e non-increasing) with floating-point slack
#
# The rtol and atol values are calibrated to the expected precision of the
# function under test and MUST be documented in the test's docstring.
# ============================================================================


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Temporary directory for test data output."""
    return tmp_path


@pytest.fixture
def seeded_rng() -> np.random.Generator:
    """Deterministic RNG for reproducible Monte Carlo tests."""
    return np.random.default_rng(42)


@pytest.fixture
def project_root() -> Path:
    """Path to the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def existing_data_dir() -> Path:
    """Path to the data/ directory (requires prior `make data`)."""
    return DATA_DIR
