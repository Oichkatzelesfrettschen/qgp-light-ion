"""
conftest.py - Shared pytest fixtures for the QGP light-ion test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is on the path for all tests.
SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


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
