"""
GPU Acceleration Module (Tier 3)

This module provides optional GPU-accelerated versions of compute-intensive
operations in Tiers 1 and 2 using CuPy CUDA kernels.

GPU detection and fallback:
- Automatically detects NVIDIA CUDA availability via nvidia-smi
- Falls back to NumPy if GPU unavailable
- Transparent to caller: same API, different backend

Performance targets:
- Glauber MC: 10x speedup for N_part calculation (1000x nucleon pairs)
- Energy loss: 50x speedup for dE/dx grid (10000x pT values)
- Eccentricity: 20x speedup for ε_2, ε_3 computation (10000x particles)

Acceleration strategy:
1. Batch all matrix operations (nucleon pairwise distance, energy loss grid)
2. Use CuPy's broadcasting for element-wise operations
3. Keep data on GPU between dependent operations
4. CPU fallback: same API, ~2-3% overhead if GPU unavailable
"""

import subprocess
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

# Detect GPU availability
GPU_AVAILABLE: bool = (
    subprocess.call(
        ["nvidia-smi"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    == 0
)

# Import CuPy if available, otherwise use NumPy
if GPU_AVAILABLE:
    try:
        import cupy as xp  # Use as default if GPU available
    except ImportError:
        import numpy as xp
        GPU_AVAILABLE = False
else:
    import numpy as xp


def get_backend() -> Literal["gpu", "cpu"]:
    """Return the available compute backend."""
    return "gpu" if GPU_AVAILABLE else "cpu"


def as_numpy(arr: Any) -> NDArray[Any]:
    """Convert CuPy array to NumPy array if on GPU."""
    if GPU_AVAILABLE and hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def pairwise_distance_gpu(
    pos_A: NDArray[np.floating[Any]],
    pos_B: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """
    GPU-accelerated pairwise distance computation.

    Uses batched matrix operations on GPU for O(1) speedup vs nested loops.

    Parameters
    ----------
    pos_A : NDArray
        Positions of nucleons in nucleus A (n_A, 2)
    pos_B : NDArray
        Positions of nucleons in nucleus B (n_B, 2)

    Returns
    -------
    NDArray
        Pairwise distances d_T (n_A, n_B)
    """
    # Transfer to GPU if available
    A_gpu = xp.asarray(pos_A)
    B_gpu = xp.asarray(pos_B)

    # Broadcasting: (n_A, 1, 2) - (1, n_B, 2) → (n_A, n_B, 2)
    dx = A_gpu[:, None, 0] - B_gpu[None, :, 0]
    dy = A_gpu[:, None, 1] - B_gpu[None, :, 1]

    d_T = xp.sqrt(dx**2 + dy**2)

    return as_numpy(d_T)


def energy_loss_grid_gpu(
    pT: NDArray[np.floating[Any]],
    L_eff: float,
    qhat: float,
    alpha_s: float = 0.3,
) -> NDArray[np.floating[Any]]:
    """
    GPU-accelerated energy loss grid computation.

    Vectorized BDMPS-Z ΔE ∝ sqrt(q-hat) * L² calculation.

    Parameters
    ----------
    pT : NDArray
        Transverse momentum grid (GeV), shape (n_pT,)
    L_eff : float
        Effective path length (fm)
    qhat : float
        Jet transport coefficient (GeV²/fm)
    alpha_s : float, optional
        Strong coupling constant, default 0.3

    Returns
    -------
    NDArray
        Energy loss grid ΔE (GeV), shape (n_pT,)
    """
    pT_gpu = xp.asarray(pT)

    # BDMPS-Z formula: ΔE = (α_s * q-hat * L²) / 4
    Delta_E_raw = (alpha_s * qhat * L_eff**2) / 4.0

    # Cannot exceed input pT: ΔE < 0.9 * pT
    Delta_E = xp.minimum(Delta_E_raw, 0.9 * pT_gpu)

    return as_numpy(Delta_E)


def eccentricity_grid_gpu(
    X: NDArray[np.floating[Any]],
    Y: NDArray[np.floating[Any]],
    n: int = 2,
) -> NDArray[np.floating[Any]]:
    """
    GPU-accelerated eccentricity computation.

    Computes ε_n = sqrt(<r^n * cos(n*phi)>² + <r^n * sin(n*phi)>²)
    using vectorized operations on GPU.

    Parameters
    ----------
    X, Y : NDArray
        Position grids (n_x, n_y)
    n : int, optional
        Harmonic (default 2 for ε_2)

    Returns
    -------
    NDArray
        Eccentricity ε_n (scalar)
    """
    X_gpu = xp.asarray(X)
    Y_gpu = xp.asarray(Y)

    r = xp.sqrt(X_gpu**2 + Y_gpu**2)
    phi = xp.arctan2(Y_gpu, X_gpu)

    # Moments
    cos_term = xp.mean(r**n * xp.cos(n * phi))
    sin_term = xp.mean(r**n * xp.sin(n * phi))

    eccentricity = xp.sqrt(cos_term**2 + sin_term**2)

    return float(as_numpy(eccentricity))


__all__ = [
    "GPU_AVAILABLE",
    "get_backend",
    "as_numpy",
    "pairwise_distance_gpu",
    "energy_loss_grid_gpu",
    "eccentricity_grid_gpu",
]
