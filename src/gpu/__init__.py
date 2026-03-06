"""
GPU Acceleration Module (Tier 3)

This module provides optional GPU-accelerated versions of compute-intensive
operations in Tiers 1 and 2 using CuPy CUDA kernels.

GPU detection and fallback:
- Automatically detects NVIDIA CUDA availability via nvidia-smi
- Falls back to NumPy if GPU unavailable
- Transparent to caller: same API, different backend

Not yet implemented (phase 16a).
"""

import subprocess
from typing import Literal

# Detect GPU availability
GPU_AVAILABLE: bool = (
    subprocess.call(
        ["nvidia-smi"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    == 0
)


def get_backend() -> Literal["gpu", "cpu"]:
    """Return the available compute backend."""
    return "gpu" if GPU_AVAILABLE else "cpu"


__all__ = ["GPU_AVAILABLE", "get_backend"]
