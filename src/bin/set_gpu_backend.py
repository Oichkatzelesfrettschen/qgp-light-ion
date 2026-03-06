#!/usr/bin/env python3
"""Set GPU_BACKEND environment variable safely.

This script prevents command injection by accepting GPU_BACKEND as a
positional argument instead of via string interpolation in Makefile.
"""

from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> None:
    """Set GPU_BACKEND environment variable and run subsequent command.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments [gpu_backend_value]
    """
    args = argv or sys.argv[1:]

    if not args:
        print("Usage: set_gpu_backend.py <gpu_backend_value>", file=sys.stderr)
        sys.exit(1)

    gpu_backend = args[0]

    # Validate input (must be "0" or "1")
    if gpu_backend not in ("0", "1"):
        print(
            f"ERROR: GPU_BACKEND must be '0' or '1', got '{gpu_backend}'",
            file=sys.stderr,
        )
        sys.exit(1)

    # Set environment variable safely (no shell expansion)
    os.environ["GPU_BACKEND"] = gpu_backend
    print(f"GPU_BACKEND={gpu_backend}")


if __name__ == "__main__":
    main()
