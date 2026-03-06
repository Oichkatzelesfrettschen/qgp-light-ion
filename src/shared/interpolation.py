"""Shared interpolation utilities (thin wrappers over scipy with physics defaults)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline


def monotone_cubic_spline(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> CubicSpline:
    """Cubic spline on strictly-increasing x; raises ValueError if not sorted.

    Parameters
    ----------
    x : NDArray[float64], shape (n,)
        Strictly increasing x values (domain points)
    y : NDArray[float64], shape (n,)
        Function values at x points

    Returns
    -------
    CubicSpline
        Cubic spline interpolator for the (x, y) data

    Raises
    ------
    ValueError
        If x is not strictly increasing
    """
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing")
    return CubicSpline(x, y)


def log_interp1d(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    x_new: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolate in log-log space for power-law physics curves.

    Fits a cubic spline through log(x), log(y) and evaluates at log(x_new),
    returning exp of the spline result. Useful for smooth power-law interpolation.

    Parameters
    ----------
    x : NDArray[float64], shape (n,)
        Strictly increasing x values (must be positive)
    y : NDArray[float64], shape (n,)
        Function values y(x) (must be positive)
    x_new : NDArray[float64], shape (m,)
        Points at which to interpolate (must be positive, within x range)

    Returns
    -------
    NDArray[float64], shape (m,)
        Interpolated values exp(spline(log(x_new)))
    """
    cs = CubicSpline(np.log(x), np.log(y))
    result = np.exp(cs(np.log(x_new)))
    return np.asarray(result, dtype=np.float64)
