"""
io_utils.py - Canonical I/O utilities for QGP data generation scripts.

All generator scripts MUST import from here instead of defining local copies.
Functions support both simple header strings and structured provenance metadata.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np


def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def make_provenance_header(
    observable: str,
    classification: str,
    model_description: str,
    model_inputs: dict | None = None,
    references: list[str] | None = None,
    notes: str | list[str] | None = None,
) -> str:
    """Generate a standard provenance header for model data files.

    WHY: Every generated .dat file should record what produced it, how, and
    with what inputs.  This header is designed to be skipped by pgfplots
    (all lines start with '#') while remaining human-readable.

    Args:
        observable: What is being calculated (e.g. "R_AA vs pT").
        classification: One of "MEASURED", "PREDICTED", "SCHEMATIC".
        model_description: Brief physics model description.
        model_inputs: Optional dict of {parameter: value} pairs.
        references: Optional list of reference strings.
        notes: Optional note string or list of note strings.

    Returns:
        Multi-line comment string (each line prefixed with '# ').
    """
    lines: list[str] = [
        "=" * 77,
        f" {observable}",
        "=" * 77,
        "",
        f"DATA TYPE: {classification} (Model-generated)",
        "",
        "GENERATOR:",
        "  Project: QGP Light-Ion Whitepaper (2025)",
        "",
        "MODEL:",
        f"  {model_description}",
    ]

    if model_inputs:
        lines.append("")
        lines.append("MODEL INPUTS:")
        for key, val in model_inputs.items():
            lines.append(f"  {key}: {val}")

    if references:
        lines.append("")
        lines.append("REFERENCES:")
        for ref in references:
            lines.append(f"  - {ref}")

    if notes:
        lines.append("")
        lines.append("NOTES:")
        if isinstance(notes, list):
            for note in notes:
                lines.append(f"  - {note}")
        else:
            lines.append(f"  {notes}")

    lines.append("")
    lines.append("=" * 77)

    return "\n".join(["# " + line for line in lines])


def save_dat(
    filename: str,
    data_dict: dict[str, Any],
    header: str = "",
    provenance: dict | None = None,
) -> None:
    """Save columnar data to a .dat file for pgfplots.

    WHY: pgfplots reads whitespace-separated columns with '#' comment lines.
    This function handles both simple headers (legacy) and structured provenance
    metadata so that every file is self-documenting.

    Args:
        filename: Absolute or relative output file path.
        data_dict: Ordered dict of {column_name: array}.
        header: Simple one-line header string (used when provenance is None).
        provenance: Dict of kwargs for make_provenance_header() (preferred).
    """
    keys = list(data_dict.keys())
    arrays = [np.atleast_1d(data_dict[k]) for k in keys]

    with open(filename, "w") as f:
        if provenance:
            f.write(make_provenance_header(**provenance) + "\n")
            f.write(f"# COLUMNS: {' '.join(keys)}\n")
        elif header:
            f.write(f"# {header}\n")
            f.write("# " + " ".join(keys) + "\n")
        else:
            f.write("# " + " ".join(keys) + "\n")
        for i in range(len(arrays[0])):
            row = [f"{arr[i]:.8e}" for arr in arrays]
            f.write(" ".join(row) + "\n")


def save_2d_grid(
    filename: str,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    header: str = "",
    provenance: dict | None = None,
) -> None:
    """Save 2D grid data for pgfplots surf/contour plots.

    WHY: pgfplots surf plots require an empty line between row-blocks.
    This function always writes that separator, regardless of how the
    caller structured X/Y/Z.

    Args:
        filename: Output file path.
        X, Y, Z: 2-D arrays with identical shapes.
        header: Simple one-line header (used when provenance is None).
        provenance: Dict of kwargs for make_provenance_header() (preferred).
    """
    with open(filename, "w") as f:
        if provenance:
            f.write(make_provenance_header(**provenance) + "\n")
            f.write("# COLUMNS: x y z\n")
        elif header:
            f.write(f"# {header}\n")
            f.write("# x y z\n")
        else:
            f.write("# x y z\n")
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                f.write(f"{X[i, j]:.6e} {Y[i, j]:.6e} {Z[i, j]:.6e}\n")
            f.write("\n")  # pgfplots requires blank line between row-blocks


def save_curve(
    filename: str,
    x: np.ndarray,
    y: np.ndarray,
    header: str = "x y",
    comments: list[str] | None = None,
    output_dir: str = "",
) -> None:
    """Save a simple x-y curve to a .dat file.

    Args:
        filename: File name (relative to output_dir if output_dir is set).
        x: x-axis array.
        y: y-axis array.
        header: Column header string (written as a comment line).
        comments: Optional extra comment lines written before the header.
        output_dir: Directory prefix for filename.  Pass "" to use filename as-is.
    """
    filepath = os.path.join(output_dir, filename) if output_dir else filename
    with open(filepath, "w") as f:
        if comments:
            for comment in comments:
                f.write(f"# {comment}\n")
        f.write(f"# {header}\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi:.6f} {yi:.6f}\n")
    print(f"  {os.path.basename(filepath)}: {len(x)} points")


def save_curve_multi(
    filename: str,
    x: np.ndarray,
    columns: list[np.ndarray],
    header: str,
    comments: list[str] | None = None,
    output_dir: str = "",
) -> None:
    """Save a curve with multiple y-columns.

    Args:
        filename: File name (relative to output_dir if output_dir is set).
        x: x-axis array.
        columns: List of y-column arrays (same length as x).
        header: Column header string.
        comments: Optional extra comment lines written before the header.
        output_dir: Directory prefix for filename.
    """
    filepath = os.path.join(output_dir, filename) if output_dir else filename
    with open(filepath, "w") as f:
        if comments:
            for comment in comments:
                f.write(f"# {comment}\n")
        f.write(f"# {header}\n")
        for i, xi in enumerate(x):
            row = f"{xi:.6f}" + "".join(f" {col[i]:.6f}" for col in columns)
            f.write(row + "\n")
    print(f"  {os.path.basename(filepath)}: {len(x)} points x {len(columns) + 1} columns")


def save_curve_with_errors(
    filename: str,
    x: np.ndarray,
    y: np.ndarray,
    err: np.ndarray,
    header: str = "x y err",
    output_dir: str = "",
) -> None:
    """Save a curve with symmetric error bars.

    Args:
        filename: File name (relative to output_dir if output_dir is set).
        x, y: Data arrays.
        err: Symmetric uncertainty array (same length as x).
        header: Column header string.
        output_dir: Directory prefix for filename.
    """
    filepath = os.path.join(output_dir, filename) if output_dir else filename
    with open(filepath, "w") as f:
        f.write(f"# {header}\n")
        for xi, yi, ei in zip(x, y, err):
            f.write(f"{xi:.6f} {yi:.6f} {ei:.6f}\n")
    print(f"  {os.path.basename(filepath)}: {len(x)} points with errors")


def save_points_with_errors(
    filename: str,
    points: Sequence[Sequence[float]],
    header: str,
    output_dir: str = "",
) -> None:
    """Save data points with (possibly asymmetric) errors.

    Each element of points is a tuple of floats; all tuples must have the
    same length.  Columns are described by header.

    Args:
        filename: File name (relative to output_dir if output_dir is set).
        points: List of tuples, one per row.
        header: Column header string.
        output_dir: Directory prefix for filename.
    """
    filepath = os.path.join(output_dir, filename) if output_dir else filename
    with open(filepath, "w") as f:
        f.write(f"# {header}\n")
        for p in points:
            f.write(" ".join(f"{v:.6f}" for v in p) + "\n")
    print(f"  {os.path.basename(filepath)}: {len(points)} points")


def load_dat(filename: str) -> dict[str, np.ndarray]:
    """Load a .dat file written by save_dat / save_curve into a dict.

    Reads the last comment line before the data to extract column names.
    Lines starting with '#' are treated as comments; blank lines are skipped.

    Args:
        filename: Path to the .dat file.

    Returns:
        Dict mapping column name -> 1-D numpy array.
    """
    col_names: list[str] = []
    data_rows: list[list[float]] = []

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # The last comment line is the column header
                col_names = line.lstrip("# ").split()
            else:
                data_rows.append([float(v) for v in line.split()])

    if not col_names:
        raise ValueError(f"No column header found in {filename}")
    if not data_rows:
        raise ValueError(f"No data rows found in {filename}")

    arr = np.array(data_rows)
    if arr.shape[1] != len(col_names):
        raise ValueError(
            f"{filename}: {arr.shape[1]} data columns but {len(col_names)} header names"
        )

    return {name: arr[:, i] for i, name in enumerate(col_names)}
