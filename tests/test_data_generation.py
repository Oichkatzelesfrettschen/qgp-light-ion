#!/usr/bin/env python3
"""
Test suite for QGP light-ion data generation and validation.

Tests cover:
1. Data file generation
2. Physical constraints validation
3. Data format correctness
4. Build artifact verification
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
BUILD_DIR = PROJECT_ROOT / "build"


class TestResult:
    """Simple test result container."""

    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name = name
        self.passed = passed
        self.message = message


def load_dat_file(filepath: Path) -> tuple[np.ndarray, list[str]]:
    """Load a .dat file and return data array and header columns."""
    with open(filepath) as f:
        header_line = f.readline().strip()
    columns = header_line.replace("#", "").strip().split()
    data = np.loadtxt(filepath)
    return data, columns


def test_data_generation_script_exists() -> TestResult:
    """Test that the data generation script exists."""
    script_path = SRC_DIR / "generate_plot_data.py"
    if script_path.exists():
        return TestResult("data_script_exists", True, f"Found {script_path}")
    return TestResult("data_script_exists", False, f"Missing {script_path}")


def test_data_generation_runs() -> TestResult:
    """Test that data generation script runs without errors."""
    script_path = SRC_DIR / "generate_plot_data.py"
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--output-dir", str(DATA_DIR)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return TestResult("data_generation_runs", True, "Script executed successfully")
        return TestResult(
            "data_generation_runs", False, f"Exit code {result.returncode}: {result.stderr}"
        )
    except subprocess.TimeoutExpired:
        return TestResult("data_generation_runs", False, "Script timed out")
    except Exception as e:
        return TestResult("data_generation_runs", False, str(e))


def test_raa_data_files_exist() -> TestResult:
    """Test that R_AA data files are generated."""
    required_files = ["RAA_OO.dat", "RAA_PbPb.dat"]
    missing = [f for f in required_files if not (DATA_DIR / f).exists()]
    if not missing:
        return TestResult("raa_files_exist", True, "All R_AA files present")
    return TestResult("raa_files_exist", False, f"Missing: {missing}")


def test_flow_data_files_exist() -> TestResult:
    """Test that flow data files are generated."""
    required_files = ["flow_v2_OO.dat", "flow_v3_OO.dat", "flow_v2_NeNe.dat"]
    missing = [f for f in required_files if not (DATA_DIR / f).exists()]
    if not missing:
        return TestResult("flow_files_exist", True, "All flow files present")
    return TestResult("flow_files_exist", False, f"Missing: {missing}")


def test_raa_physical_constraints() -> TestResult:
    """
    Test physical constraints on R_AA data.

    Physical requirements:
    - R_AA should be positive (>0)
    - R_AA should typically be <= 1.5 (allowing for Cronin enhancement)
    - At high p_T, R_AA should approach 1
    - Suppression (R_AA < 1) expected at intermediate p_T for QGP
    """
    errors = []

    for filename in ["RAA_OO.dat", "RAA_PbPb.dat"]:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            errors.append(f"{filename}: file not found")
            continue

        data, _cols = load_dat_file(filepath)
        pt = data[:, 0]
        raa = data[:, 1]

        # R_AA must be positive
        if np.any(raa <= 0):
            errors.append(f"{filename}: R_AA has non-positive values")

        # R_AA should not exceed reasonable bounds
        if np.any(raa > 1.5):
            errors.append(f"{filename}: R_AA exceeds 1.5 (unphysical)")

        # At high pT (>15 GeV), R_AA should be > 0.5 (approaching unity)
        high_pt_mask = pt > 15
        if np.any(high_pt_mask) and np.any(raa[high_pt_mask] < 0.5):
            errors.append(f"{filename}: R_AA too suppressed at high pT")

        # Should show some suppression at intermediate pT
        mid_pt_mask = (pt > 4) & (pt < 10)
        if np.any(mid_pt_mask) and np.all(raa[mid_pt_mask] >= 1.0):
            errors.append(f"{filename}: No jet quenching suppression at intermediate pT")

    if not errors:
        return TestResult("raa_physical_constraints", True, "All R_AA constraints satisfied")
    return TestResult("raa_physical_constraints", False, "; ".join(errors))


def test_flow_physical_constraints() -> TestResult:
    """
    Test physical constraints on flow coefficient data.

    Physical requirements:
    - v_n coefficients should be between -0.5 and 0.5 (typically |v_n| < 0.2)
    - v_2 should generally be larger than v_3
    - v_n should be non-negative for most centralities (can be slightly negative due to fluctuations)
    - Central collisions (0-5%) should have smaller v_2 than mid-central
    """
    errors = []

    for filename in ["flow_v2_OO.dat", "flow_v3_OO.dat", "flow_v2_NeNe.dat"]:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            errors.append(f"{filename}: file not found")
            continue

        data, _cols = load_dat_file(filepath)
        centrality = data[:, 0]
        vn = data[:, 1]

        # Flow coefficients should be bounded
        if np.any(np.abs(vn) > 0.5):
            errors.append(f"{filename}: |v_n| exceeds 0.5 (unphysical)")

        # v_n should not be significantly negative (small negative from fluctuations OK)
        if np.any(vn < -0.05):
            errors.append(f"{filename}: v_n has significant negative values (<-0.05)")

    # Compare v2 and v3 magnitudes
    v2_path = DATA_DIR / "flow_v2_OO.dat"
    v3_path = DATA_DIR / "flow_v3_OO.dat"
    if v2_path.exists() and v3_path.exists():
        v2_data, _ = load_dat_file(v2_path)
        v3_data, _ = load_dat_file(v3_path)
        v2_mean = np.mean(np.abs(v2_data[:, 1]))
        v3_mean = np.mean(np.abs(v3_data[:, 1]))
        if v3_mean > v2_mean:
            errors.append("v3 magnitude exceeds v2 on average (unusual)")

    if not errors:
        return TestResult("flow_physical_constraints", True, "All flow constraints satisfied")
    return TestResult("flow_physical_constraints", False, "; ".join(errors))


def test_data_format_consistency() -> TestResult:
    """Test that all data files have consistent format."""
    errors = []

    all_files = list(DATA_DIR.glob("*.dat"))
    for filepath in all_files:
        try:
            data, _cols = load_dat_file(filepath)

            # Must have at least 2 columns
            if data.ndim == 1 or data.shape[1] < 2:
                errors.append(f"{filepath.name}: insufficient columns")

            # Must have at least 5 data points
            if data.shape[0] < 5:
                errors.append(f"{filepath.name}: too few data points (<5)")

            # Check for NaN/Inf
            if np.any(~np.isfinite(data)):
                errors.append(f"{filepath.name}: contains NaN or Inf values")

        except Exception as e:
            errors.append(f"{filepath.name}: parse error - {e}")

    if not errors:
        return TestResult("data_format_consistency", True, f"All {len(all_files)} files valid")
    return TestResult("data_format_consistency", False, "; ".join(errors))


def test_makefile_exists() -> TestResult:
    """Test that Makefile exists and has required targets."""
    makefile = PROJECT_ROOT / "Makefile"
    if not makefile.exists():
        return TestResult("makefile_exists", False, "Makefile not found")

    content = makefile.read_text()
    required_targets = ["all", "clean", "figures", "data"]
    missing = [t for t in required_targets if ".PHONY:" not in content or t not in content]

    # Simpler check
    missing = []
    for target in required_targets:
        if f"{target}:" not in content:
            missing.append(target)

    if missing:
        return TestResult("makefile_exists", False, f"Missing targets: {missing}")
    return TestResult("makefile_exists", True, "Makefile with all required targets")


def test_latex_source_files_exist() -> TestResult:
    """Test that required LaTeX source files exist."""
    required = [
        PROJECT_ROOT / "qgp-light-ion.tex",
        PROJECT_ROOT / "references.bib",
        PROJECT_ROOT / "figures" / "RAA_visualization.tex",
        PROJECT_ROOT / "figures" / "flow_visualization.tex",
        PROJECT_ROOT / "figures" / "geometry_visualization.tex",
    ]
    missing = [str(f) for f in required if not f.exists()]
    if missing:
        return TestResult("latex_sources_exist", False, f"Missing: {missing}")
    return TestResult("latex_sources_exist", True, "All LaTeX sources present")


def run_all_tests() -> list[TestResult]:
    """Run all tests and return results."""
    tests = [
        test_data_generation_script_exists,
        test_data_generation_runs,
        test_raa_data_files_exist,
        test_flow_data_files_exist,
        test_raa_physical_constraints,
        test_flow_physical_constraints,
        test_data_format_consistency,
        test_makefile_exists,
        test_latex_source_files_exist,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
        except Exception as e:
            result = TestResult(test_func.__name__, False, f"Exception: {e}")
        results.append(result)

    return results


def main():
    """Main test runner with exit code."""
    print("=" * 60)
    print("QGP Light-Ion Test Suite")
    print("=" * 60)

    results = run_all_tests()

    passed = 0
    failed = 0

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name}")
        if r.message:
            print(f"       {r.message}")
        if r.passed:
            passed += 1
        else:
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    # Exit with error code if any tests failed
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
