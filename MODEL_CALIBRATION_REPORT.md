# QGP Light-Ion Model Calibration Report

**Date:** 2025-12-16
**Author:** Deirikr Jaiusadastra Afrauthihinngreygaard
**Analysis Script:** `src/compare_model_vs_experiment.py`

## Executive Summary

Comparison of model predictions in `src/qgp_physics.py` against 2025 experimental data from CMS and ALICE reveals systematic discrepancies requiring parameter adjustments:

1. **R_AA predictions**: Good match for O-O (χ²/ndof too high due to high-pT recovery issue), needs minor adjustment for Ne-Ne
2. **Flow predictions**: Significant underestimation of v2 in central collisions (model: ~0 vs data: ~0.06-0.07)
3. **Ne-Ne/O-O ratios**: Model incorrectly predicts increasing ratio with centrality; data shows nearly constant ~1.08

## Detailed Discrepancies

### 1. R_AA - Jet Quenching

#### O-O System
- **CMS Data (HIN-25-008)**: R_AA_min = 0.6948 at pT = 6.2 GeV
- **Model Prediction**: R_AA_min = 0.7117 at pT = 6.0 GeV
- **Discrepancy**: 0.0168 (2.4% relative error) - **ACCEPTABLE**

**Issues:**
- χ²/ndof = 28.98 (should be ~1 for good fit)
- High-pT recovery too aggressive: model → 1.0 at pT > 20 GeV, data → 0.88-0.97
- Mid-pT suppression too weak: at 8-10 GeV, model overshoots by 3-8 sigma

#### Ne-Ne System
- **CMS Estimate**: R_AA_min ≈ 0.65 at pT ≈ 6 GeV
- **Model Prediction**: R_AA_min = 0.6438 at pT = 6.0 GeV
- **Discrepancy**: 0.0062 (1% relative error) - **GOOD MATCH**

**Recommendation:** Increase Ne-Ne suppression_max slightly to better center on 0.65

---

### 2. Flow Harmonics v2 - Critical Issue

#### O-O System (CMS HIN-25-009)
- **χ²/ndof = 2692.94** - Very poor fit

| Centrality | CMS v2    | Model v2  | Δ/σ    | Status |
|------------|-----------|-----------|--------|--------|
| 0.5%       | 0.06108   | 0.00122   | 89.03  | FAIL   |
| 7.5%       | 0.06801   | 0.01790   | 66.98  | FAIL   |
| 22.5%      | 0.07266   | 0.04760   | 31.37  | FAIL   |
| 37.5%      | 0.07046   | 0.06448   | 7.07   | POOR   |

**Critical Issue:** Model predicts v2 → 0 in ultracentral collisions, but CMS observes **significant v2 ≈ 0.061** due to:
- Nuclear deformation (prolate/oblate shapes)
- Alpha clustering in O-16
- Subnucleon fluctuations

#### Ne-Ne System (CMS HIN-25-009)
- **χ²/ndof = 300.60** - Poor fit

| Centrality | CMS v2    | Model v2  | Δ/σ    | Status |
|------------|-----------|-----------|--------|--------|
| 0.5%       | 0.06718   | 0.02310   | 31.23  | FAIL   |
| 7.5%       | 0.07148   | 0.03876   | 21.80  | FAIL   |
| 22.5%      | 0.07570   | 0.06607   | 6.05   | POOR   |
| 37.5%      | 0.07330   | 0.08017   | 4.47   | POOR   |

**Issue:** Ne-Ne shows larger v2 than O-O in central collisions due to prolate deformation, but model underestimates magnitude.

---

### 3. Flow Harmonics v3

#### O-O System
- **χ²/ndof = 579.31** - Poor fit
- Model systematically underestimates v3 by ~2-3× in central/mid-central collisions
- v3 is fluctuation-driven and should be less centrality-dependent than model predicts

---

### 4. Ne-Ne/O-O Flow Ratios - Critical Issue

#### ALICE Data (arXiv:2509.06428)
- **χ²/ndof = 2605.76** - Very poor fit

| Centrality | ALICE v2(Ne/O) | Model v2(Ne/O) | Δ/σ   | Status    |
|------------|----------------|----------------|-------|-----------|
| 0%         | 1.080          | 1.000          | 4.00  | FAIL      |
| 2.5%       | 1.070          | 1.873          | 40.16 | FAIL      |
| 5%         | 1.060          | 2.746          | 84.31 | FAIL      |
| 10%        | 1.050          | 1.866          | 54.41 | FAIL      |
| 20%        | 1.030          | 1.434          | 26.90 | FAIL      |
| 30%        | 1.020          | 1.295          | 13.76 | FAIL      |

**Critical Issue:**
- **ALICE data**: Ratio is nearly constant ~1.05-1.08 across all centralities
- **Model prediction**: Ratio increases dramatically from 1.0 → 2.7 going from central to peripheral
- **Physics interpretation**: Model incorrectly assumes deformation effect grows with centrality; data shows it's strongest in central collisions

---

## Required Parameter Adjustments

### File: `src/qgp_physics.py`

#### 1. R_AA Suppression Parameters (Lines 539-546)

**Current:**
```python
params = {
    'OO': {'suppression_max': 0.31, 'pT_peak': 6.0, 'width': 4.0, 'high_pT_recovery': 40},
    'NeNe': {'suppression_max': 0.38, 'pT_peak': 6.0, 'width': 4.0, 'high_pT_recovery': 45},
    # ...
}
```

**Recommended:**
```python
params = {
    'OO': {'suppression_max': 0.31, 'pT_peak': 6.2, 'width': 4.5, 'high_pT_recovery': 60},  # Match CMS peak position, slower recovery
    'NeNe': {'suppression_max': 0.35, 'pT_peak': 6.0, 'width': 4.5, 'high_pT_recovery': 65},  # Slightly less suppression
    # ...
}
```

**Changes:**
- O-O: pT_peak 6.0→6.2 (match CMS minimum location), high_pT_recovery 40→60 (slower approach to unity)
- Ne-Ne: suppression_max 0.38→0.35 (R_AA_min closer to 0.65), high_pT_recovery 45→65

---

#### 2. Flow Response Coefficients (Lines 404-411)

**Current:**
```python
def flow_from_eccentricity(epsilon_n: float, n: int, eta_over_s: float = 0.12,
                           system_size: float = 3.0) -> float:
    # Response coefficients (from hydro simulations)
    kappa = {2: 0.25, 3: 0.15, 4: 0.10, 5: 0.05}

    # Viscous damping factor
    knudsen = eta_over_s / system_size
    damping = np.exp(-n * knudsen * 5)  # Empirical damping

    return kappa.get(n, 0.1) * epsilon_n * damping
```

**Recommended:**
```python
def flow_from_eccentricity(epsilon_n: float, n: int, eta_over_s: float = 0.12,
                           system_size: float = 3.0) -> float:
    # Response coefficients (calibrated to CMS light-ion data)
    kappa = {2: 0.35, 3: 0.25, 4: 0.15, 5: 0.08}  # Increased for all harmonics

    # Viscous damping factor (reduced for light ions)
    knudsen = eta_over_s / system_size
    damping = np.exp(-n * knudsen * 3)  # Less damping in small systems

    return kappa.get(n, 0.1) * epsilon_n * damping
```

**Changes:**
- kappa[2]: 0.25→0.35 (+40%, critical for v2 amplitude)
- kappa[3]: 0.15→0.25 (+67%, address v3 underestimation)
- kappa[4]: 0.10→0.15 (+50%)
- kappa[5]: 0.05→0.08 (+60%)
- Damping exponent: 5→3 (light ions have less viscous suppression)

**Physical justification:** Small systems show stronger hydrodynamic response than large systems due to larger gradients and faster expansion.

---

#### 3. Initial Eccentricity Model (Lines 426-448) - **CRITICAL**

**Current:**
```python
def generate_flow_vs_centrality(nucleus: Nucleus,
                                centrality_bins: np.ndarray) -> Dict[str, np.ndarray]:
    # ...
    for cent in centrality_bins:
        # ε₂ peaks around 30-40% centrality
        epsilon_2 = 0.5 * np.sin(np.pi * cent / 100) * (1 - 0.3 * cent / 100)

        # ε₃ is fluctuation-driven, roughly constant or slight increase with centrality
        epsilon_3 = 0.15 * (1 + 0.5 * cent / 100)

        # Add nuclear structure effects
        if nucleus.beta2 > 0:  # Deformed nucleus (Ne)
            # More v2 in central collisions due to deformation
            epsilon_2 += nucleus.beta2 * 0.3 * (1 - cent / 100)
```

**Recommended:**
```python
def generate_flow_vs_centrality(nucleus: Nucleus,
                                centrality_bins: np.ndarray) -> Dict[str, np.ndarray]:
    # ...
    # Base eccentricity from nuclear structure (present even in central collisions)
    if nucleus.name.startswith('Oxygen'):
        epsilon_2_base = 0.15  # Alpha clustering in O-16
        epsilon_3_base = 0.10  # Fluctuations from tetrahedral structure
    elif nucleus.name.startswith('Neon'):
        epsilon_2_base = 0.30  # Strong prolate deformation (beta_2 = 0.45)
        epsilon_3_base = 0.08  # Slightly less fluctuation than O
    else:
        epsilon_2_base = 0.0
        epsilon_3_base = 0.0

    for cent in centrality_bins:
        # Geometry-driven component (peaks at mid-peripheral)
        epsilon_2_geom = 0.4 * np.sin(np.pi * cent / 100) * (1 - 0.3 * cent / 100)

        # Total epsilon_2: base (central) + geometry (peripheral)
        # Base component decays slightly with centrality as overlap geometry dominates
        epsilon_2 = epsilon_2_base * np.exp(-cent / 50) + epsilon_2_geom

        # ε₃ is fluctuation-driven with mild centrality dependence
        epsilon_3 = epsilon_3_base * (1 + 0.3 * cent / 100)
```

**Changes:**
- **NEW:** Introduce `epsilon_2_base` for central collisions from nuclear structure
  - O-16: 0.15 (tetrahedral alpha clustering)
  - Ne-20: 0.30 (prolate deformation, beta_2 ≈ 0.45)
- **Modified:** Geometry component now adds to base, ensuring v2 > 0 even at cent=0
- **Modified:** epsilon_3 base values set per nucleus
- **Physical justification:** CMS/ALICE data prove that nuclear structure creates intrinsic eccentricity independent of collision geometry

---

#### 4. Remove Erroneous Deformation Scaling (Lines 440-442)

**Current (INCORRECT):**
```python
        # Add nuclear structure effects
        if nucleus.beta2 > 0:  # Deformed nucleus (Ne)
            # More v2 in central collisions due to deformation
            epsilon_2 += nucleus.beta2 * 0.3 * (1 - cent / 100)
```

**Action:** **DELETE** these lines - they are now replaced by the `epsilon_2_base` approach above.

**Reason:** The old approach added deformation as a multiplier that scales with (1 - cent/100), making Ne-Ne/O-O ratio explode at low centrality. The new approach uses a constant base eccentricity that decays mildly with centrality, matching ALICE's observation of nearly constant ratio.

---

#### 5. Nuclear Deformation Parameter (Line 53)

**Current:**
```python
'Ne': Nucleus('Neon-20', A=20, Z=10, R0=2.791, a=0.535, beta2=0.45),
```

**Recommended:** Keep beta2=0.45 (correct from literature)

**Note:** The issue was not the beta2 value but how it was applied in the eccentricity calculation (see #4 above).

---

## Summary of Physics Corrections

### Root Cause Analysis

The original model made a **critical physics error**:

**Incorrect assumption:** Nuclear deformation effects scale with `(1 - centrality)`, making Ne-Ne/O-O ratio increase as centrality → 0.

**Correct physics:**
- Nuclear structure (deformation, clustering) creates **intrinsic eccentricity** present in ALL collisions
- This intrinsic component is STRONGEST in central collisions where the entire nuclear volume participates
- Peripheral collisions are dominated by geometric eccentricity from impact parameter
- Therefore, Ne-Ne/O-O ratio should be **nearly constant** (~1.05-1.08), as ALICE observes

### Expected Improvements After Adjustments

| Observable | Current χ²/ndof | Expected χ²/ndof | Improvement |
|------------|-----------------|------------------|-------------|
| O-O R_AA   | 28.98           | ~3-5             | ~6× better  |
| O-O v2     | 2692.94         | ~5-10            | ~300× better|
| O-O v3     | 579.31          | ~10-20           | ~30× better |
| Ne-Ne v2   | 300.60          | ~5-10            | ~40× better |
| Ne/O ratio | 2605.76         | ~2-5             | ~600× better|

### Validation Tests After Implementation

Run these checks after applying parameter changes:

```bash
# Regenerate data
make clean
make data

# Run comparison
python3 src/compare_model_vs_experiment.py

# Expected results:
# - O-O v2(0%) should be ~0.055-0.065 (currently ~0.001)
# - Ne-Ne v2(0%) should be ~0.065-0.075 (currently ~0.023)
# - Ne/O v2 ratio should be ~1.05-1.10 across all centralities (currently 1.0→2.7)
# - R_AA high-pT recovery should asymptote to ~0.95 not 1.0
```

---

## References

1. CMS Collaboration, "Discovery of Suppressed Charged-Particle Production in Ultrarelativistic OO Collisions," CMS-HIN-25-008 (2025), https://www.hepdata.net/record/ins3068407
2. CMS Collaboration, "Observation of Long-Range Collective Flow in OO and NeNe Collisions," CMS-HIN-25-009 (2025), https://www.hepdata.net/record/ins3062822
3. ALICE Collaboration, "Evidence of nuclear geometry-driven anisotropic flow in OO and Ne-Ne collisions at sqrt(s_NN) = 5.36 TeV," arXiv:2509.06428 (2025)
4. Heinz & Snellings, "Collective flow and viscosity in relativistic heavy-ion collisions," Ann. Rev. Nucl. Part. Sci. 63 (2013) 123
5. Baier et al., "Radiative energy loss of high energy quarks and gluons in a finite-volume quark-gluon plasma," Nucl. Phys. B 484 (1997) 265

---

**End of Report**
