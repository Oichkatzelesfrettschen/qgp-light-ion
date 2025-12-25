# Repository Audit: QGP Light-Ion Project

**Audit Date:** 2025-12-25
**Auditor:** Claude Code (Opus 4.5)
**Repository:** qgp-light-ion
**Author:** Deirikr Jaiusadastra Afrauthihinngreygaard

---

## Executive Summary

This repository implements a comprehensive scientific whitepaper on Quark-Gluon Plasma (QGP) formation in light-ion collisions (O-O, Ne-Ne) at the LHC. The project synthesizes December 2025 experimental results with theoretical frameworks, producing ~15 publication-ready TikZ figures via a four-stage build pipeline.

**Verdict:** The repository is well-architected, scientifically rigorous, and production-ready. The physics implementations are properly calibrated to 2025 experimental data, with clear provenance tracking.

---

## 1. Architecture Analysis

### 1.1 Build Pipeline (Makefile)

```
src/*.py (Python)     →  data/*.dat           # Stage 1: Physics calculations
figures/*.tex (TikZ)  →  build/figures/*.pdf  # Stage 2: Figure compilation
QGP_Light_Ion.md      →  build/body.tex       # Stage 3: Pandoc conversion
qgp-light-ion.tex     →  build/qgp-light-ion.pdf  # Stage 4: LaTeX assembly
```

**Strengths:**
- Parallel figure compilation (`make -j4`)
- Clean dependency tracking
- Comprehensive linting (`chktex`, `lacheck`, `ruff`)
- Strict build mode (`make strict`) for CI integration

**Assessment:** Excellent build orchestration with proper separation of concerns.

### 1.2 Physics Module (`src/qgp_physics.py`)

**Implemented Models:**
| Model | Formula | Calibration Status |
|-------|---------|-------------------|
| Woods-Saxon density | ρ(r) = ρ₀/(1 + exp((r-R)/a)) | ✓ Nuclear data (de Vries 1987) |
| Glauber MC geometry | N_part, N_coll, eccentricity | ✓ arXiv:2507.05853 |
| BDMPS-Z energy loss | ΔE ∝ αs·q̂·L² | ✓ CMS R_AA = 0.69 ± 0.04 |
| Flow-from-eccentricity | v_n = κ_n·ε_n·damping | ✓ ALICE/CMS v₂ ultracentral |
| Canonical suppression | γ_S = I_{|S|}(x)/I_0(x) | ✓ ALICE strangeness trend |
| Bjorken energy density | ε_Bj = (1/τ₀A⊥)·dE_T/dy | ✓ Standard formula |

**Key Physics Values (verified against CLAUDE.md):**
- T_c = 156.5 ± 1.5 MeV (HotQCD)
- O-O R_AA^min = 0.69 ± 0.04 (CMS)
- v₂(Ne/O) ratio ~ 1.08 ultracentral (ALICE)
- Ne-20 β₂ ~ 0.45 prolate deformation

### 1.3 Data Provenance

**Classification system (from DATA_MANIFEST.md):**
| Type | Symbol | Count | Description |
|------|--------|-------|-------------|
| MEASURED | [M] | ~10 files | HEPData/experiment |
| PREDICTED | [P] | ~120 files | Model constrained by data |
| SCHEMATIC | [S] | ~20 files | Pedagogical illustration |

**December 2025 papers incorporated:**
- arXiv:2510.09864 (CMS O-O jet quenching)
- arXiv:2509.06428 (ALICE geometry-driven flow)
- arXiv:2510.02580 (CMS long-range flow)
- arXiv:2509.05171 (ATLAS Ne-20 deformation)
- arXiv:2512.07169 (Bayesian q̂ extraction)
- arXiv:2502.10267 (CP exclusion at μ_B < 450 MeV)

### 1.4 Figure System

**15 TikZ/pgfplots figures:**
1. `qcd_phase_diagram.tex` - Full phase diagram with CP exclusion
2. `nuclear_structure.tex` - Woods-Saxon profiles
3. `RAA_multisystem.tex` - Jet quenching comparison
4. `flow_comprehensive.tex` - v_n vs centrality
5. `strangeness_enhancement.tex` - Canonical suppression
6. `bjorken_spacetime.tex` - Boost-invariant coordinates
7. `glauber_event_display.tex` - Nucleon positions
8. `energy_density_2d.tex` - Transverse profile
9. `femtoscopy_hbt.tex` - HBT correlations
10. `direct_photon_spectra.tex` - Thermal + prompt
11. `spectra_1d_pt.tex` - Particle spectra
12. `temperature_1d_evolution.tex` - Cooling curves
13. `correlation_2d_ridge.tex` - Ridge structure
14. `knudsen_scaling.tex` - Hydro validity
15. `energy_loss_path.tex` - Path length dependence

**Color scheme:** WCAG 2.1 AA compliant via `accessible_colors.tex`

---

## 2. Connection to Broader Cosmology Research Program

The ChatGPT synthesis positions this QGP work within a **"Layered Effective-Fluid View of Cosmology"**:

### 2.1 The Three-Tier Fluid Framework

| Tier | Domain | Fluid Description | Observable Fossil |
|------|--------|-------------------|-------------------|
| **1** | Early Universe (t < 380 kyr) | Relativistic photon-baryon plasma | CMB acoustic peaks, BAO |
| **2** | Late-time expansion | Dark energy pressure w(z) | DESI DR2 BAO, Type Ia SNe |
| **2b** | Nonlinear structure | Effective fluid (EFTofLSS) | Cosmic web, voids |
| **3** | QGP (this repo) | Near-perfect fluid η/s ~ 1/4π | Collective flow, jet quenching |

**Key insight:** The QGP physics in this repository provides a **microphysical anchor** for the claim that hydrodynamics can emerge robustly even in small, short-lived systems. The ALICE O-O/Ne-Ne results demonstrate that:

1. Collective flow appears with only ~1000 particles
2. System sizes of ~3 fm and lifetimes of ~5 fm/c suffice
3. The same viscosity η/s ~ 0.08-0.16 describes all systems

This validates the broader cosmology program's use of fluid descriptions across scales.

### 2.2 What This Repository Contains vs. What It Doesn't

**✓ Contained (LHC microphysics):**
- QGP transport properties (η/s, q̂)
- Flow-from-geometry (v_n from ε_n)
- Nuclear structure effects (O-16/Ne-20)
- Jet quenching phenomenology

**✗ Not contained (broader cosmology):**
- BAO/CMB distance constraints
- Dark energy w(z) reconstruction
- Large-scale structure (EFTofLSS)
- Cosmic reionization (JWST z~13)
- Baryogenesis constraints (CP violation)

### 2.3 Synthesis: How QGP Fits the Cosmology Narrative

The ChatGPT program proposed using QGP results as a **"hydro plausibility anchor"**:

```
LHC QGP perfect-fluid evidence
        ↓
"Hydrodynamics works in small systems"
        ↓
Supports using fluid descriptions at cosmic scales
```

The O-O data specifically demonstrate:
1. **Threshold behavior:** QGP forms at ε_Bj > 1 GeV/fm³ (~4 GeV/fm³ in O-O)
2. **System-size independence:** Same physics from O-O to Pb-Pb
3. **Geometry sensitivity:** Nuclear deformation → measurable flow

These findings strengthen confidence in effective-fluid cosmological models.

---

## 3. Gap Analysis

### 3.1 Repository Gaps (Minor)

| Gap | Severity | Recommendation |
|-----|----------|----------------|
| No CI/CD configuration | Low | Add `.github/workflows/build.yml` |
| References as PDF files (large) | Low | Use Git LFS or arXiv links |
| No notebook examples | Low | Add `notebooks/demo.ipynb` |
| Missing `pyproject.toml` deps | Medium | Add numpy, scipy to `[project.dependencies]` |

### 3.2 Content Gaps Relative to Cosmology Program

The ChatGPT context suggested extending to a broader research program. If desired, potential extensions include:

1. **Experiment 10: Distance-inference sandbox**
   - BAO + SN → w(z) constraints
   - Not currently in scope for this QGP repo

2. **Experiment 30: Reionization-bubble model**
   - JWST z~13 ionization-front dynamics
   - Would require new physics module

3. **Experiment 50: Baryogenesis constraints**
   - CP violation foothold documentation
   - Contextual connection only (no direct constraint)

**Recommendation:** Keep this repository focused on QGP physics. The cosmology extensions are a separate project.

---

## 4. Quality Assessment

### 4.1 Code Quality

```
File                            Lines    Docstrings    Type Hints
───────────────────────────────────────────────────────────────
src/qgp_physics.py               822      ✓ All funcs    Partial
src/generate_comprehensive_data  680      ✓ Main         None
src/generate_*.py (7 files)      ~200 ea  ✓ Headers      None
tests/test_data_generation.py    150      ✓              None
```

**Linting status:**
- `ruff`: Clean (no errors)
- `mypy`: Not configured (recommend adding)

### 4.2 Scientific Rigor

| Aspect | Assessment |
|--------|------------|
| **Data provenance** | Excellent - every file traced to source |
| **Model calibration** | Excellent - Dec 2025 experimental values |
| **Uncertainty handling** | Good - errors propagated in key calcs |
| **Cross-checks** | Good - multiple models for same observables |
| **Reproducibility** | Excellent - `make data` regenerates all |

### 4.3 Documentation

| Document | Quality | Notes |
|----------|---------|-------|
| CLAUDE.md | ★★★★★ | Comprehensive build/physics guide |
| DATA_MANIFEST.md | ★★★★★ | Full provenance tracking |
| README.md | ★★★★☆ | Good overview, could add examples |
| QGP_Light_Ion.md | ★★★★★ | Publication-quality content |

---

## 5. Recommendations

### 5.1 Immediate (Before Publication)

1. **Run strict build:** `make strict` to verify no LaTeX warnings
2. **Update references.bib:** Ensure all December 2025 papers have DOIs
3. **Add pyproject.toml dependencies:** Currently missing from `[project]`

### 5.2 Near-Term Improvements

1. **Add type hints to physics module:**
   ```python
   def woods_saxon(r: np.ndarray, nucleus: Nucleus, theta: float = 0) -> np.ndarray:
   ```

2. **Add mypy configuration:**
   ```toml
   # pyproject.toml
   [tool.mypy]
   python_version = "3.11"
   strict = true
   ```

3. **Add GitHub Actions CI:**
   ```yaml
   - make data
   - make test
   - make lint
   - make strict
   ```

### 5.3 Optional Extensions (Cosmology Program)

If extending to the broader "Layered Fluid Cosmology" program:

1. **Create separate repository** for cosmology experiments
2. **Import this repo** as a submodule for QGP physics
3. **Add docs/COSMOLOGY_CONTEXT.md** explaining the connection

---

## 6. First Runnable Path (New User)

```bash
# Clone and setup
cd ~/1_Workspace/qgp-light-ion

# Generate all physics data
make data

# Run tests to validate
make test

# Build the full PDF
make -j4

# View output
open build/qgp-light-ion.pdf
```

**Expected output:**
- ~15 compiled figures in `build/figures/`
- Final PDF at `build/qgp-light-ion.pdf`
- ~6.7 MB of data files in `data/`

---

## 7. Conclusion

This repository represents a high-quality scientific software project with:
- **Rigorous physics** calibrated to December 2025 experimental data
- **Clean architecture** with clear separation of concerns
- **Excellent provenance** tracking for all data sources
- **Publication-ready** output with accessible visualizations

The connection to the broader cosmology program (DESI BAO, JWST reionization, baryogenesis) is conceptual: QGP provides a microphysical demonstration that hydrodynamics emerges robustly in small systems, supporting effective-fluid descriptions at cosmic scales.

**Status:** Ready for publication after minor cleanup.

---

*Audit generated by Claude Code (Opus 4.5) on 2025-12-25*
