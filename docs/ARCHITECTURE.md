# Three-Tier Hydrodynamic Universe Architecture

## Overview

The qgp-light-ion project is organized as a **three-tier scientific compute pipeline** investigating hydrodynamic transport across cosmological and microphysical scales. Each tier is independently useful but can be combined for synthesis.

```
┌─────────────────────────────────────────────────────────────────┐
│ TIER 1: QGP Microphysics (LHC scales)                          │
│ ─────────────────────────────────────────────────────────────── │
│ • Woods-Saxon nuclear profiles (O-16, Ne-20, Pb-208, Xe-129)   │
│ • Glauber Monte Carlo geometry (N_part, N_coll, eccentricity)  │
│ • BDMPS-Z radiative energy loss → R_AA(pT)                    │
│ • Viscous hydrodynamics → v_n coefficients                     │
│ • Canonical strangeness suppression                             │
│ • Output: 15 TikZ figures, publication-ready PDF               │
│                                                                 │
│ Modules: src/qgp/                                              │
│   ├── physics.py (Woods-Saxon, Glauber, BDMPS-Z)              │
│   ├── phase_diagram/ (6 submodules)                           │
│   ├── constants.py (nuclear geometry, QCD params)              │
│   ├── io_utils.py (data I/O with provenance)                  │
│   └── generate.py (comprehensive data orchestration)           │
└─────────────────────────────────────────────────────────────────┘
                            ▲
                            │ (shared infrastructure)
                            │
┌─────────────────────────────────────────────────────────────────┐
│ TIER 2: Cosmology (billion-year scales)                        │
│ ─────────────────────────────────────────────────────────────── │
│ • Reionization-bubble dynamics (JWST z~13)                     │
│ • Dark energy reconstruction (BAO + SNe)                        │
│ • Expanding ionization fronts (hydrodynamic analogy to QGP)   │
│ • Output: cosmology figures, physical insights                  │
│                                                                 │
│ Modules: src/cosmology/                                        │
│   ├── reionization.py (analogy functions)                     │
│   ├── reionization_front.py (bubble expansion solver)          │
│   ├── dark_energy_inference.py (Bayesian w(z) posterior)      │
│   ├── jwst_observations.py (redshift measurements)             │
│   ├── constants.py (cosmological parameters)                   │
│   └── io_utils.py (extends tier 1 I/O utilities)              │
└─────────────────────────────────────────────────────────────────┘
                            ▲
                            │ (optional acceleration)
                            │
┌─────────────────────────────────────────────────────────────────┐
│ TIER 3: GPU Acceleration (transparent optimization)            │
│ ─────────────────────────────────────────────────────────────── │
│ • CuPy backend for Glauber MC (100k events in <10s)           │
│ • CUDA kernels for femtoscopy grids                            │
│ • Optional: 10-50x speedup for Tiers 1 & 2                    │
│ • Falls back to NumPy automatically if GPU unavailable         │
│                                                                 │
│ Modules: src/gpu/                                              │
│   ├── __init__.py (GPU availability detection)                │
│   ├── glauber_cuda.py (CuPy-accelerated MC)                   │
│   └── femtoscopy_cuda.py (CUDA grid computation)              │
└─────────────────────────────────────────────────────────────────┘
```

## Build System Architecture

### Makefile Targets

```
make                       # Full build (Tier 1 + Tier 2 figures + PDF)
make data                  # Generate all data (Tier 1 + Tier 2, CPU)
make data-qgp              # Tier 1 data only
make data-cosmology        # Tier 2 data only
make data-gpu              # Force GPU backend (Tier 1 + 2)
make data-cpu              # Force CPU backend (fallback)
make -j4                   # Parallel builds (4 cores)

make test                  # All tests (Tier 1 + 2 + GPU)
make test-qgp              # Tier 1 tests only
make test-cosmology        # Tier 2 tests only
make test-gpu              # GPU backend tests
make coverage              # Coverage report with breakdown by tier

make lint                  # All linters (ruff + mypy + chktex)
make lint-qgp              # Tier 1 linting only
make lint-cosmology        # Tier 2 linting only
make strict                # Full lint + build + test validation

make clean                 # Remove all generated files
make clean-data            # Remove data/ (preserve experimental/)
make clean-qgp             # Remove Tier 1 data only
make clean-cosmology       # Remove Tier 2 data only
make clean-figures         # Remove compiled figures only
```

### Dependency Caching Strategy

Each tier maintains checksums to detect when regeneration is needed:

```
.generated-qgp             # Timestamp of last Tier 1 data generation
.generated-cosmology       # Timestamp of last Tier 2 data generation
.generated-figures         # Timestamp of last figure compilation

data/.checksums-qgp.txt    # SHA256 of all Tier 1 data files
data/.checksums-cosmology.txt  # SHA256 of all Tier 2 data files
```

Trigger regeneration when:
- Source code changes (src/qgp/*.py or src/cosmology/*.py)
- External data changes (experimental/*.dat, arXiv papers incorporated)
- Build options change (GPU enabled/disabled)

### Critical Path (Longest Dependency Chain)

For `make -j$(nproc)`:

1. **Tier 1 Data** (10-15 min CPU, 2-3 min GPU)
   - Glauber MC nucleon sampling
   - Energy density evolution
   - Femtoscopy correlations

2. **Tier 2 Data** (5-10 min CPU, 1-2 min GPU)
   - BAO/SNe Bayesian inference (slowest cosmology step)
   - Reionization bubble solver

3. **Figures** (5-8 min, parallelizable)
   - 15 original TikZ figures (Tier 1 data-dependent)
   - 5 new synthesis figures (Tier 1 + 2 data-dependent)

4. **PDF Assembly** (2 min)
   - Pandoc conversion (Tier 1 + 2 independent)
   - latexmk multi-pass compilation

**Estimated total:** ~20-25 min (CPU, -j4), ~8-12 min (GPU, -j8)

## Repository Structure

### Tier 1: QGP Physics

```
src/qgp/
├── __init__.py              # Package exports
├── physics.py               # Core models (Woods-Saxon, Glauber, BDMPS-Z)
├── constants.py             # Nuclear geometry + QCD constants
├── io_utils.py              # Data I/O with provenance headers
├── generate.py              # Master data generation orchestration
└── phase_diagram/           # QCD phase diagram subpackage (6 files)
    ├── __init__.py
    ├── params.py            # Transition parameter dataclass
    ├── crossover.py         # Lattice QCD crossover line
    ├── critical_point.py    # CEP exclusion region + FRG consensus
    ├── first_order.py       # First-order transition models
    ├── freeze_out.py        # Chemical freeze-out parametrization
    └── trajectories.py      # System trajectories in phase space
```

### Tier 2: Cosmology

```
src/cosmology/
├── __init__.py              # Package exports
├── constants.py             # Cosmological parameters, JWST observations
├── reionization.py          # Reionization-QGP analogy functions
├── reionization_front.py    # Bubble expansion solver (full calculation)
├── dark_energy_inference.py # BAO/SNe Bayesian posterior
├── jwst_observations.py     # Parsed JWST redshift data
└── io_utils.py              # Extends Tier 1 I/O for cosmology data
```

### Tier 3: GPU Acceleration

```
src/gpu/
├── __init__.py              # GPU detection + conditional imports
├── glauber_cuda.py          # CuPy-accelerated Glauber MC
└── femtoscopy_cuda.py       # CUDA kernel for femtoscopy grids
```

### Shared Utilities

```
src/shared/
├── __init__.py
├── math_utils.py            # Interpolation, MCMC helpers, common functions
└── visualization.py         # Plotting utilities for both tiers
```

### Data Directory Organization

```
data/
├── experimental/            # Committed experimental data (tracked in git)
│   ├── CMS_OO_RAA_HIN25008.dat
│   ├── CMS_OO_flow_HIN25009.dat
│   ├── ALICE_OO_NeNe_flow_2509.06428.dat
│   └── (more HEPData files)
│
├── qgp/                     # Tier 1 generated data (gitignored)
│   ├── nuclear_geometry/
│   ├── jet_quenching/
│   ├── flow/
│   ├── phase_diagram/
│   └── ...
│
└── cosmology/               # Tier 2 generated data (gitignored)
    ├── dark_energy/         # BAO/SNe posteriors
    ├── reionization/        # Bubble evolution
    └── ...

.generated-qgp              # Tier 1 timestamp
.generated-cosmology        # Tier 2 timestamp
.generated-figures          # Figure timestamp
```

### Test Organization

```
tests/
├── conftest.py              # Shared fixtures for all tiers
├── test_qgp/
│   ├── test_physics.py      # Woods-Saxon, Glauber, BDMPS-Z
│   ├── test_constants.py    # Nuclear parameters
│   ├── test_phase_diagram.py # Phase boundaries + trajectories
│   └── test_io_utils.py     # Data I/O routines
├── test_cosmology/
│   ├── test_reionization.py # Analogy functions + bubble solver
│   ├── test_dark_energy.py  # BAO/SNe inference
│   └── test_jwst.py         # Observation parsing
└── test_gpu/
    ├── test_gpu_equivalence.py  # CPU vs GPU output parity
    └── test_gpu_performance.py  # Speedup benchmarks
```

### Figures Directory

```
figures/
├── accessible_colors.tex    # WCAG 2.1 AA color palette

# Tier 1 figures (15 original + 5 new from phases 15b-17b)
├── qcd_phase_diagram.tex            # Phase diagram with CP
├── nuclear_structure.tex              # Woods-Saxon profiles
├── RAA_multisystem.tex                # Jet quenching R_AA
├── flow_comprehensive.tex             # v_n coefficients
├── strangeness_enhancement.tex        # Canonical suppression
├── bjorken_spacetime.tex              # Boost-invariant evolution
├── glauber_event_display.tex          # Nucleon geometry
├── energy_density_2d.tex              # 2D hot spots
├── femtoscopy_hbt.tex                 # HBT correlations
├── direct_photon_spectra.tex          # Thermal radiation
├── spectra_1d_pt.tex                  # Particle spectra
├── temperature_1d_evolution.tex       # Cooling curves
├── correlation_2d_ridge.tex           # Ridge structure
├── knudsen_scaling.tex                # Hydro validity
├── energy_loss_path.tex               # Path length dependence

# New synthesis figures from phases 15a-17b
├── qhat_posterior_credible.tex        # Phase 15b: Bayesian q-hat
├── strangeness_phase_space.tex        # Phase 15c: Threshold map
├── reionization_qgp_analogy.tex       # Phase 15a: Analogy diagram
├── dark_energy_w_of_z.tex             # Phase 17a: BAO/SNe posterior
└── reionization_bubble_dynamics.tex   # Phase 17b: Bubble evolution
```

## pyproject.toml Dependency Groups

```toml
[project.optional-dependencies]
# Tier 1: QGP physics (core)
qgp = [
    "numpy>=1.24,<3.0",
    "scipy>=1.10",
    "matplotlib>=3.7",
]

# Tier 2: Cosmology extensions
cosmology = [
    "emcee>=3.1",           # MCMC for Bayesian inference
    "corner>=2.2",          # Posterior visualization
]

# Tier 3: GPU acceleration (optional)
gpu = [
    "cupy>=12.0",           # CuPy for GPU arrays
    "pycuda>=2022.2",       # Direct CUDA kernel support
]

# Development & testing (all tiers)
dev = [
    "ruff>=0.8",
    "mypy>=1.8",
    "pytest>=7.4",
    "pytest-cov>=4.1",
]

# All-in-one installation
all = [
    "qgp-light-ion[qgp,cosmology,gpu,dev]",
]
```

## CI/CD Matrix

GitHub Actions will test multiple configurations:

```yaml
strategy:
  matrix:
    tier: [qgp, cosmology, qgp+cosmology]
    gpu: [cpu, gpu-if-available]
    python: ["3.10", "3.11", "3.12"]
```

This ensures:
- Tier 1 works independently
- Tier 2 works independently
- Combined tiers work together
- GPU backend matches CPU results
- All Python versions supported

## Development Workflow

### Adding a New Feature

1. **Identify tier:** Is this QGP (Tier 1), Cosmology (Tier 2), or optimization (Tier 3)?
2. **Add source code:** Implement in src/{qgp,cosmology,gpu}/ with full type hints
3. **Add tests:** Create test_{module}.py in tests/{qgp,cosmology,gpu}/
4. **Add data generation:** Update src/{qgp,cosmology}/generate.py to include new outputs
5. **Add figure:** If visualizable, create figures/{name}.tex
6. **Update pyproject.toml:** Add any new dependencies to appropriate group
7. **Run full validation:**
   ```bash
   make lint-{tier}
   make test-{tier}
   make data-{tier}
   make -j4  # Full build
   ```
8. **Commit:** One commit per logical feature, with Conventional Commits format

### GPU Development

If adding GPU-accelerated code:
1. Write NumPy reference version first (always required)
2. Add CuPy equivalent in src/gpu/
3. Add conditional import logic in src/gpu/__init__.py
4. Write tests in tests/test_gpu/test_gpu_equivalence.py
5. Benchmark: compare performance within 1% on both CPU and GPU

## External Data Sources

### Tier 1 (QGP)

- **Experimental:** CMS HIN-25-008, HIN-25-009 (O-O/Ne-Ne collisions)
- **Experimental:** ALICE 2509.06428 (geometry-driven flow)
- **Lattice QCD:** T_c = 156.5 MeV, κ₂ = 0.012 (HotQCD 2019, Smecca 2025)
- **Nuclear Structure:** O-16 ab initio (arXiv:2507.05853), Ne-20 deformation (NNDC)

### Tier 2 (Cosmology)

- **Observational:** DESI DR2 BAO measurements (HEPData)
- **Observational:** Pantheon+ Type Ia supernovae (Scolnic et al. 2022)
- **Observational:** JWST z~13 ionization edges (Naidu et al. 2024, Finkelstein et al. 2024)
- **Lattice:** Equation of state from lattice QCD extended to finite density

## Key Metrics

### Code Quality

- **Type coverage:** 100% of src/ files (mypy strict mode)
- **Test coverage:** >90% for each tier
- **Lint compliance:** 0 ruff errors + 0 mypy errors
- **Documentation:** >50 lines per 100 lines of code

### Performance Targets

| Operation | CPU Target | GPU Target |
|-----------|-----------|-----------|
| Glauber MC (100k events) | <2 min | <10 s |
| Energy density (3D grid) | <5 min | <30 s |
| Femtoscopy grid (100×100) | <30 s | <5 s |
| BAO/SNe MCMC (1k samples) | <10 min | <5 min (if enabled) |
| Total Tier 1 + 2 data | ~25 min | ~10 min |
| Full build (data+figures+PDF) | ~35 min | ~20 min |

## Future Extensions

Potential additions that fit the three-tier model:

- **Tier 1b:** Flow harmonics from event-by-event fluctuations
- **Tier 2b:** Large-scale structure (EFTofLSS) as third cosmological scale
- **Tier 3b:** Distributed compute (Ray) for parameter sweeps

---

*Last updated: 2026-03-05 | Architecture design for phases 15-17*
