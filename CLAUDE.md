# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Author:** Deirikr Jaiusadastra Afrauthihinngreygaard

A scientific whitepaper exploring Quark-Gluon Plasma (QGP) formation in light-ion collisions (O-O, Ne-Ne) at the LHC. The project synthesizes 2025 experimental results from CMS and ALICE with theoretical frameworks.

**Output:** `build/qgp-light-ion.pdf` (~15 publication-ready figures)

## Build Commands

```bash
make              # Full build: data → figures → PDF
make -j4          # Parallel figure compilation (4 cores)
make data         # Generate physics data only
make figures      # Compile TikZ figures only
make clean        # Remove all generated files
make test         # Run data validation tests
make lint         # Run all linters (LaTeX + Python)
make strict       # Build and fail on any LaTeX errors/warnings
make VERBOSE=1    # Show full compiler output
make help         # Show all targets
```

## Architecture

Four-stage pipeline orchestrated by Make:

```
src/*.py (Python)     →  data/*.dat           # Stage 1: Physics data generation
figures/*.tex (TikZ)  →  build/figures/*.pdf  # Stage 2: Figure compilation
QGP_Light_Ion.md      →  build/body.tex       # Stage 3: Markdown → LaTeX (Pandoc)
qgp-light-ion.tex     →  build/qgp-light-ion.pdf  # Stage 4: Document assembly (latexmk)
```

**Why this design:**
- Python (numpy) for physics calculations requiring precise numerical methods
- TikZ/pgfplots figures compile independently for parallel builds and rapid iteration
- Markdown authoring is faster than raw LaTeX; Pandoc converts to LaTeX
- latexmk handles multi-pass compilation (references, bibliography) automatically

## Key Files

| File | Purpose |
|------|---------|
| `qgp-light-ion.tex` | Main LaTeX document (includes figures, defines structure) |
| `QGP_Light_Ion.md` | Source content (Markdown, converted to body.tex) |
| `references.bib` | BibTeX bibliography |
| `src/qgp_physics.py` | Core physics models (Woods-Saxon, Glauber, BDMPS-Z, flow) |
| `src/generate_comprehensive_data.py` | Multi-stage data generation driver |
| `figures/accessible_colors.tex` | Colorblind-safe palette (all figures import this) |

## Python Data Generation

The physics module (`src/qgp_physics.py`) implements:
- Woods-Saxon nuclear density profiles with deformation (β₂ for Ne-20)
- Glauber Monte Carlo geometry (N_part, N_coll, eccentricity)
- BDMPS-Z radiative energy loss (ΔE ∝ q̂L²)
- Flow-from-eccentricity hydrodynamic response (calibrated to 2025 CMS/ALICE)
- Canonical strangeness suppression (Bessel function formalism)

**To run single data generator:**
```bash
python3 src/generate_comprehensive_data.py --subset flow  # Options: phase, geometry, flow, jet, strangeness, spacetime
```

**To run physics module self-test:**
```bash
make test-physics
```

## Figure System

15 TikZ/pgfplots figures in `figures/`. Each compiles standalone:
```bash
cd figures && pdflatex qcd_phase_diagram.tex  # Single figure
```

**Convention for figure captions:**
- `[Data]` = Actual experimental measurements
- `[Model]` = Physics model predictions constrained by data
- `[Schematic]` = Illustrative/pedagogical (not quantitatively accurate)

**Color scheme:** Uses `accessible_colors.tex` for WCAG 2.1 AA compliance and colorblind safety. System-specific colors:
- `PbPbcolor` (#0072B2) - Deep blue
- `OOcolor` (#E69F00) - Amber/orange
- `NeNecolor` (#009E73) - Teal

## Key Physics Values (Reference) — December 2025

### QCD Phase Diagram
| Observable | Value | Source |
|------------|-------|--------|
| T_c (crossover) | 156.5 ± 1.5 MeV | HotQCD PLB 795 (2019) |
| κ₂ (curvature) | 0.012(2) | Bazavov et al. (2020); Smecca PRD 112 (2025) |
| CP exclusion | μ_B < 450 MeV (2σ) | Borsányi et al. PRD 112, L111505 (2025) |
| CEP (FRG consensus) | T=110, μ_B=630 MeV (±10%) | Fu et al. arXiv:2510.11270 |

### QGP Transport
| Observable | Value | Source |
|------------|-------|--------|
| η/s (QGP) | 0.08-0.16 | 1-2× KSS bound |
| q̂/T³ | 2-4 (at T=400 MeV) | JETSCAPE arXiv:2408.08247 |
| q̂/κ ratio | 0.25-0.8 | Xue et al. arXiv:2512.07169 |

### O-O and Ne-Ne Measurements (LHC 5.36 TeV)
| Observable | Value | Source |
|------------|-------|--------|
| R_AA min (O-O) | 0.69 ± 0.04 | CMS arXiv:2510.09864 |
| R_AA min (Ne-Ne) | ~0.65 | CMS estimate |
| v₂ ultracentral (O-O) | 0.061 | CMS arXiv:2510.02580 |
| v₂(Ne/O) ratio | ~1.08 | ALICE arXiv:2509.06428 |
| Ne-20 deformation β₂ | ~0.45 | ATLAS arXiv:2509.05171 |

### Nuclear Geometry
| Observable | Value | Source |
|------------|-------|--------|
| ¹⁶O radius | R=2.608 fm, a=0.513 fm | arXiv:2507.05853 |
| ²⁰Ne deformation | β₂ ≈ 0.45 (prolate) | TGlauberMC v3.3 |

## Testing & Linting

```bash
make test         # Full test suite (regenerates data)
make test-quick   # Tests without data regeneration
make lint         # All linters (chktex, lacheck, ruff)
make lint-python  # Python only (ruff)
make fmt          # Auto-format Python with ruff
```

Tests in `tests/test_data_generation.py` validate:
- Data file existence and format
- Physical constraints (R_AA > 0, |v_n| < 0.5, etc.)
- Build artifact integrity

## Terminology

Use physics-descriptive terminology:
- **Boost-invariant coordinates** (not "Bjorken coordinates")
- **Proper time** τ = √(t² - z²)
- **Spacetime rapidity** η_s = artanh(z/t)

See Appendix in main document for complete definitions.

## Data Provenance

All generated data files include provenance headers classifying them as:
- `MEASURED` - Experimental data from HEPData/papers
- `PREDICTED` - Model constrained by experimental inputs
- `SCHEMATIC` - Illustrative only

See `DATA_MANIFEST.md` for complete data inventory and sources.
