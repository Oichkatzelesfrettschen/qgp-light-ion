# QGP Signatures in Light-Ion Collisions

A comprehensive synthesis of theoretical frameworks and experimental results from O-O and Ne-Ne collisions at the LHC, investigating quark-gluon plasma formation in small systems.

## Quick Start

```bash
make          # Build everything (data → figures → PDF)
make clean    # Remove all generated files
```

**Output:** `build/qgp-light-ion.pdf`

## Requirements

| Tool | Purpose | Installation |
|------|---------|--------------|
| Python 3 | Data generation | `brew install python` |
| NumPy | Physics calculations | `pip install numpy` |
| Pandoc | Markdown → LaTeX | `brew install pandoc` |
| LaTeX | Document compilation | TeX Live or MiKTeX |
| latexmk | Build automation | Included with TeX Live |

**Required LaTeX packages:** pgfplots, tikz, natbib, hyperref, geometry, amsmath

## Build System

### Architecture

The build system uses a four-stage pipeline:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Python Scripts │───▶│   data/*.dat    │───▶│                 │
│  (src/)         │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    │                 │
                                              │  build/figures/ │
┌─────────────────┐    ┌─────────────────┐    │     *.pdf       │
│  TikZ Sources   │───▶│   pdflatex      │───▶│                 │
│  (figures/)     │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐             │
│    Markdown     │───▶│   Pandoc        │───▶ build/body.tex
│  (*.md)         │    │                 │             │
└─────────────────┘    └─────────────────┘             │
                                                       ▼
┌───────────────────┐    ┌─────────────────┐    ┌───────────────────┐
│ qgp-light-ion.tex │───▶│   latexmk       │───▶│ qgp-light-ion.pdf │
│   references.bib  │    │   + bibtex      │    │                   │
└───────────────────┘    └─────────────────┘    └───────────────────┘
```

### Make Targets

| Target | Description |
|--------|-------------|
| `make` / `make all` | Full build (default) |
| `make data` | Generate physics data only |
| `make figures` | Compile figure PDFs only |
| `make lint` | Analyze build log for warnings |
| `make test` | Run data validation tests |
| `make clean` | Remove all generated files |
| `make help` | Show available targets |

### Why This Architecture?

1. **Separation of concerns**: Content authored in Markdown for readability; presentation in LaTeX for precision
2. **Reproducible data**: Python scripts generate consistent physics-based datasets
3. **Modular figures**: Each TikZ figure compiles independently for faster iteration
4. **Clean source tree**: All generated files go to `build/` and `data/` (gitignored)
5. **Dependency tracking**: Make rebuilds only what changed

### Parallel Builds

For faster compilation with multiple cores:

```bash
make -j4          # Compile 4 figures in parallel
make -j$(nproc)   # Use all available cores (Linux)
make -j$(sysctl -n hw.ncpu)  # Use all cores (macOS)
```

### Troubleshooting

```bash
make VERBOSE=1    # Show full compiler output
make lint         # Analyze build log for warnings
make verify-data  # Check all data files exist
```

For detailed build system documentation, see [`docs/BUILD_SYSTEM.md`](docs/BUILD_SYSTEM.md).

## Project Structure

```
.
├── qgp-light-ion.tex           # Main LaTeX document
├── QGP_Light_Ion.md            # Source content (Markdown)
├── references.bib              # Bibliography (BibTeX)
├── Makefile                    # Build automation
├── .latexmkrc                  # Latexmk configuration
│
├── figures/                    # TikZ/pgfplots figure sources
│   ├── accessible_colors.tex   # Colorblind-friendly palette
│   ├── qcd_phase_diagram.tex   # QCD phase diagram
│   ├── nuclear_structure.tex   # Nuclear density profiles
│   ├── RAA_multisystem.tex     # Jet quenching comparison
│   ├── flow_comprehensive.tex  # Anisotropic flow analysis
│   ├── strangeness_enhancement.tex
│   ├── bjorken_spacetime.tex   # Boost-invariant evolution
│   ├── glauber_event_display.tex
│   ├── energy_density_2d.tex
│   ├── femtoscopy_hbt.tex      # HBT correlations
│   └── direct_photon_spectra.tex
│
├── src/                        # Python data generation
│   ├── generate_comprehensive_data.py
│   └── qgp_physics.py          # Physics models
│
├── build/                      # Output directory (gitignored)
│   ├── qgp-light-ion.pdf       # Final document
│   ├── body.tex                # Converted Markdown
│   └── figures/                # Compiled figure PDFs
│
├── data/                       # Generated data (gitignored)
│   └── *.dat                   # Physics datasets
│
└── references/                 # Source papers (Git LFS)
    └── *.pdf
```

## Figures

The document includes 10 physics-based visualizations:

| Figure | Content |
|--------|---------|
| QCD Phase Diagram | Temperature vs. baryon chemical potential |
| Nuclear Structure | Density profiles for O, Ne, Pb |
| R_AA Multi-system | Jet quenching across collision systems |
| Flow Comprehensive | v₂, v₃ analysis with geometry |
| Strangeness Enhancement | Canonical suppression effects |
| Bjorken Spacetime | Boost-invariant evolution |
| Glauber Event Display | Participant/spectator geometry |
| Energy Density 2D | Initial-state hot spots |
| Femtoscopy/HBT | Two-particle correlations |
| Direct Photon Spectra | Thermal radiation signatures |

All figures use an accessibility-optimized color scheme defined in `figures/accessible_colors.tex`.

## Key Physics Topics

- **QGP (Quark-Gluon Plasma)**: Deconfined state of quarks and gluons
- **R_AA**: Nuclear modification factor (jet quenching signature)
- **v_n**: Anisotropic flow coefficients (collective behavior)
- **Strangeness enhancement**: Enhanced strange hadron production
- **HBT correlations**: Femtoscopic source size measurements
- **Direct photons**: Thermal radiation from QGP

## Primary References

### Theory Papers
| arXiv | Topic |
|-------|-------|
| [2504.02527](https://arxiv.org/abs/2504.02527) | Strangeness in small systems (ALICE) |
| [2503.02677](https://arxiv.org/abs/2503.02677) | Canonical strangeness treatment |
| [2205.02321](https://arxiv.org/abs/2205.02321) | Hydrodynamics in small systems |
| [2306.06047](https://arxiv.org/abs/2306.06047) | Nuclear structure effects |

### Experimental Context
- ALICE, CMS, ATLAS results from LHC O-O and Ne-Ne runs
- Lattice QCD results for T_c = 156.5 ± 1.5 MeV
- RHIC and SPS reference measurements

## License

Academic/research use.
