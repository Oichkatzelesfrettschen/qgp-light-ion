# Build System Architecture

## Overview

The QGP Light-Ion project uses a **Make-orchestrated multi-stage pipeline** that transforms Markdown content, Python-generated data, and TikZ/pgfplots figures into a final PDF document.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BUILD PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Stage 1: DATA GENERATION                                                  │
│   ┌─────────────────┐         ┌─────────────────┐                          │
│   │  Python Scripts │────────▶│   data/*.dat    │                          │
│   │     (src/)      │         │                 │                          │
│   └─────────────────┘         └────────┬────────┘                          │
│                                        │                                    │
│   Stage 2: MARKDOWN → LaTeX            │                                    │
│   ┌─────────────────┐         ┌────────▼────────┐                          │
│   │ QGP_Light_Ion.md│────────▶│ build/body.tex  │                          │
│   │    (Pandoc)     │         │                 │                          │
│   └─────────────────┘         └────────┬────────┘                          │
│                                        │                                    │
│   Stage 3: FIGURE COMPILATION          │                                    │
│   ┌─────────────────┐         ┌────────▼────────┐                          │
│   │  figures/*.tex  │────────▶│ build/figures/  │                          │
│   │   (pdflatex)    │         │    *.pdf        │                          │
│   └─────────────────┘         └────────┬────────┘                          │
│                                        │                                    │
│   Stage 4: DOCUMENT ASSEMBLY           │                                    │
│   ┌─────────────────┐         ┌────────▼────────┐    ┌──────────────────┐  │
│   │qgp-light-ion.tex│────────▶│    latexmk      │───▶│qgp-light-ion.pdf │  │
│   │  references.bib │         │    + bibtex     │    │                  │  │
│   └─────────────────┘         └─────────────────┘    └──────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Design Rationale

### Why Make + latexmk?

This project uses GNU Make for orchestration and latexmk for LaTeX compilation. This combination was chosen for:

1. **Mature Dependency Tracking**: Make's timestamp-based rebuilding ensures only changed components recompile, critical for projects with 15+ TikZ figures.

2. **Parallel Figure Compilation**: `make -j4` compiles independent figures simultaneously, reducing build time significantly.

3. **latexmk's Intelligence**: latexmk automatically determines how many passes are needed (for references, bibliography, etc.) without manual intervention.

4. **Portability**: Works on macOS, Linux, and Windows (with Make installed). No special tooling required beyond standard TeX distribution.

5. **Transparency**: Unlike GUI-based solutions, the Makefile explicitly shows the build process, making debugging straightforward.

### Why Separate Stages?

| Stage | Tool | Rationale |
|-------|------|-----------|
| **Data** | Python + NumPy | Physics calculations need precise numerical methods; Python's ecosystem (SciPy, NumPy) is ideal |
| **Content** | Pandoc | Markdown authoring is faster than raw LaTeX; Pandoc preserves semantic structure |
| **Figures** | pdflatex (standalone) | Each TikZ figure compiles independently, enabling rapid iteration and parallel builds |
| **Document** | latexmk | Handles complex multi-pass compilation (references, citations) automatically |

### Why Pandoc for Content?

The main document body is authored in **Markdown** (`QGP_Light_Ion.md`) and converted to LaTeX:

- **Faster authoring**: Markdown's minimal syntax speeds up writing
- **Version control friendly**: Markdown diffs are cleaner than LaTeX diffs
- **Flexibility**: Same source can generate HTML, DOCX, or other formats if needed
- **Semantic preservation**: Pandoc maintains document structure (headings, lists, math)

The LaTeX wrapper (`qgp-light-ion.tex`) handles:
- Document class and packages
- Title page and front matter
- Figure inclusion
- Bibliography formatting
- Custom LaTeX commands

## Alternative Build Systems Considered

### Tectonic

**What it is**: Modern, self-contained LaTeX compiler with automatic package downloading.

**Pros**:
- No need to install full TeX Live (~4GB)
- Deterministic builds (fixed package versions)
- Single binary, easy CI/CD setup

**Cons**:
- Based on XeTeX engine (not pdflatex)
- Package availability can lag behind TeX Live
- Less mature than latexmk for complex builds

**When to consider**: For CI/CD pipelines or Docker-based workflows where minimizing dependencies matters.

### Arara

**What it is**: Rule-based build system with per-file configuration via comments.

**Pros**:
- Configuration embedded in document header
- Very flexible for complex multi-tool workflows
- Can wrap latexmk for best of both worlds

**Cons**:
- Requires Java runtime
- Per-file configuration doesn't scale well for multi-file projects
- Less intuitive for Make users

**When to consider**: Single-file documents with specific tool sequences.

### latexrun

**What it is**: Lightweight alternative to latexmk with better diagnostics.

**Pros**:
- Cleaner error output
- Faster than latexmk in some cases (~30% speedup reported)
- Simpler codebase

**Cons**:
- Less widespread adoption
- Fewer configuration options
- May require manual intervention for edge cases

**When to consider**: If latexmk's output is too verbose or compilation is slow.

### Pure CMake / Ninja

**What it is**: Using CMake's LaTeX support or custom Ninja rules.

**Pros**:
- Better IDE integration (CLion, VS Code)
- Cross-platform project generation
- Fine-grained parallelism

**Cons**:
- CMake's LaTeX support is not first-class
- Significant setup overhead
- Overkill for most document projects

**When to consider**: Large documentation projects within software repositories already using CMake.

## Configuration Files

### Makefile

The Makefile defines:
- **Source files**: `MAIN_TEX`, `CONTENT_MD`, `BIB_FILE`
- **Figure lists**: `FIG_ORIGINAL`, `FIG_MULTIDIM`, `FIG_ADVANCED`
- **Build rules**: Dependencies and compilation commands
- **Convenience targets**: `data`, `figures`, `lint`, `test`

Key variables:
```makefile
MAIN_TEX    := qgp-light-ion.tex
CONTENT_MD  := QGP_Light_Ion.md
FINAL_PDF   := $(BUILD_DIR)/qgp-light-ion.pdf
```

### .latexmkrc

Latexmk configuration:
```perl
$pdf_mode = 1;           # Generate PDF directly
$out_dir = 'build';      # Output directory
$bibtex_use = 2;         # Use bibtex for bibliography
$max_repeat = 5;         # Maximum compilation passes
```

## Performance Optimization

### Current Optimizations

1. **Parallel figure compilation**: `make -j4 figures`
2. **Incremental builds**: Only changed files recompile
3. **Pregenerated data**: Physics data cached in `data/`

### Future Improvements

1. **TikZ externalization**: Cache compiled TikZ pictures
2. **PDF figure caching**: Store compiled figures between clean builds
3. **Container builds**: Docker/Podman for reproducible environments

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| Missing packages | `tlmgr install <package>` or full TeX Live install |
| Figure compilation fails | Run `make VERBOSE=1` to see full pdflatex output |
| Bibliography not appearing | Ensure `references.bib` exists and run `make clean && make` |
| Pandoc errors | Check Markdown syntax, especially math delimiters |

### Debug Mode

```bash
make VERBOSE=1          # Show full compiler output
make lint               # Analyze build log for warnings
make verify-data        # Check all data files exist
```

## References

- [latexmk documentation](https://mg.readthedocs.io/latexmk.html)
- [Pandoc User's Guide](https://pandoc.org/MANUAL.html)
- [TikZ/pgfplots manual](https://ctan.org/pkg/pgfplots)
- [GNU Make manual](https://www.gnu.org/software/make/manual/)
