# Contributing to QGP Light-Ion Project

Welcome! This guide will help you contribute effectively to the QGP light-ion collisions project.

---

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Oichkatzelesfrettschen/qgp-light-ion.git
cd qgp-light-ion

# Run bootstrap script (sets up everything)
./scripts/bootstrap.sh

# Activate virtual environment
source .venv/bin/activate

# Verify setup
make test
make lint
```

The bootstrap script will:
- ✅ Check system dependencies (Python, Pandoc, LaTeX)
- ✅ Create virtual environment
- ✅ Install all development dependencies
- ✅ Set up pre-commit hooks

### 2. Development Workflow

```bash
# 1. Create a feature branch
git checkout -b feature/your-feature-name

# 2. Make changes
# ... edit files ...

# 3. Run tests locally
make test

# 4. Check code quality
make lint

# 5. Format code
make fmt

# 6. Commit changes (pre-commit hooks run automatically)
git add .
git commit -m "feat: add your feature description"

# 7. Push and create PR
git push origin feature/your-feature-name
```

---

## Project Structure

```
qgp-light-ion/
├── src/                     # Python source code
│   ├── qgp_physics.py       # Core physics models
│   ├── generate_*.py        # Data generation scripts
│   └── z3_validators.py     # Formal validation (optional)
│
├── figures/                 # TikZ/pgfplots figures
│   ├── accessible_colors.tex
│   └── *.tex                # Individual figure sources
│
├── tests/                   # Test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   ├── properties/          # Property-based tests
│   └── test_*.py            # Test modules
│
├── docs/                    # Documentation
│   ├── ARCHITECTURE_ANALYSIS.md
│   ├── TECHNICAL_DEBT.md
│   ├── FORMAL_METHODS.md
│   └── BUILD_SYSTEM.md
│
├── scripts/                 # Utility scripts
│   └── bootstrap.sh         # Environment setup
│
├── .github/workflows/       # CI/CD pipelines
│   └── ci.yml               # GitHub Actions config
│
├── build/                   # Generated files (gitignored)
├── data/                    # Generated data (gitignored)
├── Makefile                 # Build orchestration
├── pyproject.toml           # Python project config
├── requirements-dev.txt     # Development dependencies
└── .pre-commit-config.yaml  # Pre-commit hooks
```

---

## Code Style Guidelines

### Python

We follow **PEP 8** with some physics-specific adaptations:

```python
# ✅ Good
def woods_saxon(r: FloatArray, nucleus: Nucleus, theta: float = 0) -> FloatArray:
    """
    Woods-Saxon nuclear density profile.
    
    Args:
        r: Radius array [fm]
        nucleus: Nuclear parameters
        theta: Azimuthal angle [rad]
    
    Returns:
        Normalized density ρ(r,θ)/ρ₀
    
    References:
        de Vries et al., At. Data Nucl. Data Tables 36, 495 (1987)
    """
    R_eff = nucleus.R0 * (1 + nucleus.beta2 * Y20(theta))
    return 1 / (1 + np.exp((r - R_eff) / nucleus.a))

# ❌ Bad
def ws(r,n,t=0):  # Unclear names, no types, no docstring
    return 1/(1+np.exp((r-n.R0)/n.a))
```

**Key principles:**
- **Type hints**: All function signatures must have type hints
- **Docstrings**: Google-style docstrings for all public functions
- **Variable names**: Descriptive for code, physics notation OK in comments
- **Line length**: 100 characters maximum
- **Imports**: Organized (stdlib, third-party, local)

### LaTeX

```latex
% ✅ Good - Clear structure, comments
\begin{tikzpicture}
    % Nuclear density profiles for O-O, Ne-Ne, Pb-Pb
    \begin{axis}[
        xlabel={$r$ [fm]},
        ylabel={$\rho(r)/\rho_0$},
        legend pos=north east,
    ]
        \addplot[OOcolor, thick] table {data/nuclear_density_O.dat};
        \addplot[NeNecolor, thick] table {data/nuclear_density_Ne.dat};
        \legend{$^{16}$O, $^{20}$Ne}
    \end{axis}
\end{tikzpicture}

% ❌ Bad - No comments, magic numbers
\begin{tikzpicture}
\begin{axis}[xlabel={r},ylabel={y}]
\addplot[blue] table {data1.dat};
\end{axis}
\end{tikzpicture}
```

---

## Testing Requirements

### Writing Tests

All new code must have tests. We use **pytest** with multiple test categories:

#### 1. Unit Tests (`tests/unit/`)

Test individual functions in isolation:

```python
# tests/unit/test_woods_saxon.py
import numpy as np
from src.qgp_physics import woods_saxon, NUCLEI

def test_woods_saxon_positivity():
    """Woods-Saxon density must be non-negative."""
    r = np.linspace(0, 10, 100)
    nucleus = NUCLEI['O']
    density = woods_saxon(r, nucleus)
    assert np.all(density >= 0)

def test_woods_saxon_normalization():
    """Woods-Saxon density should be <= 1 (normalized to central)."""
    r = np.linspace(0, 10, 100)
    nucleus = NUCLEI['O']
    density = woods_saxon(r, nucleus)
    assert np.all(density <= 1.0)

def test_woods_saxon_monotonic():
    """Density should decrease with radius (for spherical nuclei)."""
    r = np.linspace(3, 10, 50)  # Beyond nuclear radius
    nucleus = NUCLEI['O']
    density = woods_saxon(r, nucleus)
    assert np.all(np.diff(density) <= 0)
```

#### 2. Integration Tests (`tests/integration/`)

Test end-to-end workflows:

```python
# tests/integration/test_data_pipeline.py
def test_full_data_generation(tmp_path):
    """Test complete data generation pipeline."""
    output_dir = tmp_path / "data"
    
    # Run data generation
    result = subprocess.run(
        ["python", "src/generate_comprehensive_data.py", 
         "--output-dir", str(output_dir)],
        capture_output=True
    )
    
    assert result.returncode == 0
    assert (output_dir / "RAA_OO.dat").exists()
    assert (output_dir / "flow_v2_OO.dat").exists()
```

#### 3. Property-Based Tests (`tests/properties/`)

Use **Hypothesis** to find edge cases:

```python
# tests/properties/test_physics_invariants.py
from hypothesis import given, strategies as st

@given(
    r=st.lists(st.floats(min_value=0, max_value=20), min_size=10).map(np.array),
    nucleus=st.sampled_from(['O', 'Ne', 'Pb'])
)
def test_woods_saxon_properties(r, nucleus):
    """Property-based test for Woods-Saxon."""
    nuc = NUCLEI[nucleus]
    density = woods_saxon(r, nuc)
    
    # Properties that must always hold
    assert np.all(density >= 0)          # Non-negative
    assert np.all(density <= 1)          # Normalized
    assert np.all(np.isfinite(density))  # No NaN/Inf
```

### Running Tests

```bash
# All tests
make test

# Specific test file
pytest tests/unit/test_woods_saxon.py -v

# With coverage
pytest --cov=src --cov-report=html

# Parallel execution (faster)
pytest -n auto

# Specific marker
pytest -m "not slow"
```

### Coverage Requirements

- **Minimum:** 60% overall coverage
- **Target:** >80% for src/qgp_physics.py
- **Critical functions:** 100% coverage required

Check coverage:
```bash
pytest --cov=src --cov-report=term-missing
```

---

## Commit Message Convention

We use **Conventional Commits** for clear history:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code restructuring without behavior change
- `perf`: Performance improvement
- `test`: Adding or modifying tests
- `build`: Build system or dependencies
- `ci`: CI configuration changes
- `chore`: Other changes (e.g., .gitignore)

### Examples

```
feat(physics): add Ne-20 deformation parameter

Implement prolate deformation (β₂ = 0.45) for Neon-20 based on
ATLAS 2025 measurement (arXiv:2509.05171).

- Update NUCLEI dict with beta2 field
- Modify woods_saxon() to handle deformation
- Add test for deformed density profile

Closes #42
```

```
fix(build): resolve LaTeX compilation error in figures

The energy_density_2d.tex figure failed to compile due to missing
pgfplots library. Added \usetikzlibrary{pgfplots.colormaps}.

Fixes #38
```

```
docs: add formal methods integration guide

Document Z3, TLA+, and Design by Contract usage for physics
constraint validation and build system verification.
```

### Pre-commit Hook

Commits are automatically validated by pre-commit hooks. To bypass (not recommended):
```bash
git commit --no-verify -m "emergency fix"
```

---

## Pull Request Process

### 1. Before Opening PR

- [ ] All tests pass locally (`make test`)
- [ ] Code is formatted (`make fmt`)
- [ ] Linters pass (`make lint`)
- [ ] Documentation is updated
- [ ] CHANGELOG.md entry added (if applicable)

### 2. PR Description Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Added unit tests
- [ ] Added integration tests
- [ ] Existing tests still pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Commented hard-to-understand areas
- [ ] Updated documentation
- [ ] No new warnings introduced
- [ ] Added tests for new features
- [ ] All tests pass

## Screenshots (if applicable)
For PDF/figure changes, attach before/after screenshots.

## Related Issues
Closes #123
```

### 3. Review Process

1. **Automated checks** run via GitHub Actions CI
2. **Manual review** by maintainers
3. **Changes requested** if needed
4. **Approval** from at least one maintainer
5. **Merge** (squash and merge preferred)

### 4. After Merge

- PR branch is automatically deleted
- CI builds and publishes artifacts
- Documentation is updated (if auto-deployed)

---

## Code Review Guidelines

### As a Reviewer

**Focus on:**
- ✅ Correctness of physics calculations
- ✅ Test coverage of new code
- ✅ Clear documentation
- ✅ Performance implications
- ✅ Security considerations

**Provide:**
- Specific, actionable feedback
- Praise for good practices
- Links to relevant documentation
- Suggestions, not demands (usually)

**Example feedback:**
```
Great addition of the Ne-20 deformation! A few suggestions:

1. Add reference to ATLAS paper in docstring:
   ```python
   # Neon-20 prolate deformation (ATLAS arXiv:2509.05171)
   "Ne": Nucleus(..., beta2=0.45)
   ```

2. Consider adding a test for extreme deformation:
   ```python
   def test_woods_saxon_extreme_deformation():
       # Test with β₂ = 0.6 (near physical limit)
   ```

3. Minor: Line 142 exceeds 100 chars, consider breaking:
   ```python
   # Before
   R_eff = nucleus.R0 * (1 + nucleus.beta2 * Y20(theta) + nucleus.beta3 * Y30(theta))
   
   # After
   R_eff = nucleus.R0 * (
       1 + nucleus.beta2 * Y20(theta) + nucleus.beta3 * Y30(theta)
   )
   ```

Otherwise LGTM! 🚀
```

### As an Author

**Respond to feedback:**
- Thank reviewers for their time
- Ask clarifying questions
- Implement suggested changes or explain why not
- Update PR description if scope changes
- Request re-review after major changes

---

## Development Tools

### Make Targets

```bash
# Build system
make              # Full build (data → figures → PDF)
make data         # Generate physics data only
make figures      # Compile figures only
make clean        # Remove generated files

# Quality checks
make test         # Run test suite
make test-quick   # Tests without data regeneration
make lint         # All linters (Python + LaTeX)
make lint-python  # Python linting only
make fmt          # Auto-format Python code

# Validation
make strict       # Build with strict error checking
make verify-data  # Check all data files exist

# Help
make help         # Show all available targets
```

### Useful Commands

```bash
# Type checking
mypy src tests

# Security audit
bandit -r src
safety check
pip-audit

# Complexity analysis
radon cc src -s -n C
radon mi src -s -n B

# Docstring coverage
interrogate src -vv

# Dead code detection
vulture src --min-confidence 80

# Profile data generation
py-spy record -o profile.svg -- python src/generate_comprehensive_data.py

# Memory profiling
memray run src/generate_comprehensive_data.py
memray flamegraph memray-*.bin
```

---

## Debugging Tips

### Common Issues

#### 1. LaTeX Compilation Fails

```bash
# Show full output
make VERBOSE=1

# Check specific figure
cd figures && pdflatex qcd_phase_diagram.tex

# Check for missing packages
make lint-latex
```

#### 2. Data Generation Errors

```bash
# Run single data generator
python src/generate_comprehensive_data.py --subset flow --verbose

# Check physics module
python -c "import src.qgp_physics as qgp; print(qgp.NUCLEI['O'])"

# Validate data files
make verify-data
```

#### 3. Test Failures

```bash
# Run with verbose output
pytest -vv

# Stop at first failure
pytest -x

# Run specific test
pytest tests/unit/test_woods_saxon.py::test_woods_saxon_positivity -v

# Show print statements
pytest -s
```

#### 4. Type Check Errors

```bash
# Check specific file
mypy src/qgp_physics.py

# Ignore missing imports (development)
mypy --ignore-missing-imports src/

# Show error codes
mypy --show-error-codes src/
```

---

## Performance Optimization

### Profiling

```bash
# CPU profiling
py-spy record -o profile.svg --native -- python src/generate_comprehensive_data.py

# View profile
open profile.svg  # macOS
xdg-open profile.svg  # Linux

# Memory profiling
memray run -o memory.bin src/generate_comprehensive_data.py
memray flamegraph memory.bin
```

### Optimization Checklist

- [ ] Profile before optimizing (don't guess!)
- [ ] Vectorize NumPy operations (avoid loops)
- [ ] Use Numba JIT for hot functions
- [ ] Cache expensive calculations
- [ ] Use appropriate data types (float32 vs float64)
- [ ] Profile after changes to verify improvement

---

## Documentation

### Code Documentation

**Docstring format (Google style):**

```python
def glauber_monte_carlo(
    nucleus_a: Nucleus,
    nucleus_b: Nucleus,
    b: float,
    n_events: int = 1000
) -> tuple[FloatArray, FloatArray, dict[str, float]]:
    """
    Glauber Monte Carlo collision geometry calculation.
    
    Simulates nucleon-nucleon collisions using Woods-Saxon nuclear
    density profiles and inelastic cross section σ_NN = 70 mb.
    
    Args:
        nucleus_a: First nucleus parameters
        nucleus_b: Second nucleus parameters
        b: Impact parameter [fm]
        n_events: Number of Monte Carlo events to generate
    
    Returns:
        Tuple containing:
        - N_part: Number of participants per event
        - N_coll: Number of binary collisions per event
        - eccentricities: Dict with ε₂, ε₃, ε₄ values
    
    Raises:
        ValueError: If impact parameter is negative
        ValueError: If n_events < 1
    
    Examples:
        >>> nuc_o = NUCLEI['O']
        >>> nuc_ne = NUCLEI['Ne']
        >>> npart, ncoll, ecc = glauber_monte_carlo(nuc_o, nuc_ne, b=3.0)
        >>> assert np.mean(npart) > 0
    
    References:
        Miller et al., Ann. Rev. Nucl. Part. Sci. 57, 205 (2007)
        arXiv:2507.05853 (O-16 geometry calibration)
    """
    ...
```

### Project Documentation

When adding features, update:
- `README.md` - User-facing documentation
- `CLAUDE.md` - Build system and architecture notes
- `docs/` - Technical documentation
- Inline code comments for complex physics

---

## Getting Help

### Resources

- **Architecture:** `docs/ARCHITECTURE_ANALYSIS.md`
- **Technical Debt:** `docs/TECHNICAL_DEBT.md`
- **Build System:** `docs/BUILD_SYSTEM.md`
- **Formal Methods:** `docs/FORMAL_METHODS.md`
- **Physics References:** `references.bib`

### Contact

- **Issues:** [GitHub Issues](https://github.com/Oichkatzelesfrettschen/qgp-light-ion/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Oichkatzelesfrettschen/qgp-light-ion/discussions)
- **Email:** See repository maintainers

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

## Acknowledgments

Thank you for contributing to advancing our understanding of QGP physics in light-ion collisions! 🎉

*This guide is a living document. Suggestions for improvements are welcome!*
