#!/bin/bash
# =============================================================================
# Bootstrap Script for QGP Light-Ion Development Environment
# =============================================================================
# This script sets up a complete development environment with all tools,
# dependencies, and configurations needed for contributing to this project.
#
# Usage: ./scripts/bootstrap.sh

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
info() { echo -e "${BLUE}ℹ${NC} $*"; }
success() { echo -e "${GREEN}✓${NC} $*"; }
warning() { echo -e "${YELLOW}⚠${NC} $*"; }
error() { echo -e "${RED}✗${NC} $*"; }

# =============================================================================
# System Requirements Check
# =============================================================================

info "Checking system requirements..."

# Check Python 3.10+
if ! command -v python3 &> /dev/null; then
    error "Python 3 not found. Please install Python 3.10 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    error "Python $PYTHON_VERSION found, but 3.10+ required"
    exit 1
fi
success "Python $PYTHON_VERSION detected"

# Check Pandoc
if ! command -v pandoc &> /dev/null; then
    warning "Pandoc not found. Required for Markdown→LaTeX conversion."
    warning "Install: brew install pandoc (macOS) or apt install pandoc (Linux)"
else
    success "Pandoc $(pandoc --version | head -1 | awk '{print $2}') detected"
fi

# Check LaTeX
if ! command -v pdflatex &> /dev/null; then
    warning "LaTeX (pdflatex) not found. Required for PDF generation."
    warning "Install: TeX Live (https://www.tug.org/texlive/)"
else
    success "LaTeX detected"
fi

# Check latexmk
if ! command -v latexmk &> /dev/null; then
    warning "latexmk not found. Included with TeX Live."
else
    success "latexmk detected"
fi

# Check Make
if ! command -v make &> /dev/null; then
    error "GNU Make not found. Required for build system."
    exit 1
fi
success "Make $(make --version | head -1 | awk '{print $3}') detected"

# Check Git
if ! command -v git &> /dev/null; then
    error "Git not found. Required for version control."
    exit 1
fi
success "Git $(git --version | awk '{print $3}') detected"

# =============================================================================
# Virtual Environment Setup
# =============================================================================

info "Setting up Python virtual environment..."

if [ -d ".venv" ]; then
    warning "Virtual environment already exists at .venv"
    read -p "Remove and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .venv
        info "Removed existing .venv"
    else
        info "Using existing virtual environment"
    fi
fi

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    success "Created virtual environment at .venv"
fi

# Activate virtual environment
source .venv/bin/activate
success "Activated virtual environment"

# =============================================================================
# Python Dependencies Installation
# =============================================================================

info "Installing Python dependencies..."

# Upgrade pip, setuptools, wheel
pip install --quiet --upgrade pip setuptools wheel
success "Upgraded pip, setuptools, wheel"

# Install project with development dependencies
if [ -f "pyproject.toml" ]; then
    pip install --quiet -e ".[dev,docs]"
    success "Installed project with dev dependencies"
else
    error "pyproject.toml not found"
    exit 1
fi

# =============================================================================
# Development Tools Setup
# =============================================================================

info "Setting up development tools..."

# Install pre-commit hooks if available
if [ -f ".pre-commit-config.yaml" ]; then
    if command -v pre-commit &> /dev/null; then
        pre-commit install
        success "Installed pre-commit hooks"
    else
        warning "pre-commit not found, skipping hook installation"
    fi
fi

# =============================================================================
# Validation
# =============================================================================

info "Validating installation..."

# Check that critical Python packages are importable
python3 -c "import numpy; import scipy; import matplotlib" 2>/dev/null
if [ $? -eq 0 ]; then
    success "Core scientific packages (numpy, scipy, matplotlib) working"
else
    error "Failed to import core scientific packages"
    exit 1
fi

# Check development tools
TOOLS_MISSING=0

for tool in ruff mypy pytest bandit; do
    if command -v $tool &> /dev/null; then
        success "$tool available"
    else
        error "$tool not available"
        TOOLS_MISSING=1
    fi
done

if [ $TOOLS_MISSING -eq 1 ]; then
    warning "Some development tools missing. Run: pip install -e '.[dev]'"
fi

# =============================================================================
# Environment Information
# =============================================================================

echo ""
info "Environment setup complete!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  QGP Light-Ion Development Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Python:      $(python3 --version)"
echo "Virtual env: .venv (activated)"
echo "Project:     $(pip show qgp-light-ion | grep Version || echo 'development')"
echo ""
echo "Quick Commands:"
echo "  make              # Full build (data → figures → PDF)"
echo "  make data         # Generate physics data only"
echo "  make test         # Run test suite"
echo "  make lint         # Run all linters"
echo "  make help         # Show all available targets"
echo ""
echo "Development Workflow:"
echo "  1. Activate environment:  source .venv/bin/activate"
echo "  2. Make changes to code"
echo "  3. Run tests:             make test"
echo "  4. Check code quality:    make lint"
echo "  5. Build PDF:             make"
echo ""
echo "For more information, see:"
echo "  - CLAUDE.md              # Build system and architecture"
echo "  - docs/ARCHITECTURE_ANALYSIS.md  # Technical analysis"
echo "  - README.md              # Project overview"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# =============================================================================
# Next Steps
# =============================================================================

if ! command -v pandoc &> /dev/null || ! command -v pdflatex &> /dev/null; then
    echo ""
    warning "Optional tools missing for full PDF generation:"
    if ! command -v pandoc &> /dev/null; then
        echo "  - Pandoc: brew install pandoc (macOS) or apt install pandoc (Linux)"
    fi
    if ! command -v pdflatex &> /dev/null; then
        echo "  - LaTeX: Download TeX Live from https://www.tug.org/texlive/"
    fi
    echo ""
fi

success "Bootstrap complete! Virtual environment is activated."
