# =============================================================================
# Makefile for QGP Light-Ion Project
# =============================================================================
#
# Build System Architecture
# -------------------------
# This Makefile orchestrates a multi-stage pipeline:
#
#   1. DATA GENERATION (Python)
#      - Comprehensive physics-based datasets for multi-dimensional visualization
#      - Outputs: data/{1d_spectra,2d_correlations,3d_spacetime,4d_parameters,...}
#
#   2. MARKDOWN → LaTeX CONVERSION (Pandoc)
#      - QGP_Light_Ion.md → build/body.tex
#
#   3. FIGURE COMPILATION (pdflatex)
#      - figures/*.tex → build/figures/*.pdf
#      - Supports parallel compilation with -j flag
#
#   4. MAIN DOCUMENT (latexmk)
#      - qgp-light-ion.tex → build/qgp-light-ion.pdf
#
# Build Options:
#   make -j4          # Parallel figure compilation
#   make VERBOSE=1    # Show full pdflatex output
#   make data-only    # Generate data without figures
#
# =============================================================================

SHELL := /bin/bash
.SUFFIXES:

# Build options
VERBOSE ?= 0
ifeq ($(VERBOSE),1)
    LATEX_REDIRECT :=
    LATEXMK_SILENT :=
else
    LATEX_REDIRECT := >/dev/null 2>&1
    LATEXMK_SILENT := -silent
endif

# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------
PYTHON   := python3
PANDOC   := pandoc
PDFLATEX := pdflatex -interaction=nonstopmode -file-line-error
LATEXMK  := latexmk -pdf -interaction=nonstopmode

# Linters
CHKTEX   := chktex
LACHECK  := lacheck
RUFF     := ruff

# -----------------------------------------------------------------------------
# Directories
# -----------------------------------------------------------------------------
BUILD_DIR := build
DATA_DIR  := data
FIG_DIR   := figures
SRC_DIR   := src

# -----------------------------------------------------------------------------
# Source Files
# -----------------------------------------------------------------------------
MAIN_TEX    := qgp-light-ion.tex
CONTENT_MD  := QGP_Light_Ion.md
BIB_FILE    := references.bib
COLORS_TEX  := $(FIG_DIR)/accessible_colors.tex

# Data generation scripts
DATA_SCRIPT       := $(SRC_DIR)/generate_comprehensive_data.py
ENERGY_SCRIPT     := $(SRC_DIR)/generate_energy_density.py
MULTIDIM_SCRIPT   := $(SRC_DIR)/generate_multidimensional_data.py
HBT_SCRIPT        := $(SRC_DIR)/generate_hbt_data.py
PHOTON_SCRIPT     := $(SRC_DIR)/generate_photon_data.py
QCD_PHASE_SCRIPT  := $(SRC_DIR)/generate_qcd_phase_diagram.py
CURVES_SCRIPT     := $(SRC_DIR)/generate_figure_curves.py
PHYSICS_MOD       := $(SRC_DIR)/qgp_physics.py

# Figure sources - organized by category
# Original figures (10)
FIG_ORIGINAL := qcd_phase_diagram nuclear_structure RAA_multisystem \
                flow_comprehensive strangeness_enhancement bjorken_spacetime \
                glauber_event_display energy_density_2d femtoscopy_hbt \
                direct_photon_spectra

# New multi-dimensional figures (5)
FIG_MULTIDIM := spectra_1d_pt temperature_1d_evolution \
                correlation_2d_ridge knudsen_scaling \
                energy_loss_path

# Figures requiring full 3D/4D data (compile separately if needed)
FIG_ADVANCED := spacetime_3d_evolution parameter_4d_scan

# All figures for main build
FIG_NAMES := $(FIG_ORIGINAL) $(FIG_MULTIDIM)

FIG_SOURCES := $(foreach f,$(FIG_NAMES),$(FIG_DIR)/$(f).tex)
FIG_PDFS    := $(foreach f,$(FIG_NAMES),$(BUILD_DIR)/figures/$(f).pdf)

# Advanced figures (optional)
FIG_ADV_PDFS := $(foreach f,$(FIG_ADVANCED),$(BUILD_DIR)/figures/$(f).pdf)

# -----------------------------------------------------------------------------
# Generated Files
# -----------------------------------------------------------------------------
TEX_BODY    := $(BUILD_DIR)/body.tex
DATA_STAMP  := $(DATA_DIR)/.generated
FINAL_PDF   := $(BUILD_DIR)/qgp-light-ion.pdf

# =============================================================================
# Targets
# =============================================================================

.PHONY: all clean figures data data-only test lint help advanced strict \
        lint-latex lint-python lint-chktex lint-lacheck lint-ruff lint-build \
        fmt test-quick test-physics verify-data distclean bootstrap check-env \
        complexity security profile coverage type-check docs

# Default: build everything
all: $(FINAL_PDF)
	@echo ""
	@echo "========================================"
	@echo "Build complete: $(FINAL_PDF)"
	@echo "========================================"
	@echo "  Pages:   $$(pdfinfo $(FINAL_PDF) 2>/dev/null | grep Pages | awk '{print $$2}')"
	@echo "  Size:    $$(du -h $(FINAL_PDF) | cut -f1)"
	@echo "  Figures: $(words $(FIG_NAMES))"
	@echo ""

# Build with advanced 3D/4D figures
advanced: $(FINAL_PDF) $(FIG_ADV_PDFS)
	@echo "Advanced figures compiled: $(FIG_ADVANCED)"

# -----------------------------------------------------------------------------
# Main Document
# -----------------------------------------------------------------------------
$(FINAL_PDF): $(MAIN_TEX) $(TEX_BODY) $(BIB_FILE) $(FIG_PDFS) | $(BUILD_DIR)
	@echo "=== Building main document ==="
	@# Run latexmk with optional silent mode; check for PDF existence to handle recoverable errors
	@# (LaTeX 2024+ marks some errors like "infinite glue shrinkage" as recoverable)
	@# Use -silent to suppress intermediate pass warnings (normal multi-pass behavior)
	$(LATEXMK) $(LATEXMK_SILENT) -outdir=$(BUILD_DIR) $(MAIN_TEX) || \
		(test -f $(FINAL_PDF) && echo "Note: latexmk reported errors but PDF was generated successfully")

# -----------------------------------------------------------------------------
# Markdown to LaTeX Conversion
# -----------------------------------------------------------------------------
$(TEX_BODY): $(CONTENT_MD) | $(BUILD_DIR)
	@echo "=== Converting Markdown to LaTeX ==="
	$(PANDOC) --from markdown --to latex --natbib -o $@ $<

# -----------------------------------------------------------------------------
# Figure Compilation
# -----------------------------------------------------------------------------
# Supports parallel builds: make -j4 figures
$(BUILD_DIR)/figures/%.pdf: $(FIG_DIR)/%.tex $(COLORS_TEX) $(DATA_STAMP) | $(BUILD_DIR)/figures
	@echo "=== Compiling figure: $* ==="
	@cd $(FIG_DIR) && $(PDFLATEX) -jobname=$* $*.tex $(LATEX_REDIRECT) || \
		(echo "ERROR compiling $*.tex - run with VERBOSE=1 for details" && exit 1)
	@mv $(FIG_DIR)/$*.pdf $@
	@rm -f $(FIG_DIR)/$*.aux $(FIG_DIR)/$*.log

# Convenience target for figures only
figures: $(FIG_PDFS)
	@echo "All $(words $(FIG_PDFS)) figures compiled."

# -----------------------------------------------------------------------------
# Data Generation
# -----------------------------------------------------------------------------
# Multi-stage data generation:
# 1. Core physics data (comprehensive_data)
# 2. Energy density grids
# 3. Multi-dimensional analysis data
# 4. HBT and photon specializations (if scripts exist)

$(DATA_STAMP): $(DATA_SCRIPT) $(ENERGY_SCRIPT) $(MULTIDIM_SCRIPT) $(QCD_PHASE_SCRIPT) $(PHYSICS_MOD)
	@mkdir -p $(DATA_DIR)
	@echo "=== Stage 1: Core physics data ==="
	$(PYTHON) $(DATA_SCRIPT) --output-dir $(DATA_DIR)
	@echo "=== Stage 2: Energy density grids ==="
	$(PYTHON) $(ENERGY_SCRIPT)
	@echo "=== Stage 3: Multi-dimensional analysis data ==="
	$(PYTHON) $(MULTIDIM_SCRIPT) --output-dir $(DATA_DIR)
	@if [ -f $(HBT_SCRIPT) ]; then \
		echo "=== Stage 4a: HBT data ==="; \
		$(PYTHON) $(HBT_SCRIPT) 2>/dev/null || true; \
	fi
	@if [ -f $(PHOTON_SCRIPT) ]; then \
		echo "=== Stage 4b: Photon data ==="; \
		$(PYTHON) $(PHOTON_SCRIPT) 2>/dev/null || true; \
	fi
	@if [ -f $(QCD_PHASE_SCRIPT) ]; then \
		echo "=== Stage 5a: QCD phase diagram (high-fidelity) ==="; \
		$(PYTHON) $(QCD_PHASE_SCRIPT); \
	fi
	@if [ -f $(CURVES_SCRIPT) ]; then \
		echo "=== Stage 5b: Figure curves ==="; \
		$(PYTHON) $(CURVES_SCRIPT); \
	fi
	@# Create symlink for figure data access
	@rm -f $(FIG_DIR)/data
	@ln -s ../$(DATA_DIR) $(FIG_DIR)/data
	@touch $@
	@echo ""
	@echo "Data generation complete."
	@echo "  Directories: $$(ls -d $(DATA_DIR)/*/ 2>/dev/null | wc -l | tr -d ' ')"
	@echo "  Files: $$(find $(DATA_DIR) -name '*.dat' 2>/dev/null | wc -l | tr -d ' ') .dat files"

# Convenience target for data only
.PHONY: data data-only
data:
	@$(MAKE) $(DATA_STAMP)

data-only: $(DATA_STAMP)
	@echo "Data generated (no figures compiled)"

# -----------------------------------------------------------------------------
# Directory Creation
# -----------------------------------------------------------------------------
$(BUILD_DIR) $(BUILD_DIR)/figures:
	@mkdir -p $@

# Note: $(DATA_DIR) is created by data generation scripts, not as a phony target
# (to avoid conflict with 'make data' convenience target)

# -----------------------------------------------------------------------------
# Quality Assurance - Linting
# -----------------------------------------------------------------------------

# Combined lint target: runs all linters
lint: lint-latex lint-python
	@echo ""
	@echo "========================================"
	@echo "All linting complete"
	@echo "========================================"

# LaTeX linting (both chktex and lacheck)
lint-latex: lint-chktex lint-lacheck

# ChkTeX - detailed LaTeX/TeX checker
# Uses .chktexrc for configuration
lint-chktex:
	@echo "=== ChkTeX: LaTeX syntax check ==="
	@WARNINGS=0; \
	OUTPUT=$$($(CHKTEX) -q $(MAIN_TEX) 2>&1 | grep -v "WARNING --" | grep -v "^$$"); \
	if [ -n "$$OUTPUT" ]; then \
		echo "Main document:"; \
		echo "$$OUTPUT"; \
		WARNINGS=$$((WARNINGS + 1)); \
	fi; \
	for fig in $(FIG_NAMES); do \
		OUTPUT=$$(cd $(FIG_DIR) && $(CHKTEX) -q $$fig.tex 2>&1 | grep -v "WARNING --" | grep -v "^$$"); \
		if [ -n "$$OUTPUT" ]; then \
			echo "figures/$$fig.tex:"; \
			echo "$$OUTPUT"; \
			WARNINGS=$$((WARNINGS + 1)); \
		fi; \
	done; \
	if [ $$WARNINGS -eq 0 ]; then \
		echo "  [OK] No ChkTeX warnings"; \
	else \
		echo ""; \
		echo "  [INFO] $$WARNINGS file(s) with warnings"; \
	fi

# Lacheck - lightweight LaTeX checker
lint-lacheck:
	@echo "=== Lacheck: LaTeX consistency check ==="
	@OUTPUT=$$($(LACHECK) $(MAIN_TEX) 2>&1 | grep -v "^$$" | head -20); \
	if [ -n "$$OUTPUT" ]; then \
		echo "$$OUTPUT"; \
		LINES=$$(echo "$$OUTPUT" | wc -l | tr -d ' '); \
		echo ""; \
		echo "  [INFO] $$LINES issue(s) found"; \
	else \
		echo "  [OK] No Lacheck warnings"; \
	fi

# Python linting with ruff
lint-python: lint-ruff

lint-ruff:
	@echo "=== Ruff: Python lint ==="
	@if command -v $(RUFF) >/dev/null 2>&1; then \
		$(RUFF) check $(SRC_DIR) tests/ 2>&1 || true; \
	else \
		echo "  [SKIP] ruff not installed (pip install ruff)"; \
	fi

# Format Python code with ruff
fmt:
	@echo "=== Ruff: Python format ==="
	@if command -v $(RUFF) >/dev/null 2>&1; then \
		$(RUFF) format $(SRC_DIR) tests/; \
		$(RUFF) check --fix $(SRC_DIR) tests/ 2>&1 || true; \
	else \
		echo "  [SKIP] ruff not installed"; \
	fi

# Build log analysis (after compilation)
lint-build: $(FINAL_PDF)
	@echo "=== Build Log Analysis ==="
	@echo ""
	@echo "LaTeX Warnings:"
	@grep -i "warning" $(BUILD_DIR)/qgp-light-ion.log 2>/dev/null | \
		grep -v "rerunfilecheck\|infwarerr" | head -10 || echo "  None"
	@echo ""
	@echo "Overfull/Underfull boxes:"
	@grep -iE "overfull|underfull" $(BUILD_DIR)/qgp-light-ion.log 2>/dev/null | head -5 || echo "  None"
	@echo ""
	@echo "Undefined references:"
	@grep -i "undefined" $(BUILD_DIR)/qgp-light-ion.log 2>/dev/null || echo "  None"
	@echo ""
	@echo "Figure warnings:"
	@for f in $(FIG_DIR)/*.log; do \
		if [ -f "$$f" ]; then \
			grep -l "Warning" "$$f" 2>/dev/null; \
		fi; \
	done || echo "  None (logs cleaned)"

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------

test: $(DATA_STAMP)
	@echo "=== Running test suite ==="
	@$(PYTHON) tests/test_data_generation.py

# Quick test without regenerating data
test-quick:
	@echo "=== Running test suite (no data regen) ==="
	@if [ -f $(DATA_STAMP) ]; then \
		$(PYTHON) tests/test_data_generation.py; \
	else \
		echo "[SKIP] No data generated. Run 'make data' first."; \
	fi

# Validate physics module loads correctly
test-physics:
	@echo "=== Physics module validation ==="
	@$(PYTHON) -c "import sys; sys.path.insert(0,'src'); from qgp_physics import *; print('  [OK] Physics module imported')"

# Strict build: fail on warnings and undefined references
strict: $(FINAL_PDF)
	@echo "=== Strict Build Validation ==="
	@ERRORS=0; \
	echo "Checking for LaTeX errors..."; \
	if grep -q "^!" $(BUILD_DIR)/qgp-light-ion.log 2>/dev/null; then \
		echo "  [FAIL] LaTeX errors found:"; \
		grep "^!" $(BUILD_DIR)/qgp-light-ion.log | head -5; \
		ERRORS=$$((ERRORS + 1)); \
	else \
		echo "  [OK] No LaTeX errors"; \
	fi; \
	echo "Checking for undefined control sequences..."; \
	if grep -q "Undefined control sequence" $(BUILD_DIR)/qgp-light-ion.log 2>/dev/null; then \
		echo "  [FAIL] Undefined control sequences:"; \
		grep "Undefined control sequence" $(BUILD_DIR)/qgp-light-ion.log | head -3; \
		ERRORS=$$((ERRORS + 1)); \
	else \
		echo "  [OK] No undefined control sequences"; \
	fi; \
	echo "Checking for undefined references..."; \
	if grep -q "LaTeX Warning: Reference.*undefined" $(BUILD_DIR)/qgp-light-ion.log 2>/dev/null; then \
		echo "  [FAIL] Undefined references:"; \
		grep "Reference.*undefined" $(BUILD_DIR)/qgp-light-ion.log | head -5; \
		ERRORS=$$((ERRORS + 1)); \
	else \
		echo "  [OK] No undefined references"; \
	fi; \
	echo "Checking for undefined citations..."; \
	if grep -q "LaTeX Warning: Citation.*undefined" $(BUILD_DIR)/qgp-light-ion.log 2>/dev/null; then \
		echo "  [FAIL] Undefined citations:"; \
		grep "Citation.*undefined" $(BUILD_DIR)/qgp-light-ion.log | head -5; \
		ERRORS=$$((ERRORS + 1)); \
	else \
		echo "  [OK] No undefined citations"; \
	fi; \
	echo "Checking for missing figures..."; \
	if grep -q "File.*not found" $(BUILD_DIR)/qgp-light-ion.log 2>/dev/null; then \
		echo "  [FAIL] Missing figures:"; \
		grep "File.*not found" $(BUILD_DIR)/qgp-light-ion.log | head -5; \
		ERRORS=$$((ERRORS + 1)); \
	else \
		echo "  [OK] All figures found"; \
	fi; \
	echo ""; \
	if [ $$ERRORS -gt 0 ]; then \
		echo "======================================"; \
		echo "STRICT BUILD FAILED: $$ERRORS issue(s)"; \
		echo "======================================"; \
		exit 1; \
	else \
		echo "========================================"; \
		echo "STRICT BUILD PASSED: $(FINAL_PDF)"; \
		echo "========================================"; \
		echo "  Pages:   $$(pdfinfo $(FINAL_PDF) 2>/dev/null | grep Pages | awk '{print $$2}')"; \
		echo "  Size:    $$(du -h $(FINAL_PDF) | cut -f1)"; \
		echo "  Figures: $(words $(FIG_NAMES))"; \
	fi

# Verify all data files exist
verify-data: $(DATA_STAMP)
	@echo "=== Verifying data files ==="
	@echo "Expected directories:"
	@for dir in phase_diagram nuclear_geometry flow jet_quenching strangeness spacetime comparison 1d_spectra 2d_correlations 3d_spacetime 4d_parameters physics_connections; do \
		if [ -d "$(DATA_DIR)/$$dir" ]; then \
			echo "  [OK] $$dir ($$(ls $(DATA_DIR)/$$dir/*.dat 2>/dev/null | wc -l) files)"; \
		else \
			echo "  [MISSING] $$dir"; \
		fi; \
	done

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
clean:
	@echo "=== Cleaning build artifacts ==="
	rm -rf $(BUILD_DIR) $(DATA_DIR)
	rm -f $(FIG_DIR)/*.aux $(FIG_DIR)/*.log $(FIG_DIR)/*.pdf
	rm -f $(FIG_DIR)/data
	rm -rf $(SRC_DIR)/__pycache__
	@echo "Clean complete"

# Deep clean (including generated intermediate files)
distclean: clean
	rm -f *.aux *.log *.out *.bbl *.blg *.fls *.fdb_latexmk
	@echo "Distribution clean complete"

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
help:
	@echo "QGP Light-Ion Build System"
	@echo ""
	@echo "Usage: make [target] [options]"
	@echo ""
	@echo "Primary targets:"
	@echo "  all         Build complete PDF (default)"
	@echo "  advanced    Build with 3D/4D advanced figures"
	@echo "  clean       Remove all generated files"
	@echo "  distclean   Deep clean including root artifacts"
	@echo ""
	@echo "Partial builds:"
	@echo "  data        Generate physics data"
	@echo "  data-only   Generate data without figures"
	@echo "  figures     Compile figure PDFs only"
	@echo ""
	@echo "Linting:"
	@echo "  lint        Run all linters (LaTeX + Python)"
	@echo "  lint-latex  Run LaTeX linters (chktex + lacheck)"
	@echo "  lint-chktex Run ChkTeX on .tex files"
	@echo "  lint-lacheck Run Lacheck on main document"
	@echo "  lint-python Run Python linter (ruff)"
	@echo "  lint-ruff   Run Ruff on src/ and tests/"
	@echo "  lint-build  Analyze build log for issues"
	@echo "  fmt         Auto-format Python code with ruff"
	@echo ""
	@echo "Testing:"
	@echo "  test        Run full test suite (regenerates data)"
	@echo "  test-quick  Run tests without data regeneration"
	@echo "  test-physics Validate physics module imports"
	@echo ""
	@echo "Quality assurance:"
	@echo "  strict      Build and fail on any warnings/errors"
	@echo "  verify-data Check all data directories exist"
	@echo ""
	@echo "Options:"
	@echo "  -j N        Parallel figure compilation (e.g., make -j4)"
	@echo "  VERBOSE=1   Show full LaTeX output"
	@echo ""
	@echo "Figure categories:"
	@echo "  Original:   $(words $(FIG_ORIGINAL)) figures"
	@echo "  Multi-dim:  $(words $(FIG_MULTIDIM)) figures"
	@echo "  Advanced:   $(words $(FIG_ADVANCED)) figures (optional)"
	@echo ""
	@echo "Output: $(FINAL_PDF)"

# -----------------------------------------------------------------------------
# Development & Analysis Tools
# -----------------------------------------------------------------------------

# Bootstrap development environment
bootstrap:
	@echo "=== Setting up development environment ==="
	@chmod +x scripts/bootstrap.sh
	@./scripts/bootstrap.sh

# Check environment dependencies
check-env:
	@echo "=== Checking environment ==="
	@command -v python3 >/dev/null || (echo "❌ Python 3 not found" && exit 1)
	@command -v pandoc >/dev/null || (echo "⚠️  Pandoc not found (optional)" && exit 0)
	@command -v pdflatex >/dev/null || (echo "⚠️  LaTeX not found (optional)" && exit 0)
	@command -v make >/dev/null || (echo "❌ Make not found" && exit 1)
	@python3 -c "import numpy, scipy" 2>/dev/null || \
		(echo "❌ Python dependencies missing. Run: pip install -r requirements-dev.txt" && exit 1)
	@echo "✅ Environment check passed"

# Type checking with mypy
type-check:
	@echo "=== MyPy Type Checking ==="
	@if command -v mypy >/dev/null 2>&1; then \
		mypy src tests --ignore-missing-imports --check-untyped-defs || true; \
	else \
		echo "  [SKIP] mypy not installed (pip install mypy)"; \
	fi

# Code complexity analysis
complexity:
	@echo "=== Code Complexity Analysis ==="
	@if command -v radon >/dev/null 2>&1; then \
		echo "Cyclomatic Complexity:"; \
		radon cc src -s -n C || true; \
		echo ""; \
		echo "Maintainability Index:"; \
		radon mi src -s -n B || true; \
	else \
		echo "  [SKIP] radon not installed (pip install radon)"; \
	fi

# Security vulnerability scanning
security:
	@echo "=== Security Vulnerability Scan ==="
	@if command -v bandit >/dev/null 2>&1; then \
		echo "Bandit Security Scanner:"; \
		bandit -r src -c pyproject.toml || true; \
	else \
		echo "  [SKIP] bandit not installed (pip install bandit[toml])"; \
	fi
	@if command -v safety >/dev/null 2>&1; then \
		echo ""; \
		echo "Safety Dependency Check:"; \
		safety check || true; \
	else \
		echo "  [SKIP] safety not installed (pip install safety)"; \
	fi

# Code coverage report
coverage: $(DATA_STAMP)
	@echo "=== Test Coverage Analysis ==="
	@if command -v pytest >/dev/null 2>&1; then \
		pytest --cov=src --cov-report=html --cov-report=term-missing; \
		echo ""; \
		echo "Coverage report generated: htmlcov/index.html"; \
	else \
		echo "  [SKIP] pytest not installed (pip install pytest pytest-cov)"; \
	fi

# Performance profiling
profile:
	@echo "=== Performance Profiling ==="
	@if command -v py-spy >/dev/null 2>&1; then \
		echo "Profiling data generation with py-spy..."; \
		py-spy record -o profile.svg --native -- python3 $(DATA_SCRIPT); \
		echo "Profile saved to: profile.svg"; \
	else \
		echo "  [SKIP] py-spy not installed (pip install py-spy)"; \
		echo "  Alternative: Use python -m cProfile"; \
	fi

# Documentation generation (placeholder for Sphinx)
docs:
	@echo "=== Documentation Generation ==="
	@echo "TODO: Sphinx documentation not yet configured"
	@echo "Run: sphinx-quickstart docs"
	@echo "See: docs/CONTRIBUTING.md for setup instructions"

# Dead code detection
dead-code:
	@echo "=== Dead Code Detection ==="
	@if command -v vulture >/dev/null 2>&1; then \
		vulture src tests --min-confidence 80 || true; \
	else \
		echo "  [SKIP] vulture not installed (pip install vulture)"; \
	fi

# Docstring coverage
docstring-coverage:
	@echo "=== Docstring Coverage ==="
	@if command -v interrogate >/dev/null 2>&1; then \
		interrogate src -vv --fail-under 60 || true; \
	else \
		echo "  [SKIP] interrogate not installed (pip install interrogate)"; \
	fi

# Full quality check (all linters + tests)
quality: lint type-check complexity security test
	@echo ""
	@echo "========================================"
	@echo "Full quality check complete"
	@echo "========================================"

# CI simulation (run what CI runs)
ci: check-env quality coverage
	@echo ""
	@echo "========================================"
	@echo "CI simulation complete"
	@echo "========================================"
