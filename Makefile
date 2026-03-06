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

# Data generation scripts (Tier 1: QGP)
DATA_SCRIPT       := $(SRC_DIR)/qgp/generate.py
ENERGY_SCRIPT     := $(SRC_DIR)/qgp/generate_energy_density.py
MULTIDIM_SCRIPT   := $(SRC_DIR)/qgp/generate_multidimensional_data.py
HBT_SCRIPT        := $(SRC_DIR)/qgp/generate_hbt_data.py
PHOTON_SCRIPT     := $(SRC_DIR)/qgp/generate_photon_data.py
QCD_PHASE_SCRIPT  := $(SRC_DIR)/qgp/generate_qcd_phase_diagram.py
CURVES_SCRIPT     := $(SRC_DIR)/qgp/generate_figure_curves.py
PHYSICS_MOD       := $(SRC_DIR)/qgp/physics.py

# Data generation scripts (Tier 2: Cosmology) - optional, not yet implemented
COSMOLOGY_SCRIPT  := $(SRC_DIR)/cosmology/generate.py

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
QGP_STAMP   := $(DATA_DIR)/.generated
COSMOLOGY_STAMP := $(DATA_DIR)/.generated-cosmology
FINAL_PDF   := $(BUILD_DIR)/qgp-light-ion.pdf

# =============================================================================
# Targets
# =============================================================================

.PHONY: all clean figures data data-only data-qgp data-cosmology data-gpu data-cpu \
        test lint help advanced strict \
        lint-latex lint-python lint-chktex lint-lacheck lint-ruff lint-mypy lint-build \
        fmt test-quick test-physics test-unit test-qgp test-cosmology test-gpu verify-data distclean \
        generate-checksums verify-checksums validate-physics coverage lint-qgp lint-cosmology \
        data-fetch data-verify-experimental

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
		$(PYTHON) $(HBT_SCRIPT) || echo "WARNING: HBT data generation failed (non-fatal)"; \
	fi
	@if [ -f $(PHOTON_SCRIPT) ]; then \
		echo "=== Stage 4b: Photon data ==="; \
		$(PYTHON) $(PHOTON_SCRIPT) || echo "WARNING: Photon data generation failed (non-fatal)"; \
	fi
	@if [ -f $(QCD_PHASE_SCRIPT) ]; then \
		echo "=== Stage 5a: QCD phase diagram (high-fidelity) ==="; \
		$(PYTHON) $(QCD_PHASE_SCRIPT); \
	fi
	@if [ -f $(CURVES_SCRIPT) ]; then \
		echo "=== Stage 5b: Figure curves ==="; \
		$(PYTHON) $(CURVES_SCRIPT); \
	fi
	@# Seed committed experimental data into data/experimental/
	@if [ -d experimental ]; then \
		mkdir -p $(DATA_DIR)/experimental; \
		cp experimental/*.dat $(DATA_DIR)/experimental/; \
		echo "  Seeded experimental data: $$(ls experimental/*.dat | wc -l | tr -d ' ') files"; \
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
data:
	@$(MAKE) $(DATA_STAMP)

data-only: $(DATA_STAMP)
	@echo "Data generated (no figures compiled)"

# =============================================================================
# GPU Detection and Tier-Aware Data Generation
# =============================================================================

# Detect GPU availability (NVIDIA CUDA)
GPU_AVAILABLE := $(shell $(PYTHON) -c "import subprocess; ret = subprocess.call(['which', 'nvidia-smi'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); print('1' if ret == 0 else '0')")

# GPU_BACKEND control: use GPU if available, unless explicitly disabled
# Set GPU_BACKEND=0 to force CPU, or GPU_BACKEND=1 to force GPU
GPU_BACKEND ?= $(GPU_AVAILABLE)

# Tier-specific data generation (Tier 1: QGP physics)
data-qgp:
	@echo "=== Tier 1: QGP Physics Data Generation (GPU=$(GPU_BACKEND)) ==="
	@$(PYTHON) -c "import os; os.environ['GPU_BACKEND'] = '$(GPU_BACKEND)'" && \
	$(PYTHON) $(SRC_DIR)/qgp/generate.py --output-dir $(DATA_DIR) || \
	echo "=== Tier 1 data generation not yet available (using legacy pipeline) ===" && \
	$(MAKE) $(DATA_STAMP)

# Cosmology stamp rule: hard dependency on generate.py
$(COSMOLOGY_STAMP): $(SRC_DIR)/cosmology/generate.py
	@mkdir -p $(DATA_DIR)/cosmology
	$(PYTHON) $(SRC_DIR)/cosmology/generate.py --output-dir $(DATA_DIR)
	@touch $@

# Tier-specific data generation (Tier 2: Cosmology)
data-cosmology: $(COSMOLOGY_STAMP)

# Force GPU backend (requires NVIDIA CUDA and cupy)
data-gpu: GPU_BACKEND = 1
data-gpu: data
	@if [ "$(GPU_AVAILABLE)" = "0" ]; then \
		echo "WARNING: GPU requested but nvidia-smi not found"; \
	fi

# Force CPU backend (fallback)
data-cpu: GPU_BACKEND = 0
data-cpu: data
	@echo "Data generation completed (CPU backend)"

# Fetch all experimental data from HEPData (requires internet access)
data-fetch:
	@echo "=== Fetching experimental data from HEPData ==="
	$(PYTHON) src/tools/fetch_experimental_data.py

# Verify committed experimental data matches CHECKSUMS.sha256
data-verify-experimental:
	@echo "=== Verifying experimental data checksums ==="
	$(PYTHON) src/tools/fetch_experimental_data.py --verify-only

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

# Tier-specific linting (Tier 1: QGP)
lint-qgp:
	@echo "=== Tier 1: QGP Physics Linting ==="
	@$(PYTHON) -m mypy src/qgp/
	@$(RUFF) check src/qgp/ tests/test_qgp/

# Tier-specific linting (Tier 2: Cosmology)
lint-cosmology:
	@echo "=== Tier 2: Cosmology Linting ==="
	@if [ -d src/cosmology ]; then \
		$(PYTHON) -m mypy src/cosmology/; \
		$(RUFF) check src/cosmology/ tests/test_cosmology/; \
	else \
		echo "  [SKIP] Cosmology module not yet implemented"; \
	fi

# Combined lint target: runs all linters
lint: lint-latex lint-python lint-mypy
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
		$(RUFF) check $(SRC_DIR) tests/; \
	else \
		echo "  [SKIP] ruff not installed (pip install ruff)"; \
	fi

# Mypy type checking
lint-mypy:
	@echo "=== Mypy: Python type check ==="
	@$(PYTHON) -m mypy src/

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
	@echo "=== Running full test suite ==="
	@$(PYTHON) -m pytest tests/ -v --tb=short

# Quick unit tests without data regeneration (no make data dependency)
test-quick:
	@echo "=== Running unit tests (no data regen) ==="
	@$(PYTHON) -m pytest tests/test_qgp/test_qgp_physics.py tests/test_qgp/test_constants.py -v --tb=short

# Validate physics module loads correctly
test-physics:
	@echo "=== Physics module validation ==="
	@$(PYTHON) -c "import sys; sys.path.insert(0,'src'); from qgp.physics import woods_saxon, NUCLEI; print('  [OK] Physics module imported')"

# Run only unit tests (no I/O, no data dependency)
test-unit: test-quick

# Tier-specific tests (Tier 1: QGP physics)
test-qgp: $(DATA_STAMP)
	@echo "=== Tier 1: QGP Physics Tests ==="
	@if [ -d tests/test_qgp ]; then \
		$(PYTHON) -m pytest tests/test_qgp/ -v --tb=short; \
	else \
		$(PYTHON) -m pytest tests/test_qgp_physics.py tests/test_constants.py tests/test_phase_diagram.py tests/test_io_utils.py -v --tb=short; \
	fi

# Tier-specific tests (Tier 2: Cosmology)
test-cosmology: $(DATA_STAMP)
	@echo "=== Tier 2: Cosmology Tests ==="
	@if [ -d tests/test_cosmology ]; then \
		$(PYTHON) -m pytest tests/test_cosmology/ -v --tb=short; \
	else \
		echo "  [SKIP] Cosmology tests not yet available"; \
	fi

# GPU backend tests
test-gpu:
	@echo "=== Tier 3: GPU Backend Tests ==="
	@if [ -d tests/test_gpu ]; then \
		$(PYTHON) -m pytest tests/test_gpu/ -v --tb=short; \
	else \
		echo "  [SKIP] GPU tests not yet available"; \
	fi

# Test coverage report
coverage:
	@echo "=== Running tests with coverage ==="
	@$(PYTHON) -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html:build/coverage-html
	@echo "Coverage report: build/coverage-html/index.html"

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
	echo "Checking Python lint (ruff)..."; \
	if $(RUFF) check $(SRC_DIR) tests/ >/dev/null 2>&1; then \
		echo "  [OK] Ruff: no issues"; \
	else \
		echo "  [FAIL] Ruff found issues:"; \
		$(RUFF) check $(SRC_DIR) tests/ 2>&1 | head -5; \
		ERRORS=$$((ERRORS + 1)); \
	fi; \
	echo "Checking Python types (mypy)..."; \
	if $(PYTHON) -m mypy src/ >/dev/null 2>&1; then \
		echo "  [OK] Mypy: no issues"; \
	else \
		echo "  [FAIL] Mypy found issues:"; \
		$(PYTHON) -m mypy src/ 2>&1 | tail -5; \
		ERRORS=$$((ERRORS + 1)); \
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

# Generate sha256 checksums for all data files (run after make data)
generate-checksums: $(DATA_STAMP)
	@echo "=== Generating data checksums ==="
	find $(DATA_DIR) -name '*.dat' | sort | xargs sha256sum > $(DATA_DIR)/CHECKSUMS.sha256
	@echo "Checksums written to $(DATA_DIR)/CHECKSUMS.sha256"

# Verify data files match previously generated checksums
verify-checksums: $(DATA_DIR)/CHECKSUMS.sha256
	@echo "=== Verifying data checksums ==="
	sha256sum --check $(DATA_DIR)/CHECKSUMS.sha256
	@echo "All checksums verified"

# Run chi-squared model vs experiment validation
validate-physics: $(DATA_STAMP)
	@echo "=== Running physics validation ==="
	@# Ensure experimental data is available (may not be present if data/ was partially cleaned)
	@if [ -d experimental ]; then \
		mkdir -p $(DATA_DIR)/experimental; \
		cp experimental/*.dat $(DATA_DIR)/experimental/; \
	fi
	@PYTHONPATH=$(SRC_DIR) python3 $(SRC_DIR)/compare_model_vs_experiment.py
	@echo "Physics validation complete"

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
	@echo "  lint        Run all linters (LaTeX + Python + mypy)"
	@echo "  lint-latex  Run LaTeX linters (chktex + lacheck)"
	@echo "  lint-chktex Run ChkTeX on .tex files"
	@echo "  lint-lacheck Run Lacheck on main document"
	@echo "  lint-python Run Python linter (ruff)"
	@echo "  lint-ruff   Run Ruff on src/ and tests/"
	@echo "  lint-mypy   Run mypy type checker on src/"
	@echo "  lint-build  Analyze build log for issues"
	@echo "  fmt         Auto-format Python code with ruff"
	@echo ""
	@echo "Testing:"
	@echo "  test        Run full test suite (regenerates data)"
	@echo "  test-quick  Run tests without data regeneration"
	@echo "  test-physics Validate physics module imports"
	@echo "  coverage    Run tests with coverage report"
	@echo ""
	@echo "Quality assurance:"
	@echo "  strict             Build and fail on any warnings/errors"
	@echo "  verify-data        Check all data directories exist"
	@echo "  generate-checksums Generate sha256 checksums for data files"
	@echo "  verify-checksums   Verify data files match checksums"
	@echo "  validate-physics   Run chi-squared model vs experiment validation"
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
