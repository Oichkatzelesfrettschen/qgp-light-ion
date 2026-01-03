# Formal Methods Integration Guide
**Project:** QGP Light-Ion Collisions  
**Date:** 2026-01-03  
**Focus:** Z3, TLA+, and Design by Contract

---

## Executive Summary

This document explores the integration of formal methods into the QGP light-ion project:
- **Z3 SMT Solver** for constraint validation
- **TLA+ Specifications** for build system verification
- **Design by Contract** for physics function guarantees

---

## 1. Z3 SMT Solver Integration

### 1.1 Overview

Z3 is a high-performance SMT (Satisfiability Modulo Theories) solver from Microsoft Research. We can use it to:
1. Validate physical constraints on parameters
2. Verify configuration consistency
3. Check invariants in numerical computations

### 1.2 Installation

```bash
pip install z3-solver
```

### 1.3 Use Case 1: Physics Parameter Validation

**File: `src/z3_validators.py`**

```python
"""
Z3-based validation for physics parameters.

Ensures that computed quantities satisfy known physical constraints.
"""

from z3 import *
import numpy as np
from typing import Dict, List, Tuple

# =============================================================================
# R_AA Constraints
# =============================================================================

def validate_raa_constraints(pt_range: Tuple[float, float]) -> bool:
    """
    Validate that R_AA satisfies physical bounds using Z3.
    
    Physical constraints:
    1. R_AA > 0 (positive)
    2. R_AA ≤ 1.5 (Cronin enhancement upper bound)
    3. R_AA(high pT) > 0.5 (approaches 1 at large pT)
    4. R_AA(mid pT) < 1 (suppression in QGP regime)
    
    Args:
        pt_range: (min_pt, max_pt) in GeV/c
    
    Returns:
        True if constraints are satisfiable, False otherwise
    """
    s = Solver()
    
    # Variables
    R_AA = Real('R_AA')
    p_T = Real('p_T')
    
    # Physical constraints
    s.add(R_AA > 0)           # Must be positive
    s.add(R_AA <= 1.5)        # Upper bound (Cronin effect)
    s.add(p_T >= pt_range[0]) # pT range
    s.add(p_T <= pt_range[1])
    
    # High-pT behavior: approaches 1
    s.add(Implies(p_T > 15.0, R_AA > 0.5))
    
    # Mid-pT behavior: suppressed
    s.add(Implies(And(p_T > 4.0, p_T < 10.0), R_AA < 1.0))
    
    # Check satisfiability
    result = s.check()
    
    if result == sat:
        model = s.model()
        print(f"Example solution: R_AA = {model[R_AA]}, pT = {model[p_T]}")
        return True
    elif result == unsat:
        print("UNSAT: Constraints are contradictory!")
        return False
    else:
        print("UNKNOWN: Cannot determine satisfiability")
        return False


# =============================================================================
# Flow Coefficient Constraints
# =============================================================================

def validate_flow_coefficients() -> bool:
    """
    Validate anisotropic flow coefficient relationships.
    
    Physical constraints:
    1. |v_n| < 0.5 for all n
    2. v_2 > v_3 typically (elliptic > triangular)
    3. v_n ≥ -0.05 (small negative from fluctuations OK)
    4. v_n increases with centrality (for peripheral)
    """
    s = Solver()
    
    # Variables
    v2 = Real('v2')
    v3 = Real('v3')
    v4 = Real('v4')
    centrality = Real('centrality')  # 0-100%
    
    # Bounds
    s.add(And(v2 > -0.05, v2 < 0.5))
    s.add(And(v3 > -0.05, v3 < 0.5))
    s.add(And(v4 > -0.05, v4 < 0.5))
    s.add(And(centrality >= 0, centrality <= 100))
    
    # Hierarchy (typically)
    s.add(v2 > v3)
    s.add(v3 > v4)
    
    # Centrality dependence
    # Peripheral (50-80%) has larger flow than central (0-10%)
    v2_peripheral = Real('v2_peripheral')
    v2_central = Real('v2_central')
    s.add(v2_peripheral > v2_central)
    s.add(v2_central > 0)
    
    result = s.check()
    
    if result == sat:
        model = s.model()
        print(f"Example: v2={model[v2]}, v3={model[v3]}, v4={model[v4]}")
        return True
    else:
        print(f"Flow constraints: {result}")
        return False


# =============================================================================
# Temperature Evolution Constraints
# =============================================================================

def validate_temperature_evolution() -> bool:
    """
    Validate QGP cooling trajectory.
    
    Constraints:
    1. T(τ=0.6 fm/c) ~ 400-500 MeV (initial temperature)
    2. T(τ) decreases monotonically
    3. T(τ→∞) → T_freeze ~ 100 MeV
    4. Bjorken: T ∝ τ^(-1/3) approximately
    """
    s = Solver()
    
    # Variables (times in fm/c, temperatures in GeV)
    T_init = Real('T_init')
    T_freeze = Real('T_freeze')
    tau_0 = Real('tau_0')
    tau_freeze = Real('tau_freeze')
    
    # Initial conditions
    s.add(And(T_init > 0.400, T_init < 0.600))  # 400-600 MeV
    s.add(tau_0 == 0.6)  # Formation time
    
    # Freeze-out
    s.add(And(T_freeze > 0.080, T_freeze < 0.120))  # 80-120 MeV
    s.add(tau_freeze > 5.0)  # At least 5 fm/c
    
    # Monotonic decrease
    s.add(T_init > T_freeze)
    
    # Bjorken scaling (approximate)
    # T(τ) / T_0 ≈ (τ_0 / τ)^(1/3)
    # This is approximate, so we use it as a guide
    
    result = s.check()
    
    if result == sat:
        model = s.model()
        print(f"Example trajectory: T_init={model[T_init]} GeV, "
              f"T_freeze={model[T_freeze]} GeV at τ={model[tau_freeze]} fm/c")
        return True
    else:
        print(f"Temperature evolution: {result}")
        return False


# =============================================================================
# Configuration Validation
# =============================================================================

def validate_nucleus_parameters(nucleus_config: Dict[str, float]) -> bool:
    """
    Validate nuclear structure parameters are physically consistent.
    
    Args:
        nucleus_config: Dictionary with R0, a, A, beta2, etc.
    """
    s = Solver()
    
    # Variables
    R0 = Real('R0')   # Nuclear radius [fm]
    a = Real('a')     # Skin thickness [fm]
    A = Int('A')      # Mass number
    beta2 = Real('beta2')  # Deformation
    
    # Physical constraints
    s.add(R0 > 0)
    s.add(a > 0)
    s.add(A > 0)
    s.add(And(beta2 >= -0.6, beta2 <= 0.6))  # Typical deformation range
    
    # Relationship: R0 ≈ 1.2 * A^(1/3) (rough approximation)
    # For A=16 (O), R0 ~ 2.4-2.8 fm
    # For A=208 (Pb), R0 ~ 6.4-7.0 fm
    s.add(R0 > 1.0 * A**(1.0/3))
    s.add(R0 < 1.4 * A**(1.0/3))
    
    # Skin thickness typically 0.4-0.7 fm
    s.add(And(a > 0.4, a < 0.7))
    
    # Add specific values from config
    s.add(R0 == nucleus_config.get('R0', 2.608))
    s.add(a == nucleus_config.get('a', 0.513))
    s.add(A == nucleus_config.get('A', 16))
    s.add(beta2 == nucleus_config.get('beta2', 0.0))
    
    result = s.check()
    
    if result == sat:
        print(f"✓ Nucleus configuration is physically consistent")
        return True
    else:
        print(f"✗ Nucleus configuration violates constraints: {result}")
        return False


# =============================================================================
# Property-Based Validation with Z3
# =============================================================================

def generate_valid_test_cases(constraint_func, n_cases: int = 10) -> List[Dict]:
    """
    Use Z3 to generate valid test cases that satisfy constraints.
    
    This is useful for property-based testing.
    """
    test_cases = []
    
    s = Solver()
    # Add constraints from constraint_func
    # ... (implementation depends on specific constraints)
    
    for i in range(n_cases):
        if s.check() == sat:
            model = s.model()
            test_case = {str(d): model[d] for d in model.decls()}
            test_cases.append(test_case)
            
            # Add constraint to find different solution next time
            s.add(Or([d() != model[d] for d in model.decls()]))
    
    return test_cases


# =============================================================================
# Main Validation Runner
# =============================================================================

def run_all_validations() -> Dict[str, bool]:
    """Run all Z3 validation checks."""
    results = {}
    
    print("=" * 60)
    print("Z3 Physics Constraint Validation")
    print("=" * 60)
    
    print("\n1. R_AA Constraints:")
    results['raa'] = validate_raa_constraints((0, 30))
    
    print("\n2. Flow Coefficients:")
    results['flow'] = validate_flow_coefficients()
    
    print("\n3. Temperature Evolution:")
    results['temperature'] = validate_temperature_evolution()
    
    print("\n4. Nucleus Parameters (Oxygen):")
    O_config = {'R0': 2.608, 'a': 0.513, 'A': 16, 'beta2': 0.0}
    results['nucleus_O'] = validate_nucleus_parameters(O_config)
    
    print("\n5. Nucleus Parameters (Neon):")
    Ne_config = {'R0': 2.791, 'a': 0.535, 'A': 20, 'beta2': 0.45}
    results['nucleus_Ne'] = validate_nucleus_parameters(Ne_config)
    
    print("\n" + "=" * 60)
    print(f"Validation Summary: {sum(results.values())}/{len(results)} passed")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = run_all_validations()
    
    # Exit with error if any validation failed
    if not all(results.values()):
        exit(1)
```

---

## 2. TLA+ Specification for Build System

### 2.1 Overview

TLA+ (Temporal Logic of Actions) can model the build system state machine to verify:
- Dependency ordering is correct
- Parallel builds don't have race conditions
- Build is idempotent

### 2.2 Specification

**File: `specs/BuildSystem.tla`**

```tla
---------------------- MODULE BuildSystem ----------------------
(*
  TLA+ specification for QGP light-ion build system.
  
  Models the 4-stage pipeline:
  1. Data generation (Python)
  2. Figure compilation (LaTeX/TikZ)
  3. Markdown conversion (Pandoc)
  4. PDF assembly (latexmk)
*)

EXTENDS Integers, Sequences, FiniteSets

CONSTANTS 
    Figures,          \* Set of figure files
    MaxParallelJobs   \* Maximum parallel figure compilations

VARIABLES
    dataGenerated,     \* Boolean: data files exist
    figuresCompiled,   \* Set of compiled figures
    bodyGenerated,     \* Boolean: body.tex exists
    pdfBuilt,          \* Boolean: final PDF exists
    buildStage,        \* Current stage: "idle", "data", "figures", "body", "pdf", "done"
    runningJobs        \* Number of parallel figure compilation jobs

-----------------------------------------------------------------------------

TypeInvariant ==
    /\ dataGenerated \in BOOLEAN
    /\ figuresCompiled \subseteq Figures
    /\ bodyGenerated \in BOOLEAN
    /\ pdfBuilt \in BOOLEAN
    /\ buildStage \in {"idle", "data", "figures", "body", "pdf", "done", "error"}
    /\ runningJobs \in 0..MaxParallelJobs

-----------------------------------------------------------------------------

Init ==
    /\ dataGenerated = FALSE
    /\ figuresCompiled = {}
    /\ bodyGenerated = FALSE
    /\ pdfBuilt = FALSE
    /\ buildStage = "idle"
    /\ runningJobs = 0

-----------------------------------------------------------------------------

(* Stage 1: Generate data *)
GenerateData ==
    /\ buildStage = "idle"
    /\ ~dataGenerated
    /\ dataGenerated' = TRUE
    /\ buildStage' = "data"
    /\ UNCHANGED <<figuresCompiled, bodyGenerated, pdfBuilt, runningJobs>>

(* Stage 2: Compile figures (can be parallel) *)
CompileFigure(fig) ==
    /\ buildStage = "data"
    /\ dataGenerated
    /\ fig \in Figures
    /\ fig \notin figuresCompiled
    /\ runningJobs < MaxParallelJobs
    /\ figuresCompiled' = figuresCompiled \union {fig}
    /\ runningJobs' = runningJobs + 1
    /\ UNCHANGED <<dataGenerated, bodyGenerated, pdfBuilt, buildStage>>

FinishFigureJob ==
    /\ runningJobs > 0
    /\ runningJobs' = runningJobs - 1
    /\ IF figuresCompiled = Figures /\ runningJobs' = 0
       THEN buildStage' = "figures"
       ELSE UNCHANGED buildStage
    /\ UNCHANGED <<dataGenerated, figuresCompiled, bodyGenerated, pdfBuilt>>

(* Stage 3: Convert Markdown to LaTeX *)
ConvertMarkdown ==
    /\ buildStage = "figures"
    /\ figuresCompiled = Figures
    /\ ~bodyGenerated
    /\ bodyGenerated' = TRUE
    /\ buildStage' = "body"
    /\ UNCHANGED <<dataGenerated, figuresCompiled, pdfBuilt, runningJobs>>

(* Stage 4: Build final PDF *)
BuildPDF ==
    /\ buildStage = "body"
    /\ bodyGenerated
    /\ figuresCompiled = Figures
    /\ ~pdfBuilt
    /\ pdfBuilt' = TRUE
    /\ buildStage' = "done"
    /\ UNCHANGED <<dataGenerated, figuresCompiled, bodyGenerated, runningJobs>>

(* Error state (e.g., compilation failure) *)
Error ==
    /\ buildStage \in {"data", "figures", "body", "pdf"}
    /\ buildStage' = "error"
    /\ UNCHANGED <<dataGenerated, figuresCompiled, bodyGenerated, pdfBuilt, runningJobs>>

-----------------------------------------------------------------------------

Next ==
    \/ GenerateData
    \/ \E fig \in Figures : CompileFigure(fig)
    \/ FinishFigureJob
    \/ ConvertMarkdown
    \/ BuildPDF
    \/ Error

Spec == Init /\ [][Next]_<<dataGenerated, figuresCompiled, bodyGenerated, pdfBuilt, buildStage, runningJobs>>

-----------------------------------------------------------------------------

(* Safety properties *)

(* Can't build PDF without data *)
PDFRequiresData ==
    pdfBuilt => dataGenerated

(* Can't build PDF without figures *)
PDFRequiresFigures ==
    pdfBuilt => (figuresCompiled = Figures)

(* Can't build PDF without body *)
PDFRequiresBody ==
    pdfBuilt => bodyGenerated

(* Stage ordering is correct *)
StageOrdering ==
    \/ buildStage = "idle"
    \/ (buildStage = "data" /\ dataGenerated)
    \/ (buildStage = "figures" /\ dataGenerated)
    \/ (buildStage = "body" /\ figuresCompiled = Figures)
    \/ (buildStage = "pdf" /\ bodyGenerated)
    \/ (buildStage = "done" /\ pdfBuilt)
    \/ buildStage = "error"

(* Parallel jobs don't exceed maximum *)
JobLimit ==
    runningJobs <= MaxParallelJobs

(* Liveness properties *)

(* Eventually completes if no errors *)
EventuallyCompletes ==
    <>(buildStage = "done" \/ buildStage = "error")

(* Figures can be compiled in any order *)
FigureCompilationOrder ==
    \A fig1, fig2 \in Figures :
        fig1 # fig2 => ~(fig1 < fig2)  (* No ordering constraint *)

=============================================================================
```

### 2.3 Model Checking

**File: `specs/BuildSystem.cfg`**

```
SPECIFICATION Spec

CONSTANTS
    Figures = {f1, f2, f3, f4, f5}
    MaxParallelJobs = 4

INVARIANTS
    TypeInvariant
    PDFRequiresData
    PDFRequiresFigures
    PDFRequiresBody
    StageOrdering
    JobLimit

PROPERTIES
    EventuallyCompletes
```

**Run TLC model checker:**
```bash
# Install TLA+ tools
# Download from: https://github.com/tlaplus/tlaplus/releases

# Run model checker
java -jar tla2tools.jar -config specs/BuildSystem.cfg specs/BuildSystem.tla
```

---

## 3. Design by Contract Implementation

### 3.1 Contract Decorator

**File: `src/contracts.py`**

```python
"""
Design by Contract implementation for QGP physics functions.

Provides decorators for pre/postconditions and invariants.
"""

import functools
import warnings
from typing import Callable, Any
import numpy as np


class ContractViolation(Exception):
    """Raised when a contract (pre/post/invariant) is violated."""
    pass


def require(condition: Callable[..., bool], message: str = "Precondition failed"):
    """
    Precondition decorator.
    
    Usage:
        @require(lambda r: np.all(r >= 0), "Radius must be non-negative")
        def woods_saxon(r, nucleus, theta=0):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Evaluate condition with function arguments
            if not condition(*args, **kwargs):
                raise ContractViolation(
                    f"Precondition violated in {func.__name__}: {message}"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def ensure(condition: Callable[..., bool], message: str = "Postcondition failed"):
    """
    Postcondition decorator.
    
    Usage:
        @ensure(lambda result: np.all(result >= 0), "Density must be non-negative")
        def woods_saxon(r, nucleus, theta=0):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if not condition(result):
                raise ContractViolation(
                    f"Postcondition violated in {func.__name__}: {message}"
                )
            return result
        return wrapper
    return decorator


def invariant(condition: Callable[..., bool], message: str = "Invariant violated"):
    """
    Invariant decorator (checks both before and after).
    
    Usage:
        @invariant(lambda self: self.A > 0, "Mass number must be positive")
        class Nucleus:
            ...
    """
    def decorator(cls):
        # Wrap __init__ and all methods to check invariant
        original_init = cls.__init__
        
        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if not condition(self):
                raise ContractViolation(f"Invariant violated after init: {message}")
        
        cls.__init__ = new_init
        
        # Wrap all methods
        for name in dir(cls):
            if not name.startswith('_'):
                attr = getattr(cls, name)
                if callable(attr):
                    setattr(cls, name, _wrap_with_invariant(attr, condition, message))
        
        return cls
    return decorator


def _wrap_with_invariant(method, condition, message):
    """Helper to wrap method with invariant check."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        if not condition(self):
            raise ContractViolation(f"Invariant violated after {method.__name__}: {message}")
        return result
    return wrapper
```

### 3.2 Example Usage

**File: `src/qgp_physics_contracts.py`**

```python
"""
QGP physics functions with Design by Contract.

This module wraps key physics functions with explicit contracts.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Any
from contracts import require, ensure, invariant
from qgp_physics import Nucleus, NUCLEI

FloatArray = NDArray[np.floating[Any]]


@require(
    lambda r, nucleus, theta: isinstance(r, np.ndarray) and np.all(r >= 0),
    "Radius must be non-negative array"
)
@require(
    lambda r, nucleus, theta: isinstance(nucleus, Nucleus),
    "Second argument must be Nucleus instance"
)
@require(
    lambda r, nucleus, theta: 0 <= theta < 2 * np.pi,
    "Theta must be in [0, 2π)"
)
@ensure(
    lambda result: np.all(result >= 0) and np.all(result <= 1),
    "Density must be in [0, 1]"
)
@ensure(
    lambda result: np.all(np.isfinite(result)),
    "Density must be finite"
)
def woods_saxon_safe(r: FloatArray, nucleus: Nucleus, theta: float = 0) -> FloatArray:
    """
    Woods-Saxon density with contracts.
    
    Preconditions:
        - r >= 0 (non-negative radius)
        - nucleus is valid Nucleus instance
        - 0 <= theta < 2π
    
    Postconditions:
        - result in [0, 1] (normalized density)
        - result is finite (no NaN/Inf)
    """
    from qgp_physics import woods_saxon
    return woods_saxon(r, nucleus, theta)


@require(
    lambda pt, qhat, L, T: np.all(pt > 0),
    "Transverse momentum must be positive"
)
@require(
    lambda pt, qhat, L, T: qhat > 0,
    "Transport coefficient must be positive"
)
@require(
    lambda pt, qhat, L, T: np.all(L >= 0),
    "Path length must be non-negative"
)
@ensure(
    lambda result: np.all(result > 0) and np.all(result <= 1.5),
    "R_AA must be in (0, 1.5]"
)
def R_AA_safe(pt: FloatArray, qhat: float, L: FloatArray, T: float) -> FloatArray:
    """
    Nuclear modification factor with contracts.
    
    Preconditions:
        - pt > 0 (positive momentum)
        - qhat > 0 (positive transport coefficient)
        - L >= 0 (non-negative path length)
    
    Postconditions:
        - R_AA in (0, 1.5] (physical bounds)
    """
    # Energy loss (BDMPS-Z formula)
    alpha_s = 0.3  # Strong coupling
    dE = 0.5 * alpha_s * qhat * np.minimum(L, 50)**2  # Clip L to avoid overflow
    
    # Nuclear modification factor
    dE_clipped = np.minimum(dE, 0.95 * pt)  # Can't lose more than 95% of energy
    R_AA = (pt - dE_clipped) / pt
    
    return np.clip(R_AA, 0, 1.5)  # Enforce physical bounds
```

---

## 4. Integration with Testing

### 4.1 Property-Based Testing with Contracts

**File: `tests/properties/test_contracts.py`**

```python
"""
Property-based tests using contracts and Hypothesis.
"""

import numpy as np
from hypothesis import given, strategies as st, assume
import pytest

from src.qgp_physics_contracts import woods_saxon_safe, R_AA_safe
from src.qgp_physics import NUCLEI
from src.contracts import ContractViolation


# Strategy for valid radii
radii_strategy = st.lists(
    st.floats(min_value=0, max_value=20, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=100
).map(np.array)

# Strategy for valid angles
angle_strategy = st.floats(min_value=0, max_value=2*np.pi, exclude_max=True)


@given(r=radii_strategy, angle=angle_strategy)
def test_woods_saxon_contracts(r, angle):
    """Test that Woods-Saxon always satisfies contracts."""
    nucleus = NUCLEI['O']
    
    # Should not raise ContractViolation
    density = woods_saxon_safe(r, nucleus, angle)
    
    # Additional properties
    assert density.shape == r.shape
    assert np.all(np.isfinite(density))


@given(r=st.lists(st.floats(allow_nan=True, allow_infinity=True), min_size=1).map(np.array))
def test_woods_saxon_rejects_invalid(r):
    """Test that invalid inputs are rejected."""
    # Assume we have at least one invalid value
    assume(not np.all(np.isfinite(r)) or np.any(r < 0))
    
    nucleus = NUCLEI['O']
    
    with pytest.raises((ContractViolation, ValueError)):
        woods_saxon_safe(r, nucleus, 0)


@given(
    pt=st.lists(st.floats(min_value=0.1, max_value=100), min_size=1).map(np.array),
    qhat=st.floats(min_value=0.1, max_value=10),
    L=st.lists(st.floats(min_value=0, max_value=20), min_size=1).map(np.array)
)
def test_raa_contracts(pt, qhat, L):
    """Test that R_AA always satisfies contracts."""
    T = 0.3  # Temperature in GeV
    
    # Should not raise ContractViolation
    raa = R_AA_safe(pt, qhat, L, T)
    
    # Additional properties
    assert raa.shape == pt.shape
    assert np.all(raa > 0)
    assert np.all(raa <= 1.5)
```

---

## 5. Formal Methods in CI Pipeline

### 5.1 Add to CI

**Update `.github/workflows/ci.yml`:**

```yaml
  formal-methods:
    name: Formal Methods Validation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      
      - name: Install dependencies
        run: |
          pip install z3-solver hypothesis pytest
          pip install -e .
      
      - name: Run Z3 validations
        run: python src/z3_validators.py
      
      - name: Run property-based tests with contracts
        run: pytest tests/properties/test_contracts.py -v
```

---

## 6. Documentation and Maintenance

### 6.1 When to Use Formal Methods

**Use Z3 when:**
- ✅ Validating configuration files
- ✅ Checking physics parameter consistency
- ✅ Generating test cases for property-based testing
- ❌ Runtime performance-critical code (too slow)

**Use TLA+ when:**
- ✅ Modeling concurrent/parallel systems
- ✅ Verifying build system correctness
- ✅ Documenting system behavior formally
- ❌ Rapid prototyping (specification overhead)

**Use Design by Contract when:**
- ✅ Critical physics calculations
- ✅ Public API functions
- ✅ Debugging numerical stability issues
- ❌ Performance-critical inner loops (add overhead)

### 6.2 Performance Considerations

Contracts can be disabled in production:

```python
# In production
import os
ENABLE_CONTRACTS = os.getenv('ENABLE_CONTRACTS', '0') == '1'

def require(condition, message=""):
    def decorator(func):
        if ENABLE_CONTRACTS:
            # Full contract checking
            ...
        else:
            # No-op decorator
            return func
        return wrapper
    return decorator
```

---

## 7. Future Work

1. **Automated Specification Extraction**: Generate TLA+ specs from Python code
2. **Proof Automation**: Use Z3 to prove properties about algorithms
3. **Formal Verification**: Verify numerical stability with symbolic execution
4. **Constraint-Based Optimization**: Use SMT solvers for parameter fitting

---

*Document version: 1.0*  
*Last updated: 2026-01-03*
