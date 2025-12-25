# Cosmology Context: QGP as Microphysical Anchor

**Date:** 2025-12-25
**Status:** Conceptual framework document

---

## Overview

This document places the QGP light-ion research within a broader "Layered Effective-Fluid Cosmology" framework. The key insight: **the same hydrodynamic description that works for QGP droplets at femtometer scales also underpins cosmological fluid models at megaparsec scales**.

---

## 1. The Three-Fluidity Stack

### Tier 1: Early Universe (Literal Relativistic Fluid)

The early universe is accurately described as a relativistic fluid:
- **Photon-baryon plasma** (t < 380 kyr): coupled by Thomson scattering
- **Sound waves** propagate with speed c_s = c/√3
- **Acoustic oscillations** freeze at recombination

**Observable fossils:**
| Probe | What it measures | Status |
|-------|------------------|--------|
| CMB acoustic peaks | Sound horizon at z=1090 | Planck precision ~0.1% |
| BAO (DESI DR2) | Sound horizon at z=0.3-2.3 | March 2025 results |

### Tier 2: Late-Time Expansion (Pressure-Sensitive)

The Friedmann equations treat the universe as a perfect fluid:
```
H² = (8πG/3)(ρ_m + ρ_r + ρ_DE)
```

**Dark energy pressure:**
- ΛCDM: w = -1 (cosmological constant)
- DESI DR2: 3.1σ preference for w₀ > -1, w_a < 0 (evolving)

### Tier 2b: Nonlinear Structure (Effective Fluid)

At late times, nonlinear gravitational collapse creates:
- **Cosmic web:** filaments, walls, voids
- **"Eddies":** not literal turbulence, but shell-crossing vorticity

**Effective descriptions:**
- **EFTofLSS:** systematic expansion with fluid-like parameters
- **Adhesion/Burgers model:** intuition for caustics and voids

### Tier 3: QGP Microphysics (This Repository)

Heavy-ion collisions produce a quark-gluon plasma that:
- Behaves as a **near-perfect fluid** (η/s ~ 1/4π)
- Exhibits **collective flow** (v₂, v₃) from geometry
- Shows **jet quenching** from energy loss

**Why it matters for cosmology:**
The O-O and Ne-Ne results demonstrate that hydrodynamics emerges robustly even in:
- **Small systems:** ~3 fm radius (10⁻¹⁵ m)
- **Short lifetimes:** ~5 fm/c (10⁻²³ s)
- **Modest multiplicities:** ~1000 particles

This provides an empirical anchor: if hydrodynamics works here, it plausibly works at larger scales.

---

## 2. December 2025 Experimental Landscape

### 2.1 DESI DR2 BAO Results

**Source:** [arXiv:2503.14738](https://arxiv.org/abs/2503.14738)

| Finding | Value | Implication |
|---------|-------|-------------|
| w₀ > -1 preference | 3.1σ (BAO+CMB) | Dark energy may be evolving |
| w_a < 0 | Quadrant preference | Phantom crossing possible |
| Σm_ν constraint | < 0.064 eV (ΛCDM) | Stringent neutrino mass limit |
| ΛCDM tension | 2.3σ | Standard model under pressure |

### 2.2 ALICE/CMS O-O and Ne-Ne Flow

**Sources:** [arXiv:2509.06428](https://arxiv.org/abs/2509.06428), [arXiv:2510.02580](https://arxiv.org/abs/2510.02580)

| Finding | Value | Implication |
|---------|-------|-------------|
| v₂ observed | Sizable in O-O/Ne-Ne | Collective flow in small systems |
| v₂(Ne/O) ratio | ~1.08 ultracentral | Nuclear geometry drives flow |
| Hydro agreement | Excellent | Same physics as Pb-Pb |
| R_AA^min (O-O) | 0.69 ± 0.04 | Jet quenching in light ions |

### 2.3 JWST z~13 Reionization

**Source:** [Nature (2025)](https://www.nature.com/articles/s41586-025-08779-5) - JADES-GS-z13-1

| Finding | Value | Implication |
|---------|-------|-------------|
| Lyα detection | z = 13.0 | Earliest ionized region |
| Timing | 330 Myr post-BB | Reionization started earlier |
| Bubble size | Local ionized region | Inhomogeneous reionization |

**Fluid interpretation:** Ionization fronts propagating through neutral IGM - radiation-hydrodynamics at cosmic scales.

### 2.4 LHCb CP Violation in Baryons

**Source:** [CERN Press Release](https://home.cern/news/press-release/physics/new-piece-matter-antimatter-puzzle)

| Finding | Value | Implication |
|---------|-------|-------------|
| Λ_b asymmetry | 2.45% | 5.2σ detection |
| SM explanation | Orders of magnitude too small | Beyond-SM physics needed |

**Cosmology context:** The observed matter-antimatter asymmetry requires CP violation, but the Standard Model provides insufficient amounts. This keeps baryogenesis as an open problem.

---

## 3. Falsifiable Predictions

### From the Fluid Framework

| Prediction | Test | Status |
|------------|------|--------|
| QGP flow in O-O | v₂ > 0 with hydro hierarchy | ✓ Confirmed |
| Geometry → flow | Ne/O ratio ~ deformation | ✓ Confirmed |
| w(z) deviation stable | Multiple SN samples agree | ⚠ Partially (SN calibration debates) |
| EFTofLSS describes voids | Match to N-body stats | ⚠ Active research |
| Early reionization | More z>10 Lyα detections | ⚠ JWST ongoing |

### What Would Falsify This Framework

1. **QGP tier:** If O-O showed no flow despite high multiplicity → hydrodynamics has a sharp onset threshold
2. **Dark energy tier:** If w(z) deviation vanishes with better SN calibration → ΛCDM survives
3. **Structure tier:** If EFTofLSS cannot reproduce void statistics → effective fluid breaks down

---

## 4. How This Repository Fits

### What This Repo Provides

1. **Calibrated physics models** for QGP in light ions
2. **Quantitative validation** of hydro in small systems
3. **Nuclear structure probes** (O-16, Ne-20)
4. **Threshold behavior** for QGP formation

### What This Repo Does NOT Address

1. BAO/CMB distance constraints
2. Dark energy reconstruction
3. Large-scale structure (LSS)
4. Baryogenesis constraints

### The Connection (Analogy vs. Direct Constraint)

| Connection Type | Example | Strength |
|-----------------|---------|----------|
| **Direct constraint** | QGP η/s constrains strongly-coupled QCD | Strong |
| **Analogical support** | QGP hydro validates small-system fluid models | Moderate |
| **Context only** | CP violation reminds us baryogenesis is open | Weak |

The QGP work provides **analogical support** for cosmological fluid descriptions, not direct constraints on cosmological parameters.

---

## 5. Key References for the Cosmology Program

### Early Universe / BAO / Dark Energy

- DESI DR2 Results II: [arXiv:2503.14738](https://arxiv.org/abs/2503.14738)
- DESI DR2 Lyα BAO: [arXiv:2503.14739](https://arxiv.org/abs/2503.14739)
- Planck 2018: [A&A 641, A6 (2020)](https://www.aanda.org/articles/aa/pdf/2020/09/aa33910-18.pdf)
- Pantheon+ SN: [ApJ 938, 110 (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...938..110B)

### QGP / Heavy Ions (This Repository)

- ALICE O-O/Ne-Ne flow: [arXiv:2509.06428](https://arxiv.org/abs/2509.06428)
- CMS O-O flow: [arXiv:2510.02580](https://arxiv.org/abs/2510.02580)
- CMS O-O jet quenching: [arXiv:2510.09864](https://arxiv.org/abs/2510.09864)
- JETSCAPE q̂: [arXiv:2408.08247](https://arxiv.org/abs/2408.08247)

### Effective Fluid Descriptions

- EFTofLSS: [JHEP 09 (2012) 082](https://link.springer.com/article/10.1007/JHEP09(2012)082)
- Adhesion model review: [Physics-Uspekhi (2012)](https://www.mathnet.ru/eng/ufn2380)

### Cosmic Dawn / Reionization

- JWST z=13 Lyα: [Nature (2025)](https://www.nature.com/articles/s41586-025-08779-5)
- JADES overview: [arXiv:2306.02465](https://arxiv.org/abs/2306.02465)

### Microphysics Anchors

- LHCb CP violation: [Nature (2025)](https://www.nature.com/articles/s41586-025-09119-3)
- Lattice QCD T_c: [PLB 795 (2019) 15](https://doi.org/10.1016/j.physletb.2019.05.013)

---

## 6. Summary

The qgp-light-ion repository provides rigorous physics for **Tier 3** of a broader cosmological research program:

```
Tier 1: Early Universe Fluid (CMB/BAO)
         ↓ supports
Tier 2: Late-Time Pressure (DESI w(z))
         ↓ analogizes
Tier 3: QGP Microphysics (this repo) ← ANCHOR
         ↓ validates
Tier 2b: Nonlinear Structure (EFTofLSS)
```

The O-O and Ne-Ne results demonstrate that:
1. **Hydrodynamics is robust** even in tiny systems
2. **System-size scaling** follows expected physics
3. **Transport properties** (η/s) are universal

This strengthens confidence in using effective-fluid descriptions across the cosmological hierarchy.

---

*Document generated 2025-12-25*
