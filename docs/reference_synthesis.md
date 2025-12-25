# Reference Synthesis: QGP in Light-Ion Collisions

This document consolidates the key findings from all analyzed reference materials for the QGP Light-Ion project.

---

## Table of Contents

1. [Validated References](#validated-references)
2. [Workshop Summary: OO and pO Opportunities](#1-workshop-summary-oo-and-po-opportunities)
3. [Canonical Strangeness Treatment](#2-canonical-strangeness-treatment)
4. [ALICE Strangeness Enhancement Results](#3-alice-strangeness-enhancement-results)
5. [Model Predictions for O+O Collisions](#4-model-predictions-for-oo-collisions)
6. [Synthesized Key Equations](#synthesized-key-equations)
7. [Unified Physics Framework](#unified-physics-framework)
8. [Invalid References](#invalid-references-to-replace)

---

## Validated References

| arXiv ID | Title | Relevance |
|----------|-------|-----------|
| 2103.01939 | Opportunities of OO and pO collisions at the LHC | Workshop summary, experimental program |
| 2503.02677 | Canonical treatment of strangeness and light nuclei production | Statistical hadronization theory |
| 2504.02527 | Strangeness enhancement in small collision systems with ALICE | Experimental results |
| 2507.16266 | Strangeness production in O+O at 7 TeV | EPOS4/AMPT model predictions |

---

## 1. Workshop Summary: OO and pO Opportunities

**Source:** arXiv:2103.01939 (Brewer, Mazeliauskas, van der Schee)

### Key Physics Opportunities

#### Flow and Collectivity
- **v₂ hierarchy expected:** v₂(OO) < v₂(PbPb) due to less eccentric geometry
- **v₃ comparable:** v₃(OO) ~ v₃(PbPb) due to fluctuation-driven nature
- **Compact geometry:** OO produces higher temperature at same multiplicity vs PbPb
- **Thermal photon enhancement:** Factor 2× in OO vs PbPb at same multiplicity

#### Hard Probes
- **Energy loss signal:** 5-10% suppression expected for hadrons with pT > 20 GeV
- **Key question:** Why does pPb show no energy loss despite collective behavior?

#### Nuclear Structure Effects
- **Alpha clustering:** ¹⁶O may have tetrahedral 4-alpha structure
- **Weak eccentricity effect:** Few-percent changes in central events
- **Sub-nucleonic structure:** More important for flow harmonics than clustering

### Experimental Parameters

| Parameter | OO Value | pO Value |
|-----------|----------|----------|
| √s_NN | 7 TeV | 9.9 TeV |
| Target luminosity | 0.5 nb⁻¹ | 2-5 nb⁻¹ |
| Run duration | 1 day physics | 2.5-3 days physics |

### Critical Observables

1. **v₂²-⟨pT⟩ correlator:** Sign change at dNch/dη ~ 10 distinguishes initial vs final state
2. **Forward π⁰ fraction:** Critical for cosmic ray shower modeling
3. **Triggered jet spectra:** ΔpT ~ 160 MeV shifts detectable

---

## 2. Canonical Strangeness Treatment

**Source:** arXiv:2503.02677 (N. Sharma)

### Theoretical Framework

#### Grand Canonical vs Canonical Ensemble

**Grand Canonical (large systems):**
```
Z_GC = Tr[exp(-(H - μ·Q)/T)]
```
- Conserves quantum numbers on average
- Valid when statistical fluctuations are small

**Canonical (small systems):**
```
Z^C_{S=0} = Tr[exp(-H/T) δ(S,0)]
```
- Enforces exact strangeness conservation
- Required for small systems and low energies

### Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| T_ch | 156.5 MeV | LQCD calculations |
| γ_s | 1 | Fixed |
| μ | 0 | LHC energies |

### Volume Parametrization (pp at √s = 7 TeV)

```
V_A = 1.55 + 3.02 × (dNch/dη)  [fm³]  (Acceptance volume)
V_C = 12.32 + 3.02 × (dNch/dη) [fm³]  (Correlation volume)
```

**Critical finding:** V_C > V_A for low multiplicities, converging at dNch/dη ≥ 15

### Particle Multiplicity with Canonical Suppression

```
⟨N^s_k⟩_A ≃ V_A · n^s_k(T) · [I_s(S₁)/I₀(S₁)]
```

Where:
- I_s, I₀ are modified Bessel functions
- S₁ = V_C Σ_k n(k,T)
- Suppression increases with strangeness content |S|

### Light Nuclei: Baryon Canonical Effect

For particles with baryon number b:
```
⟨N^b_k⟩_A ≃ V_A · n^b_k(T) · [I_b(B₁)/I₀(B₁)]
```

**Off-equilibrium factor:** λ = 0.45 ± 0.03 for B > 2 particles

---

## 3. ALICE Strangeness Enhancement Results

**Source:** arXiv:2504.02527 (S. Pucillo for ALICE)

### Core Observations

1. **Continuous enhancement:** Strange/non-strange ratios increase with multiplicity
2. **System independence:** Pattern extends from Pb-Pb through pp collisions
3. **Strangeness hierarchy:** Enhancement scales with strangeness content
   - Ω shows strongest enhancement
   - Multi-strange > single-strange hadrons

### Transverse Spherocity Analysis

**Definition:**
```
S₀^(pT=1) = (π²/4) min_n̂ (Σᵢ|p̂_Ti × n̂|/⟨N_trks⟩)²
```

**Range:** 0 (jet-like) → 1 (isotropic)

**Key Results:**
- Jet-like events: Suppress strangeness
- Isotropic events: Enhance strangeness
- Pattern scales with strangeness content

### Effective Energy Analysis

**Concept:** Available energy = √s - forward energy (ZDC)

**Finding:** Lower forward energy → higher strangeness at fixed multiplicity

**Interpretation:** Initial-state partonic energy affects strangeness independently of final-state multiplicity

### Event-by-Event Distributions P(nS)

**First measurements in pp at √s = 5.02 TeV:**

| Hadron | Max per event |
|--------|--------------|
| K⁰S | 7 |
| Λ | 5 |
| Ξ | 4 |
| Ω | 2 |

### Model Comparisons

| Model | Performance |
|-------|-------------|
| PYTHIA 8 Monash | Underestimates all ratios |
| PYTHIA 8 Ropes | Best qualitative agreement |
| EPOS LHC | Fails at high multiplicity |

---

## 4. Model Predictions for O+O Collisions

**Source:** arXiv:2507.16266 (Singh et al.)

### Models Compared

#### EPOS4
- 3+1D viscous hydrodynamics
- Core-corona approach
- Lattice QCD equation of state
- vHLLE + UrQMD cascade
- ~3 million minimum-bias events

#### AMPT (Default and String Melting)
- HIJING initialization
- ZPC parton cascade
- String fragmentation (Def) or quark coalescence (SM)
- ART hadronic rescattering
- ~6 million events per version

### Key Predictions for O+O at √s_NN = 7 TeV

#### Charged Particle Multiplicity
- EPOS4: ~5 to ~230 (0-100% centrality)
- AMPT-SM: ~5 to ~185

#### Strangeness Production

**AMPT-Def:** Weak multiplicity dependence (fragmentation-dominated)

**AMPT-SM:** Strong multiplicity dependence (coalescence mechanism)

**EPOS4:**
- High multiplicity: Core-dominated → enhanced strangeness
- Low multiplicity: Corona-dominated → suppressed strangeness

### System Size Overlap

O+O predictions show multiplicity overlap with:
- **Small systems:** pp at 7 TeV, p-Pb at 5.02 TeV
- **Large systems:** Pb-Pb at 2.76 TeV

**Implication:** O+O bridges the gap between elementary and heavy-ion collisions

---

## Synthesized Key Equations

### 1. Bjorken Energy Density
```
ε_Bj(τ₀) ≈ (1/τ₀·A_⊥) · (dE_T/dy)|_{y=0}
```

### 2. Nuclear Modification Factor
```
R_AA(pT) = (1/N_coll) · (dN_AA/dpT) / (dN_pp/dpT)
```

### 3. Flow Coefficient Definition
```
dN/dφ ∝ 1 + 2·Σₙ vₙ·cos[n(φ - Ψₙ)]
```

### 4. Canonical Suppression Factor
```
Suppression = I_{|S|}(x) / I₀(x)
```

### 5. BDMPS-Z Energy Loss
```
ΔE ∝ q̂ · L²
```

### 6. Volume Scaling (Statistical Hadronization)
```
V = V₀ + α · (dNch/dη)
```

---

## Unified Physics Framework

### QGP Formation Threshold

**Emerging consensus from all sources:**

1. **No sharp threshold:** QGP formation is gradual, not discrete
2. **Multiplicity as driver:** dNch/dη is the critical parameter
3. **Small system QGP:** High-multiplicity pp may form QGP droplets
4. **O+O as bridge:** Connects small and large system physics

### Key Physics Questions Addressed by O+O

| Question | O+O Contribution |
|----------|-----------------|
| Minimum QGP size? | Test threshold between pPb (no quenching) and PbPb |
| Origin of v_n in small systems? | Better geometry control than pPb |
| Energy loss mechanism? | Detect 5-10% suppression predicted |
| Nuclear structure effects? | Probe alpha clustering in ¹⁶O |

### Hadronization Mechanisms

**Two competing pictures:**

1. **String fragmentation:** Dominates in AMPT-Def, weak multiplicity dependence
2. **Quark coalescence:** Dominates in AMPT-SM and EPOS4, strong multiplicity dependence

**Canonical suppression:** Bridges both pictures through statistical mechanics

---

## Invalid References (To Replace)

The following files in `references/` are mismatched and should be replaced with correct physics papers:

| File | Actual Content | Should Be |
|------|---------------|-----------|
| 2205.02321.pdf | ML/neural network pruning | QGP/hydrodynamics paper |
| 2306.06047.pdf | Solar corona MHD | Nuclear structure paper |
| 2307.16266.pdf | 4-manifold topology | O+O collision predictions |

**Recommendation:** Download correct papers from arXiv or remove these entries from references.bib.

---

## Citation Keys for references.bib

```bibtex
@article{Brewer:2021,
    author = "Brewer, Jasmine and Mazeliauskas, Aleksas and van der Schee, Wilke",
    title = "{Opportunities of OO and pO collisions at the LHC}",
    eprint = "2103.01939",
    archivePrefix = "arXiv",
    year = "2021"
}

@article{Sharma:2025,
    author = "Sharma, Natasha",
    title = "{Canonical treatment of strangeness and light nuclei production}",
    eprint = "2503.02677",
    archivePrefix = "arXiv",
    year = "2025"
}

@article{Pucillo:2025,
    author = "Pucillo, Sara and {ALICE Collaboration}",
    title = "{Recent results on strangeness enhancement in small collision systems with ALICE}",
    eprint = "2504.02527",
    archivePrefix = "arXiv",
    year = "2025"
}

@article{Singh:2025,
    author = "Singh, J. and Ashraf, M. U. and Khan, A. M. and Kabana, S.",
    title = "{Strangeness production in O+O collisions at √sNN=7 TeV}",
    eprint = "2507.16266",
    archivePrefix = "arXiv",
    year = "2025"
}
```
