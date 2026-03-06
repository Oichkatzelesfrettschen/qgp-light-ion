"""
constants.py - Single source of truth for physics constants.

Every constant used across the QGP light-ion project is defined here with:
  - Value and uncertainty
  - Units
  - Literature citation

Importing modules MUST use these values instead of hardcoding their own.
"""

from __future__ import annotations

# =============================================================================
# QCD PHASE TRANSITION
# =============================================================================

# Crossover temperature at mu_B = 0
# HotQCD Collaboration, Phys. Lett. B 795, 15 (2019)
T_C0_MEV: float = 156.5  # MeV
T_C0_MEV_ERR: float = 1.5  # MeV
T_C0_GEV: float = 0.1565  # GeV

# Crossover curvature: T_c(mu_B)/T_c(0) = 1 - kappa2*(mu_B/T_c)^2 - kappa4*(mu_B/T_c)^4
#
# ---- kappa2 determinations (three independent lattice methods) ----
#
# (A) Taylor expansion at mu_B=0, HISQ staggered fermions [PRIMARY]:
#     Bazavov et al. (HotQCD), Phys. Lett. B 795, 15 (2019); arXiv:1812.08235
#     kappa2 = 0.0120(20) [subtracted condensate]
#     kappa2 = 0.0123(30) [disconnected chiral susceptibility]
#     kappa4 = 0.000(4)   [consistent with zero]
#
# (B) Imaginary mu_B analytic continuation, stout staggered fermions:
#     Borsanyi et al. (Wuppertal-Budapest), PRL 125, 052001 (2020); arXiv:2002.02821
#     kappa2 = 0.0153(18), kappa4 = 0.00032(67)
#     Uses chiral susceptibility vs condensate "universal curve" for T_pc;
#     Nt=10,12,16 continuum extrapolation.
#
# (C) Mesonic correlation functions (V-AV degeneracy), Wilson fermions:
#     Smecca et al. (FASTSUM), PRD 112, 114509 (2025); arXiv:2412.20922
#     kappa = 0.0131(23)(23) [Gen 2 ensembles, stat+syst]
#     First hadronic-observable extraction. Not at physical pion mass;
#     no continuum extrapolation. Exploratory but consistent with (A) and (B).
#
# The ~25% spread (0.012 to 0.015) across methods reflects:
#   - Different analytic continuation schemes (Taylor vs imaginary mu_B)
#   - Different T_pc definitions (susceptibility peak vs universal curve)
#   - Different fermion discretizations (HISQ vs stout vs Wilson)
# All results are compatible within combined uncertainties (~1.3 sigma).
# We adopt (A) as the primary value because the Taylor method has the
# most controlled systematics at small mu_B and is the community default.
KAPPA2: float = 0.012
KAPPA2_ERR: float = 0.002

# Higher-order curvature coefficient.
# Note: kappa4 = 0.00032(67) comes from Borsanyi et al. (2020) [method B above].
# The HotQCD value (method A) gives kappa4 = 0.000(4), consistent with zero.
# We use the Wuppertal-Budapest central value as it has smaller uncertainty.
# Borsanyi et al., PRL 125, 052001 (2020); arXiv:2002.02821
KAPPA4: float = 0.00032

# Alternative extraction from mesonic correlation functions (method C above).
# Central value 0.0131, rounded to 0.013 here; within 1-sigma of primary value.
KAPPA2_MESONIC: float = 0.0131
KAPPA2_MESONIC_ERR_STAT: float = 0.0023
KAPPA2_MESONIC_ERR_SYST: float = 0.0023

# =============================================================================
# CRITICAL POINT EXCLUSION / ESTIMATES
# =============================================================================

# Yang-Lee edge singularity analysis EXCLUDES CP at mu_B < 450 MeV (2sigma)
# Borsanyi et al., PRD 112, L111505 (2025); arXiv:2502.10267
MU_B_CP_EXCLUDED_2SIGMA: float = 450.0  # MeV

# FRG consensus (QM2025): current best theoretical estimate
# Fu, Pawlowski, Rennecke, arXiv:2510.11270
T_CP_FRG_MEV: float = 110.0  # MeV (+/- 10%)
MU_B_CP_FRG_MEV: float = 630.0  # MeV (+/- 10%)

# =============================================================================
# QGP TRANSPORT COEFFICIENTS
# =============================================================================

# Shear viscosity to entropy density ratio
# Near KSS bound: 1/(4*pi) ~ 0.08
# JETSCAPE Collaboration; Bernhard et al., Nature Phys. 15, 1113 (2019)
ETA_OVER_S: float = 0.12  # dimensionless (1.5x KSS bound)
ETA_OVER_S_RANGE: tuple[float, float] = (0.08, 0.16)

# Jet transport coefficient at T ~ T_c
# JETSCAPE arXiv:2408.08247
QHAT_0: float = 2.0  # GeV^2/fm
QHAT_OVER_T3_RANGE: tuple[float, float] = (2.0, 4.0)  # at T=400 MeV

# Strong coupling constant (for BDMPS-Z energy loss)
ALPHA_S: float = 0.3

# =============================================================================
# FUNDAMENTAL / CONVERSION CONSTANTS
# =============================================================================

HBARC: float = 0.197  # hbar*c [GeV*fm]
FM_TO_GEV_INV: float = 1.0 / 0.197  # 1 fm = 1/0.197 GeV^-1

# KSS bound
KSS_BOUND: float = 0.0795775  # 1/(4*pi)

# =============================================================================
# NUCLEAR GEOMETRY
# =============================================================================

# Nucleon-nucleon inelastic cross section at LHC (sqrt(s) = 5.36 TeV)
SIGMA_NN_FM2: float = 7.0  # fm^2 (~70 mb)

# O-16 radius parameters
# ab initio + electron scattering: arXiv:2507.05853
O16_R0: float = 2.608  # fm
O16_A: float = 0.513  # fm (skin thickness)
O16_MASS_NUMBER: int = 16
O16_ATOMIC_NUMBER: int = 8

# Ne-20 radius parameters
# TGlauberMC v3.3; ATLAS arXiv:2509.05171
NE20_R0: float = 2.791  # fm
NE20_A: float = 0.535  # fm
NE20_BETA2: float = 0.45  # prolate deformation
NE20_MASS_NUMBER: int = 20
NE20_ATOMIC_NUMBER: int = 10

# Ar-40
AR40_R0: float = 3.427  # fm
AR40_A: float = 0.569  # fm
AR40_MASS_NUMBER: int = 40
AR40_ATOMIC_NUMBER: int = 18

# Xe-129
XE129_R0: float = 5.36  # fm
XE129_A: float = 0.59  # fm
XE129_BETA2: float = 0.18
XE129_MASS_NUMBER: int = 129
XE129_ATOMIC_NUMBER: int = 54

# Pb-208
PB208_R0: float = 6.62  # fm
PB208_A: float = 0.546  # fm
PB208_MASS_NUMBER: int = 208
PB208_ATOMIC_NUMBER: int = 82

# Nuclear radius scaling parameter
R0_SCALING: float = 1.25  # fm, R = r0 * A^(1/3)

# =============================================================================
# HYDRODYNAMIC FLOW RESPONSE
# =============================================================================

# Flow response coefficients: v_n = kappa_n * epsilon_n * damping
# Calibrated to 2025 CMS O-O/Ne-Ne measurements:
#   CMS arXiv:2510.02580 (O-O ultracentral v2 ~ 0.061)
#   ALICE arXiv:2509.06428 (Ne/O ratio ~ 1.08)
FLOW_KAPPA: dict[int, float] = {
    2: 0.35,
    3: 0.25,
    4: 0.15,
    5: 0.08,
}

# =============================================================================
# EXPERIMENTAL REFERENCE VALUES (LHC 5.36 TeV)
# =============================================================================

# O-O measurements
# CMS arXiv:2510.09864
RAA_MIN_OO: float = 0.69
RAA_MIN_OO_ERR: float = 0.04

# CMS arXiv:2510.02580
V2_ULTRACENTRAL_OO: float = 0.061

# Ne-Ne estimates
RAA_MIN_NENE: float = 0.65  # CMS estimate
V2_ULTRACENTRAL_NENE: float = 0.066

# ALICE arXiv:2509.06428
V2_NENE_OO_RATIO: float = 1.08

# =============================================================================
# THERMAL / FREEZE-OUT PARAMETERS
# =============================================================================

# Chemical freeze-out temperature
T_CHEM_GEV: float = 0.156  # GeV (approximately = T_c0)

# Kinetic freeze-out temperature
T_KIN_GEV: float = 0.100  # GeV

# Critical energy density for deconfinement
EPSILON_C: float = 1.0  # GeV/fm^3

# Default formation time
TAU_0: float = 0.6  # fm/c

# Typical q-hat for central Pb-Pb
QHAT_PBPB: float = 2.5  # GeV^2/fm

# =============================================================================
# PARTICLE MASSES (PDG 2024)
# =============================================================================

M_PION: float = 0.140  # GeV
M_KAON: float = 0.494  # GeV
M_PROTON: float = 0.938  # GeV
