# Data Manifest: QGP Light-Ion Project

**Author:** Deirikr Jaiusadastra Afrauthihinngreygaard
**Last Updated:** 2025-12-20
**Total Files:** 153
**Total Size:** 6.7 MB

---

## December 2025 Data Source Update

**Latest arXiv papers incorporated (December 2025):**

| Category | Paper | Key Result |
|----------|-------|------------|
| **Lattice QCD** | arXiv:2512.09415 | Finite density QCD without extrapolations (canonical formulation) |
| **Lattice QCD** | arXiv:2512.01126 | Net-baryon cumulants rule out CP at BES-II μ_B/T |
| **Lattice QCD** | arXiv:2412.20922 (Phys. Rev. D 112, 114509) | κ₂ ≈ 0.014 from mesonic correlators |
| **CP Exclusion** | arXiv:2502.10267 (Phys. Rev. D 112, L111505) | μ_B < 450 MeV excluded at 2σ |
| **FRG CEP** | arXiv:2510.11270 | CEP at T=110 MeV, μ_B=630 MeV (±10%) |
| **O-O R_AA** | arXiv:2510.09864 (CMS-HIN-25-008) | R_AA^min = 0.69 ± 0.04 |
| **O-O/Ne-Ne Flow** | arXiv:2509.06428 (ALICE) | Geometry-driven flow, v₂(Ne/O) ~ 1.08 |
| **O-O/Ne-Ne Flow** | arXiv:2510.02580 (CMS) | v₂{4} < v₂{2} confirms fluctuations |
| **O-O/Ne-Ne Flow** | arXiv:2509.05171 (ATLAS) | Prolate Ne-20 deformation confirmed |
| **Jet Quenching** | arXiv:2512.07169 | Bayesian q̂/κ = 0.25-0.8 from D-mesons |
| **Jet Quenching** | arXiv:2512.12715 | Thermal recoil jets (novel mechanism) |
| **Strangeness** | arXiv:2512.00671 | String closepacking mechanism |
| **Strangeness** | arXiv:2511.10413 | ALICE P(n_s) distributions |
| **Freeze-out** | arXiv:2511.15707 | PCA-Bayesian freeze-out extraction |
| **Freeze-out** | arXiv:2412.20517 | Two-component (2CFO) scenario |

---

## Data Classification System

All data in this project falls into three categories:

| Classification | Symbol | Description |
|----------------|--------|-------------|
| **MEASURED** | `[M]` | Experimental data from published results (HEPData, papers) |
| **PREDICTED** | `[P]` | Model calculations constrained by experimental inputs |
| **SCHEMATIC** | `[S]` | Illustrative data for pedagogical figures |

---

## 1. Experimental Data (`data/experimental/`)

**Classification: MEASURED**

These files contain actual experimental measurements from LHC collaborations.

### CMS O-O Jet Quenching (2025)

| File | Observable | Source |
|------|------------|--------|
| `CMS_OO_RAA_HIN25008.dat` | R_AA(pT) charged hadrons | CMS-HIN-25-008 |

**Provenance:**
- **Collaboration:** CMS
- **Report:** CMS-HIN-25-008
- **HEPData:** https://www.hepdata.net/record/ins3068407
- **DOI:** 10.17182/hepdata.165512
- **Conditions:** O-O at sqrt(s_NN) = 5.36 TeV, minimum-bias, |eta| < 1.0
- **Luminosity:** 6.1 nb^-1 (O-O), 1.02 pb^-1 (pp reference)

### CMS O-O and Ne-Ne Collective Flow (2025)

| File | Observable | Source |
|------|------------|--------|
| `CMS_OO_flow_HIN25009.dat` | v2, v3 vs N_trk^offline | CMS-HIN-25-009 |
| `CMS_NeNe_flow_HIN25009.dat` | v2, v3 vs N_trk^offline | CMS-HIN-25-009 |

**Provenance:**
- **Collaboration:** CMS
- **Report:** CMS-HIN-25-009, CERN-EP-2025-222
- **HEPData:** https://www.hepdata.net/record/ins3062822
- **DOI:** 10.17182/hepdata.165514
- **arXiv:** 2510.02580
- **Conditions:** O-O and Ne-Ne at sqrt(s_NN) = 5.36 TeV
- **Method:** Two-particle correlations with |Delta_eta| > 2

### ALICE O-O and Ne-Ne Anisotropic Flow (2025)

| File | Observable | Source |
|------|------------|--------|
| `ALICE_OO_NeNe_flow_2509.06428.dat` | v2{2}, v3{2}, v2{4} ratios | arXiv:2509.06428 |

**Provenance:**
- **Collaboration:** ALICE
- **Report:** CERN-EP-2025-203
- **arXiv:** 2509.06428
- **Status:** Preliminary (HEPData upload pending after publication)
- **Conditions:** O-O and Ne-Ne at sqrt(s_NN) = 5.36 TeV
- **Statistics:** ~3 billion O-O events, ~400 million Ne-Ne events
- **Method:** Two- and four-particle correlations, |eta| < 0.8

**Key Results:**
- First observation of geometry-driven hydrodynamic flow in small systems
- v2{2}(Ne-Ne/O-O) ~ 1.08 (ultracentral) → 1.05 (10% centrality)
- Confirms prolate "bowling pin" shape of 20Ne nucleus
- Agreement with hydrodynamic models (Trajectum, IP-Glasma+JIMWLK)

---

## 2. Model Data Overview

All files outside `data/experimental/` are model-generated using `src/qgp_physics.py` and `src/generate_comprehensive_data.py`. Model parameters are constrained by experimental inputs where available.

### Classification Summary by Directory

| Directory | Classification | Physics Basis |
|-----------|---------------|---------------|
| `1d_spectra/` | PREDICTED | Blast-wave + thermal models |
| `2d_correlations/` | PREDICTED | Two-particle correlations |
| `3d_spacetime/` | PREDICTED | (2+1)D hydrodynamic evolution |
| `4d_parameters/` | SCHEMATIC | Parameter space exploration |
| `comparison/` | PREDICTED | Bjorken formula calculations |
| `femtoscopy/` | PREDICTED | HBT correlation functions |
| `flow/` | PREDICTED | Hydro + transport hybrid |
| `jet_quenching/` | PREDICTED | BDMPS-Z energy loss |
| `nuclear_geometry/` | PREDICTED | Woods-Saxon + Glauber |
| `phase_diagram/` | SCHEMATIC | QCD phase boundaries |
| `photons/` | PREDICTED | Thermal + prompt production |
| `physics_connections/` | SCHEMATIC | Scaling relations |
| `spacetime/` | PREDICTED | Hydrodynamic profiles |
| `strangeness/` | PREDICTED | Canonical suppression |

---

## 3. Detailed File Inventory

### 3.1 Jet Quenching (`data/jet_quenching/`)

**Classification: PREDICTED**
**Physics Model:** BDMPS-Z radiative energy loss with geometry from Glauber

| File | Observable | System | Notes |
|------|------------|--------|-------|
| `RAA_PbPb.dat` | [P] R_AA(pT) | Pb-Pb 5.02 TeV | Constrained by ALICE/CMS data |
| `RAA_OO.dat` | [P] R_AA(pT) | O-O 5.36 TeV | Scaled from Pb-Pb via path length |
| `RAA_NeNe.dat` | [P] R_AA(pT) | Ne-Ne 5.36 TeV | Includes deformation effects |
| `RAA_XeXe.dat` | [P] R_AA(pT) | Xe-Xe 5.44 TeV | Validated against CMS data |
| `RAA_pPb.dat` | [P] R_AA(pT) | p-Pb 5.02 TeV | Cold nuclear matter baseline |
| `RAA_pp.dat` | [P] R_AA(pT) | pp reference | Unity (by definition) |
| `RAA_vs_Npart.dat` | [P] R_AA vs N_part | Multiple | System-size scaling |
| `energy_loss_qhat*.dat` | [S] Delta_E(L) | Pb-Pb | q-hat sensitivity study |

**Model Inputs:**
- Transport coefficient: q-hat = 1.5-4.5 GeV^2/fm (Pb-Pb)
- Initial temperature: T_0 = 300-400 MeV
- Thermalization time: tau_0 = 0.6-1.0 fm/c

### 3.2 Collective Flow (`data/flow/`)

**Classification: PREDICTED**
**Physics Model:** (2+1)D viscous hydrodynamics + UrQMD afterburner

| File | Observable | System | Notes |
|------|------------|--------|-------|
| `vn_vs_cent_Pb.dat` | [P] v2, v3, v4 vs centrality | Pb-Pb | eta/s = 0.08-0.16 |
| `vn_vs_cent_O.dat` | [P] v2, v3, v4 vs centrality | O-O | Alpha-cluster geometry |
| `vn_vs_cent_Ne.dat` | [P] v2, v3, v4 vs centrality | Ne-Ne | Prolate deformation |
| `v2_pT_*_pion.dat` | [P] v2(pT) | O/Pb | Mass ordering |
| `v2_pT_*_kaon.dat` | [P] v2(pT) | O/Pb | Mass ordering |
| `v2_pT_*_proton.dat` | [P] v2(pT) | O/Pb | Mass ordering |
| `azimuthal_*.dat` | [S] dN/dphi | - | Illustrative |

**Model Inputs:**
- Shear viscosity: eta/s = 1-2 x (1/4pi) KSS bound
- Initial eccentricity: epsilon_2, epsilon_3 from Glauber MC
- Freeze-out temperature: T_fo = 150 MeV

### 3.3 Nuclear Geometry (`data/nuclear_geometry/`)

**Classification: PREDICTED**
**Physics Model:** Woods-Saxon + Glauber Monte Carlo

| File | Observable | System | Notes |
|------|------------|--------|-------|
| `woods_saxon_O.dat` | [P] rho(r) | ^16O | R = 2.608 fm, a = 0.513 fm |
| `woods_saxon_Ne.dat` | [P] rho(r) | ^20Ne | Prolate deformation beta_2 = 0.45 |
| `woods_saxon_Pb.dat` | [P] rho(r) | ^208Pb | R = 6.62 fm, a = 0.546 fm |
| `woods_saxon_Xe.dat` | [P] rho(r) | ^129Xe | Reference system |
| `woods_saxon_Ar.dat` | [P] rho(r) | ^40Ar | Future LHC run |
| `glauber_nucleons_*.dat` | [P] (x,y) positions | O/Ne/Pb | Event samples |
| `density_2d_*.dat` | [P] rho(x,y) | O/Ne/Pb | 2D projections |
| `eccentricity_dist_*.dat` | [P] P(epsilon_n) | O/Ne | Fluctuation distributions |
| `oxygen_alpha_clusters.dat` | [P] alpha positions | ^16O | 4-alpha structure |
| `nuclear_parameters.dat` | [P] Summary table | All | R, a, beta_2 values |

**Primary Sources:**
- Oxygen: ab initio nuclear structure (arXiv:2507.05853)
- Neon: NNDC evaluated data + deformation from Moller-Nix
- Lead: ALICE Glauber parameters

### 3.4 Femtoscopy (`data/femtoscopy/`)

**Classification: PREDICTED**
**Physics Model:** Gaussian HBT correlation functions

| File | Observable | System | Notes |
|------|------------|--------|-------|
| `correlation_*.dat` | [P] C(q) | pp/O-O/Pb-Pb | 1D correlation |
| `correlation_3d_*.dat` | [P] C(q_out/side/long) | Pb-Pb | 3D decomposition |
| `hbt_radii_vs_cent_*.dat` | [P] R_out/side/long(cent) | All | Centrality dependence |
| `hbt_system_size_scaling.dat` | [P] R vs (dN/deta)^{1/3} | All | Universal scaling |
| `hbt_radius_vs_kT_*.dat` | [P] R(k_T) | O-O/Pb-Pb | m_T scaling |

**Model Inputs:**
- Source lifetime: tau ~ R/c
- Homogeneity lengths from hydro freeze-out

### 3.5 Direct Photons (`data/photons/`)

**Classification: PREDICTED**
**Physics Model:** Thermal emission + prompt NLO pQCD

| File | Observable | System | Notes |
|------|------------|--------|-------|
| `direct_photon_spectrum_*.dat` | [P] dN/dpT | All | Thermal + prompt |
| `photon_decomposition_*.dat` | [P] Components | O-O/Pb-Pb | Thermal/prompt breakdown |
| `photon_ratio_R_gamma_*.dat` | [P] R_gamma | All | Excess over pQCD |
| `photon_v2_PbPb.dat` | [P] v2^gamma(pT) | Pb-Pb | Flow coefficient |
| `effective_temperature_vs_system.dat` | [P] T_eff | All | Inverse slope |
| `thermal_photon_inverse_slope_fit.dat` | [S] Fit results | - | Illustrative |

**Model Inputs:**
- Initial temperature: T_0 from Bjorken estimate
- QGP emission rates: Arnold-Moore-Yaffe (AMY)

### 3.6 Strangeness Enhancement (`data/strangeness/`)

**Classification: PREDICTED**
**Physics Model:** Canonical statistical mechanics

| File | Observable | System | Notes |
|------|------------|--------|-------|
| `canonical_suppression_S*.dat` | [P] gamma_s(V) | - | S=1,2,3 suppression |
| `enhancement_vs_mult.dat` | [P] Yield ratios | All | Multi-strange baryons |
| `systems_on_curve.dat` | [P] System positions | All | On universal curve |

**Model Inputs:**
- Canonical correlation volume V_c
- Chemical freeze-out: T = 156 MeV, mu_B ~ 0

### 3.7 Phase Diagram (`data/phase_diagram/`)

**Classification: SCHEMATIC**
**Purpose:** Pedagogical illustration of QCD phase structure

| File | Observable | Notes |
|------|------------|-------|
| `crossover.dat` | [S] T_c(mu_B) | Lattice QCD constrained (mu_B < 300 MeV) |
| `first_order.dat` | [S] T_c(mu_B) | High mu_B extrapolation |
| `critical_point.dat` | [S] (mu_B, T) position | Uncertain location |
| `collision_systems.dat` | [S] System trajectories | LHC/RHIC/SPS |

**Note:** Phase boundary location at high mu_B is not experimentally established.

### 3.8 Spacetime Evolution (`data/3d_spacetime/`, `data/spacetime/`)

**Classification: PREDICTED**
**Physics Model:** (2+1)D boost-invariant hydrodynamics

| File | Observable | Notes |
|------|------------|-------|
| `epsilon_xy_tau*.dat` | [P] epsilon(x,y,tau) | Energy density evolution |
| `temperature_xy_tau*.dat` | [P] T(x,y,tau) | Temperature profiles |
| `flow_velocity_tau*.dat` | [P] u^mu(x,y) | Radial flow |
| `freeze_out_surface.dat` | [P] Sigma^mu | Hypersurface |
| `tau_values.dat` | [P] Proper time grid | tau = 0.6-12 fm/c |
| `energy_density_*_b*.dat` | [P] epsilon(x,y) | Impact parameter scans |
| `qgp_lifetime.dat` | [P] tau_QGP(system) | Lifetime vs system size |
| `temperature_evolution_*.dat` | [P] T(tau) at origin | Cooling curves |

**Model Inputs:**
- EOS: s95p-v1 (lattice QCD + hadron resonance gas)
- Initial conditions: Glauber entropy deposition

### 3.9 1D Spectra (`data/1d_spectra/`)

**Classification: PREDICTED**
**Physics Model:** Blast-wave + Cooper-Frye

| File | Observable | Notes |
|------|------------|-------|
| `pt_spectrum_*.dat` | [P] dN/dpT | pi/K/p spectra |
| `rapidity_dist_*.dat` | [P] dN/dy | Pb-Pb/O-O |
| `temperature_evolution_*.dat` | [P] T(tau) | Central region |
| `energy_loss_dist_*.dat` | [P] P(Delta_E) | Parton energy loss |
| `mult_dist_*_central.dat` | [P] P(N_ch) | Multiplicity fluctuations |

### 3.10 2D Correlations (`data/2d_correlations/`)

**Classification: PREDICTED**

| File | Observable | Notes |
|------|------------|-------|
| `C2_deta_dphi_*.dat` | [P] C(Delta_eta, Delta_phi) | Ridge structure |
| `HBT_qout_qside_*.dat` | [P] C(q_out, q_side) | 2D HBT |
| `v2_v3_scatter_*.dat` | [P] Event-by-event (v2, v3) | Flow correlations |
| `v2_v3_hist_*.dat` | [P] P(v2), P(v3) | Distributions |
| `RAA_cent_pT_PbPb.dat` | [P] R_AA(cent, pT) | 2D suppression map |

### 3.11 Parameter Studies (`data/4d_parameters/`)

**Classification: SCHEMATIC**
**Purpose:** Multi-observable correlations for visualization

| File | Observable | Notes |
|------|------------|-------|
| `v2_etaS_cent_*.dat` | [S] v2(eta/s, cent) | Parameter sensitivity |
| `multiobs_events.dat` | [S] Multi-dimensional data | PCA input |
| `correlation_matrix.dat` | [S] Observable correlations | Covariance |
| `pca_*.dat` | [S] Principal components | Dimensionality reduction |
| `parallel_coords.dat` | [S] Parallel coordinates | Visualization |

### 3.12 Physics Connections (`data/physics_connections/`)

**Classification: SCHEMATIC**
**Purpose:** Pedagogical scaling relations

| File | Observable | Notes |
|------|------------|-------|
| `knudsen_scaling.dat` | [S] v2/epsilon_2 vs Kn^{-1} | Hydro validity |
| `knudsen_systems.dat` | [S] System Knudsen numbers | pp to Pb-Pb |
| `v2_vs_eps2_*.dat` | [S] Linear response | Flow/geometry relation |
| `energy_loss_vs_L.dat` | [S] Delta_E(L) | Path length dependence |
| `canonical_suppression_vs_V.dat` | [S] gamma_s(V) | Strangeness canonical |
| `qgp_signature_onset.dat` | [S] Observable thresholds | Signature onset vs system size |

### 3.13 System Comparison (`data/comparison/`)

**Classification: PREDICTED**

| File | Observable | Notes |
|------|------------|-------|
| `bjorken_energy_density.dat` | [P] epsilon_Bj(system) | From measured dE_T/dy |
| `multiplicity_scaling.dat` | [P] dN_ch/deta scaling | System comparison |

---

## 4. Root-Level Data Files

| File | Classification | Observable | Notes |
|------|---------------|------------|-------|
| `energy_density_2d.dat` | [P] | epsilon(x,y) | Main visualization |
| `energy_density_2d_50.dat` | [P] | epsilon(x,y) at 50% | Time slice |
| `energy_density_2d_75.dat` | [P] | epsilon(x,y) at 75% | Time slice |
| `energy_density_2d_100.dat` | [P] | epsilon(x,y) at 100% | Final state |
| `RAA_OO.dat` | [P] | R_AA(pT) O-O | Quick access |
| `RAA_PbPb.dat` | [P] | R_AA(pT) Pb-Pb | Quick access |
| `flow_v2_OO.dat` | [P] | v2(cent) O-O | Quick access |
| `flow_v2_NeNe.dat` | [P] | v2(cent) Ne-Ne | Quick access |
| `flow_v3_OO.dat` | [P] | v3(cent) O-O | Quick access |

---

## 5. Data Generation

### 5.1 Generator Scripts

| Script | Purpose | Output Directory |
|--------|---------|------------------|
| `src/generate_comprehensive_data.py` | Main data generator | All subdirectories |
| `src/qgp_physics.py` | Physics model library | (imported) |

### 5.2 Regenerating Data

```bash
make data          # Regenerate all model data
make clean-data    # Remove generated data (preserves experimental/)
make               # Full rebuild including data
```

### 5.3 Adding Experimental Data

1. Download from HEPData or extract from publications
2. Create file in `data/experimental/` with provenance header
3. Include: DATA TYPE, SOURCE (collaboration, report, DOI), CONDITIONS, COLUMNS
4. Update this manifest

---

## 6. External Data Sources

### 6.1 HEPData Records Used

| Record | Collaboration | Observable | Our File |
|--------|--------------|------------|----------|
| ins3068407 | CMS | O-O R_AA | `CMS_OO_RAA_HIN25008.dat` |
| ins3062822 | CMS | O-O/Ne-Ne v_n | `CMS_OO_flow_HIN25009.dat`, `CMS_NeNe_flow_HIN25009.dat` |

### 6.2 Additional HEPData Records (Not Yet Downloaded)

| Record | Collaboration | Observable | Status |
|--------|--------------|------------|--------|
| ins2864789 | ALICE | O-O multiplicity | Available |
| TBD | ALICE | O-O/Ne-Ne flow | arXiv:2509.06428 |
| TBD | ALICE | pi0 R_AA | arXiv:2511.22139 |

### 6.3 Lattice QCD

- **Crossover temperature:** T_c = 156.5 ± 1.5 MeV (HotQCD PLB 795, 2019)
- **Curvature coefficient:** κ₂ = 0.012(2) (Bazavov et al. 2020; confirmed by Smecca et al. PRD 112, 2025)
- **CP exclusion:** μ_B < 450 MeV excluded at 2σ (Borsányi et al. PRD 112, L111505, 2025)
- **FRG consensus CEP:** T = 110 MeV, μ_B = 630 MeV ± 10% (Fu et al. arXiv:2510.11270)
- **Finite density:** Canonical formulation (Adam et al. arXiv:2512.09415) eliminates Taylor expansion
- **BES-II constraints:** Net-baryon cumulants rule out CP at μ_B/T ≤ 2 (Goswami & Karsch arXiv:2512.01126)
- **EOS:** s95p-v1 parameterization; extended EOS coverage to μ_B/T ~ 3.5 (arXiv:2504.01881)

### 6.4 Nuclear Structure

- Oxygen: ab initio (arXiv:2507.05853, Hagen et al.)
- Neon: NNDC + Moller-Nix mass table
- Lead: ALICE Glauber MC standard parameters

---

## 7. Version Control Notes

- `data/experimental/` tracked in git (small files with provenance)
- `data/*.dat` and subdirectories gitignored (regenerated by `make data`)
- `.generated` timestamp file indicates last regeneration

---

## 8. Quality Assurance

### 8.1 Experimental Data Validation

All experimental data files include:
- Full provenance header with DOI/HEPData link
- Column definitions with units
- Systematic and statistical uncertainties
- Notes on experimental conditions

### 8.2 Model Data Validation

Model outputs are validated against:
- Published experimental results where available
- Consistency checks (e.g., R_AA -> 1 at high pT)
- Physical constraints (0 < v_n < 1, R_AA > 0)

---

*This manifest was generated on 2025-12-16. For updates, regenerate with the latest data.*
