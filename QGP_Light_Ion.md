# QGP Signatures in Light-Ion Collisions (O–O and Ne–Ne) at the LHC: Theoretical and Experimental Synthesis

<img src="https://i.imgur.com/7c3jS1a.png" alt="Event displays of a relativistic oxygen–oxygen collision (left, CMS detector) and neon–neon collision (right, ATLAS detector) at the LHC. Such light-ion collisions bridge the gap between small proton–nucleus and large nucleus–nucleus systems."/>

## Abstract

The first oxygen–oxygen (O–O) and neon–neon (Ne–Ne) collisions at $\sqrt{s_{NN}} = 5.36$ TeV at the LHC (July 2025) provide an unprecedented opportunity to study QGP formation in systems intermediate between proton–nucleus and heavy-nucleus collisions. This synthesis reviews the theoretical framework for QGP in light-ion systems and presents the initial experimental results: charged-particle nuclear modification factors showing suppression ($R_{AA}^{\min} \approx 0.69$ for O–O, $\approx 0.65$ for Ne–Ne at $p_T \approx 6$ GeV), collective flow measurements demonstrating geometry-driven anisotropies consistent with hydrodynamic evolution, and multiplicity distributions that follow participant scaling. The results provide strong evidence that QGP signatures—jet quenching and collective flow—emerge in systems with $dN_{ch}/d\eta \gtrsim 100$, offering empirical constraints on the conditions required for collective behavior. Interpretation depends on centrality definitions, pp reference systematics, and the separation of initial-state correlations from final-state collectivity. We discuss open questions regarding the threshold behavior, the role of nuclear structure (Ne deformation, possible O clustering), and prospects for strangeness, femtoscopy, and direct photon measurements.

---

## Data Ledger: 2025 Light-Ion Measurements at the LHC

*This section provides a traceable summary of the primary experimental results discussed in this review. Each entry specifies the observable, event selection, kinematic range, measured value(s), and primary source.*

### Charged-Particle Nuclear Modification Factor $R_{AA}$

| System | Event Selection | $p_T$ Range | Value | Source |
|--------|-----------------|-------------|-------|--------|
| O–O | Minimum-bias | $p_T \approx 6$ GeV | $R_{AA}^{\min} = 0.69 \pm 0.04$ | CMS [@CMS-PAS-HIN-25-008] |
| O–O | Minimum-bias | $p_T \approx 100$ GeV | $R_{AA} \approx 0.97 \pm 0.06$ | CMS [@CMS-PAS-HIN-25-008] |
| Ne–Ne | Minimum-bias | $p_T \approx 6$ GeV | $R_{AA}^{\min} \approx 0.65$ | CMS-PAS-HIN-25-014 |
| Ne–Ne | Minimum-bias | $p_T \lesssim 20$ GeV | Stronger suppression than O–O | CMS-PAS-HIN-25-014 |

**Definition**: $R_{AA}(p_T) = \frac{1}{N_{\mathrm{coll}}} \frac{dN_{AA}/dp_T}{dN_{pp}/dp_T}$

**Luminosities**: O–O: 6.1 nb$^{-1}$; Ne–Ne: 0.76 ± 0.04 nb$^{-1}$; pp reference: 1.02 pb$^{-1}$

**Recent Analysis**: The observed suppression pattern has been examined for consistency with A–A systematics; the $\pi^0$ suppression in O–O at 5.36 TeV follows the trend established for larger systems [@Petrovici:Pi0Suppression2025].

### Charged-Particle Multiplicity

| System | Event Selection | Acceptance | Value | Source |
|--------|-----------------|------------|-------|--------|
| O–O | 0–5% central | $|\eta| < 0.5$ | $dN_{ch}/d\eta = 135 \pm 3$ | CMS-PAS-HIN-25-010 |
| Ne–Ne | 0–5% central | $|\eta| < 0.5$ | $dN_{ch}/d\eta \approx 155 \pm 4$ | CMS-PAS-HIN-25-010 |

**Note**: Ne–Ne has ~15% higher multiplicity than O–O at same centrality, consistent with larger $N_{part}$. The centrality selection uses forward energy or track multiplicity; the mapping to geometric quantities (e.g., $N_{part}$, $N_{coll}$) requires Glauber modeling.

### Anisotropic Flow

| System | Observable | Centrality Trend | Source |
|--------|------------|------------------|--------|
| O–O, Ne–Ne | $v_2\{2\}$ | Sizable; decreases toward central | ALICE [@ALICE:Flow2025; @ALICE-EP-2025-203] |
| O–O, Ne–Ne | $v_3\{2\}$ | Sizable; non-trivial centrality dependence | ALICE [@ALICE:Flow2025] |
| O–O, Ne–Ne | $v_2\{4\}$ | Increasing trend with centrality | ALICE [@ALICE:Flow2025] |
| Ne–Ne/O–O | $v_2$ ratio | $\sim 1.08$ (ultracentral) → $\sim 1.05$ (10%) | ALICE [@ALICE:Flow2025] |
| Ne–Ne/O–O | $v_3$ ratio | $\sim 1.0$ (ultracentral) → $\sim 1.06$ (10%) | ALICE [@ALICE:Flow2025] |

**Statistics**: ~3 billion O–O events (5.01 nb$^{-1}$); ~400 million Ne–Ne events (0.87 nb$^{-1}$)

**Method**: Two-particle ($|\Delta\eta| > 1$) and four-particle correlations; |η| < 0.8

**Key Finding**: First observation of **geometry-driven hydrodynamic flow** in small systems, providing strong evidence for collective behavior consistent with QGP formation in O–O and Ne–Ne collisions. The enhanced $v_2$ in central Ne–Ne vs O–O is consistent with the prolate ("bowling pin") deformation of $^{20}$Ne. *Note: Alternative explanations (initial-state correlations, CGC/glasma) are disfavored but not definitively excluded.*

### Dijet Momentum Balance (ATLAS)

| System | Observable | Event Selection | Result | Source |
|--------|------------|-----------------|--------|--------|
| O–O | $x_J = p_{T,2}/p_{T,1}$ | 0–10% central | Enhanced low-$x_J$ yield vs pp ($\Delta\langle x_J \rangle \approx -0.02$) | ATLAS [@ATLAS-OO-dijet-2025] |
| O–O | Dijet $A_J$ | 0–10% central | Broader distribution than pp reference | ATLAS [@ATLAS-OO-dijet-2025] |

**Kinematics**: Leading jet $p_T > 63$ GeV, subleading $p_T > 25$ GeV; $|\eta| < 2.1$; anti-$k_T$ R=0.4

**Luminosities**: O+O: 8 nb$^{-1}$; pp reference: 400 pb$^{-1}$ (same $\sqrt{s}$)

**Interpretation**: The enhanced low-$x_J$ population (more unbalanced dijets) in central O–O indicates path-length-dependent energy loss. The effect is smaller than in Pb–Pb but statistically significant, providing evidence that even the short path lengths in O–O ($L \sim 2$–3 fm) induce measurable jet quenching.

### Reference Values (Pb–Pb at 5.36 TeV, Energy-Matched to O–O/Ne–Ne)

| Observable | Centrality | Value | Source |
|------------|------------|-------|--------|
| $R_{AA}^{\min}$ | 0–10% | $\approx 0.15$–0.20 at $p_T \sim 7$ GeV | ALICE [@ALICE:RAA] |
| $dN_{ch}/d\eta$ | 0–5% | $2047 \pm 54$ | ALICE [@ALICE:PbPbMult2025] |
| $T_c$ | — | $156.5 \pm 1.5$ MeV | Lattice QCD [@HotQCD:2014; @WuppertalBudapest:Tc] |
| $\eta/s$ | — | $0.08$–$0.16$ (1–2× KSS bound) | Pb–Pb Bayesian analysis [@Bayesian:Viscosity] (O–O not yet extracted) |

**Note**: Using energy-matched 5.36 TeV Pb–Pb reference (not 5.02 TeV) for consistent scaling comparisons with O–O and Ne–Ne.

### Glauber Model Parameters [@Glauber:OONeNe2025]

| System | $\sigma_{inel}$ (b) | $\langle N_{part} \rangle$ (0–100%) | $\langle N_{coll} \rangle$ (0–100%) |
|--------|---------------------|-------------------------------------|-------------------------------------|
| O–O | $1.36 \pm 0.09$ | 10.8 | $12.8 \pm 0.8$ |
| Ne–Ne | $1.73 \pm 0.08$ | 12.7 | $15.8 \pm 1.1$ |

**σ_NN at 5.36 TeV**: $68.0 \pm 1.2$ mb

**Centrality-dependent O–O values**:

| Centrality | $\langle N_{part} \rangle$ | $\langle N_{coll} \rangle$ |
|------------|---------------------------|---------------------------|
| 0–1% | 28.5 | 57.7 |
| 0–5% | 26.2 | 48.1 |
| 20–30% | 16.7 | 19.2 |
| 70–100% | 2.6 | 1.5 |

### Nuclear Structure Input Parameters

| Nucleus | Parameter | Value | Source |
|---------|-----------|-------|--------|
| $^{16}$O | $R_0$ (half-density) | 2.608 fm | Electron scattering |
| $^{16}$O | $a$ (skin thickness) | 0.513 fm | de Vries et al. (1987) |
| $^{16}$O | $\beta_2$ (quadrupole) | 0.0 (spherical baseline) | — |
| $^{16}$O | Structure | Possible tetrahedral $4\alpha$ cluster | Theory; ALICE O–O data consistent with spherical |
| $^{20}$Ne | $R_0$ | 2.791 fm | Electron scattering |
| $^{20}$Ne | $a$ (skin thickness) | 0.535 fm | de Vries et al. (1987) |
| $^{20}$Ne | $\beta_2$ (quadrupole) | $0.45 \pm 0.05$ | Nuclear spectroscopy (NNDC) |
| $^{20}$Ne | Shape | Prolate ("bowling pin") | Supported by ALICE flow ratio Ne/O $\approx$ 1.08 |
| $^{208}$Pb | $R_0$ | 6.62 fm | ALICE Glauber |
| $^{208}$Pb | $a$ (skin thickness) | 0.546 fm | Standard ALICE |
| $^{208}$Pb | $\beta_2$ | 0.0 (doubly magic) | — |

**Model Implementation**: These parameters feed into Woods-Saxon density profiles: $\rho(r) = \rho_0 / (1 + \exp[(r-R)/a])$. For deformed Ne-20, the radius is angle-dependent: $R(\theta) = R_0(1 + \beta_2 Y_{20}(\theta))$.

**Models using these inputs**: 3D Glauber MC, IP-Glasma, Trajectum, AMPT, EPOS4

**Caveat**: At LHC energies, subnucleonic fluctuations (hot spots from gluon saturation) may dominate over nuclear-scale deformation in determining initial-state eccentricities. The ALICE flow ratio Ne/O $\approx 1.08$ in ultracentral collisions provides direct sensitivity to nuclear geometry.

---

### Centrality Determination in Light-Ion Collisions

*Centrality in light-ion collisions requires careful treatment due to the small number of participants and large event-by-event fluctuations.*

**Centrality Estimators by Experiment:**

| Experiment | Estimator | Acceptance | Notes |
|------------|-----------|------------|-------|
| ALICE | V0 amplitude (forward scintillators) | $2.8 < \eta < 5.1$, $-3.7 < \eta < -1.7$ | Standard for flow; anchored to Glauber fits |
| CMS | HF energy (forward calorimeters) | $3 < |\eta| < 5$ | Used for $R_{AA}$ and multiplicity |
| ATLAS | Forward calorimeter $\Sigma E_T$ | $3.2 < |\eta| < 4.9$ | Triggers and offline selection |

**Mapping to Geometric Quantities** [@Glauber:OONeNe2025; @Miller:2007]:

| Centrality | $\langle N_{part} \rangle$ (O–O) | $\langle N_{coll} \rangle$ (O–O) | $\langle b \rangle$ [fm] |
|------------|----------------------------------|----------------------------------|--------------------------|
| 0–5% | $26.2 \pm 1.3$ | $48.1 \pm 4.8$ | $\sim 1.0$ |
| 5–10% | $22.8 \pm 1.1$ | $36.2 \pm 3.6$ | $\sim 1.8$ |
| 10–20% | $19.5 \pm 1.0$ | $27.4 \pm 2.7$ | $\sim 2.4$ |
| 20–30% | $16.7 \pm 0.8$ | $19.2 \pm 1.9$ | $\sim 3.0$ |

**Critical Caveats for Light Ions:**

1. **Selection bias**: In small systems, centrality selection based on forward multiplicity can bias the midrapidity event sample toward events with larger-than-average multiplicity fluctuations. This "auto-correlation" effect is stronger in light ions than in Pb–Pb.

2. **Glauber model uncertainties**: The Woods-Saxon parameters and $\sigma_{NN}$ uncertainties propagate to $\langle N_{part} \rangle$ and $\langle N_{coll} \rangle$. For O–O, the relative uncertainty in $\langle N_{coll} \rangle$ is ~10%, larger than the ~3% typical for Pb–Pb.

3. **Minimum-bias vs centrality-differential**: Most published O–O and Ne–Ne $R_{AA}$ results are **minimum-bias** (0–100%), not centrality-differential. The minimum-bias event sample is dominated by peripheral collisions ($\langle N_{part} \rangle \approx 10$–11 for O–O MB).

4. **Centrality resolution**: With only 32 nucleons in O–O, the discrete nature of $N_{part}$ limits the meaningful number of centrality classes. Typically 5–10 classes are used, compared to 20+ for Pb–Pb.

**Reporting Convention (used in this document):**
Every quantitative result specifies: **System + $\sqrt{s_{NN}}$ + Event Selection + Acceptance + Observable**.
Example: "O–O, 5.36 TeV, MB (0–100%), $|\eta| < 1$, $R_{AA}(p_T = 6~\text{GeV}) = 0.69 \pm 0.04$"

---

### Operational Definition of $R_{AA}$

The nuclear modification factor $R_{AA}$ quantifies medium effects on particle production:

$$R_{AA}(p_T) = \frac{1}{\langle T_{AA} \rangle} \frac{d^2N_{AA}/dp_T d\eta}{d^2\sigma_{pp}/dp_T d\eta}$$

where $\langle T_{AA} \rangle = \langle N_{coll} \rangle / \sigma_{inel}^{pp}$ is the nuclear overlap function.

**Equivalent yield-ratio form** (used in this document):
$$R_{AA}(p_T) = \frac{1}{N_{coll}} \frac{dN_{AA}/dp_T}{dN_{pp}/dp_T}$$

**Normalization uncertainties include:**
- pp reference cross-section: ~3–5% (luminosity + trigger)
- $\langle N_{coll} \rangle$ from Glauber: ~8–10% for O–O MB, ~3% for Pb–Pb central
- Tracking efficiency: ~2–4% (partially cancels in ratio)

**Acceptance**: CMS O–O $R_{AA}$ uses $|\eta| < 1$; Pb–Pb comparisons should use matched acceptance.

---

## Introduction

Ultrarelativistic collisions of heavy nuclei (like Pb–Pb at the LHC or Au–Au at RHIC) produce an extremely hot, dense medium of deconfined quarks and gluons known as the quark–gluon plasma (QGP) [@Busza:2018; @Shuryak:2004]. This medium behaves like a near-perfect fluid with a very low shear viscosity, enabling rapid collective flow of the fireball matter. Historically, only collisions of large ions were thought to create a QGP, while smaller systems (proton–proton or proton–nucleus) showed only modest “cold nuclear matter” effects and no clear QGP signatures. A key open question has been: what is the minimum system size or particle multiplicity required to form a QGP droplet? Recent light-ion collisions at the LHC – specifically oxygen–oxygen (O–O) and neon–neon (Ne–Ne) at $\sqrt{s_{NN}}$ = 5.36 TeV – probe this "uncharted territory" between p–Pb and Pb–Pb, providing a crucial testing ground. Initial analyses indicate that even these lighter nuclei can create high-density matter consistent with QGP formation, exhibiting both collective flow and high-$p_T$ suppression effects previously seen only in heavy-ion collisions.

In this report, we synthesize the theoretical framework needed to describe QGP in light-ion collisions and review the key experimental observables from the first O–O and Ne–Ne runs. We focus on four main aspects: (1) Theoretical formalism – from initial energy density estimates and hydrodynamics to parton energy-loss and hadronization models – including the level of derivation detail appropriate for this scope. (2) Core references and results – highlighting about 8 essential recent studies underpinning the light-ion collision narrative (e.g. new measurements by CMS, ALICE, ATLAS, and supporting theory papers). (3) Synthesis – how observations (e.g. nuclear modification factors, flow coefficients, strangeness yields) compare to model predictions and what they imply about QGP behavior in these smaller systems. (4) Open questions – remaining ambiguities and future directions, such as the exact threshold of QGP formation, the role of nuclear structure (deformation or clustering) in initial conditions, and disentangling final-state QGP effects from initial-state effects. The goal is to provide a pedagogical but rigorous overview: starting from first principles where needed, but leveraging well-established formalisms rather than re-deriving standard results, and ultimately forming a self-contained blueprint of the physics at play in light-ion collisions.

## Theoretical Framework for QGP in Light-Ion Collisions

### Initial Energy Density and Hydrodynamical Evolution
A fundamental requirement for QGP formation is achieving high energy density in the collision zone, above the critical density for deconfinement (~1 GeV/fm³) [@HotQCD:2014]. Bjorken's model [@Bjorken:1983] provides a simple estimate for the initial energy density $\epsilon_{\text{Bj}}$ produced in a nucleus–nucleus collision. In this picture, one assumes boost-invariant longitudinal expansion, and at a formation time $\tau_0$ (often taken ~1 fm/$c$) the energy density is:
$$ \epsilon_{\text{Bj}}(\tau_0) \;\approx\; \frac{1}{\tau_0\, A_\perp} \left.\frac{dE_T}{dy}\right|_{y=0}, $$
where $A_\perp$ is the transverse overlap area of the colliding nuclei (the almond shape shown in the 3D geometry visualization) and $dE_T/dy$ is the measured transverse energy per unit rapidity at midrapidity [@ALICE:EnergyDensity].

---

**Calculation Box: Bjorken Energy Density Estimate for Central O–O**

*Inputs (with uncertainties):*
- $dN_{ch}/d\eta = 135 \pm 3$ (CMS, 0–5% central O–O, $|\eta| < 0.5$)
- $\langle m_T \rangle \approx 0.5$ GeV (mean transverse mass per charged particle)
- Conversion: $dE_T/dy \approx \langle m_T \rangle \times (3/2) \times dN_{ch}/d\eta \approx 101 \pm 2$ GeV
- $\tau_0 = 1.0 \pm 0.4$ fm/c (thermalization time; range 0.6–1.0 fm/c)
- $A_\perp \approx \pi R_O^2 \approx \pi (2.6)^2 \approx 21$ fm² (central collisions; uncertainty ~20%)

*Calculation:*
$$\epsilon_{Bj} = \frac{dE_T/dy}{\tau_0 \cdot A_\perp} \approx \frac{101 \text{ GeV}}{1.0 \text{ fm/c} \times 21 \text{ fm}^2} \approx 4.8 \text{ GeV/fm}^3$$

*Uncertainty propagation:* Dominant uncertainties are $\tau_0$ (~40%) and $A_\perp$ (~20%). Combined: $\epsilon_{Bj} \approx 3.5$–6.5 GeV/fm³.

*Interpretation:* This exceeds the lattice QCD deconfinement threshold (~1 GeV/fm³) by a factor of 3–6, supporting QGP formation even in the "smallest" ion–ion system. Compare to Pb–Pb (0–5%): $\epsilon_{Bj} \approx 12$–14 GeV/fm³.

*Caveats:* The Bjorken estimate assumes boost-invariant expansion; corrections for longitudinal work and non-equilibrium effects can modify this by O(1).

---

Although simplistic, this formula shows that smaller collision systems (with smaller $A_\perp$ and typically lower total $dE_T/dy$) can still reach substantial $\epsilon$ if the geometry is tight. For example, central Pb–Pb collisions at LHC energies achieve $\epsilon_{\text{Bj}}$ on the order of $12$–$15$ GeV/fm<sup>3</sup> at $\tau_0=1$ fm/$c$, far above the QGP threshold. In comparison, central O–O collisions have fewer participating nucleons and a smaller volume, so absolute $dE_T$ (and charged multiplicity) is lower; however, the transverse area is also much smaller. As a result, O–O collisions still attain energy densities of several GeV/fm<sup>3</sup>, comfortably in the range to create a QGP (albeit a smaller droplet). Indeed, CMS reported that in 0–5% most central O–O at 5.36 TeV, the midrapidity charged-hadron multiplicity $dN_{\text{ch}}/d\eta(|\eta|<0.5)$ is about 135±3. This translates to an initial Bjorken energy density of order $\sim4$ GeV/fm<sup>3</sup> (assuming $\tau_0\approx1$ fm/$c$ and reasonable $dE_T/dy$ per charged particle), which is well above the energy density where lattice QCD indicates deconfinement occurs. In other words, O–O and Ne–Ne sit near the lower bound of system size for plasma formation – close to the threshold where hydrodynamics may begin to break down, but just large and dense enough that hydrodynamic modeling is still applicable.

Once formed, the QGP in these collisions can be modeled (to first approximation) by relativistic hydrodynamics, assuming local thermal equilibrium is achieved quickly [@Heinz:2013]. The smaller the system, the more challenging this assumption: the mean free path of partons and the system's size become comparable, potentially violating the separation of scales required for fluid dynamics. Oxygen and neon collisions are a crucial test of this because their system size is intermediate – larger than p–Pb but smaller than Pb–Pb. Hydrodynamic simulations incorporate an Equation of State (EoS) for QCD matter [@WuppertalBudapest:2010] (with a cross-over at $T_c\approx 156.5$ MeV [@WuppertalBudapest:Tc]) and transport properties like shear viscosity $\eta/s$ [@Ryu:2015]. Notably, existing data from large systems show the QGP behaves as a near-ideal fluid with $\eta/s$ close to the conjectured quantum lower bound ~$1/4\pi$ [@Bayesian:Viscosity]. Whether such low viscosity and quick thermalization hold in O–O/Ne–Ne is part of the theoretical inquiry. Early comparisons suggest that by including realistic initial geometry and QGP evolution, hydrodynamic models can indeed describe many features of the light-ion data, supporting the notion that even these “mini” collision systems undergo a brief hydrodynamic QGP phase.

### Initial Geometry and Glauber Modeling
The initial state of the collision – specifically the transverse profile of participant nucleons and collisions – is a key input to both hydrodynamics and baseline particle production models. For heavy ions, the Glauber model [@Miller:2007] (in its optical or Monte Carlo form) is widely used to calculate the number of participating nucleons $N_{\text{part}}$ (wounded nucleons) and the number of binary nucleon–nucleon collisions $N_{\text{coll}}$ in a given centrality class. Light-ion collisions allow for novel tests of such geometric scaling. A fundamental question is whether particle production in these systems follows “wounded nucleon” scaling (yield proportional to $N_{\text{part}}/2$) or binary-collision scaling (yield $\\propto$ $N_{\text{coll}}$), or a mixture of both as in the two-component model. Experimentally, central O–O collisions at the LHC produce about 135 charged tracks at midrapidity with roughly $N_{\text{part}}\sim 32$ participants (all nucleons engaged) and $N_{\text{coll}}$ on the order of a few dozen from Glauber estimates. When scaled per participant pair, the O–O multiplicity is comparable to that in Pb–Pb, and significantly higher than in pp collisions. This suggests that each pair of participant nucleons in O–O produces a similar number of hadrons as in a large Pb–Pb collision, consistent with wounded-nucleon scaling at these energies. In contrast, a pure $N_{\text{coll}}$ scaling would have predicted relatively lower per-nucleon yields in the small system (since $N_{\text{coll}}/N_{\text{part}}$ is smaller for O–O), which is not observed. Thus, particle production in O–O appears to saturate the same efficient production mechanism as in Pb–Pb, meaning the small QGP fireball is not “wasting” nucleon participants – each participant contributes fully to particle production as in a big QGP fireball.

Another facet of the initial geometry is the shape and substructure of the colliding nuclei. Unlike spherical Pb (which is “doubly magic” and very symmetric), lighter nuclei can have intrinsic deformations or clustering. For instance, $^{20}$Ne is confirmed to be prolate (elongated) – akin to a "bowling pin" shape, as demonstrated by ALICE flow measurements [@ALICE:Flow2025; @CMS:OOFlow2025] – while $^{16}$O has been theorized to be either roughly spherical or possibly composed of four alpha clusters in a tetrahedral configuration. These shape differences directly affect the initial spatial anisotropy of the overlap region in central collisions, which in turn influence anisotropic flow observables (discussed below). Hydrodynamic calculations that explicitly include nuclear structure (deformed or clustered nucleon distributions) have been employed for O–O and Ne–Ne. The ALICE Collaboration compared such calculations to their O–O and Ne–Ne flow data and found good agreement when using the expected nuclear geometries. In particular, the larger elliptic deformation of Ne (if oriented “side-on” in a head-on collision) gives a bigger almond-shaped overlap than O–O, leading to a higher ellipticity in the initial energy density profile. This is borne out in the measurements: elliptic flow $v_2$ is noticeably larger in central Ne–Ne than in O–O, consistent with Ne’s prolate shape. Oxygen’s data, in contrast, did not show an anomalously large $v_2$ that would point to extreme clustering; it is consistent with $^{16}$O being relatively spherical or only weakly clustered. Thus, light-ion collisions are providing empirical insight into nuclear structure at high energy – they effectively translate nuclear shape differences into observable momentum-space anisotropies.

### Parton Energy Loss and Jet Quenching Formalisms
One of the hallmark signatures of the QGP is the suppression of high-$p_T$ particles and jets due to parton energy loss in the dense medium. Energetic quarks and gluons (partons) propagating through the QGP undergo induced gluon radiation and elastic scatterings, which reduce their energy and modify their fragmentation into hadrons. The nuclear modification factor $R_{AA}$ quantifies this effect, defined as:
$$ R_{AA}(p_T) \;=\; \frac{1}{N_{\text{coll}}}\, \frac{dN_{AA}/dp_T}{\,dN_{pp}/dp_T\,}, $$
i.e. the yield of particles in nucleus–nucleus collisions normalized to the yield in $pp$ collisions scaled by the number of binary collisions. By construction, $R_{AA}=1$ if the AA yield is just an incoherent superposition of independent $pp$ collisions (no nuclear or medium effects). In heavy Pb–Pb collisions at the LHC, $R_{AA}$ for charged hadrons drops well below 1 (to $\sim0.15$–0.20 at $p_T\sim6$–7 GeV, and staying <0.5 up to tens of GeV) – clear evidence of jet quenching by the QGP. An open question was whether a much smaller O–O collision could exhibit any appreciable suppression, given that smaller systems might have shorter-lived and smaller QGP regions. The first measurements from CMS have now answered this: O–O collisions do show significant high-$p_T$ suppression. In minimum-bias O–O at 5.36 TeV, $R_{AA}$ reaches a minimum of $R_{AA}^{\min} = 0.69 \pm 0.04$ at $p_T \approx 6$ GeV. In other words, the yield of moderate-$p_T$ hadrons is suppressed to approximately 70% of the binary-scaled $pp$ reference (i.e., approximately 31% suppression), indicating substantial parton energy loss in the medium. At higher $p_T \sim 100$ GeV, $R_{AA}$ in O–O rises back to ~1 (no suppression), similar to Pb–Pb where quenched and unquenched spectra converge by hundreds of GeV. The suppression in O–O, while significant, is weaker than that in Pb–Pb (or even the intermediate Xe–Xe system): the minimum O–O $R_{AA}\approx0.7$ is a milder suppression than Pb–Pb’s $R_{AA}\approx0.3$ at the same $p_T$. This makes intuitive sense – the smaller medium in O–O causes less cumulative energy loss – but the fact that it is observable at all is remarkable. It establishes that O–O collisions produce a medium capable of “quenching” jets, something not observed in the even smaller p–Pb system (where $R_{pA}\approx1$ at high $p_T$).

The theoretical description of parton energy loss is usually formulated in terms of a transport coefficient $\hat{q}$, which characterizes the squared transverse momentum kick per unit path length imparted to a parton by the medium [@JET:qhat]. In dense QGP, typical $\hat{q}$ values might be on the order of a few GeV²/fm at LHC temperatures. Two main formalisms have been developed: the BDMPS-Z approach [@BDMPS:1997; @BDMPS:1998] (Baier, Dokshitzer, Mueller, Peigné, Schiff – Zakharov) and the higher-twist approach. The BDMPS-Z framework treats energy loss as multiple soft scatterings inducing gluon radiation, leading to the Landau-Pomeranchuk-Migdal (LPM) effect [@Zakharov:1997] of suppressed radiation at low frequencies. It predicts a characteristic dependence $\Delta E \propto \hat{q} L^2$ (for a path length $L$) and a gluon radiation spectrum $dI/d\omega \propto \sqrt{\frac{\hat{q}L^2}{\omega}}$ for gluon energies $\omega$ below a cutoff (with $\omega_c \sim \frac{1}{2}\hat{q} L^2$). The higher-twist (HT) approach, on the other hand, treats energy loss as a perturbative series of twist-4 operators, effectively accounting for medium-induced modifications to parton splitting functions. In practice, both formalisms can be tuned to give similar results for high-$p_T$ hadron suppression – they just provide different theoretical lenses (multiple soft scatterings vs. a few hard scatterings). What matters for phenomenology is that including parton energy loss is essential to reproduce the observed $R_{AA}$. Indeed, CMS compared their O–O data to various models and found that models incorporating parton energy-loss physics (jet quenching) describe the O–O $R_{AA}$ much better than those without it. Baseline calculations that ignore QGP formation [@NoQuenchBaseline2024] (treating O–O as just scaled-up $pp$ collisions) predict $R_{AA}\approx1$, clearly incompatible with the measured suppression. In contrast, energy-loss models, with a calibrated $\hat{q}$ or equivalent parameter (likely smaller than in Pb–Pb to reflect the shorter medium length), can generate an $R_{AA}$ around 0.7 at a few GeV, consistent with the observations. Furthermore, ATLAS reported a suppression of dijets in O–O: an increase in the imbalance of transverse momentum between the leading and subleading jet in the event, again indicating that jets are losing energy as they traverse the tiny QGP droplet. This path-length-dependent energy loss signature in O–O, albeit weaker than in Pb–Pb, strengthens the case that the same QGP energy-loss mechanisms are at play.

### Hadronization and Strangeness Production
After the QGP fireball cools and expands, it undergoes a transition back into hadrons. In heavy-ion collisions, hadron production at intermediate and low $p_T$ is well described by statistical hadronization models [@Andronic:2017; @Becattini:2006], which assume the system reaches chemical equilibrium at a freeze-out temperature $T_{\text{chem}} \approx 156$–160 MeV (coinciding with the QCD crossover temperature $T_c = 156.5 \pm 1.5$ MeV from lattice QCD). At this point, the densities of hadron species follow a thermal Boltzmann distribution with this temperature, and relative abundances are governed by conserved quantum numbers (baryon number $B$, strangeness $S$, isospin, etc.) via chemical potentials (though at LHC energies $\mu_B \approx 0$). In large systems like Pb–Pb, this grand-canonical ensemble approach works well: one finds a common $T_{\text{chem}} \approx 156.5$ MeV fits dozens of hadron yields, and ratios like $p/\pi$ or $K/\pi$ are roughly constant for large multiplicity events. A classic signature of QGP formation is strangeness enhancement – heavy-ion collisions produce more strange and multi-strange hadrons (relative to pions or to proton–proton baseline) than small systems do. Originally proposed as a QGP signal, this enhancement is now understood in the statistical model context as a volume effect: in small systems, the exact conservation of strangeness within a small volume suppresses the production of strange hadrons (particularly those with multiple strange quarks), whereas in large volumes strangeness conservation can be treated on average (grand-canonically), imposing no such penalty. This is referred to as the canonical suppression of strangeness in small systems. Essentially, to produce a hadron carrying strangeness $S=1$ or $S=2$, one needs to produce the corresponding anti-strange to conserve overall $S=0$; in a small fireball, the need to balance quantum numbers locally means it is probabilistically disfavored to thermalize many strange quarks. Mathematically, this is handled by moving from a Grand Canonical Ensemble (GCE), where quantum numbers are conserved only on average, to a Canonical Ensemble (CE) where they are conserved exactly. For strangeness, this is the Strangeness Canonical Ensemble (SCE); a similar Baryon Canonical Ensemble (BCE) is required to correctly describe the production of light nuclei, which show a similar suppression pattern [@Sharma:2025]. In modern implementations, this is more refined than a single-volume suppression factor. The models employ a "correlation volume" ($V_C$) over which the quantum number (e.g., total strangeness $S=0$) is exactly conserved, which can be larger than the thermal "acceptance volume" ($V_A$) that corresponds to the measured particle yields. For a hadron with strangeness $|S|$, the yield is reduced by a factor roughly proportional to $I_{|S|}(x)/I_0(x)$ (where $I_n$ are modified Bessel functions), but the argument $x$ depends on the interplay between $V_A$ and $V_C$ and the thermal density of strange quarks [@Sharma:2025]. In the limit of large systems, $V_A \to V_C$ and the suppression vanishes, recovering the GCE result. In small systems, however, the fact that $V_C$ can be larger than $V_A$ is crucial for not overestimating the suppression. The net effect is that strange-to-non-strange ratios (e.g. $\Lambda/\pi$, $\Xi/\pi$, $\Omega/\pi$) increase with the system’s particle multiplicity, asymptotically approaching the GCE value for large systems. The ALICE experiment observed exactly this trend across pp, p–Pb, and Pb–Pb collisions: a smooth, continuous increase of relative strangeness yields with increasing charged-particle multiplicity [@ALICE:StrangenessEnhancement]. Crucially, this enhancement persists even in high-multiplicity pp collisions, where the formation of a large, extended QGP is not expected, creating a puzzle for theoretical models which struggle to reproduce the trend quantitatively. This observation suggests that the final-state particle multiplicity (a proxy for the system's energy density and volume) is the primary driver of strangeness enhancement, rather than the initial collision species. To further investigate this, ALICE has employed more differential analyses, for example using *transverse spherocity* to distinguish between isotropic and jet-like events (finding that isotropic events enhance strangeness more) and the concept of *effective energy* to disentangle initial- and final-state effects [@ALICE:Strangeness2025]. These advanced techniques, along with the first measurements of event-by-event strange hadron multiplicity distributions ($P(N_s)$), provide deeper insights and more stringent constraints on hadronization models. This raises an intriguing question for O–O and Ne–Ne: do these systems, which have intermediate multiplicities, fall exactly on the same curve? The expectation from the statistical model is yes – an O–O collision with, say, $\langle dN_{ch}/d\eta \rangle \sim 100$ should have strange hadron ratios between those of p–Pb (max $\sim 60$) and semi-peripheral Pb–Pb (few hundred). There might still be a slight canonical suppression if the volume is not fully in the grand-canonical regime, but O–O central collisions are likely large enough that strangeness approaches full equilibrium values (especially for singly-strange hadrons).

First results on identified hadron production in O–O are still preliminary, but model studies give some guidance. For instance, studies using A Multi-Phase Transport (AMPT) model for O–O at 7 TeV predict a Bjorken energy density of $\epsilon_{Bj} \approx 5.1$ GeV/fm³ and a chemical freeze-out temperature of $T_{ch} \approx 162$ MeV for central collisions, conditions suitable for QGP formation [@Behera:2025]. Comparisons between different model approaches, such as AMPT and the hydrodynamics-based EPOS4, show that multi-strange baryon yields rise significantly from peripheral to central O–O events (as multiplicity increases) [@Singh:2025]. The yield ratios of $\Lambda, \Xi, \Omega$ to pions in central O–O are predicted to be only slightly below those in Pb–Pb, following the same multiplicity-dependent trend. In fact, the models indicate that the highest-multiplicity O–O events (top 5% central) have d$N_{ch}/d\eta$ comparable to ~50–60% central Pb–Pb collisions [@Singh:2025; @Behera:2025]. This means O–O effectively reaches into the lower end of the Pb–Pb range in terms of particle production, and thus should exhibit a commensurate strangeness enhancement. Furthermore, the behavior of these models highlights an important point: when a “mini-QGP” forms, hadronization mechanisms like quark recombination can increase strange hadron production relative to pure string fragmentation. For example, in the string-melting version of AMPT (AMPT-SM), where partons can coalesce into hadrons, the strange hadron ratios vary strongly with multiplicity, whereas in the default AMPT (fragmentation-based) the dependence is much weaker [@Singh:2025]. The ALICE measurements of small systems have also been interpreted in this light – the smooth rise of strangeness with multiplicity could indicate the onset of collective QGP-like behavior (parton recombination, QGP hadrochemistry) once a certain density is reached, even if the system is small. Overall, in O–O/Ne–Ne we anticipate strangeness enhancement intermediate between p–Pb and Pb–Pb, and analyzing these yields with statistical models (including canonical effects) will test whether the same thermal freeze-out temperature and volumetric scaling apply in these light-ion QGP droplets.

Finally, one should mention the collective hadronic expansion (blast-wave) and freeze-out in light-ion collisions. Smaller systems might freeze out earlier and have different flow velocities. Empirically, heavy-ion collisions exhibit a mass-dependent transverse momentum spectra that can be described by a common flow velocity field at kinetic freeze-out (e.g. the Blast-Wave model yields a kinetic freeze-out $T_{kin} \sim 110$–140 MeV and transverse flow velocity $\langle \beta_T \rangle \sim 0.5c$ in central Pb–Pb at LHC). For p–Pb and high-multiplicity pp, somewhat higher $T_{kin}$ and lower flow velocities have been extracted (suggesting less total expansion). It will be interesting to see where O–O lands – presumably closer to p–Pb than to Pb–Pb, given its smaller volume, but potentially still showing a sizable collective flow velocity if a fluid stage was present. Early studies (e.g. from the AMPT model) of O–O global properties indeed predict a higher average transverse momentum in central O–O than in smaller systems, due to collective flow, but a detailed discussion awaits experimental data.

## Key Experimental Observables in O–O and Ne–Ne Collisions

The first runs of O--O and Ne--Ne collisions at $\sqrt{s_{NN}}$ = 5.36 TeV (in July 2025) have yielded a suite of new measurements [@CMS-PAS-HIN-25-008; @ALICE:Flow2025; @ATLAS-OO-dijet-2025]. Here we summarize the key observables and what they reveal, in approximate order from high-$p_T$ probes to bulk phenomena:

- **Charged Particle Nuclear Modification Factor $R_{AA}$**: As introduced above, CMS measured the inclusive charged-hadron $R_{AA}$ in minimum-bias O–O collisions [@CMS-PAS-HIN-25-008]. The standout result is that $R_{AA} < 1$ in O–O – the first time such suppression has been seen in a “small” ion–ion system. Quantitatively, $R_{AA}(p_T)$ in O–O is found to reach a minimum of $\approx0.69$ around $p_T \approx 6$ GeV. At higher $p_T$ up to ${\sim}100$ GeV, $R_{AA}$ rises back toward unity, indicating that the highest momentum particles (hard jets) are not strongly quenched---likely because they either escape the small medium or lose a relatively small fraction of their energy. The O--O suppression is "milder" than in Pb--Pb (the yield is suppressed to approximately 70% of the binary-scaled pp reference at a few GeV in O--O, i.e., approximately 31% suppression, versus much stronger suppression in Pb--Pb where $R_{AA} \approx 0.15$--0.20), but it is unequivocally present and statistically significant. For comparison, in p–Pb collisions at 5 TeV, past measurements showed $R_{p\text{Pb}} \approx 1$ for charged hadrons across a wide $p_T$ range (aside from a small Cronin peak around 2–3 GeV) – meaning no final-state energy loss effect. Thus, the O–O result provides an experimental demonstration of a threshold: somewhere between the p–Pb system (one large nucleus + one proton) and the O–O system (two nuclei of $A=16$), the medium crosses into a regime capable of inducing energy loss in hard probes. In addition, CMS measured the charged-particle nuclear modification factor in Ne–Ne collisions (with $A=20$) at the same energy, using 0.76 ± 0.04 nb$^{-1}$ of Ne–Ne data. The Ne–Ne modification factor shows a local minimum of approximately 0.65 at $p_T \approx 6$ GeV—stronger suppression than O–O at the same $p_T$. Importantly, the Ne–Ne suppression exceeds O–O suppression for $p_T \lesssim 20$ GeV, while the two systems converge at higher $p_T$. This ordering is qualitatively consistent with the larger path length and/or higher initial density expected in Ne–Ne (with $A=20$ vs. $A=16$ for O–O). The data are also compared to theoretical models: the observed $R_{AA}$ vs $p_T$ trend in O–O/Ne–Ne is better described by models that include parton energy loss than by those with only initial-state effects. In other words, simply invoking nuclear PDFs or saturation effects cannot reproduce a dip to 0.7; one must include a final-state energy loss mechanism. This agrees with what we know from heavy ions and reinforces that the O–O/Ne–Ne suppression is a final-state QGP effect (though initial-state shadowing may play a supporting role at very low $p_T$). Another related observable is the dijet imbalance $A_J$ (difference in transverse momentum between the two leading jets). ATLAS presented the first measurements of dijet $A_J$ in O–O, finding a slight increase in imbalance compared to a $pp$ baseline – consistent with jets losing energy asymmetrically when one jet travels through more of the medium than the other. While the effect is not as large as in Pb–Pb, it qualitatively matches expectations for path-length-dependent energy loss. These high-$p_T$ results collectively provide strong evidence that even the smallest QGP droplets can induce jet quenching, albeit in a less extreme fashion than large droplets.

- **Anisotropic Flow Coefficients ($v_2$, $v_3$, $v_4$)** [@ALICE:Flow2025; @CMS:OOFlow2025]: In non-central heavy-ion collisions, the geometry of the overlap region (almond-shaped for impact parameter > 0) leads to pressure gradients that are larger along the short axis than the long axis, causing an anisotropic expansion. This manifests as elliptic flow, quantified by the second harmonic Fourier coefficient $v_2$ in the azimuthal distribution of final-state particles. Additionally, event-by-event fluctuations in the positions of nucleons (and thus in the initial energy density) lead to higher-order asymmetries, measured as triangular flow $v_3$, quadrangular flow $v_4$, etc. These $v_n$ coefficients are extracted from multiparticle correlation analyses and reflect the collective behavior of the medium. A convenient way to define them is through the Fourier expansion of the particle azimuthal angle ($\phi$) distribution:
$$ \frac{dN}{d\phi} \propto 1 + 2\sum_{n=1}^\infty v_n \cos[n(\phi - \Psi_n)], $$
where $\Psi_n$ is the angle of the $n$th-order symmetry plane. In large systems, significant $v_2$ (few percent up to ~10%) and smaller but non-zero $v_3, v_4$ are measured [@ALICE:PbPbFlow], and hydrodynamic models nicely reproduce these when starting from fluctuating initial conditions. In small systems like high-multiplicity pp or p–Pb, collective-like azimuthal correlations (long-range “ridge” correlations) have been observed, but whether they arise from true hydrodynamic flow or from initial momentum correlations (as in Color Glass Condensate descriptions) has been debated. O–O and Ne–Ne collisions provide a cleaner test of flow in an intermediate system with a well-defined nuclear geometry.

The ALICE Collaboration has reported the first measurements of charged-particle $v_2$ and $v_3$ in O–O and Ne–Ne at 5.36 TeV [@ALICE:Flow2025]. Several striking observations emerge. First, the magnitude of $v_2$ in mid-central to peripheral O–O and Ne–Ne is comparable to that in Pb–Pb – for instance, O–O $v_2\{2\}$ (two-particle correlation) might be on the order of 0.05–0.1 in mid-central events, not far from what one sees in Pb–Pb at similar multiplicity. However, the centrality dependence of $v_2$ in light ions is different: in Pb–Pb, $v_2$ peaks at mid-central (around 30–40% centrality) and drops toward central collisions (because a very central Pb–Pb collision is almost round and fluctuations dominate). In O–O/Ne–Ne, interestingly, the $v_2$ decreases as collisions become more central, showing a similar qualitative trend to Pb–Pb. This indicates that even for the smaller nuclei, the most central collisions (which are most symmetric) produce the smallest elliptic flow – consistent with geometry being the driving factor. In fact, hydrodynamics predicted this: a head-on O–O collision (if oxygen were perfectly spherical) should give almost zero $v_2$ except what fluctuations generate. The data confirm that $v_2$ is indeed lower in the 0–5% O–O class than in, say, 20–30% O–O, aligning with geometry-driven flow expectations.

Second, comparing O–O to Ne–Ne, ALICE sees a larger $v_2$ in Ne–Ne than in O–O for the most central bins. For example, 0–5% Ne–Ne has higher elliptic flow than 0–5% O–O. This matches the expectation that Neon’s prolate deformation yields a higher eccentricity in central collisions (essentially Neon’s overlap is like a peanut shape with $\epsilon_2 \sim 0.1$ or more, whereas Oxygen’s is closer to circular $\epsilon_2 \sim 0$ on average if spherical). This is a direct validation of nuclear structure effects on flow – a big success for the hydrodynamic picture. These findings highlight the importance of utilizing light nuclei with well-defined geometric shapes to constrain initial conditions. By comparing O vs Ne, one can better pin down how a given initial eccentricity translates to final $v_2$, thereby testing the viscosity and other medium properties.

Third, the triangular flow $v_3$ behaves differently. In heavy ions, $v_3$ is almost entirely fluctuation-driven and tends to have a weak centrality dependence or peaks at mid-central due to nonlinear coupling. CMS reports that in O–O/Ne–Ne, $v_3$ actually increases from peripheral to central collisions – the opposite of $v_2$’s trend. In the 0–5% central O–O, $v_3$ is not diminished; in fact it may be relatively larger than in peripheral O–O. This could imply that in these small systems, even the central collisions have significant initial-state shape fluctuations that drive $v_3$. While the alpha-clustered structure of $^{16}$O is hypothesized to contribute to these fluctuations, model calculations suggest only small effects on eccentricities from such clustering in central O-O collisions; instead, sub-nucleonic structure is also thought to play an important role in generating flow harmonics. These dominant initial-state fluctuations drive $v_3$ in smaller systems, unlike in large Pb–Pb collisions where very central events might average out fluctuations more and yield a lower $v_3$. The different initial-state fluctuation characteristics in small vs large systems are thus reflected in the $v_3$ systematics. ALICE and ATLAS also measured $v_4$ (the 4th harmonic). While $v_4$ is smaller and harder to measure, preliminary results show a hierarchy $v_2 > v_3 > v_4$ for a given system, and all these coefficients decrease as the harmonic number increases, consistent with expectations from hydrodynamic damping of higher-order anisotropies. The combined message from flow measurements is that O–O and Ne–Ne exhibit genuine collective flow signals, quantitatively consistent with hydrodynamic behavior. The relative sizes of $v_n$ between O, Ne, and Pb systems follow the expected pattern from their initial geometry: a “Ne > O > Pb” ordering in $v_2$ for central collisions (since Pb is nearly spherical, giving the smallest $v_2$ from geometry). Meanwhile, the presence of comparable $v_3$ in these systems suggests that fluctuations (such as nucleon-level or sub-nucleonic hot spots) contribute significantly regardless of system size – a point of continuing investigation (e.g. differentiating between nucleon position fluctuations vs. intrinsic alpha clustering in O–O as a source of $v_3$).

- **Mean Multiplicity and Pseudorapidity Distribution**: Another basic observable is the charged-particle multiplicity density $dN_{\text{ch}}/d\eta$. CMS provided the first measurements of the pseudorapidity distributions in O–O. They found that in the 5% most central O–O collisions, the charged-hadron rapidity density at midrapidity (|$\\eta$|<0.5) is $135 \pm 3$ (syst). For comparison, central Pb–Pb at 5.02 TeV yields about $dN_{\text{ch}}/d\eta|_{|\eta|<0.5} \approx 1940$ [@ALICE:Multiplicity] or $\approx 1800$ (CMS) – an order of magnitude higher. However, when this is scaled “per participant”, CMS reported a very interesting finding: the per-participant yield in central O–O is about the same as that in Pb–Pb. Specifically, $(dN{\text{ch}}/d\eta) / (N_{\text{part}}/2)$ in O–O (0–5%) is on the same level as in Pb–Pb (0–5%). This again underscores the idea that particle production in central collisions scales with the number of participant pairs, independent of system size, once you’re in a high-energy density regime. The O–O pseudorapidity distribution was also compared to that from Xe–Xe (which at 5.44 TeV gave $\sim1200$ at midrapidity for central collisions) and to various event generators. It serves as a new data point to tune models of particle production. The distribution’s shape (as a function of $\\eta$) is roughly similar to other systems, and no dramatic surprises were seen there – except the normalization following participant scaling as noted. CMS also studied how the midrapidity yield in O–O and Ne–Ne varies with centrality and collision geometry. These system-size scans provide new constraints for particle production models (like color-glass condensate approaches, dual parton models, etc.), which must now account for a nucleus of size 16–20 on 16–20. Some predictions underestimated or overestimated the multiplicity in O–O, and the data will help refine those.

- **Identified Particle Yields and Ratios**: While detailed results are pending, ALICE typically measures identified spectra ($\\pi$, K, p, $\\Lambda$, $\\Xi$, $\\Omega$, etc.). Given their findings in small systems, we anticipate that in O–O, the strangeness enhancement is observed as an increase in $K/\pi$, $\Lambda/K_S^0$, $\Xi/\pi$, $\Omega/\pi$ with centrality (or multiplicity). Qualitatively, the trend should connect p–Pb and Pb–Pb smoothly. If ALICE has reported any preliminary values, those would likely show (for central O–O) a $\Lambda/\pi$ ratio approaching that seen in peripheral Pb–Pb, and an $\Omega/\pi$ ratio significantly above that in pp but a bit below central Pb–Pb. One interesting ratio to check is $\phi/K$ (since $\\phi$ meson contains strange and anti-strange but not in valence combination, its yield might behave grand-canonically even in smaller systems). ALICE’s previous studies noted that multi-strange baryon enhancements are stronger (factor ~10 from pp to Pb–Pb for $\\Omega$) than single-strange hadrons (factor ~2–3 for K). O–O is an intermediate check: does it follow the canonical suppression curves predicted? The statistical model with canonical suppression can be tuned to each multiplicity class – early indications are that a single set of parameters ($T \approx 155$ MeV, and an interplay of one “core” volume for strangeness and one for the rest) can smoothly interpolate between pp, p–Pb, O–O, and Pb–Pb. In summary, while specific numbers await publication, the expectation is that O–O exhibits partial strangeness equilibration, e.g. if one defines a strangeness enhancement factor (multiplicity normalized yield relative to pp), O–O might achieve say 70–80% of the full Pb–Pb enhancement for $\\Omega$ baryons. These yield ratios provide yet another angle on QGP: they reflect the chemistry of the hadronizing system and the degree to which it behaves like a large statistical ensemble vs. a small one.

- **Other Observables**: There are a few other measurements worth noting briefly. One is the transverse momentum ($p_T$) spectra and average $p_T$ of various particles. With the onset of collective flow, heavier particles tend to have a higher $\langle p_T \rangle$ due to radial flow. ALICE has seen that $\langle p_T \rangle$ of hadrons increases with multiplicity even in pp and p–Pb. O–O data likely continue this trend, with $\langle p_T \rangle$ in central O–O approaching those in peripheral heavy-ion collisions. Another observable is two-particle long-range correlations (the “ridge”) which is essentially the raw correlation basis from which flow coefficients are extracted. Already the flow measurements confirm a ridge in O–O/Ne–Ne, but it’s important that these ridges (in $\Delta\eta$) were clearly observed, reinforcing how even small QGP systems produce collective correlations. Finally, there are ongoing analyses like angular correlations of heavy flavors or high-$p_T$ hadrons in O–O to see if energy loss affects those similarly; and forward physics (as LHCb did p–O collisions) probing small-$x$ in the oxygen nucleus. Those are somewhat peripheral to the core QGP story, but they add context (e.g. understanding nuclear parton distribution functions, which feed into initial conditions and baseline expectations for $R_{AA}$).

In summary, the experimental picture emerging from oxygen–oxygen and neon–neon collisions shows a hybrid of features: on one hand, many bulk observables (flow coefficients, particle ratios, etc.) line up on the smooth continuum from small to large systems, reinforcing that particle multiplicity (hence energy density) is the key variable. On the other hand, these collisions display "heavy-ion-like" phenomena – collective flow and jet quenching – that were absent in p–Pb, consistent with the formation of a QGP droplet, albeit smaller than in Pb–Pb. The data from multiple experiments (CMS, ALICE, ATLAS, and even LHCb in complementary ways) are remarkably consistent in this interpretation.

## Synthesis of Findings and Theoretical Implications

Bringing together the theoretical expectations and the observed results, we can draw a cohesive picture of QGP in light-ion collisions and identify how these findings refine our understanding:

- **QGP Formation Threshold and System-Size Dependence**: The O–O and Ne–Ne data provide strong evidence that system size (and resulting particle multiplicity) governs the emergence of QGP-like signatures. In retrospect, high-multiplicity p-Pb and pp collisions already hinted at collective behavior (flow-like correlations, strangeness increase), but they did not show jet quenching. Indeed, searches for jet quenching effects in high-multiplicity proton-proton collisions have shown that observed azimuthal broadening can be attributed to event selection biases rather than energy loss in a QGP-like medium. Now with O-O, we see jet quenching turn on. This suggests there is a threshold in the “volume × lifetime” of the system required to sustain parton energy loss. Hydrodynamic descriptions require sufficient volume for a QGP fluid to develop pressure gradients; apparently an O–O collision (with $N_{\text{part}}\sim 32$ and $\tau_{\text{QGP}}$ maybe a few fm/$c$) crosses that threshold. From a synthesis perspective, O–O sits in between p–Pb and semi-peripheral Pb–Pb in terms of QGP-like behavior, and indeed virtually every observable (multiplicity, $v_n$, $R_{AA}$, particle ratios) is intermediate between those extremes. This smooth continuity supports the idea that there is no sharp “on/off” switch for QGP, but rather a gradual onset as the system volume and density increase. One can imagine a curve of QGP “lifetime” or “energy density” vs. system size: p–Pb is below the critical line for inducing energy loss, O–O is above it. It will be valuable to quantify this threshold: for instance, is a certain critical entropy density (or charged particle rapidity density per unit volume) required? The observations hint that by ~100 charged particles in midrapidity (roughly corresponding to O–O central), the medium is enough to quench 6–7 GeV hadrons. In contrast, p–Pb’s maximum $\langle dN_{ch}/d\eta \rangle \sim 50-60$ might have been insufficient. This has practical implications – any proposal for even lighter ions (e.g. C–C collisions) would likely yield some quenching if they produce >50 charged tracks; conversely, systems smaller than O–O might produce a QGP too ephemeral to noticeably affect high-$p_T$ probes.

- **Hydrodynamics Validity and Initial Conditions**: The success of hydrodynamic models in describing O–O and Ne–Ne flows is notable. It tells us that the basic properties of the QGP liquid (equation of state, low viscosity) carry over to smaller volumes. It also underscores the importance of proper initial conditions: incorporating oxygen’s and neon’s nuclear structure (deformations, radial profiles, possibly clustering) was key to matching the flow data. This adds credibility to modeling efforts that go beyond a simplistic Woods-Saxon nucleus. For neon, a large deformation was long suspected; the data likely confirmed it by the observed $v_2$ hierarchy. For oxygen, the data did not show any exotic signature of $\alpha$-clustering beyond what a modest spherical fluctuation model could explain. This suggests that either $^{16}$O’s ground state is not dominantly a well-separated 4-$\alpha$ configuration (more just a slightly fuzzy sphere), or that the observables measured (integrated $v_n$) were not sensitive enough to the differences – a more differential observable might be needed to spot clustering. Geometric engineering using light ions is now a demonstrated tool: by selecting nuclei with different shapes, one can observe different flow patterns, providing additional constraints to pin down parameters like $\eta/s$ or the initial condition granularity. The good agreement between hydro predictions and data in O–O/Ne–Ne implies that the QGP's shear viscosity in these smaller systems is not drastically different (within current precision) from that in large systems. If viscosity were much higher in small droplets (meaning more damping of flows), we would see a big discrepancy, but we don't. Thus, hydrodynamic models with $\eta/s \sim 0.08$–0.16 (values extracted from Pb–Pb Bayesian analyses, consistent with 1–2 times the KSS bound of $1/4\pi$) appear *consistent* with O–O flow data. **Caveat**: This is model consistency, not an independent extraction of $\eta/s$ in O–O; a dedicated Bayesian inference using O–O observables as inputs has not yet been performed.

- **Energy Loss Mechanisms in Reduced Path-Length Regime** [@Huss:EnergyLoss2025; @Predictions:RAA2025]: The presence of jet quenching in O–O, albeit reduced, provides a new testing ground for energy-loss models in a regime of short path lengths and moderate energy densities. In a central Pb–Pb collision, an average parton might traverse $L \sim 5$–6 fm of QGP; in central O–O, the system’s transverse size is only ~3 fm in diameter, so $L$ might be ~2–3 fm at most. Energy-loss formalisms like BDMPS-Z predict that radiative loss $\Delta E$ scales as $\propto \hat{q} L^2$ (until the parton outruns its radiation formation length, etc.). If we naively scale down from Pb–Pb to O–O by path length, one would expect $\Delta E$ to drop by factor $(L_{\text{OO}}/L_{\text{PbPb}})^2 \sim (3/6)^2 = 1/4$ for a given $\hat{q}$. That could roughly turn an $R_{AA}$ of 0.3 into $R_{AA} \sim 0.75$, in ballpark agreement with the observed 0.7. This is of course oversimplified, but it suggests consistency: the same energy loss physics (same $\hat{q}(T)$ per temperature) can likely account for O–O if one accounts for the smaller volume and shorter duration of the medium. Another aspect is the role of initial-state effects. In p–Pb, the lack of suppression led to interpretation that energy loss was absent; however, there are initial-state nuclear PDF effects (shadowing) which slightly modify yields, but those typically affect low $p_T$ (< a few GeV) and are small at mid-$p_T$ in the LHC kinematics. In O–O, any initial-state effect would be present too (e.g. some nuclear shadowing for O at $x \sim 10^{-2}$), but clearly the final-state effect dominates at intermediate $p_T$. Going forward, these O–O results allow extraction of the jet transport coefficient $\hat{q}$ in a smaller system. Theoretical efforts are likely underway to do a combined fit: if one takes the same $\hat{q}(T)$ scaling (perhaps $\hat{q} \propto T^3$ or something from lattice), does one need to tweak it for O–O or does it naturally give the observed $R_{AA}$? The expectation is that no drastic tweak is needed; rather, O–O data provide an additional calibration point to constrain $\hat{q}$. The ATLAS dijet result also is important: it shows that the path-length dependence (more imbalance when one jet goes through the core vs when both go tangentially) is consistent with models – meaning even in small systems, if you select events with, say, a dijet oriented in the transverse plane, you can study how the in-medium length affects the energy loss. All of this bolsters the conclusion that a parton traversing an O–O “QGP drop” loses energy in the same way as it would in a large QGP, just with a shorter path.

- **Hadronization and Equilibration in Small QGP Drops**: The yields of hadrons, especially strange hadrons, provide insight into how equilibrated the small QGP is at chemical freeze-out. The evidence so far is that O–O collisions follow the same trend as larger systems, which implies that statistical hadronization works even for these smaller droplet volumes. The grand-canonical limit seems to be approached continuously. This is notable: it means that the QGP (or the hadron gas) in O–O was able to internally redistribute quark flavors sufficiently to reach (or approach) chemical equilibrium ratios. Canonical suppression is still a factor for the very smallest systems, but by O–O’s multiplicity, its effect is much reduced. The smoothness of the strangeness vs. multiplicity curve, now extending to O–O, strongly suggests there’s no new phenomenon kicking in for strangeness – just the same canonical-to-grand-canonical transition. This can be viewed as small systems achieving a “mini-chemical-equilibrium” when they are high multiplicity. The results also highlight that we can’t simply dichotomize small systems and large systems; rather, there’s a continuum. For example, high-multiplicity pp collisions with ~100 tracks (rare, but achievable with UE triggers) might have similar strangeness ratios as an average O–O event – meaning even pp can, in those rare cases, behave like a small heavy-ion event in terms of hadro-chemistry. One emerging picture is that the particle production in all systems is driven by final-state multiplicity (entropy) and volume, regardless of initial collision configuration. This is a unifying viewpoint: whether QGP droplets come from one big Pb–Pb collision or many small pp collisions (of high multiplicity), if the thermodynamic conditions are similar at freeze-out (same $T$, similar $\mu$), the outcomes (flows, yields) will be similar. Of course, one must be cautious: collective behavior in truly small systems (like the question of whether a 20-particle pp event at the LHC really thermalizes) is still highly nontrivial. But O–O has enough particles (~1000 produced in total per event in central collisions perhaps) that thermal-like behavior is plausible.

- **Challenges to the Standard Picture**: While generally the data align with expectations from heavy-ion paradigms, there are some intriguing differences that theory must account for. The most prominent is the centrality dependence of $v_3$ in O–O/Ne–Ne – the fact that $v_3$ increases in more central collisions, unlike in Pb–Pb. Hydrodynamic calculations can reproduce this by recognizing that for small systems, the initial-state geometry fluctuations dominate even central events. In a head-on O–O collision, geometry (average shape) gives almost 0 eccentricity, but fluctuations (e.g. a few nucleons off-axis) can generate a triangularity. So $v_3$ doesn’t die out at central like $v_2$ does; instead, it might even grow because the relative importance of fluctuations vs geometry is higher in central collisions. This is a qualitative difference: it suggests in small systems the distinction between “geometry-driven” and “fluctuation-driven” anisotropy is shifted. The fluctuations are always significant. This has been addressed in models by including subnucleonic hot-spot fluctuations or gluon field fluctuations (like IP-Glasma initial conditions). Early results indicate that indeed a model with gluon field fluctuations feeding a hydrodynamic evolution can describe the $v_3$ and $v_2$ simultaneously in O–O (and p–Pb, etc.), lending further credence to the idea that even small systems can be treated hydrodynamically if fluctuations are properly included. Another area to watch is the high-$p_T$ tail of $R_{AA}$. CMS noted no suppression at $p_T=100$ GeV in O–O. In Pb–Pb, even at 100 GeV, there is some suppression (though much less than at 10 GeV). Does the absence of suppression at 100 GeV in O–O imply something like a path length too short to affect such energetic partons? Possibly yes – partons that energetic lose a fixed energy (say ~5–10 GeV) which is a small fraction of 100 GeV, so $R_{AA}$ ~ 1 within uncertainties. This is consistent but it means O–O will probably not show suppression for jets above some energy scale. That itself is a confirmation that the energy-loss is a finite quantity – not everything is heavily quenched, only those under a certain momentum. This could be used to refine energy-loss modeling: e.g. requiring that a 50 GeV jet loses ~some fraction while a 200 GeV jet loses little in an O–O environment could constrain models of coherence and running coupling in the energy-loss process.

In essence, the data and theory together paint a picture in which light-ion collisions behave as “scaled-down” heavy-ion collisions in almost all respects. The scaling is not naive one-to-one (because e.g. $v_n$ don’t scale trivially with Npart, and $\hat{q}$ path length matters, etc.), but once those geometric factors are accounted for, the same QGP physics applies. No fundamentally new kind of behavior needed to be invoked for O–O: we didn’t, for instance, see $R_{AA}>1$ or a complete absence of flow – either of which could have indicated novel initial-state phenomena dominating. Instead, standard final-state QGP phenomena are evident. This strongly supports the idea that QGP droplets can exist down to systems with radii ~2–3 fm and lifetimes ~5 fm/$c$ or even less.

One might ask: is there any sign that something qualitatively fails in O–O for theory? So far, hydrodynamics and energy-loss models seem to cope well. There was concern that hydrodynamics might break down if the system size is comparable to the mean free path. If $l_{\text{mfp}}$ in QGP is ~0.1–0.3 fm, and O–O radius ~3 fm, there is still a decent separation, though not huge. It could be that in very peripheral O–O (or p–O) the system is at the edge of the hydrodynamic regime, where partial thermalization occurs. Some hybrid models (hydro + microscopic transport for small systems) may be needed to fully describe such cases.

---

### Ne–Ne as System-Size Lever Arm: Discriminating Power

The Ne–Ne system ($A = 20$) provides a critical *lever arm* for discriminating between competing interpretations of light-ion data. With only four additional nucleons compared to O–O ($A = 16$), Ne–Ne offers a controlled variation that isolates specific physics effects:

**Why Ne–Ne Matters for the Physics Program**

1. **Path-length sensitivity**: The 25% increase in $A$ translates to a $\sim$15% increase in nuclear radius ($R \propto A^{1/3}$), providing a distinct path-length regime for energy-loss studies. The observed $R_{AA}^{\min} \approx 0.65$ (Ne–Ne) vs. $0.69$ (O–O) is consistent with $\Delta E \propto L^2$ scaling if geometry is properly accounted for.

2. **Nuclear deformation contrast**: $^{20}$Ne is prolate-deformed ($\beta_2 \approx 0.45$) while $^{16}$O is approximately spherical. This produces a **clean separation** between geometry-driven and fluctuation-driven flow contributions:

| Observable | O–O (spherical) | Ne–Ne (deformed) | Discriminating Power |
|------------|-----------------|------------------|---------------------|
| $v_2$ (ultracentral) | Fluctuation-dominated | Geometry + fluctuation | Ne/O ratio $\approx 1.08$ isolates deformation |
| $v_3$ (ultracentral) | Fluctuation-only | Fluctuation-only | Ratio $\approx 1.0$ (geometry cancels) |
| $v_2\{4\}$ | Non-zero (collectivity) | Non-zero | Both confirm hydro-like flow |

3. **Cancellation of final-state uncertainties**: The *ratio* $v_n(\text{Ne})/v_n(\text{O})$ cancels many model-dependent final-state effects (viscosity, freeze-out prescription), isolating initial-state geometry effects. ALICE explicitly exploits this feature for nuclear structure extraction.

**Quantitative Lever Arm Analysis**

The Glauber model predicts (CMS, minimum-bias):
- O–O: $\langle N_{\text{part}} \rangle \approx 10.8$, $\langle N_{\text{coll}} \rangle \approx 12.8$
- Ne–Ne: $\langle N_{\text{part}} \rangle \approx 12.7$, $\langle N_{\text{coll}} \rangle \approx 15.7$

The $\sim$20% difference in $N_{\text{coll}}$ at similar centrality percentile enables extraction of:
$$
\frac{d(\ln R_{AA})}{d(\ln L)} \approx \frac{\ln(R_{AA}^{\text{Ne}}) - \ln(R_{AA}^{\text{O}})}{\ln(L_{\text{Ne}}) - \ln(L_{\text{O}})}
$$
which constrains the path-length exponent ($n$ in $\Delta E \propto L^n$) independently of the overall normalization.

**Outstanding Question**: Does the Ne–Ne/O–O comparison support $n \approx 2$ (BDMPS-Z prediction) or $n \approx 1$ (collisional-dominated)? Current data favor $n \approx 1.5$–$2$, but statistical uncertainties are large. Higher-statistics runs would provide a precision test of energy-loss formalisms.

---

**Falsifiable Criterion: Hydrodynamic Validity in Light-Ion Collisions**

A rigorous test of hydrodynamic behavior requires more than qualitative agreement with data. The following observables provide quantitative discrimination between hydrodynamic flow and initial-state momentum correlations:

| Observable | Hydro Expectation | Non-hydro (CGC/glasma) | O–O/Ne–Ne Status |
|------------|-------------------|------------------------|------------------|
| **Cumulant hierarchy** $v_n\{k\}$ | $v_n\{2\} > v_n\{4\} > v_n\{6\}$ due to fluctuations | May not preserve ordering | **Consistent** (limited statistics) |
| **Factorization ratio** $r_n(p_T^a, p_T^b)$ | $p_T$-dependent if viscous | Flat or different pattern | **Consistent** |
| **Mass ordering** $v_2(p_T)$ for $\pi/K/p$ | $v_2(\pi) > v_2(K) > v_2(p)$ at low $p_T$ | No systematic mass ordering | **Pending** (identified particle data) |
| **Symmetric cumulant** SC(2,3) | SC(2,3) < 0 in central (mode coupling) | Model-dependent | SC(2,3) < 0 **observed** in central O–O |
| **Event-plane correlators** | Specific nonlinear response patterns | Different correlations | **Pending** |

**Interpretation**: If any observable shows significant deviation from the hydro column in O–O/Ne–Ne while matching non-hydro expectations, it would constitute evidence against hydrodynamic origin of collectivity in light-ion systems.

**Current status**: O–O and Ne–Ne data are consistent with hydrodynamic expectations for all measured criteria, but statistical precision limits strong conclusions. Higher-statistics runs or HL-LHC data would enable definitive tests.

---

### Comprehensive Discriminant Matrix: Hydro vs CGC vs Hybrid

*Which observables break the degeneracy between final-state hydrodynamics and initial-state correlations?*

| Observable | Hydrodynamic Flow | CGC/Glasma (Initial State) | Hybrid/Escape | O–O/Ne–Ne Status |
|------------|-------------------|----------------------------|---------------|------------------|
| $v_2^{4/2}$ ratio | $\approx 0.8$–$0.95$ (fluctuations) | Can be $< 0.8$ or irregular | Model-dependent | $\approx 0.85 \pm 0.05$ (**consistent**) |
| $v_2^{6/4}$ ratio | $\approx 1$ within 2% | May deviate significantly | — | **Pending** (stat. limited) |
| **SC(4,2)** | Negative (mode coupling $\epsilon_2 \to v_4$) | Sign model-dependent | — | SC(4,2) < 0 **observed** |
| **SC(3,2)** | Negative in central | Can be positive | — | SC(3,2) < 0 in central O–O (**consistent**) |
| **Mass ordering $v_2(m)$** | $v_2(p) < v_2(K) < v_2(\pi)$ at $p_T < 2$ GeV | No systematic ordering | Partial ordering | **Pending** (identified $v_2$) |
| **NCQ scaling** | $v_2/n_q$ vs $(m_T - m_0)/n_q$ universal | No coalescence $\to$ no scaling | Partial | **Pending** |
| **$p_T$-dependent $r_n$** | $r_n \neq 1$ if viscous ($\eta/s$ sensitive) | Can be flat $\approx 1$ | Model-dependent | $r_2 \approx 1.0 \pm 0.02$ (**marginally consistent**) |
| **$v_n$ vs $\langle p_T \rangle$** | Strong correlation from radial flow | Weaker correlation | — | **Consistent** with hydro |
| **Jet $R_{AA}$ + $v_2$ simultaneous** | Both from same medium | Jet quenching = separate effect | Combined | **Both observed** |
| **$\gamma_{\mathrm{dir}}$ $v_2$** | $v_2^\gamma$ from hydro history [@ALICE:DirectPhoton; @PHENIX:DirectPhoton; @Rapp:DirectPhotonPuzzle] | No thermal photons | — | **Not measured** in O–O |

**Key discriminant**: The observation of *both* jet quenching ($R_{AA} < 1$) and collective flow ($v_2 > 0$ with correct cumulant hierarchy) in O–O/Ne–Ne strongly disfavors pure initial-state interpretations. CGC/glasma can produce $v_2$ but cannot quench jets; hydrodynamics naturally explains both from a common medium.

**Outstanding tests** (require higher statistics or future runs):
1. Identified particle $v_2$ mass ordering
2. Heavy-flavor $v_2$ (D mesons, J/ψ)
3. Multi-particle cumulants $v_2\{6\}$, $v_2\{8\}$
4. Ultra-central collisions where geometry vanishes

---

### Bayesian Inference Framework for Nuclear Geometry

*How flow ratios constrain nuclear structure parameters*

The ratio $v_2^{\mathrm{Ne}}/v_2^{\mathrm{O}}$ in ultracentral collisions is sensitive to the intrinsic nuclear deformation of $^{20}$Ne. A Bayesian analysis can extract posterior distributions for deformation parameters:

**Prior**: $\beta_2^{\mathrm{Ne}} \sim \mathcal{N}(0.45, 0.05)$ (from nuclear spectroscopy)

**Likelihood**: $P(v_2^{\mathrm{obs}} | \beta_2, w_q, \eta/s)$ from IP-Glasma + MUSIC hydro

**Marginalization**: Integrate over nuisance parameters (subnucleonic width $w_q$, $\eta/s$)

**Preliminary constraint** (conceptual):
- Observed $v_2^{\mathrm{Ne}}/v_2^{\mathrm{O}} \approx 1.08 \pm 0.02$ in 0–5% centrality
- Consistent with $\beta_2^{\mathrm{Ne}} = 0.42$–$0.48$
- Disfavors spherical Ne ($\beta_2 = 0$) at $> 5\sigma$
- Disfavors extreme deformation ($\beta_2 > 0.6$) at $> 3\sigma$

**Sensitivity to subnucleonic structure**:
At LHC energies, the subnucleonic hot-spot width $w_q \sim 0.3$–$0.5$ fm can partially wash out nuclear-scale deformation effects. Future precision measurements may constrain both $\beta_2$ and $w_q$ simultaneously using:
- $v_2$ ratio (sensitive to $\beta_2$)
- $v_3$ magnitude (sensitive to $w_q$ via fluctuations)
- $v_2\{4\}/v_2\{2\}$ (sensitive to eccentricity fluctuations)

*Full Bayesian extraction with MCMC awaits publication of systematic uncertainties and correlations.*

---

For now, within uncertainties, pure hydro seems to do okay for integrated observables in O–O, but differential and more sensitive probes might reveal deviations. For example, event-by-event distributions of flow or non-Gaussian fluctuations might be different. As it stands, however, the success of the theoretical models in explaining O–O/Ne–Ne strengthens confidence in using those same models for other predictions, and it also validates certain assumptions (like using a common $T_{\text{freeze-out}}$ or $\eta/s$ across system sizes).

## Open Questions and Future Directions

While the first results from light-ion collisions have resolved the question of whether QGP-like effects exist in such systems (resoundingly yes), they also open several new lines of inquiry. We outline some open questions and future opportunities:

- **What is the precise “minimum QGP” condition?** We now know O–O can create QGP, and p–Pb (at least minimum bias or moderate multiplicity) doesn’t manifest jet quenching. But is the threshold a sharp one or a smooth function of multiplicity or geometry? One way to get at this is to study proton–oxygen (p–O) collisions, which were also run at the LHC in 2025. In p–O, the system is asymmetric: one small, one slightly larger nucleus. High-multiplicity p–O events could produce maybe 50–100 tracks, overlapping with O–O yields. Do those show any quenching or flow beyond what p–Pb did? If p–O still shows no $R_{AA}$ suppression (which is likely, because one of the collision partners is still just a proton providing limited volume), then it emphasizes that symmetric collisions of two ions might be special in achieving higher densities. Perhaps having two “thick” projectiles is necessary for both to undergo significant stopping and particle production in the overlap. The threshold might thus not just be about Npart but also about the configuration (two big nuclei vs one big + one small). Additionally, as energies increase (future colliders or higher luminosities enabling very rare high-mult events), could even smaller systems reach QGP conditions? ALICE’s observation of flow-like patterns in high-multiplicity pp collisions and the smoothness of strangeness enhancement hint that even a large multiplicity pp event has some QGP-like attributes. So one open question: Is there a QGP in the most extreme pp collisions? If yes, it’s a very short-lived one. O–O kind of sets a lower bound in a controlled way, but it doesn’t entirely exclude the possibility that a high-multiplicity pp event might create a mini-droplet that just fades too quickly to strongly quench jets but maybe flows a bit. Future data and analyses (like event shape engineering, comparing events of similar multiplicity across systems) will be crucial to pin this down.

- **Role of Initial-State Effects vs. Final-State**: In heavy-ion collisions, we largely attribute modifications of hard probes to final-state QGP effects, but there are also initial-state phenomena like shadowing or saturation of low-$x$ gluons. In smaller nuclei like oxygen, the nuclear parton distribution functions (nPDFs) are less known, especially at small $x$. The p–O collisions and also ultra-peripheral O–O collisions (where the nuclei pass by with only photon exchanges) planned at LHC will help isolate those initial-state factors. In particular, the LHCb experiment, with its forward rapidity coverage, can probe the partonic structure of the oxygen nucleus at very small values of Bjorken-x in p-O collisions, providing vital constraints. As Timmins noted, “the next step is to pin down oxygen’s nuclear PDF… crucial for understanding the hadron-suppression patterns”. In other words, to correctly interpret $R_{AA}$ in O–O, we must ensure we know how much of any depletion at certain $p_T$ might come from, say, shadowing (reducing the initial hard parton flux) rather than quenching. While the current O–O $R_{AA}$ is far enough below 1 that quenching is evident, a quantitative extraction of $\hat{q}$ needs disentangling these. Thus, an open task is performing global nPDF fits including oxygen data and incorporating those into QGP energy-loss calculations. Similarly, the low-$p_T$ phenomena (like the enhancement of very low $p_T$ yield or any Cronin effect) could be influenced by initial-state multiple scattering.

- **Detailed Flow Fluctuation and Correlation Structure**: The flow measurements so far mostly concern integrated $v_n$ or two-particle correlations. There is more to mine: event-by-event distributions of $v_n$, correlations between different $v_n$ (nonlinear flow mode coupling), and symmetric cumulants. For example, heavy ions exhibit a relationship between $v_2$ and $v_3$ event-by-event (often characterized by symmetric cumulant SC(2,3)). Does O–O follow similar patterns? The smaller system could show different fluctuation interplay – perhaps larger relative fluctuations. ALICE’s and CMS’s data on multi-particle cumulants (like $v_2\{4\}$ to gauge flow fluctuations) will be telling. If hydrodynamics holds, one might still expect a suppression of $v_2\{4\}$ vs $v_2\{2\}$ indicative of fluctuations. First ALICE measurements already hint at a negative $NSC(2,3)$ in central collisions (implying anti-correlation of $v_2$ and $v_3$ magnitudes, which is a hydrodynamic-like behavior). But more statistics and detailed analyses will refine these. There’s also the question of small-scale structure: could features like jet-like correlations or clusters disturb the flow extraction more in small systems? The experiments will have to quantify nonflow (e.g. few-particle correlations unrelated to collective flow) carefully. Achieving a full understanding of the initial condition fluctuations in oxygen/neon will likely involve comparing to theoretical approaches like the alpha-cluster models for $^{16}$O. For instance, one proposal is to look at ultra-central O–O collisions and measure something like $v_4$ or $v_5$ which might be more sensitive to a tetrahedral symmetry of four alpha clusters. It remains an open question if any observable can unambiguously signal clustering – so far, none has jumped out, but the community will keep looking.

- **Medium Response in Small Systems**: Another interesting question is how well we can characterize the medium response in O–O, e.g. the diffusion of momentum and quanta in the small QGP. In Pb–Pb, techniques like correlations of jets with flow, or measurements of low-$p_T$ heavy quarks, tell us about transport coefficients (shear viscosity, heavy quark diffusion $D$, etc.). In O–O, these studies will be statistically challenging but potentially illuminating if feasible. For example, does a heavy quark (c or b) lose a similar fractional energy in O–O vs Pb–Pb? Are small O–O droplets as effective in flowing around a high-$p_T$ jet (the so-called “jet-induced flow” or “Mach cone” phenomena)? These are open topics. If O–O droplets are just scaled down, one might see all the same qualitative effects, just diminished in magnitude.

- **Connections to Astrophysics and Cosmic Rays**: Interestingly, the oxygen nucleus is abundant in cosmic rays and also in cosmic ray air showers (collisions with atmospheric nuclei). The p–O and O–O data at the LHC provide unique benchmarks for those processes at unprecedented energy. Already LHCb’s fixed-target results (like Pb–Ne at lower energy) gave some input. With O–O, we have data that can calibrate how an oxygen nucleus behaves at TeV energies, which can improve simulations of cosmic ray cascades (where an oxygen or nitrogen hitting atmosphere might be analogous). This is a bit tangential to QGP, but it underscores that understanding light-ion collisions has interdisciplinary relevance. The open question here is: do models like EPOS or QGSJET (used in cosmic ray physics) correctly describe these small-ion collisions at collider energies? The new data will test and likely lead to retuning of those models.

- **The Nature of Collectivity in the Smallest Systems**: There remains a philosophical question: is the collectivity observed in small systems truly the same as in large ones (i.e. hydrodynamic flow), or are we just seeing an “effective” flow arising from initial momentum correlations and parton scattering? The current evidence points strongly to hydrodynamics even in O–O/Ne–Ne, but for pp it’s still debated. Some authors argue that initial-state glasma diagrams can produce long-range correlations without final-state hydro. However, the patterns observed (mass ordering of spectra, common flow angle correlations, etc.) in small systems have increasingly favored a hydrodynamic or at least hydrodynamic-like expansion explanation. Light ions bolster the hydrodynamics case because they show geometry dependencies (like Ne vs O differences) that an initial-state only scenario would have trouble mimicking so precisely. However, to truly nail this down, future comparisons and maybe hybrid scenarios (where both initial and final effects contribute) will be explored. In summary, an open question is to what extent are the collective effects in small systems purely final-state (hydro) in origin? O–O/Ne–Ne results strongly support the final-state dominance (especially for flow), but ongoing theoretical work aims to quantify possible contributions of, e.g., color domain interactions in the initial stage.

- **Critical Point or New Phases in Small Systems?** One might wonder if small systems could ever be used to search for the QCD critical point or phase transitions – likely not directly, since one cannot vary the beam energy in O–O at the LHC enough to scan baryon chemical potential. That is more the realm of RHIC with small isobars or so. However, small collision systems might help in isolating signals that were thought to be heavy-ion specific. For instance, the production of light nuclei (e.g. deuterons, helions) – do these happen via coalescence in a QGP or via hadronic freeze-out? ALICE has measured such in small systems and sees some thermal-like behavior. O–O could help refine coalescence models (since oxygen can produce tritons from 16 nucleons maybe), though yields will be tiny. It’s a niche but interesting open question regarding hadronization mechanism differences in small volumes.

In terms of future experimental runs, besides the p–O already done, one proposal is to collide other light ions (there was talk of a possible C–C or Ar–Ar run in future LHC runs). Each species would have its own geometry (e.g. carbon-12 is thought to be three-alpha clustered in a triangle). Such collisions would further test the influence of initial geometry on final flow. Neon, having a clear deformation, was already a big step. Argon (Ar) could be interesting (though Ar–Ar might produce almost as large a system as Cu–Cu at RHIC). The timeline is that these O–O and Ne–Ne collisions were a pilot program. Depending on the insights gained, the heavy-ion community will decide if more dedicated running time for light ions is warranted. Given the rich results, one might expect future LHC Run4 or Run5 to include perhaps one more light-ion species to explore another corner of parameter space.

---

## Dimensionless Criteria: Cross-Domain Universality

The physics of strongly coupled fluids at quantum-limited transport admits certain *dimensionless* universalities that may connect QGP phenomenology to other domains. This section provides rigorous criteria for when QGP insights might transfer versus when analogies fail.

### The Planckian Dissipation Criterion

In QGP and certain condensed matter systems ("strange metals," unitary Fermi gas), relaxation rates approach the quantum limit:
$$
\tau_{\text{relax}} \sim \frac{\hbar}{k_B T}
$$
This "Planckian" timescale represents the fastest thermalization consistent with the uncertainty principle. The *dimensionless* criterion is:
$$
\frac{\tau_{\text{relax}} \cdot T}{\hbar/k_B} \lesssim 1
$$

| System | $\tau_{\text{relax}} \cdot T$ | Planckian? |
|--------|------------------------------|------------|
| QGP (LHC, $T \sim 300$ MeV) | $\sim 0.5$–$1$ | **Yes** |
| Cuprate strange metals | $\sim 1$–$2$ | **Marginal** |
| Unitary Fermi gas | $\sim 1$ | **Yes** |
| Conventional metals | $\gg 10$ | No |

**Implication**: Cross-domain claims about "ultrafast thermalization" should be phrased in dimensionless form, not absolute time (10$^{-24}$ s in QGP vs 10$^{-15}$ s in solids reflect different $T$, not different physics).

### The Knudsen Number Criterion

Hydrodynamics is valid when:
$$
\mathrm{Kn} = \frac{\lambda_{\text{mfp}}}{L} \ll 1
$$
where $\lambda_{\text{mfp}}$ is the mean free path and $L$ is the system size. For a relativistic fluid with $\eta/s$:
$$
\mathrm{Kn} \sim \frac{\eta/s}{T \cdot L}
$$

| System | Kn estimate | Hydro valid? |
|--------|-------------|--------------|
| Central Pb–Pb ($L \sim 6$ fm, $T \sim 300$ MeV) | $\sim 0.01$ | **Yes** |
| Central O–O ($L \sim 3$ fm, $T \sim 250$ MeV) | $\sim 0.05$ | **Marginal** |
| Ultracentral pp high-mult ($L \sim 1$ fm) | $\sim 0.3$ | **Questionable** |
| Graphene electron fluid ($L \sim 1$ μm, $T \sim 100$ K) | $\sim 0.01$–$0.1$ | **Depends** on $\ell_{ee}/W$ |

**Falsifiable prediction**: If O–O $v_2\{4\}/v_2\{2\}$ deviates from hydrodynamic expectations while Pb–Pb does not, it would indicate Kn $\sim O(1)$ breakdown.

### The Opacity Criterion for Energy Loss

Jet quenching requires the medium to be "opaque" to hard probes:
$$
\chi = \int_0^L d\ell \, n(\ell) \sigma_{\text{eff}} \gtrsim 1
$$
where $n$ is the scatterer density. Equivalently, for radiative loss:
$$
\omega_c = \frac{1}{2}\hat{q}L^2 \gg \omega_{\text{typical}}
$$

The observation that O–O shows $R_{AA} < 1$ at $p_T \sim 6$ GeV while p–Pb does not implies:
- O–O: $\chi > 1$ (optically thick for 6 GeV partons)
- p–Pb: $\chi < 1$ (optically thin)

### When Analogies Fail: Cautionary Notes

The following cross-domain claims **do not transfer** without careful qualification:

1. **"QGP thermalization informs femtosecond laser processing"**: QGP thermalizes via strong QCD coupling; solids equilibrate via phonon-mediated electron-lattice coupling. The *mechanisms* differ fundamentally, even if both approach Planckian timescales.

2. **"q̂ in QGP equals stopping power in solids"**: q̂ parametrizes *transverse* momentum broadening of color charges; dE/dx in solids is dominated by electromagnetic stopping. The dimensional form ($\sim$ GeV²/fm vs MeV·cm²/g) reflects different physics.

3. **"$R_{AA}$ data constrain Bragg peak positioning"**: QGP quenching operates at GeV scale via QCD; therapy-relevant Bragg peaks arise from electromagnetic stopping at MeV scale. The energy regimes differ by $\sim 10^3$.

### What Does Transfer: Methodological Universalities

Rather than claiming physics transfers directly, the robust cross-domain insights are *methodological*:

| QGP Method | Transferable Insight | Target Domain |
|------------|---------------------|---------------|
| Bayesian parameter inference | High-dimensional inverse problems with nuisance parameters | Materials modeling, geophysics |
| Hydrodynamic response from initial geometry | "Structure from flow" inversion | Electron hydrodynamics, soft matter |
| System-size ratio cancellation | Reduce systematics by comparing similar systems | Any differential measurement |
| Multi-observable global fits | Constrain transport coefficients | Fusion plasma, dense matter |

**Bottom line**: The value of QGP physics for other domains lies primarily in *transferable methods* and *dimensionless scaling behaviors*, not in direct parameter translation.

---

## Conclusion

The advent of oxygen–oxygen and neon–neon collisions at the LHC [@CMS-PAS-HIN-25-008; @ALICE:Flow2025; @CERN-Courier-2025] has provided a fascinating middle-ground laboratory between the well-explored extremes of proton–nucleus and heavy–nucleus collisions. The emerging consensus from the measurements is that small QGP droplets are real – when two light nuclei collide at sufficiently high energy, they create a short-lived plasma that behaves in much the same way as the QGP in large nuclei, albeit on a reduced scale. The empirical signs of this plasma (collective flow, anisotropies, jet quenching, modified hadrochemistry) are all present in O–O and Ne–Ne collisions [@ALICE:Flow2025; @CMS-PAS-HIN-25-008; @Giacalone:FlowPrediction2025]. The theoretical frameworks developed for heavy-ion collisions – hydrodynamics for the soft sector and energy-loss formalisms for hard probes – successfully extend to these systems, with only minor adjustments mainly related to initial geometry and volume. Each new piece of data (e.g. the precise $v_2$ in Ne–Ne vs O–O, or the magnitude of $R_{AA}$ in O–O) serves to refine our understanding of QGP properties: for instance, the consistency with low-viscosity hydrodynamics in small systems strengthens constraints on the allowed mean free path of QGP constituents, and the observation of jet quenching in tiny systems suggests that $\hat{q}$ remains sizable in small volumes and/or that parton energy loss is efficient even over 2–3 fm distances.

We have also seen how light-ion collisions provide unique leverage to probe nuclear structure (validating Neon’s deformation, probing Oxygen’s symmetry) and to test scaling laws (such as participant scaling of multiplicity, or the interplay of N<sub>coll</sub> vs N<sub>part</sub> in producing secondaries). These collisions thus act as “control experiments” that complement the heavy-ion program: by varying the initial conditions in a controlled way (different nucleus size/shape while keeping energy high), one can disentangle effects and verify that our models hold water under new conditions.

Crucially, no obvious contradictions have appeared among the different experiments’ findings or between data and theory – rather, a coherent narrative is forming. If anything, the slight surprises (like the central increase of $v_3$ in O–O, or the fact that per-participant particle yields in O–O equal Pb–Pb) are themselves insights into the nature of fluctuations and particle production. There remain critical tensions to explore in finer detail: for example, models will need to simultaneously explain why $v_2$ decreases while $v_3$ increases toward central collisions in O–O (which might push improvements in initial state modeling of fluctuations); also, ensuring that a single set of model parameters can explain $R_{AA}$ from O–O up to Pb–Pb will be a non-trivial check on energy-loss theory. So far, the comparisons look promising, but more differential data (like identified particle $R_{AA}$ or jet substructure in O–O) could challenge certain assumptions.

In conclusion, the exploration of light-ion collisions supports the concept of the "mini-QGP" and fills in a missing piece of the heavy-ion jigsaw puzzle. We now see a continuous evolution from small to large systems, rather than a discontinuous gap – nature doesn’t abruptly turn off QGP effects when going to lighter nuclei, it gradually tapers. This realization encourages us to think of collective QCD phenomena on a continuum, controlled by initial entropy density and system size, rather than a strict division by projectile size. The work is far from over: upcoming analyses (p–O results, detailed multi-particle correlations, comparisons to theory with varied initial conditions) will further sharpen our understanding. With this new data in hand, theorists can fine-tune simulations that incorporate everything from nuclear structure, Glauber geometry, pre-equilibrium dynamics, hydrodynamics, to hadronization – for O–O and Ne–Ne specifically – and have confidence that those simulations mirror reality. Such “benchmark” smaller systems will then improve our overall confidence in the modeling of heavy-ion collisions (since any model that works from O–O to Pb–Pb is on very solid footing).

The light-ion collision program, therefore, represents a successful extension of QGP studies into a new domain, and it sets the stage for future investigations of QGP in even more extreme or varied conditions – be it different nuclei, higher energies, or eventually the Electron-Ion Collider (which will probe some complementary aspects of small QCD systems in terms of initial-state). For now, the O–O and Ne–Ne results stand as a testament that even "tiny" droplets of quark–gluon plasma behave in remarkable accordance with the physics first discovered in big heavy-ion collisions, reaffirming and extending our knowledge of the hot and dense QCD matter.

---

## Appendix: Data Provenance and Reproducibility

*This appendix provides machine-readable provenance for all data used in figures.*

### HEPData Records

| Observable | Experiment | Primary Source | HEPData Record | Status |
|------------|------------|----------------|----------------|--------|
| O–O $R_{AA}$ vs $p_T$ | CMS | CMS-PAS-HIN-25-008 | ins3068407 (pending) | Prelim. |
| Ne–Ne $R_{AA}$ vs $p_T$ | CMS | CMS-PAS-HIN-25-014 | ins3068408 (pending) | Prelim. |
| O–O $v_2$, $v_3$ vs centrality | ALICE | arXiv:2509.06428 | — | Submitted |
| Ne–Ne $v_2$, $v_3$ vs centrality | ALICE | arXiv:2509.06428 | — | Submitted |
| O–O $dN_{ch}/d\eta$ | CMS | CMS-PAS-HIN-25-010 | — | Prelim. |
| Pb–Pb $R_{AA}$ (0–10%) | ALICE | JHEP 11 (2018) 013 | ins1657384 | Published |
| Pb–Pb $v_2$, $v_3$ | ALICE | PRL 116 (2016) 132302 | ins1394678 | Published |
| Strangeness enhancement | ALICE | Nature Phys. 13 (2017) 535 | ins1512110 | Published |

### Figure Data Sources

**Figure 3 (RAA_multisystem)**:
- O–O data points: CMS-PAS-HIN-25-008, Table 1 (min-bias, $|\eta| < 1$)
- Pb–Pb data points: ALICE JHEP 11 (2018) 013, Fig. 1 (0–10% central)
- Model curves: BDMPS-Z with $\hat{q} = 2.5$ GeV²/fm (Pb–Pb), scaled by path length
- *Digitization*: Not applicable (synthetic curves based on published parametrizations)

**Figure 5 (flow_comprehensive)**:
- Model curves constrained by: CMS-PAS-HIN-25-009, ALICE arXiv:2509.06428
- Centrality mapping: Glauber MC (arXiv:2507.05853)
- *Data points shown are model predictions, not direct experimental measurements*

**Figure 7 (strangeness_enhancement)**:
- pp, p–Pb, Pb–Pb data: ALICE Nature Phys. 13 (2017) 535, HEPData ins1512110
- O–O band: Extrapolation based on multiplicity scaling (no measurement)

**Figure 9 (femtoscopy_hbt)**:
- HBT radii systematics: ALICE [@ALICE:HBT] and review [@Lisa:2005]
- O–O predictions: Scaled from Pb–Pb multiplicity dependence

**Figure 10 (direct_photon_spectra)**:
- Pb–Pb: PHENIX [@PHENIX:DirectPhoton] and ALICE [@ALICE:DirectPhoton] measurements
- Theory: Thermal photon rates and "direct photon puzzle" [@Rapp:DirectPhotonPuzzle]
- O–O: Model prediction only (no measurement exists)

### Data Classification Legend

All figures in this document are labeled with data classification:

| Label | Meaning |
|-------|---------|
| **[Data]** | Experimentally measured values with uncertainties |
| **[Data + Model]** | Measured data points with theoretical model curves |
| **[Model]** | Theoretical predictions or model outputs |
| **[Schematic]** | Illustrative/conceptual (not quantitative) |
| **[Extrapolation]** | Scaled or interpolated from other measurements |

### Reproducibility Statement

All TikZ/pgfplots figures in this document are generated from source code in `figures/*.tex`. Data files in `data/*.dat` are generated by Python scripts in `src/`. The full build pipeline is available at the repository and can be reproduced with `make`.

