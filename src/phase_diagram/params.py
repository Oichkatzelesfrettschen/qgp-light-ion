"""Shared dataclasses and constants for QCD phase diagram submodules."""

from __future__ import annotations

from dataclasses import dataclass

from constants import KAPPA2, KAPPA2_ERR, KAPPA4, T_C0_MEV, T_C0_MEV_ERR


@dataclass
class PhaseTransitionParams:
    """QCD phase transition parameters from lattice QCD (December 2025 update)."""

    # Crossover temperature at mu_B = 0 (HotQCD 2019, Phys. Lett. B 795)
    T_c0: float = T_C0_MEV
    T_c0_err: float = T_C0_MEV_ERR

    # Curvature of crossover line (HotQCD, arXiv:1812.08235)
    # T_c(mu_B)/T_c(0) = 1 - kappa2*(mu_B/T_c)^2 - kappa4*(mu_B/T_c)^4
    kappa2: float = KAPPA2
    kappa2_err: float = KAPPA2_ERR
    kappa4: float = KAPPA4

    # Critical point exclusion -- Borsanyi et al. PRD 112 L111505 (Dec 2025)
    mu_B_excluded_2sigma: float = 450.0  # MeV - CP excluded below this at 2sigma
    mu_B_excluded_1sigma: float = 400.0  # MeV - CP excluded below this at 1sigma

    # Superseded Clarke et al. 2024 estimate (arXiv:2405.10196) -- NOW EXCLUDED
    T_cp_clarke: float = 105.0
    T_cp_clarke_err_up: float = 8.0
    T_cp_clarke_err_down: float = 18.0
    mu_B_cp_clarke: float = 422.0
    mu_B_cp_clarke_err_up: float = 80.0
    mu_B_cp_clarke_err_down: float = 35.0

    # FRG consensus (QM2025) -- Fu, Pawlowski, Rennecke arXiv:2510.11270
    T_cp_frg: float = 110.0
    T_cp_frg_err: float = 11.0
    mu_B_cp_frg: float = 630.0
    mu_B_cp_frg_err: float = 63.0


@dataclass
class FreezeOutPoint:
    """Chemical freeze-out measurement."""

    sqrt_s_NN: float  # GeV
    T: float  # MeV
    T_err: float  # MeV
    mu_B: float  # MeV
    mu_B_err: float  # MeV
    experiment: str
    system: str


@dataclass
class CollisionSystem:
    """Collision system with approximate phase diagram location."""

    name: str
    sqrt_s_NN: float  # GeV
    mu_B: float  # MeV (approximate at freeze-out)
    T: float  # MeV (freeze-out temperature)
    T_initial: float | None = None  # Initial temperature estimate
    marker: str = "square"
    color_key: str = "PbPbcolor"


# Experimental freeze-out data from thermal model fits
# Sources: Andronic et al. Nature 561 (2018) 321; STAR BES publications
FREEZE_OUT_DATA = [
    FreezeOutPoint(4.85, 125.0, 4.0, 420.0, 25.0, "AGS", "Au-Au"),
    FreezeOutPoint(6.3, 136.0, 4.0, 380.0, 20.0, "NA49", "Pb-Pb"),
    FreezeOutPoint(7.7, 148.0, 4.0, 340.0, 18.0, "NA49/STAR", "Au-Au"),
    FreezeOutPoint(8.8, 150.0, 4.0, 310.0, 16.0, "NA49", "Pb-Pb"),
    FreezeOutPoint(12.3, 156.0, 4.0, 260.0, 14.0, "NA49", "Pb-Pb"),
    FreezeOutPoint(17.3, 158.0, 4.0, 220.0, 12.0, "NA49", "Pb-Pb"),
    FreezeOutPoint(19.6, 160.0, 4.0, 195.0, 12.0, "STAR", "Au-Au"),
    FreezeOutPoint(27.0, 162.0, 4.0, 160.0, 10.0, "STAR", "Au-Au"),
    FreezeOutPoint(39.0, 164.0, 4.0, 115.0, 8.0, "STAR", "Au-Au"),
    FreezeOutPoint(62.4, 166.0, 4.0, 75.0, 6.0, "STAR", "Au-Au"),
    FreezeOutPoint(130.0, 167.0, 4.0, 40.0, 5.0, "STAR", "Au-Au"),
    FreezeOutPoint(200.0, 166.0, 4.0, 25.0, 4.0, "STAR", "Au-Au"),
    FreezeOutPoint(2760.0, 156.5, 3.0, 1.0, 1.0, "ALICE", "Pb-Pb"),
    FreezeOutPoint(5020.0, 156.5, 3.0, 0.7, 0.7, "ALICE", "Pb-Pb"),
]

COLLISION_SYSTEMS = [
    CollisionSystem(
        "LHC Pb-Pb", 5020, 0.7, 156.5, T_initial=400, marker="square*", color_key="PbPbcolor"
    ),
    CollisionSystem(
        "LHC O-O", 7000, 0.5, 156.5, T_initial=350, marker="triangle*", color_key="OOcolor"
    ),
    CollisionSystem(
        "LHC Ne-Ne", 6500, 0.5, 156.5, T_initial=360, marker="diamond*", color_key="NeNecolor"
    ),
    CollisionSystem(
        "RHIC Au-Au", 200, 25, 166, T_initial=340, marker="pentagon*", color_key="accentpurple"
    ),
    CollisionSystem(
        "SPS Pb-Pb", 17.3, 250, 158, T_initial=260, marker="star", color_key="accentred"
    ),
    CollisionSystem(
        "AGS Au-Au", 4.85, 526, 125, T_initial=180, marker="oplus", color_key="textmid"
    ),
]

# RHIC BES isentropes (entropy/baryon values)
ISENTROPE_VALUES = [
    (420, "sqrt_s = 200 GeV"),
    (144, "sqrt_s = 62.4 GeV"),
    (70, "sqrt_s = 27 GeV"),
    (51, "sqrt_s = 19.6 GeV"),
    (30, "sqrt_s = 11.5 GeV"),
]

# Future heavy-ion facilities coverage regions
FUTURE_FACILITIES = {
    "FAIR_CBM": {
        "name": "FAIR/CBM",
        "sqrt_s_range": (2.7, 4.9),
        "mu_B_range": (500, 800),
        "T_range": (50, 150),
        "description": "High-mu_B frontier (2025+)",
    },
    "NICA_MPD": {
        "name": "NICA/MPD",
        "sqrt_s_range": (4, 11),
        "mu_B_range": (300, 600),
        "T_range": (100, 170),
        "description": "CP search region (2024+)",
    },
    "RHIC_BES2": {
        "name": "RHIC BES-II",
        "sqrt_s_range": (7.7, 27),
        "mu_B_range": (150, 450),
        "T_range": (130, 165),
        "description": "CP search ongoing",
    },
}

COLLISION_SQRT_S = {
    "LHC_PbPb": {"sqrt_s": 5020, "mu_B": 0.7, "T": 156.5, "label": "5.02 TeV"},
    "LHC_OO": {"sqrt_s": 7000, "mu_B": 0.5, "T": 156.5, "label": "7 TeV"},
    "RHIC_200": {"sqrt_s": 200, "mu_B": 25, "T": 166, "label": "200 GeV"},
    "SPS": {"sqrt_s": 17.3, "mu_B": 220, "T": 158, "label": "17.3 GeV"},
    "AGS": {"sqrt_s": 4.85, "mu_B": 420, "T": 125, "label": "4.85 GeV"},
}
