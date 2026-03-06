"""
QGP Light-Ion Physics Module (Tier 1)

This module implements core quark-gluon plasma physics models:
- Nuclear density profiles (Woods-Saxon, deformation)
- Glauber Monte Carlo geometry
- BDMPS-Z radiative energy loss
- Viscous hydrodynamic response
- QCD phase diagram

All physical constants are imported from constants.py (source of truth).
"""

from . import constants
from .constants import (
    KAPPA2,
    KAPPA2_ERR,
    KAPPA4,
    T_C0_MEV,
    T_C0_MEV_ERR,
)
from .io_utils import (
    ensure_dir,
    load_dat,
    make_provenance_header,
    save_2d_grid,
    save_curve,
    save_curve_multi,
    save_curve_with_errors,
    save_dat,
    save_points_with_errors,
)
from .physics import (
    Nucleus,
    bdmps_energy_loss,
    bjorken_energy_density,
    calculate_participants,
    get_nuclear_profile_2d,
    raa_model,
    sample_nucleon_positions,
    woods_saxon,
)
from .phase_diagram import (
    PhaseTransitionParams,
    critical_point_box_excluded,
    critical_point_ellipse_excluded,
    critical_point_exclusion_boundary,
    critical_point_exclusion_region,
    critical_point_frg_box,
    critical_point_frg_ellipse,
    crossover_temperature,
    crossover_uncertainty_band,
    first_order_consensus_band,
    first_order_frg,
    first_order_line,
    first_order_njl,
    first_order_pqm,
    freeze_out_from_sqrt_s,
    freeze_out_parametrization,
    freeze_out_uncertainty_band,
)

__all__ = [
    # Constants
    "constants",
    "KAPPA2",
    "KAPPA2_ERR",
    "KAPPA4",
    "T_C0_MEV",
    "T_C0_MEV_ERR",
    # Physics models
    "Nucleus",
    "woods_saxon",
    "get_nuclear_profile_2d",
    "bdmps_energy_loss",
    "raa_model",
    "sample_nucleon_positions",
    "calculate_participants",
    "bjorken_energy_density",
    # Phase diagram
    "PhaseTransitionParams",
    "crossover_temperature",
    "crossover_uncertainty_band",
    "critical_point_exclusion_region",
    "critical_point_exclusion_boundary",
    "critical_point_frg_ellipse",
    "critical_point_frg_box",
    "critical_point_ellipse_excluded",
    "critical_point_box_excluded",
    "first_order_line",
    "first_order_njl",
    "first_order_pqm",
    "first_order_frg",
    "first_order_consensus_band",
    "freeze_out_parametrization",
    "freeze_out_from_sqrt_s",
    "freeze_out_uncertainty_band",
    # I/O
    "save_dat",
    "load_dat",
    "save_2d_grid",
    "save_curve",
    "save_curve_multi",
    "save_curve_with_errors",
    "save_points_with_errors",
    "make_provenance_header",
    "ensure_dir",
]
