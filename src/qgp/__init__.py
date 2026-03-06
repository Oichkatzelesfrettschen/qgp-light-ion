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

__all__ = [
    "KAPPA2",
    "KAPPA2_ERR",
    "KAPPA4",
    "T_C0_MEV",
    "T_C0_MEV_ERR",
    "Nucleus",
    "PhaseTransitionParams",
    "bao_measurement",
    "bdmps_energy_loss",
    "bjorken_energy_density",
    "calculate_participants",
    "comoving_distance",
    "constants",
    "critical_point_box_excluded",
    "critical_point_ellipse_excluded",
    "critical_point_exclusion_boundary",
    "critical_point_exclusion_region",
    "critical_point_frg_box",
    "critical_point_frg_ellipse",
    "crossover_temperature",
    "crossover_uncertainty_band",
    "distance_modulus",
    "ensure_dir",
    "first_order_consensus_band",
    "first_order_frg",
    "first_order_line",
    "first_order_njl",
    "first_order_pqm",
    "freeze_out_from_sqrt_s",
    "freeze_out_parametrization",
    "freeze_out_uncertainty_band",
    "get_nuclear_profile_2d",
    "load_dat",
    "make_provenance_header",
    "raa_model",
    "sample_nucleon_positions",
    "save_2d_grid",
    "save_curve",
    "save_curve_multi",
    "save_curve_with_errors",
    "save_dat",
    "save_points_with_errors",
    "woods_saxon",
]
