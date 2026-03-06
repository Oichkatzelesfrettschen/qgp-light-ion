"""First-order phase transition lines from multiple theoretical models."""

from __future__ import annotations

import numpy as np

from .params import PhaseTransitionParams


def first_order_line(mu_B: np.ndarray, params: PhaseTransitionParams) -> np.ndarray:
    """
    First-order phase transition line beyond the critical point (schematic).

    UPDATED December 2025: Uses FRG consensus CP location (110, 630) MeV.
    Connects CP to nuclear matter ground state at T=0, mu_B=930 MeV.
    """
    mu_cp = params.mu_B_cp_frg   # 630 MeV
    T_cp = params.T_cp_frg       # 110 MeV
    mu_nm = 930.0                # Nuclear matter (MeV)
    T_nm = 0.0

    mask = mu_B >= mu_cp
    T = np.zeros_like(mu_B)
    T[mask] = T_cp - (T_cp - T_nm) / (mu_nm - mu_cp) * (mu_B[mask] - mu_cp)
    T[~mask] = np.nan

    return T


def first_order_njl(mu_B: np.ndarray, params: PhaseTransitionParams) -> np.ndarray:
    """
    First-order line from Nambu--Jona-Lasinio (NJL) model.

    NJL critical point typically: T_CP ~ 70-90 MeV, mu_B ~ 300-350 MeV.
    Source: Buballa, Phys. Rep. 407 (2005) 205.
    """
    T_cp_njl = 80.0
    mu_cp_njl = 330.0

    mask = mu_B >= mu_cp_njl
    T = np.full_like(mu_B, np.nan)
    T[mask] = T_cp_njl * np.exp(-((mu_B[mask] - mu_cp_njl) ** 2) / (2 * 400**2))

    return T


def first_order_pqm(mu_B: np.ndarray, params: PhaseTransitionParams) -> np.ndarray:
    """
    First-order line from Polyakov-Quark-Meson (PQM) model.

    PQM critical point: T_CP ~ 90-110 MeV, mu_B ~ 350-400 MeV.
    Source: Schaefer et al., PRD 76 (2007) 074023.
    """
    T_cp_pqm = 95.0
    mu_cp_pqm = 370.0

    mask = mu_B >= mu_cp_pqm
    T = np.full_like(mu_B, np.nan)
    T[mask] = T_cp_pqm - (T_cp_pqm / (930 - mu_cp_pqm)) * (mu_B[mask] - mu_cp_pqm)
    T[mask] = np.maximum(T[mask], 0)

    return T


def first_order_frg(mu_B: np.ndarray, params: PhaseTransitionParams) -> np.ndarray:
    """
    First-order line from Functional Renormalization Group (FRG).

    UPDATED: Uses QM2025 FRG consensus (Fu, Pawlowski, Rennecke arXiv:2510.11270).
    CEP at (T=110, mu_B=630) MeV -- beyond the lattice exclusion region.
    """
    T_cp_frg = params.T_cp_frg     # 110 MeV
    mu_cp_frg = params.mu_B_cp_frg  # 630 MeV

    mask = mu_B >= mu_cp_frg
    T = np.full_like(mu_B, np.nan)
    T[mask] = T_cp_frg * (1 - ((mu_B[mask] - mu_cp_frg) / (930 - mu_cp_frg)) ** 1.5)
    T[mask] = np.maximum(T[mask], 0)

    return T


def first_order_consensus_band(
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the consensus band for first-order transition predictions.

    Encompasses the spread of NJL, PQM, and FRG theoretical predictions.
    The band represents theoretical uncertainty across models.

    Returns: (mu_B_range, T_frg_upper, T_njl_lower)
    """
    params = PhaseTransitionParams()
    mu_B_range = np.linspace(630, 920, n_points)

    # FRG upper boundary (QM2025 consensus)
    T_frg = params.T_cp_frg * (
        1 - ((mu_B_range - params.mu_B_cp_frg) / (930 - params.mu_B_cp_frg)) ** 1.5
    )
    T_frg = np.maximum(T_frg, 0)

    # NJL lower boundary (predicts lower T at the same mu_B)
    T_njl_at_630 = 80.0 * np.exp(-((630 - 330) ** 2) / (2 * 400**2))
    T_njl = T_njl_at_630 * np.exp(-((mu_B_range - 630) ** 2) / (2 * 250**2))
    T_njl = np.maximum(T_njl, 0)

    return mu_B_range, T_frg, T_njl
