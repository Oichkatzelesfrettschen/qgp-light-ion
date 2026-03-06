"""
Cosmology Tier 2: Large-scale structure and reionization.

This tier connects QGP physics (Tier 1) to cosmological structure formation
(early universe, dark energy, reionization fronts).

Modules:
- reionization_bubble: Bubble growth and overlap statistics in cosmic reionization
- dark_energy: Dark energy equation of state and acceleration history
- bao_distance: Baryon acoustic oscillation and sound horizon measurements

Physical context:
The early universe QGP transition (~1 microsecond after Big Bang) sets the
initial conditions for baryon-photon coupling that leads to acoustic oscillations
in the primordial plasma. After recombination (z~1000), the universe becomes
optically thin, and by z~6 galaxies reionize the intergalactic medium through
UV radiation. The reionization bubble growth follows similar percolation physics
to QGP bubble nucleation in heavy-ion collisions.

References:
- Reionization: Gnedin, ApJ 535 (2000) L75; Planck Collaboration XXVII (2018)
- BAO: Eisenstein et al., ApJ 633 (2005) 560; DESI DR2 (2025)
- Dark energy: Perlmutter et al., ApJ 517 (1999) 565; JWST time-domain cosmology
"""

from .dark_energy import (
    DarkEnergyModel,
    bao_measurement,
    comoving_distance,
    distance_modulus,
)
from .reionization_bubble import (
    ReionizationBubble,
    bubble_growth_rate,
    overlap_probability,
)

__all__ = [
    "DarkEnergyModel",
    "ReionizationBubble",
    "bao_measurement",
    "bubble_growth_rate",
    "comoving_distance",
    "distance_modulus",
    "overlap_probability",
]
