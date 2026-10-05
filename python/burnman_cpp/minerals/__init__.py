"""Native C++ mineral and solution factories for the equilibration examples.

HP_2011_ds62, SLB_2011, JH_2015, mb50NCKFMASHTO, mp50NCKFMASHTO and HGP_2018_ds633
expose the implemented subset of each
published dataset. Factories create independent, validated native objects.
The binary solution factories provide fixed reduced chemical systems and
require no Python BurnMan. The HGP melt factory is Cr-free and corrects
the jadeite pseudo-species spelling to retain its intended unit occupancy.
"""

from .._core.minerals import (
    SLB_2024,
    HP_2011_ds62,
    SLB_2011,
    JH_2015,
    mb50NCKFMASHTO,
    mp50NCKFMASHTO,
    HGP_2018_ds633,
)

__all__ = [
    "SLB_2024",
    "HP_2011_ds62",
    "SLB_2011",
    "JH_2015",
    "mb50NCKFMASHTO",
    "mp50NCKFMASHTO",
    "HGP_2018_ds633",
]

# SLB_2024 includes all 15 unrelaxed solution models and 74 endmembers.
# Equilibrating the full endmember coordinates also equilibrates spin states.
