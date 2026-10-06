"""Complete native C++ catalogues copied from the pinned Python BurnMan.

HP_2011_ds62, SLB_2011, JH_2015, mb50NCKFMASHTO, mp50NCKFMASHTO and HGP_2018_ds633
expose all public mineral and solution definitions, including aliases and
combined endmembers. Factories create independent, validated native objects.
The binary solution factories provide fixed reduced chemical systems and
require no Python BurnMan. HGP silicate_melt includes Cr; silicate_melt_cr_free
retains the earlier example model. Melt factories correct the jadeite
pseudo-species spelling to retain its intended unit occupancy.
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

# SLB_2024 also provides the two relaxed spin models.
# Equilibrating the full endmember coordinates also equilibrates spin states.
