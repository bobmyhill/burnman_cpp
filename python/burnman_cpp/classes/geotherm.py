# ------------------------------------------------------
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.
#
# This file is part of BurnMan.
#
# Detailed license information governing the source code
# and contributions can be found in the LICENSE document.
#
# ------------------------------------------------------

"""Native geotherms following BurnMan's classes.geotherm hierarchy.

Depths use m, pressures Pa and temperatures K. Tabulated profiles raise
ValueError outside their data range and retain the shallower value at an exact
discontinuity. Queries accept scalars and arrays, returning floating-point
temperatures with the input shape (including integer coordinate inputs).

Pressure geotherms default to the shared seismic.prem_model and also accept
objects providing pressure(depths). Adiabats solve for constant molar entropy
at fixed composition and leave the material at the anchor pressure/temperature,
as in Python BurnMan. Empty queries return empty arrays.
"""

from .._core.geotherm import (
    Geotherm,
    BrownShankland,
    Anderson,
    StaceyContinental,
    StaceyOceanic,
    Katsura2022,
    Plesa2022Mars6cm3,
    GeothermFromPressures,
    Anzellini2013,
    AdiabaticGeotherm,
    adiabatic_profile,
)

__all__ = [
    "Geotherm",
    "BrownShankland",
    "Anderson",
    "StaceyContinental",
    "StaceyOceanic",
    "Katsura2022",
    "Plesa2022Mars6cm3",
    "GeothermFromPressures",
    "Anzellini2013",
    "AdiabaticGeotherm",
    "adiabatic_profile",
]
