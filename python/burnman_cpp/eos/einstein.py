# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Einstein lattice functions and derivatives, excluding zero-point energy.

Scalar and broadcast array inputs are supported, with molar SI units and
temperatures in kelvin. ``n`` may be fractional.
"""

from .._core.einstein import (
    thermal_energy,
    molar_heat_capacity_v,
    helmholtz_energy,
    entropy,
    dmolar_heat_capacity_v_dT,
    dhelmholtz_dTheta,
    d2helmholtz_dTheta2,
    dentropy_dTheta,
)

__all__ = [
    "thermal_energy",
    "molar_heat_capacity_v",
    "helmholtz_energy",
    "entropy",
    "dmolar_heat_capacity_v_dT",
    "dhelmholtz_dTheta",
    "d2helmholtz_dTheta2",
    "dentropy_dTheta",
]
