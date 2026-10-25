# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Debye lattice functions and derivatives, excluding zero-point energy.

Scalar and broadcast array inputs are supported. Temperatures are in kelvin;
energies and entropies use molar SI units. ``n`` may be fractional.
"""

from .._core.debye import (
    debye_fn,
    debye_fn_cheb,
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
    "debye_fn",
    "debye_fn_cheb",
    "thermal_energy",
    "molar_heat_capacity_v",
    "helmholtz_energy",
    "entropy",
    "dmolar_heat_capacity_v_dT",
    "dhelmholtz_dTheta",
    "d2helmholtz_dTheta2",
    "dentropy_dTheta",
]
