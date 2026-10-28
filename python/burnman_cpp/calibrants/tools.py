# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Conversion between pressure calibrations of the same material."""

from .._core.calibrants.tools import pressure_to_pressure as _pressure_to_pressure


def pressure_to_pressure(
    old_calibrant, new_calibrant, pressure, temperature, PT_covariance=None
):
    """Convert pressure scales, optionally returning covariance in [P,T] order.

    Preserve volume and temperature using the old and new calibrations of the
    same material. The supplied 2x2 covariance has pressure/temperature units.
    """
    return _pressure_to_pressure(
        old_calibrant._as_native(),
        new_calibrant._as_native(),
        pressure,
        temperature,
        PT_covariance,
    )
