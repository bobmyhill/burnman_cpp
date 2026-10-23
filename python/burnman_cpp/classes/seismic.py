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

"""Native one-dimensional seismic models, following BurnMan classes.seismic.

Depth/radius are in m, pressure/moduli in Pa, density in kg/m³ and velocities
in m/s. Property queries accept scalars or NumPy arrays. ``evaluate`` returns
an array with properties by rows and depths by columns. Missing pressure and
gravity profiles are integrated lazily and cached; this requires density over
the entire planet. Tables can be assigned on a ``SeismicTable`` subclass.

``radius(pressure)`` uses ``earth_radius - depth(pressure)`` with the same
pressure bounds as ``depth``, correcting the reversed interpolation upstream.
"""

from .._core.seismic import (
    Seismic1DModel,
    SeismicTable,
    PREM,
    Slow,
    Fast,
    STW105,
    IASP91,
    AK135,
    prem_model,
)

__all__ = [
    "Seismic1DModel",
    "SeismicTable",
    "PREM",
    "Slow",
    "Fast",
    "STW105",
    "IASP91",
    "AK135",
    "prem_model",
]
