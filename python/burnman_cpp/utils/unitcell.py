# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Native cell conversions and molar volume from crystallographic volumes."""

from ..constants import Avogadro
from .._core import (
    cell_parameters_to_vectors,
    cell_vectors_to_parameters,
    rotation_to_crystallographic_frame,
)


def molar_volume_from_unit_cell_volume(unit_cell_v, z):
    """Convert Angstrom cubed per cell to cubic metres per mole (z formulae/cell)."""
    return unit_cell_v * Avogadro / 1.0e30 / z
