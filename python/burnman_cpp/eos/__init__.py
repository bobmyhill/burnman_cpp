# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Native thermal EoS components, using Python BurnMan names and molar SI units."""

import sys as _sys

from .. import (
    debye_temperature_models,
    anharmonic_prefactor_models,
    anharmonic_thermal_models,
)
from . import debye, einstein, bukowinski_electronic, anharmonic_debye

__all__ = [
    "debye",
    "einstein",
    "bukowinski_electronic",
    "anharmonic_debye",
    "debye_temperature_models",
    "anharmonic_prefactor_models",
    "anharmonic_thermal_models",
]

for _name in (
    "debye_temperature_models",
    "anharmonic_prefactor_models",
    "anharmonic_thermal_models",
):
    _sys.modules[f"{__name__}.{_name}"] = globals()[_name]
