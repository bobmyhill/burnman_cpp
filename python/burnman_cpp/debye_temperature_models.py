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

"""Native volume dependence models for the Debye temperature."""

from ._core.debye_temperature_models import (
    DebyeTemperatureModelBase,
    SLB,
    PowerLawGammaSimple,
    PowerLawGamma,
)

__all__ = ["DebyeTemperatureModelBase", "SLB", "PowerLawGammaSimple", "PowerLawGamma"]
