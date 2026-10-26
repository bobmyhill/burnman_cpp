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

"""Helpers for fitting material equations of state."""

from .._core.optimize.eos_fitting import (
    default_mle_tolerances,
    MineralFit,
    fit_PTV_data,
    fit_PTp_data,
)
from .nonlinear_fitting import NonLinearModel

NonLinearModel.register(MineralFit)

__all__ = ["default_mle_tolerances", "MineralFit", "fit_PTV_data", "fit_PTp_data"]
