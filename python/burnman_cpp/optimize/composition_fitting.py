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

"""Fit nonnegative phase amounts to a bulk chemical composition."""

from .._core.optimize.composition_fitting import (
    fit_phase_proportions_to_bulk_composition,
)

__all__ = ["fit_phase_proportions_to_bulk_composition"]
