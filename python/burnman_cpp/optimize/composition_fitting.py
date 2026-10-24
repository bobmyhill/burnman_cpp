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

"""Chemical compositions, site occupancies and nonnegative phase amounts."""

from .._core.optimize.composition_fitting import (
    DummyCompositionSolution,
    fit_composition_to_solution,
    fit_phase_proportions_to_bulk_composition,
)

__all__ = [
    "DummyCompositionSolution",
    "fit_composition_to_solution",
    "fit_phase_proportions_to_bulk_composition",
]
