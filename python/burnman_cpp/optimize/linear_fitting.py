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

"""Weighted least-squares fitting with linear equality and inequality constraints."""

from .._core.optimize.linear_fitting import weighted_constrained_least_squares

__all__ = ["weighted_constrained_least_squares"]
