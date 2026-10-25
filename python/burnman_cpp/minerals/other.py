# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Additional native mineral models."""

from .. import _core


def liquid_iron():
    """AA1994 liquid Fe with the existing Cv-based Helmholtz formulation.

    Uses the Figure 5 heat capacity and the legacy unfitted S_0=100 J/mol/K.
    """
    return _core._liquid_iron()
