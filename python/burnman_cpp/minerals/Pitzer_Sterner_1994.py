# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Pitzer-Sterner water with Python BurnMan's original Debye thermal reference.

The 1995 paper repeats the 1994 water coefficient table. Absolute thermal
properties use the Debye fit in Python's Pitzer_Sterner_1994 dataset.
For the HP2011 ideal-gas reference, use HP_2011_fluids.H2O instead.
"""

from .._core import _ps1994_water as H2O_Pitzer_Sterner

__all__ = ["H2O_Pitzer_Sterner"]
