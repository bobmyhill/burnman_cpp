# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Dubrovinsky_1998."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Al2O3_corundum(Calibrant):
    "The Al2O3 pressure standard reported by\nDubrovinsky (1998; https://dx.doi.org/10.1007/s002690050133)."

    def __init__(self):
        self._set_native(_native.Dubrovinsky_1998.Al2O3_corundum())
