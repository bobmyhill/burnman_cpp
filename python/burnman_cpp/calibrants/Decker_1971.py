# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Decker_1971."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class NaCl_B1(Calibrant):
    "The NaCl (B1 structured) pressure standard reported by\nDecker (1971; https://doi.org/10.1063/1.1660714).\n\n.. note:: This calibrant is not exactly the same as that\n    proposed by Decker. The cold compression curve has here been\n    approximated by a 4th order Birch-Murnaghan EoS, as described in\n    Matsui et al. (2012; https://doi.org/10.2138/am.2012.4136)."

    def __init__(self):
        self._set_native(_native.Decker_1971.NaCl_B1())
