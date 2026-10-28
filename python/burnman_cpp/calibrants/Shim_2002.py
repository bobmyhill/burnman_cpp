# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Shim_2002."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Au(Calibrant):
    "The Au pressure standard reported by\nShim et al. (2002; https://doi.org/10.1016/S0012-821X(02)00917-2)."

    def __init__(self):
        self._set_native(_native.Shim_2002.Au())
