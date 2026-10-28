# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Zha_2004."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Re(Calibrant):
    "The Re pressure standard reported by\nZha et al. (2004; https://doi.org/10.1063/1.1765752)."

    def __init__(self):
        self._set_native(_native.Zha_2004.Re())
