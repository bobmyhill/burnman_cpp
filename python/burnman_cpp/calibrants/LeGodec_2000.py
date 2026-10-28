# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from LeGodec_2000."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class hBN(Calibrant):
    "The hexagonal boron nitride pressure standard reported by\nLeGodec et al. (2000; http://dx.doi.org/10.1080/08957950008200304)."

    def __init__(self):
        self._set_native(_native.LeGodec_2000.hBN())
