# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Chidester_2021."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class KCl_B2(Calibrant):
    "The B2 KCl pressure standard reported by\nChidester et al. (2021; https://doi.org/10.1103/PhysRevB.104.094107)."

    def __init__(self):
        self._set_native(_native.Chidester_2021.KCl_B2())
