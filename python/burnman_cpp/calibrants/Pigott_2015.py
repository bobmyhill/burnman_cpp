# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Pigott_2015."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Ni(Calibrant):
    "The Ni pressure standard reported by\nPigott (2015; https://doi.org/10.1002/2015GL066577)."

    def __init__(self):
        self._set_native(_native.Pigott_2015.Ni())
