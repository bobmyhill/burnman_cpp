# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Campbell_2009."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Ni_fcc(Calibrant):
    "The FCC Ni pressure standard reported by\nCampbell (2009; https://doi.org/10.1016/j.epsl.2009.07.022)."

    def __init__(self):
        self._set_native(_native.Campbell_2009.Ni_fcc())
