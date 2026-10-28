# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Speziale_2001."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class MgO(Calibrant):
    "The MgO pressure standard reported by\nSpeziale (2001; https://doi.org/10.1029/2000JB900318)."

    def __init__(self):
        self._set_native(_native.Speziale_2001.MgO())
