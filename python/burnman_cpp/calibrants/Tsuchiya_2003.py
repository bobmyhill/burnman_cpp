# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Tsuchiya_2003."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Au(Calibrant):
    "The Au pressure standard reported by\nTsuchiya (2003; https://doi.org/10.1029/2003JB002446)."

    def __init__(self):
        self._set_native(_native.Tsuchiya_2003.Au())
