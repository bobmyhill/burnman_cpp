# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Matsui_2010."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Au(Calibrant):
    "The Au pressure standard reported by\nMatsui (2010; https://dx.doi.org/10.1088/1742-6596/215/1/012197)."

    def __init__(self):
        self._set_native(_native.Matsui_2010.Au())
