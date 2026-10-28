# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Matsui_2012."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class NaCl_B1(Calibrant):
    "The NaCl (B1 structured) pressure standard reported by\nMatsui (2012; https://doi.org/10.2138/am.2012.4136)."

    def __init__(self):
        self._set_native(_native.Matsui_2012.NaCl_B1())
