# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Fei_2007."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Pt(Calibrant):
    "The Pt pressure standard reported by\nFei et al. (2007; https://doi.org/10.1073/pnas.0609013104)."

    def __init__(self):
        self._set_native(_native.Fei_2007.Pt())


class Au(Calibrant):
    "The Au pressure standard reported by\nFei et al. (2007; https://doi.org/10.1073/pnas.0609013104)."

    def __init__(self):
        self._set_native(_native.Fei_2007.Au())
