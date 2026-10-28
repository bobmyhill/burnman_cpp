# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Holmes_1989."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Pt(Calibrant):
    "The Pt pressure standard reported by\nHolmes et al. (1989; https://doi.org/10.1063/1.344177)."

    def __init__(self):
        self._set_native(_native.Holmes_1989.Pt())
