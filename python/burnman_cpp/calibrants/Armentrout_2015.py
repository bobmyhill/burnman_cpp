# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Armentrout_2015."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Co_fcc(Calibrant):
    "The FCC Cobalt pressure standard reported by Armentrout\n(2015; https://doi.org/10.1063/1.4935087)."

    def __init__(self):
        self._set_native(_native.Armentrout_2015.Co_fcc())
