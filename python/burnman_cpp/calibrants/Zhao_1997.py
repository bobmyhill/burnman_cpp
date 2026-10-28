# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Zhao_1997."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class hBN(Calibrant):
    "The hBN pressure standard reported by\nZhao et al. (1997; https://doi.org/10.1080/08957959708240481)."

    def __init__(self):
        self._set_native(_native.Zhao_1997.hBN())
