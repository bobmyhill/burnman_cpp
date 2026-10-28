# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Zhao_2000."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Mo(Calibrant):
    "The Mo pressure standard reported by\nZhao et al. (2000; https://doi.org/10.1103/PhysRevB.62.8766)."

    def __init__(self):
        self._set_native(_native.Zhao_2000.Mo())
