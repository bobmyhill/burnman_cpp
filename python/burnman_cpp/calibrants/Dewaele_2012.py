# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Dewaele_2012."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class KBr_B2(Calibrant):
    "The B2 KBr pressure standard reported by\nDewaele et al. (2012; https://doi.org/10.1103/PhysRevB.85.214105)."

    def __init__(self):
        self._set_native(_native.Dewaele_2012.KBr_B2())


class KCl_B2(Calibrant):
    "The B2 KCl pressure standard reported by\nDewaele et al. (2012; https://doi.org/10.1103/PhysRevB.85.214105)."

    def __init__(self):
        self._set_native(_native.Dewaele_2012.KCl_B2())
