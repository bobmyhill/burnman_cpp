# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Dewaele_2008."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Ni(Calibrant):
    "The Ni pressure standard reported by\nDewaele (2008; Table IV; https://doi.org/10.1103/PhysRevB.78.104102)."

    def __init__(self):
        self._set_native(_native.Dewaele_2008.Ni())
