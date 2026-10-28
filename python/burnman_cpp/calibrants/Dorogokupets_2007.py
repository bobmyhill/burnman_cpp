# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Dorogokupets_2007."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Ag(Calibrant):
    "The Ag pressure standard reported by\nDorogokupets and Oganov (2007; https://doi.org/10.1103/PhysRevB.75.024115)."

    def __init__(self):
        self._set_native(_native.Dorogokupets_2007.Ag())


class Al(Calibrant):
    "The Al pressure standard reported by\nDorogokupets and Oganov (2007; https://doi.org/10.1103/PhysRevB.75.024115)."

    def __init__(self):
        self._set_native(_native.Dorogokupets_2007.Al())


class Au(Calibrant):
    "The Au pressure standard reported by\nDorogokupets and Oganov (2007; https://doi.org/10.1103/PhysRevB.75.024115)."

    def __init__(self):
        self._set_native(_native.Dorogokupets_2007.Au())


class Cu(Calibrant):
    "The Cu pressure standard reported by\nDorogokupets and Oganov (2007; https://doi.org/10.1103/PhysRevB.75.024115)."

    def __init__(self):
        self._set_native(_native.Dorogokupets_2007.Cu())


class diamond(Calibrant):
    "The diamond pressure standard reported by\nDorogokupets and Oganov (2007; https://doi.org/10.1103/PhysRevB.75.024115)."

    def __init__(self):
        self._set_native(_native.Dorogokupets_2007.diamond())


class MgO(Calibrant):
    "The MgO pressure standard reported by\nDorogokupets and Oganov (2007; https://doi.org/10.1103/PhysRevB.75.024115)."

    def __init__(self):
        self._set_native(_native.Dorogokupets_2007.MgO())


class Pt(Calibrant):
    "The Pt pressure standard reported by\nDorogokupets and Oganov (2007; https://doi.org/10.1103/PhysRevB.75.024115)."

    def __init__(self):
        self._set_native(_native.Dorogokupets_2007.Pt())


class Ta(Calibrant):
    "The Ta pressure standard reported by\nDorogokupets and Oganov (2007; https://doi.org/10.1103/PhysRevB.75.024115)."

    def __init__(self):
        self._set_native(_native.Dorogokupets_2007.Ta())


class W(Calibrant):
    "The W pressure standard reported by\nDorogokupets and Oganov (2007; https://doi.org/10.1103/PhysRevB.75.024115)."

    def __init__(self):
        self._set_native(_native.Dorogokupets_2007.W())
