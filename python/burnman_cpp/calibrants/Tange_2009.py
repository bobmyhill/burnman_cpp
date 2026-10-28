# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Tange_2009."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class MgO_BM3(Calibrant):
    "The MgO pressure standard reported by\nTange et al. (2009; https://doi.org/10.1029/2008JB005813).\nBM3 version."

    def __init__(self):
        self._set_native(_native.Tange_2009.MgO_BM3())


class MgO_Vinet(Calibrant):
    "The MgO pressure standard reported by\nTange et al. (2009; https://doi.org/10.1029/2008JB005813).\nVinet version."

    def __init__(self):
        self._set_native(_native.Tange_2009.MgO_Vinet())
