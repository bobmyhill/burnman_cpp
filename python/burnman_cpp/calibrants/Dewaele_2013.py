# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Dewaele_2013."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Al2O3_corundum(Calibrant):
    "The alpha-Al2O3 (corundum) pressure standard reported by\nDewaele and Torrent (2013; https://doi.org/10.1103/PhysRevB.88.064107).\nThe thermal model is from Shi (2022)."

    def __init__(self):
        self._set_native(_native.Dewaele_2013.Al2O3_corundum())
