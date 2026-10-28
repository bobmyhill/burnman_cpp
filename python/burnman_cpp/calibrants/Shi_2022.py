# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Shi_2022."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Al2O3_corundum(Calibrant):
    "The corundum-structured Al2O3 pressure standard reported by\nShi (2022; https://doi.org/10.1029/2021JB023805)."

    def __init__(self):
        self._set_native(_native.Shi_2022.Al2O3_corundum())
