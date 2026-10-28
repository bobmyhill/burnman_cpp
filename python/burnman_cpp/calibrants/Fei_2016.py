# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Fei_2016."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Fe_hcp(Calibrant):
    "The HCP Fe pressure standard reported by\nFei et al. (2016; https://doi.org/10.1002/2016GL069456)."

    def __init__(self):
        self._set_native(_native.Fei_2016.Fe_hcp())
