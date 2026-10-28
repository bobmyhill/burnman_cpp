# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Miozzi_2020."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Fe_hcp(Calibrant):
    "The HCP Fe pressure standard reported by\nMiozzi (2020; https://doi.org/10.3390/min10020100)."

    def __init__(self):
        self._set_native(_native.Miozzi_2020.Fe_hcp())
