# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Tateno_2019."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class KCl_B2(Calibrant):
    "The B2 KCl pressure standard reported by\nTateno (2019; https://doi.org/10.2138/am-2019-6779)."

    def __init__(self):
        self._set_native(_native.Tateno_2019.KCl_B2())
