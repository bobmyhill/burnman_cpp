# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Zha_2008."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Pt(Calibrant):
    "The Pt pressure standard reported by\nZha (2008; https://doi.org/10.1063/1.2844358)."

    def __init__(self):
        self._set_native(_native.Zha_2008.Pt())
