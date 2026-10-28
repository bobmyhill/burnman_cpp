# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Matsui_2009."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Pt(Calibrant):
    "The Pt pressure standard reported by\nMatsui (2009; https://doi.org/10.1063/1.3054331)."

    def __init__(self):
        self._set_native(_native.Matsui_2009.Pt())
