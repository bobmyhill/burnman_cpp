# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Ono_2022."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Re(Calibrant):
    "The Re pressure standard reported by\nOno (2022; https://doi.org/10.1155/2022/7545777)."

    def __init__(self):
        self._set_native(_native.Ono_2022.Re())
