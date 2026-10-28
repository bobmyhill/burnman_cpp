# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Sokolova_2013."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Mo(Calibrant):
    "The Mo pressure standard reported by Sokolova et al. (2013),\nas reported by Huang et al. (2016; https://doi.org/10.1038/srep19923)."

    def __init__(self):
        self._set_native(_native.Sokolova_2013.Mo())
