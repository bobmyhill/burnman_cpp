# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Dewaele_2020."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class CsCl(Calibrant):
    "The CsCl pressure standard reported by\nDewaele (2020; https://doi.org/10.1080/08957959.2020.1774754).\nThe thermal model is from Decker (1971)."

    def __init__(self):
        self._set_native(_native.Dewaele_2020.CsCl())
