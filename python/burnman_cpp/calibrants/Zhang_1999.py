# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Zhang_1999."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Fe_bcc(Calibrant):
    "The BCC Fe pressure standard reported by\nZhang et al. (1999; https://dx.doi.org/10.1007/s002690050178)."

    def __init__(self):
        self._set_native(_native.Zhang_1999.Fe_bcc())
