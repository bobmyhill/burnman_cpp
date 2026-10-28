# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Dorogokupets_2017."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Fe_bcc(Calibrant):
    "The BCC Fe pressure standard reported by\nDorogokupets (2017; https://doi.org/10.1038/srep41863)."

    def __init__(self):
        self._set_native(_native.Dorogokupets_2017.Fe_bcc())


class Fe_fcc(Calibrant):
    "The FCC Fe pressure standard reported by\nDorogokupets (2017; https://doi.org/10.1038/srep41863)."

    def __init__(self):
        self._set_native(_native.Dorogokupets_2017.Fe_fcc())


class Fe_hcp(Calibrant):
    "The HCP Fe pressure standard reported by\nDorogokupets (2017; https://doi.org/10.1038/srep41863)."

    def __init__(self):
        self._set_native(_native.Dorogokupets_2017.Fe_hcp())
