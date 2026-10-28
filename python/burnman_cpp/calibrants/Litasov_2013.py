# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native pressure standards from Litasov_2013."""

from ..classes.calibrant import Calibrant
from .._core import calibrants as _native


class Mo_bcc(Calibrant):
    "The BCC Mo pressure standard reported by\nLitasov et al. (2013; https://doi.org/10.1063/1.4794127)."

    def __init__(self):
        self._set_native(_native.Litasov_2013.Mo_bcc())


class W_bcc(Calibrant):
    "The BCC W pressure standard reported by\nLitasov et al. (2013; https://doi.org/10.1063/1.4799018)."

    def __init__(self):
        self._set_native(_native.Litasov_2013.W_bcc())
