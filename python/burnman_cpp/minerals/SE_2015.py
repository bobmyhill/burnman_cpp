# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Saxena–Eriksson iron phases using the full Brosh CALPHAD cold curve.

HCP b_1=2.49614 comes from Perple_X SE15ver.dat (FeHCP, b6); the paper's
Table 1 gives 5.5. FCC uses the converted AFM parameters 67 K and 0.7 mu_B.
"""

from .. import _core


def bcc_iron():
    return _core._se2015_bcc_iron()


def fcc_iron():
    return _core._se2015_fcc_iron()


def hcp_iron():
    return _core._se2015_hcp_iron()


def liquid_iron():
    return _core._se2015_liquid_iron()
