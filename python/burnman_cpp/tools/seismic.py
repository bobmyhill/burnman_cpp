# ------------------------------------------------------
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.
#
# This file is part of BurnMan.
#
# Detailed license information governing the source code
# and contributions can be found in the LICENSE document.
#
# ------------------------------------------------------

"""Seismic velocity corrections, following BurnMan tools.seismic."""

from .. import _core


def attenuation_correction(v_p, v_s, v_phi, Qs, Qphi):
    """Return attenuation-corrected P, S and bulk velocities (Matas et al. 2007).

    Velocities use m/s and quality factors are dimensionless. Inputs may be
    scalars or broadcastable NumPy arrays; inputs are left unchanged.
    """
    return (
        _core._attenuation_correction(v_p, v_s, v_phi, Qs, Qphi),
        _core._attenuation_correction_s(v_p, v_s, v_phi, Qs, Qphi),
        _core._attenuation_correction_phi(v_p, v_s, v_phi, Qs, Qphi),
    )


__all__ = ["attenuation_correction"]
