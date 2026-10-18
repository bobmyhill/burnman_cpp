# ------------------------------------------------------
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#
# This file is part of BurnMan.
#
# Detailed license information governing the source code
# and contributions can be found in the LICENSE document.
#
# ------------------------------------------------------
"""Quadratic reproduction on physical P-S axes, including edges and corners."""

from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")
smooth = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "examples/gaussian_moment_corrected.py")
)["gaussian_filter_moment_corrected"]


def quadratic_grids():
    pressure = np.linspace(1e9, 4e9, 15)
    entropy = np.linspace(100, 120, 13)
    p, s = np.meshgrid((pressure - 1e9) / 1e9, (entropy - 100) / 10)
    return SimpleNamespace(
        pressure=pressure,
        entropy=entropy,
        enthalpy=1000 + 20*p + 3*s - .7*p*p + .2*p*s + .4*s*s,
        dH_dP=(20 - 1.4*p + .2*s) / 1e9,
        dH_dS=(3 + .2*p + .8*s) / 10,
        d2H_dP2=np.full(p.shape, -1.4 / 1e18),
        d2H_dSdP=np.full(p.shape, .2 / 1e10),
        d2H_dS2=np.full(p.shape, .8 / 100),
    )


@pytest.mark.parametrize("sigma", [2., (.9, 2.1), (12., 15.)])
def test_physical_quadratic_reproduction_at_every_point(sigma):
    grids = quadratic_grids()
    before = {name: value.copy() for name, value in vars(grids).items()}
    widths = np.broadcast_to(np.asarray(sigma, dtype=float), (2,))
    variances = []
    for width, spacing in zip(widths, (np.diff(grids.entropy)[0], np.diff(grids.pressure)[0])):
        radius = int(4 * width + .5)
        offsets = np.arange(-radius, radius + 1)
        weights = np.exp(-.5 * (offsets / width)**2)
        variances.append(np.average((offsets * spacing)**2, weights=weights))
    expected = grids.enthalpy + .5 * (
        variances[0] * grids.d2H_dS2 + variances[1] * grids.d2H_dP2
    )
    result = smooth(grids, sigma)
    np.testing.assert_allclose(result, expected, rtol=0, atol=2e-11)
    for name, original in before.items():
        np.testing.assert_array_equal(getattr(grids, name), original)


@pytest.mark.parametrize("sigma", [0., -1., np.nan, (1., 0.), (1., 2., 3.)])
def test_invalid_width_is_rejected(sigma):
    with pytest.raises(ValueError, match="sigma"):
        smooth(quadratic_grids(), sigma)


def test_unfilled_derivative_is_rejected():
    grids = quadratic_grids()
    grids.d2H_dSdP[0, 0] = np.nan
    with pytest.raises(ValueError, match="Infill"):
        smooth(grids, 2.)


def test_nonuniform_pressure_axis_is_rejected():
    grids = quadratic_grids()
    grids.pressure[3] += 1e6
    with pytest.raises(ValueError, match="pressure.*uniformly spaced"):
        smooth(grids, 2.)
