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

import numpy as np
import pytest
import burnman_cpp as bm

pytest.importorskip("scipy")
smooth = lambda grids, sigma: bm.smooth_potential_grid(grids, sigma).values


def quadratic_grids(n_x=15, n_y=13):
    x = np.linspace(1e9, 4e9, n_x)
    y = np.linspace(100, 120, n_y)
    p, s = np.meshgrid((x - 1e9) / 1e9, (y - 100) / 10)
    return bm.PotentialGrid(
        potential="H",
        x=x,
        y=y,
        values=1000 + 20 * p + 3 * s - 0.7 * p * p + 0.2 * p * s + 0.4 * s * s,
        d_dx=(20 - 1.4 * p + 0.2 * s) / 1e9,
        d_dy=(3 + 0.2 * p + 0.8 * s) / 10,
        d2_dx2=np.full(p.shape, -1.4 / 1e18),
        d2_dxdy=np.full(p.shape, 0.2 / 1e10),
        d2_dy2=np.full(p.shape, 0.8 / 100),
    )


@pytest.mark.parametrize("sigma", [2.0, (0.9, 2.1), (12.0, 15.0)])
def test_physical_quadratic_reproduction_at_every_point(sigma):
    grids = quadratic_grids()
    before = {
        name: value.copy()
        for name, value in vars(grids).items()
        if isinstance(value, np.ndarray)
    }
    widths = np.broadcast_to(np.asarray(sigma, dtype=float), (2,))
    variances = []
    for width, spacing in zip(widths, (np.diff(grids.y)[0], np.diff(grids.x)[0])):
        radius = int(4 * width + 0.5)
        offsets = np.arange(-radius, radius + 1)
        weights = np.exp(-0.5 * (offsets / width) ** 2)
        variances.append(np.average((offsets * spacing) ** 2, weights=weights))
    expected = grids.values + 0.5 * (
        variances[0] * grids.d2_dy2 + variances[1] * grids.d2_dx2
    )
    result = smooth(grids, sigma)
    np.testing.assert_allclose(result, expected, rtol=0, atol=2e-11)
    for name, original in before.items():
        np.testing.assert_array_equal(getattr(grids, name), original)


@pytest.mark.parametrize("sigma", [0.0, -1.0, np.nan, (1.0, 0.0), (1.0, 2.0, 3.0)])
def test_invalid_width_is_rejected(sigma):
    with pytest.raises(ValueError, match="sigma"):
        smooth(quadratic_grids(), sigma)


def test_unfilled_derivative_is_rejected():
    grids = quadratic_grids()
    grids.d2_dxdy[0, 0] = np.nan
    with pytest.raises(ValueError, match="Infill"):
        smooth(grids, 2.0)


def test_nonuniform_x_axis_is_rejected():
    grids = quadratic_grids()
    grids.x[3] += 1e6
    with pytest.raises(ValueError, match="P.*uniformly spaced"):
        smooth(grids, 2.0)


@pytest.mark.parametrize("shape", [(3, 3), (13, 15)])
def test_smoothed_values_derivatives_match_quadratic_including_edges(shape):
    process = bm.smooth_potential_grid
    grids = quadratic_grids(shape[1], shape[0])
    grids.equilibrium_success = np.ones(shape, dtype=bool)
    original = {
        name: value.copy()
        for name, value in vars(grids).items()
        if isinstance(value, np.ndarray)
    }
    result = process(grids)  # The requested default is sigma=2.
    np.testing.assert_allclose(result.values, smooth(grids, 2.0), rtol=0, atol=0)
    for name in ("d_dy", "d_dx", "d2_dy2", "d2_dxdy", "d2_dx2"):
        np.testing.assert_allclose(getattr(result, name), original[name], rtol=1e-8)
        assert getattr(result, name) is not getattr(grids, name)
    assert result.equilibrium_success is grids.equilibrium_success
    for name, value in original.items():
        np.testing.assert_array_equal(getattr(grids, name), value)


def test_smoothed_curvature_is_recomputed_from_values():
    process = bm.smooth_potential_grid
    x = np.linspace(2.4e9, 5.6e9, 33)
    y = np.arange(65.0) + 100.0
    p, s = np.meshgrid((x - 4e9) / 1e9, y - 132.0)
    bump = 10.0 * np.exp(-0.5 * (s / 3.0) ** 2)
    grids = bm.PotentialGrid(
        potential="H",
        x=x,
        y=y,
        values=1000.0 + 3.0 * p - 0.1 * p * p + 1000.0 * s + s * s + bump,
        d_dx=(3.0 - 0.2 * p) / 1e9,
        d_dy=1000.0 + 2.0 * s - s * bump / 9.0,
        d2_dx2=np.full(p.shape, -0.2 / 1e18),
        d2_dxdy=np.zeros(p.shape),
        d2_dy2=2.0 + (s * s / 81.0 - 1.0 / 9.0) * bump,
    )
    result = process(grids)
    # Broadening the Gaussian peak reduces its negative curvature at the
    # centre. Copying the unsmoothed Hessian would fail this physical check.
    assert grids.d2_dy2[32, 16] < result.d2_dy2[32, 16] < 2.0
