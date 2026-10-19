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
"""Natural-potential signs, edge infilling and full two-dimensional smoothing."""

import numpy as np
import pytest

import burnman_cpp as bm
from burnman_cpp.tools.potential_grid import DERIVATIVES, POTENTIALS

# Independent thermodynamic expectations, rather than values read back from
# the implementation being tested: axes, then the two stability curvatures.
NATURAL_POTENTIALS = {
    "E": (("V", "S"), (1, 1)),
    "F": (("V", "T"), (1, -1)),
    "G": (("P", "T"), (-1, -1)),
    "H": (("P", "S"), (-1, 1)),
}


def test_natural_variables_and_curvatures_match_thermodynamics():
    assert {
        name: values[:2] for name, values in POTENTIALS.items()
    } == NATURAL_POTENTIALS


def quadratic(symbol):
    (vx, vy), (cx, cy) = NATURAL_POTENTIALS[symbol]
    sx, sy = {"V": 1e-5, "P": 1e9}[vx], {"S": 10.0, "T": 1000.0}[vy]
    x, y = np.linspace(1, 2, 7) * sx, np.linspace(1, 2, 9) * sy
    xx, yy = np.meshgrid(x / sx - 1, y / sy - 1)
    slope = -4.0 if vx == "V" else 4.0
    return bm.PotentialGrid(
        symbol,
        x,
        y,
        20 + slope * xx + 3 * yy + 0.3 * cx * xx**2 + 0.2 * xx * yy + 0.5 * cy * yy**2,
        (slope + 0.6 * cx * xx + 0.2 * yy) / sx,
        (3 + 0.2 * xx + cy * yy) / sy,
        np.full(xx.shape, 0.6 * cx / sx**2),
        np.full(xx.shape, 0.2 / (sx * sy)),
        np.full(xx.shape, cy / sy**2),
        field_numbers=np.ones(xx.shape, int),
    )


@pytest.mark.parametrize("symbol", list(NATURAL_POTENTIALS))
def test_reconstruct_missing_derivatives_at_edges_on_all_natural_potentials(symbol):
    grid = quadratic(symbol)
    truth = {name: getattr(grid, name).copy() for name in DERIVATIVES}
    for name in DERIVATIVES:
        getattr(grid, name)[0, 0] = getattr(grid, name)[4, 3] = np.nan
    result, info = bm.infill_potential_grid(
        grid, fill_derivatives=True, return_info=True
    )
    assert info["success"]
    for name in DERIVATIVES:
        np.testing.assert_allclose(getattr(result, name), truth[name], rtol=2e-9)
    np.testing.assert_array_equal(result.values, grid.values)


@pytest.mark.parametrize("symbol", list(NATURAL_POTENTIALS))
def test_infill_corner_value_and_derivatives_with_correct_thermodynamic_signs(symbol):
    grid = quadratic(symbol)
    grid.values[0, 0] = np.nan
    for name in DERIVATIVES:
        getattr(grid, name)[0, 0] = np.nan
    result, info = bm.infill_potential_grid(
        grid, fill_derivatives=True, return_info=True
    )
    assert info["success"] and info["estimated_masks"]["values"][0, 0]
    for name in ("values", *DERIVATIVES):
        assert np.isfinite(getattr(result, name)).all()
        known = np.isfinite(getattr(grid, name))
        np.testing.assert_array_equal(
            getattr(result, name)[known], getattr(grid, name)[known]
        )
    for axis, sign in ((1, grid.curvature[0]), (0, grid.curvature[1])):
        assert (sign * np.diff(result.values, n=2, axis=axis) >= -1e-7).all()
    assert np.isnan(grid.values[0, 0])


@pytest.mark.parametrize("symbol", list(NATURAL_POTENTIALS))
def test_smoothing_derivatives_reproduce_physical_quadratics_for_all_potentials(symbol):
    grid = quadratic(symbol)
    result = bm.smooth_potential_grid(grid, sigma=(1.5, 2.0))
    # Full Gaussian convolution adds a constant to a quadratic. Its derivatives
    # must therefore agree, including at asymmetric edges and corners.
    assert np.ptp(result.values - grid.values) < 1e-12
    for name in DERIVATIVES:
        np.testing.assert_allclose(
            getattr(result, name), getattr(grid, name), rtol=2e-9
        )
    assert result.field_numbers is grid.field_numbers
    assert result.values is not grid.values


def test_grid_rejects_invalid_axes_and_shapes():
    with pytest.raises(ValueError, match="axes"):
        bm.PotentialGrid("G", [1, 0], [1, 2], np.zeros((2, 2)))
    with pytest.raises(ValueError, match="axes"):
        bm.PotentialGrid("G", 1.0, [1, 2], np.zeros((2, 2)))
    with pytest.raises(ValueError, match="shape"):
        bm.PotentialGrid("G", [1, 2], [1, 2], np.zeros((3, 2)))


@pytest.mark.parametrize("symbol", list(NATURAL_POTENTIALS))
def test_potential_arrays_round_trip_without_models_or_pickle(tmp_path, symbol):
    grids = quadratic(symbol)
    grids.values[0, 0] = np.nan
    grids.d2_dy2[-1, -1] = np.inf
    grids.diagnostics = ["Saved diagnostic"]
    path = bm.save_potential_grid(grids, tmp_path / "raw.npz")
    saved = bm.load_potential_grid(path)
    assert saved.potential == symbol and saved.diagnostics == grids.diagnostics
    for name in ("x", "y", "values", *DERIVATIVES, "field_numbers"):
        np.testing.assert_array_equal(getattr(saved, name), getattr(grids, name))
    with np.load(path, allow_pickle=False) as arrays:
        assert all(not arrays[name].dtype.hasobject for name in arrays.files)
    with pytest.raises(ValueError, match="disagrees"):
        bm.load_potential_grid(path, potential="G" if symbol != "G" else "H")
