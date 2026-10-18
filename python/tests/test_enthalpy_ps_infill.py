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
"""Synthetic values surfaces for field-local derivative reconstruction."""

import numpy as np
import pytest
import burnman_cpp as bm

pytest.importorskip("scipy")
infill_potential_grid = bm.infill_potential_grid
DERIVATIVES = ("d_dx", "d_dy", "d2_dx2", "d2_dxdy", "d2_dy2")


def quadratic_grids(x_scale=1.0, y_scale=1.0, size=7):
    p = np.linspace(0, 1, size)
    s = np.linspace(0, 1, size)
    P, S = np.meshgrid(p, s)
    return bm.PotentialGrid(
        potential="H",
        x=p * x_scale,
        y=s * y_scale,
        values=10 + 4 * P + 3 * S - 0.3 * P**2 + 0.2 * P * S + 0.5 * S**2,
        d_dx=(4 - 0.6 * P + 0.2 * S) / x_scale,
        d_dy=(3 + 0.2 * P + S) / y_scale,
        d2_dx2=np.full(P.shape, -0.6 / x_scale**2),
        d2_dxdy=np.full(P.shape, 0.2 / (x_scale * y_scale)),
        d2_dy2=np.full(P.shape, 1 / y_scale**2),
        field_numbers=np.ones(P.shape, dtype=int),
        equilibrium_success=np.ones(P.shape, dtype=bool),
    )


@pytest.mark.parametrize("scales", [(1.0, 1.0), (2e9, 10.0)])
def test_quadratic_derivatives_at_edges_corners_and_interior(scales):
    grids = quadratic_grids(*scales)
    truth = {name: getattr(grids, name).copy() for name in DERIVATIVES}
    targets = ((0, 0), (0, 3), (3, 3), (6, 6))
    for name in DERIVATIVES:
        for point in targets:
            getattr(grids, name)[point] = np.nan
    # Also reconstruct a single missing coefficient while fixing the others.
    grids.d2_dxdy[2, 1] = np.nan
    original = {name: getattr(grids, name).copy() for name in DERIVATIVES}
    result, info = infill_potential_grid(grids, fill_derivatives=True, return_info=True)
    assert info["success"] and not info["derivative_fit_failures"]
    assert info["n_missing"] == 0  # Derivative-only gaps still need processing.
    np.testing.assert_array_equal(result.values, grids.values)
    assert result.equilibrium_success is grids.equilibrium_success
    for name in DERIVATIVES:
        np.testing.assert_allclose(
            getattr(result, name), truth[name], rtol=1e-8, atol=0
        )
        np.testing.assert_array_equal(getattr(grids, name), original[name])
        np.testing.assert_array_equal(
            info["estimated_masks"][name], np.isnan(original[name])
        )
        known = np.isfinite(original[name])
        np.testing.assert_array_equal(
            getattr(result, name)[known], original[name][known]
        )


def test_missing_derivative_attributes_use_the_same_quadratic_surface():
    truth = quadratic_grids()
    grids = bm.PotentialGrid(potential="H", x=truth.x, y=truth.y, values=truth.values)
    result, info = infill_potential_grid(grids, fill_derivatives=True, return_info=True)
    assert info["success"]
    for name in DERIVATIVES:
        np.testing.assert_allclose(
            getattr(result, name), getattr(truth, name), rtol=1e-9
        )
        assert info["estimated_masks"][name].all()
        assert getattr(grids, name) is None


def test_values_and_derivative_corner_gap_preserves_original_observations():
    grids = quadratic_grids(size=5)
    grids.values[0, 0] = np.nan
    for name in DERIVATIVES:
        getattr(grids, name)[0, 0] = np.nan
    result, info = infill_potential_grid(
        grids,
        fill_derivatives=True,
        return_info=True,
        tol=1e-10,
    )
    assert info["success"]
    assert np.isfinite(result.values).all()
    assert (np.diff(result.values, axis=1) >= -1e-8).all()
    assert (np.diff(result.values, axis=0) >= -1e-8).all()
    assert (np.diff(result.values, n=2, axis=1) <= 1e-8).all()
    assert (np.diff(result.values, n=2, axis=0) >= -1e-8).all()
    known = np.isfinite(grids.values)
    np.testing.assert_array_equal(result.values[known], grids.values[known])
    assert np.isnan(grids.values[0, 0])
    assert info["estimated_masks"]["values"][0, 0]
    for name in DERIVATIVES:
        assert np.isfinite(getattr(result, name)).all()
        assert np.isnan(getattr(grids, name)[0, 0])
    assert (result.d_dx >= 0).all() and (result.d_dy >= 0).all()
    assert (result.d2_dx2 <= 0).all() and (result.d2_dy2 >= 0).all()


def test_known_phase_boundary_keeps_each_curvature_branch():
    grids = quadratic_grids()
    P, S = np.meshgrid(grids.x, grids.y)
    hinge = np.maximum(S - 0.5, 0)
    grids.values += 0.4 * hinge**2 - 0.5 * S**2
    grids.d_dy += 0.8 * hinge - S
    grids.d2_dy2[:] = 0.8 * (S > 0.5)
    grids.field_numbers[:] = np.where(S > 0.5, 2, 1)
    grids.d2_dy2[2, 3] = grids.d2_dy2[4, 3] = np.nan
    result, info = infill_potential_grid(grids, fill_derivatives=True, return_info=True)
    assert info["success"]
    assert result.d2_dy2[2, 3] == pytest.approx(0, abs=1e-8)
    assert result.d2_dy2[4, 3] == pytest.approx(0.8, abs=1e-8)


def test_bounds_constrain_estimates_without_adjusting_observations():
    grids = quadratic_grids()
    P, S = np.meshgrid(grids.x, grids.y)
    grids.values += 0.6 * P**2
    grids.d_dx += 1.2 * P
    grids.d2_dx2[:] = 0.6  # Deliberately inconsistent observed curvature.
    grids.d2_dx2[0, 0] = np.nan
    result, info = infill_potential_grid(grids, fill_derivatives=True, return_info=True)
    assert info["success"]
    assert result.d2_dx2[0, 0] <= 0
    np.testing.assert_array_equal(result.d2_dx2[1:], grids.d2_dx2[1:])


def test_underdetermined_fit_remains_missing_and_is_reported():
    truth = quadratic_grids(size=2)
    grids = bm.PotentialGrid(potential="H", x=truth.x, y=truth.y, values=truth.values)
    result, info = infill_potential_grid(grids, fill_derivatives=True, return_info=True)
    assert not info["success"]
    assert len(info["derivative_fit_failures"]) == 4
    assert np.isnan(result.d2_dy2).all()
    assert not info["estimated_masks"]["d2_dy2"].any()


def test_default_return_type_and_known_values_are_unchanged():
    grids = quadratic_grids()
    result, info = infill_potential_grid(grids, return_info=True)
    assert isinstance(result, bm.PotentialGrid)
    assert info["success"]
    assert "estimated_masks" not in info
    np.testing.assert_array_equal(result.values, grids.values)
    assert result.values is not grids.values


def test_public_processing_preserves_equilibrium_masks():
    grids = quadratic_grids()
    for name in DERIVATIVES:
        getattr(grids, name)[0, 0] = np.nan
    masks = grids.equilibrium_success.copy()
    result, info = infill_potential_grid(grids, fill_derivatives=True, return_info=True)
    assert info["success"]
    np.testing.assert_array_equal(result.equilibrium_success, masks)
    for name in DERIVATIVES:
        assert np.isnan(getattr(grids, name)[0, 0])
        assert np.isfinite(getattr(result, name)[0, 0])
