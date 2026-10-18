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
"""Synthetic enthalpy surfaces for field-local derivative reconstruction."""

from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")
infill_enthalpy_ps = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "examples/enthalpy_ps_infill.py")
)["infill_enthalpy_ps"]
DERIVATIVES = ("dH_dP", "dH_dS", "d2H_dP2", "d2H_dSdP", "d2H_dS2")


def quadratic_grids(pressure_scale=1.0, entropy_scale=1.0, size=7):
    p = np.linspace(0, 1, size)
    s = np.linspace(0, 1, size)
    P, S = np.meshgrid(p, s)
    return SimpleNamespace(
        pressure=p * pressure_scale,
        entropy=s * entropy_scale,
        enthalpy=10 + 4 * P + 3 * S - .3 * P**2 + .2 * P * S + .5 * S**2,
        dH_dP=(4 - .6 * P + .2 * S) / pressure_scale,
        dH_dS=(3 + .2 * P + S) / entropy_scale,
        d2H_dP2=np.full(P.shape, -.6 / pressure_scale**2),
        d2H_dSdP=np.full(P.shape, .2 / (pressure_scale * entropy_scale)),
        d2H_dS2=np.full(P.shape, 1 / entropy_scale**2),
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
    grids.d2H_dSdP[2, 1] = np.nan
    original = {name: getattr(grids, name).copy() for name in DERIVATIVES}
    result, info = infill_enthalpy_ps(grids, fill_derivatives=True, return_info=True)
    assert info["success"] and not info["derivative_fit_failures"]
    assert info["n_missing"] == 0  # Derivative-only gaps still need processing.
    np.testing.assert_array_equal(result.enthalpy, grids.enthalpy)
    assert result.equilibrium_success is grids.equilibrium_success
    for name in DERIVATIVES:
        np.testing.assert_allclose(getattr(result, name), truth[name], rtol=1e-8, atol=0)
        np.testing.assert_array_equal(getattr(grids, name), original[name])
        np.testing.assert_array_equal(info["estimated_masks"][name], np.isnan(original[name]))
        known = np.isfinite(original[name])
        np.testing.assert_array_equal(getattr(result, name)[known], original[name][known])


def test_missing_derivative_attributes_use_the_same_quadratic_surface():
    truth = quadratic_grids()
    grids = SimpleNamespace(
        pressure=truth.pressure, entropy=truth.entropy, enthalpy=truth.enthalpy
    )
    result, info = infill_enthalpy_ps(grids, fill_derivatives=True, return_info=True)
    assert info["success"]
    for name in DERIVATIVES:
        np.testing.assert_allclose(getattr(result, name), getattr(truth, name), rtol=1e-9)
        assert info["estimated_masks"][name].all()
        assert not hasattr(grids, name)


def test_enthalpy_and_derivative_corner_gap_preserves_original_observations():
    grids = quadratic_grids(size=5)
    grids.enthalpy[0, 0] = np.nan
    for name in DERIVATIVES:
        getattr(grids, name)[0, 0] = np.nan
    result, info = infill_enthalpy_ps(
        grids, fill_derivatives=True, return_info=True, tol=1e-10,
    )
    assert info["success"]
    assert np.isfinite(result.enthalpy).all()
    assert (np.diff(result.enthalpy, axis=1) >= -1e-8).all()
    assert (np.diff(result.enthalpy, axis=0) >= -1e-8).all()
    assert (np.diff(result.enthalpy, n=2, axis=1) <= 1e-8).all()
    assert (np.diff(result.enthalpy, n=2, axis=0) >= -1e-8).all()
    known = np.isfinite(grids.enthalpy)
    np.testing.assert_array_equal(result.enthalpy[known], grids.enthalpy[known])
    assert np.isnan(grids.enthalpy[0, 0])
    assert info["estimated_masks"]["enthalpy"][0, 0]
    for name in DERIVATIVES:
        assert np.isfinite(getattr(result, name)).all()
        assert np.isnan(getattr(grids, name)[0, 0])
    assert (result.dH_dP >= 0).all() and (result.dH_dS >= 0).all()
    assert (result.d2H_dP2 <= 0).all() and (result.d2H_dS2 >= 0).all()


def test_known_phase_boundary_keeps_each_curvature_branch():
    grids = quadratic_grids()
    P, S = np.meshgrid(grids.pressure, grids.entropy)
    hinge = np.maximum(S - .5, 0)
    grids.enthalpy += .4 * hinge**2 - .5 * S**2
    grids.dH_dS += .8 * hinge - S
    grids.d2H_dS2[:] = .8 * (S > .5)
    grids.field_numbers[:] = np.where(S > .5, 2, 1)
    grids.d2H_dS2[2, 3] = grids.d2H_dS2[4, 3] = np.nan
    result, info = infill_enthalpy_ps(grids, fill_derivatives=True, return_info=True)
    assert info["success"]
    assert result.d2H_dS2[2, 3] == pytest.approx(0, abs=1e-8)
    assert result.d2H_dS2[4, 3] == pytest.approx(.8, abs=1e-8)


def test_bounds_constrain_estimates_without_adjusting_observations():
    grids = quadratic_grids()
    P, S = np.meshgrid(grids.pressure, grids.entropy)
    grids.enthalpy += .6 * P**2
    grids.dH_dP += 1.2 * P
    grids.d2H_dP2[:] = .6  # Deliberately inconsistent observed curvature.
    grids.d2H_dP2[0, 0] = np.nan
    result, info = infill_enthalpy_ps(grids, fill_derivatives=True, return_info=True)
    assert info["success"]
    assert result.d2H_dP2[0, 0] <= 0
    np.testing.assert_array_equal(result.d2H_dP2[1:], grids.d2H_dP2[1:])


def test_underdetermined_fit_remains_missing_and_is_reported():
    truth = quadratic_grids(size=2)
    grids = SimpleNamespace(
        pressure=truth.pressure, entropy=truth.entropy, enthalpy=truth.enthalpy
    )
    result, info = infill_enthalpy_ps(grids, fill_derivatives=True, return_info=True)
    assert not info["success"]
    assert len(info["derivative_fit_failures"]) == 4
    assert np.isnan(result.d2H_dS2).all()
    assert not info["estimated_masks"]["d2H_dS2"].any()


def test_default_return_type_and_known_values_are_unchanged():
    grids = quadratic_grids()
    result, info = infill_enthalpy_ps(grids, return_info=True)
    assert isinstance(result, np.ndarray)
    assert info["success"]
    assert "estimated_masks" not in info
    np.testing.assert_array_equal(result, grids.enthalpy)
    assert result is not grids.enthalpy


def test_pyrolite_processing_wrapper_changes_only_missing_thermodynamic_entries():
    fill = runpy.run_path(str(
        Path(__file__).resolve().parents[2]
        / "examples/build_pyrolite_sp_enthalpy_grid.py"
    ))["fill_enthalpy_gaps"]
    grids = quadratic_grids()
    snapshots = {name: getattr(grids, name).copy() for name in DERIVATIVES}
    for name in DERIVATIVES:
        getattr(grids, name)[0, 0] = np.nan
    masks = grids.equilibrium_success.copy()
    info = fill(grids, fill_derivatives=True)
    assert info["success"]
    for name in DERIVATIVES:
        np.testing.assert_allclose(getattr(grids, name), snapshots[name], rtol=1e-9)
    np.testing.assert_array_equal(grids.equilibrium_success, masks)
