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

"""Weighted linear fits, constraints and covariance estimates."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import burnman_cpp as bm
from burnman_cpp.optimize.linear_fitting import (
    weighted_constrained_least_squares as fit,
)


def test_imports_and_closed_form_fit():
    assert bm.optimize.linear_fitting.weighted_constrained_least_squares is fit
    A = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0, 2.0])
    originals = A.copy(), b.copy()
    parameters, covariance, residual = fit(A, b)
    assert_allclose(parameters, [2.0 / 3.0, 0.5], rtol=1e-13)
    assert_allclose(covariance, [[7.0 / 3.0, -1.0], [-1.0, 0.5]], rtol=1e-13)
    assert_allclose(residual, 1.0 / 6.0, rtol=1e-13)
    assert_array_equal(A, originals[0])
    assert_array_equal(b, originals[1])
    parameters[:] = 0
    covariance[:] = 0
    assert_allclose(fit(A, b)[0], [2.0 / 3.0, 0.5], rtol=1e-13)


def test_correlated_observations_match_closed_form_weighted_fit():
    A = np.array([[1, 0], [1, 1], [1, 2], [1, 3]], dtype=float)
    b = np.array([1, 2, 2, 4], dtype=float)
    covariance = np.array(
        [[2, 0.3, 0, 0], [0.3, 1, 0.2, 0], [0, 0.2, 3, 0.1], [0, 0, 0.1, 2]]
    )
    information = A.T @ np.linalg.solve(covariance, A)
    expected_covariance = np.linalg.inv(information)
    expected = np.linalg.solve(information, A.T @ np.linalg.solve(covariance, b))
    parameters, fitted_covariance, residual = fit(A, b, covariance)
    assert_allclose(parameters, expected, rtol=1e-13)
    assert_allclose(fitted_covariance, expected_covariance, rtol=1e-13)
    error = A @ expected - b
    assert_allclose(residual, error @ np.linalg.solve(covariance, error), rtol=1e-13)


@pytest.mark.parametrize("redundant", [False, True])
def test_equality_constraints_and_unconditional_covariance(redundant):
    A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    b = np.array([7, 8, 9], dtype=float)
    C, c = [[1, 1]], [0.5]
    if redundant:
        C, c = [[1, 1], [2, 2], [0, 0]], [0.5, 1, 0]
    parameters, covariance, residual = fit(A, b, equality_constraints=[C, c])
    assert_allclose(parameters, [-6.0, 6.5], rtol=1e-13)
    assert_allclose(covariance, np.linalg.inv(A.T @ A), rtol=1e-13)
    assert residual < 1e-24


def test_inequality_constraints_allow_signed_parameters():
    A = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
    b = np.array([1, 2, 2], dtype=float)
    parameters, covariance, residual = fit(
        A, b, inequality_constraints=[-np.eye(2), [-2, 0]]
    )
    assert_allclose(parameters, [2.0, 0.0], atol=1e-10)
    assert_allclose(covariance, np.linalg.inv(A.T @ A), rtol=1e-13)
    assert_allclose(residual, 1.0, rtol=1e-10)
    assert_allclose(fit(np.eye(2), [-2, 3])[0], [-2, 3], atol=1e-13)


@pytest.mark.parametrize(
    "D,d,expected",
    [
        ([[-1, 0], [0, -1], [-1, -1]], [0, 0, 0], [0, 0]),
        ([[1, 0], [-1, 0]], [0, 0], [0, -1]),
        ([[-1, 0], [-2, 0], [0, -1]], [0, 0, 0], [0, 0]),
    ],
)
def test_dependent_and_opposing_active_inequalities(D, d, expected):
    parameters, _, _ = fit(np.eye(2), [-1, -1], inequality_constraints=[D, d])
    assert_allclose(parameters, expected, atol=1e-9)


def test_mixed_equalities_and_inequalities():
    parameters, covariance, residual = fit(
        np.eye(2),
        [-1, 2],
        equality_constraints=[[[1, 1]], [1]],
        inequality_constraints=[-np.eye(2), [0, 0]],
    )
    assert_allclose(parameters, [0, 1], atol=1e-10)
    assert_allclose(covariance, np.eye(2), atol=1e-14)
    assert_allclose(residual, 2.0, rtol=1e-10)


def test_fully_fixed_and_empty_constraints():
    A, b = np.eye(2), np.array([1.0, 2.0])
    assert_allclose(fit(A, b, equality_constraints=[A, [0, 1]])[0], [0, 1], atol=1e-14)
    for empty in [[np.empty((0, 2)), np.empty(0)], [[], []]]:
        assert_allclose(fit(A, b, equality_constraints=empty)[0], b, atol=1e-14)
        assert_allclose(fit(A, b, inequality_constraints=empty)[0], b, atol=1e-14)
    with pytest.raises(ValueError, match="infeasible"):
        fit(
            A,
            b,
            equality_constraints=[A, b],
            inequality_constraints=[A, [0, 0]],
        )


def test_rank_deficiency_and_minimum_norm_unconstrained_fit():
    A = np.array([[1, 2], [2, 4], [3, 6]], dtype=float)
    b = np.array([1, 2, 3], dtype=float)
    with pytest.raises(ValueError, match="rank-deficient"):
        fit(A, b)
    parameters, covariance, residual = fit(A, b, allow_rank_deficient=True)
    assert_allclose(A @ parameters, b, rtol=1e-13)
    assert_allclose(parameters, np.linalg.pinv(A) @ b, rtol=1e-13)
    assert_allclose(covariance, np.linalg.pinv(A.T @ A), rtol=1e-13)
    assert residual < 1e-25
    # Rank checking precedes constraints, even if these specify every parameter.
    with pytest.raises(ValueError, match="rank-deficient"):
        fit(A, b, equality_constraints=[np.eye(2), [0, 0.5]])
    assert_allclose(
        fit([[1, 2, 3]], [2], allow_rank_deficient=True)[0],
        np.array([1, 2, 3]) / 7.0,
        rtol=1e-13,
    )


def test_singular_covariance_zero_weight_and_constrained_unobserved_parameter():
    covariance = np.diag([1, 0, 4])
    with pytest.raises(ValueError, match="rank-deficient"):
        fit(np.eye(3), [2, 100, 3], covariance)
    parameters, fitted_covariance, residual = fit(
        np.eye(3), [2, 100, 3], covariance, allow_rank_deficient=True
    )
    assert_allclose(parameters, [2, 0, 3], atol=1e-13)
    assert_allclose(fitted_covariance, covariance, atol=1e-13)
    assert residual < 1e-25
    parameters, _, _ = fit(
        np.eye(3),
        [2, 100, 3],
        covariance,
        equality_constraints=[[[0, 1, 0]], [2]],
        allow_rank_deficient=True,
    )
    assert_allclose(parameters, [2, 2, 3], atol=1e-13)
    parameters, fitted_covariance, residual = fit(
        np.eye(2),
        [1, 2],
        np.zeros((2, 2)),
        inequality_constraints=[-np.eye(2), [-1, -2]],
        allow_rank_deficient=True,
    )
    assert (parameters >= np.array([1, 2]) - 1e-10).all()
    assert_array_equal(fitted_covariance, np.zeros((2, 2)))
    assert residual == 0


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_weight_and_constraint_scaling(scale):
    A = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
    b = np.array([1, 2, 2], dtype=float)
    parameters, covariance, residual = fit(
        A,
        b,
        np.eye(3) * scale,
        inequality_constraints=[-np.diag([scale, 1 / scale]), [-2 * scale, 0]],
    )
    assert_allclose(parameters, [2, 0], atol=1e-9)
    assert_allclose(covariance / scale, np.linalg.inv(A.T @ A), rtol=1e-13)
    assert_allclose(residual * scale, 1.0, rtol=1e-9)


def test_constraint_tolerances_ignore_unrelated_parameter_scales():
    parameters, _, residual = fit(
        np.eye(2), [1e12, 2], inequality_constraints=[[[0, 1]], [0]]
    )
    assert_allclose(parameters, [1e12, 0], atol=1e-10)
    assert_allclose(residual, 4.0, rtol=1e-12)
    with pytest.raises(ValueError, match="infeasible"):
        fit(
            np.eye(2),
            [0, 0],
            equality_constraints=[[[1, 0], [0, 0]], [1e12, 1]],
        )


@pytest.mark.parametrize("direction", [[1.0, 1.0], [1.0, 2.0, 3.0]])
def test_correlated_singular_covariance_ignores_roundoff_directions(direction):
    direction = np.array(direction)
    n = len(direction)
    covariance = np.outer(direction, direction)
    observations = np.arange(1.0, n + 1)[::-1]
    with pytest.raises(ValueError, match="rank-deficient"):
        fit(np.eye(n), observations, covariance)
    parameters, fitted_covariance, residual = fit(
        np.eye(n), observations, covariance, allow_rank_deficient=True
    )
    expected = direction * (direction @ observations) / (direction @ direction)
    assert_allclose(parameters, expected, rtol=1e-13)
    assert_allclose(fitted_covariance, covariance, rtol=1e-13)
    assert residual < 1e-25


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_covariance_validation_remains_relative_at_extreme_scales(scale):
    with pytest.raises(ValueError, match="symmetric"):
        fit(np.eye(2), [1, 2], np.array([[2, 1], [0, 2]]) * scale)
    parameters, covariance, residual = fit(np.eye(2), [1, 2], np.eye(2) * scale)
    assert_allclose(parameters, [1, 2], rtol=1e-13)
    assert_allclose(covariance / scale, np.eye(2), atol=1e-13)
    assert residual == 0


@pytest.mark.parametrize("seed", range(10))
def test_constructed_convex_optima_satisfy_constraints_and_stationarity(seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(12, 5))
    L = rng.normal(size=(12, 12))
    covariance = L @ L.T + np.eye(12)
    expected = rng.normal(size=5)
    D = rng.normal(size=(20, 5))
    d = D @ expected + np.r_[np.zeros(3), rng.uniform(0.5, 2.0, 17)]
    # Choose observations whose objective gradient is balanced by positive
    # multipliers on the first three constraints. The optimum is unique.
    force = D[:3].T @ np.array([1.0, 2.0, 3.0])
    b = A @ expected + covariance @ A @ np.linalg.solve(A.T @ A, force)
    parameters, _, residual = fit(A, b, covariance, inequality_constraints=[D, d])
    assert_allclose(parameters, expected, atol=2e-7, rtol=2e-7)
    assert (D @ parameters - d < 1e-8).all()
    error = A @ expected - b
    assert_allclose(residual, error @ np.linalg.solve(covariance, error), rtol=1e-10)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"equality_constraints": [[[1, 1], [2, 2]], [1, 3]]},
        {"equality_constraints": [[[0, 0]], [1]]},
        {"inequality_constraints": [[[0, 0]], [-1]]},
        {"inequality_constraints": [[[1, 0], [-1, 0]], [0, -1]]},
        {
            "equality_constraints": [[[1, 0]], [1]],
            "inequality_constraints": [[[1, 0]], [0]],
        },
    ],
)
def test_infeasible_constraints_raise(kwargs):
    with pytest.raises(ValueError, match="infeasible"):
        fit(np.eye(2), [0, 0], **kwargs)


@pytest.mark.parametrize(
    "A,b,kwargs",
    [
        (np.empty((0, 2)), [], {}),
        (np.eye(2), [1], {}),
        (np.eye(2), [1, np.nan], {}),
        ([[1, np.inf], [0, 1]], [1, 2], {}),
        (np.eye(2), [1, 2], {"Cov_b": np.eye(3)}),
        (np.eye(2), [1, 2], {"Cov_b": [[1, 0.5], [0, 1]]}),
        (np.eye(2), [1, 2], {"Cov_b": [[1, 2], [2, 1]]}),
        (np.eye(2), [1, 2], {"Cov_b": [[1, 0], [0, np.nan]]}),
        (np.eye(2), [1, 2], {"equality_constraints": [[[1, 1]], [1, 2]]}),
        (np.eye(2), [1, 2], {"equality_constraints": [np.empty((1, 0)), []]}),
        (np.eye(2), [1, 2], {"inequality_constraints": [[[1, 1, 1]], [1]]}),
        (np.eye(2), [1, 2], {"inequality_constraints": [[[np.inf, 0]], [1]]}),
    ],
)
def test_invalid_inputs_raise(A, b, kwargs):
    with pytest.raises(ValueError):
        fit(A, b, **kwargs)


@pytest.mark.parametrize(
    "case", ["ordinary", "correlated", "equality", "inequality", "mixed", "singular"]
)
def test_optional_python_burnman_parity(case):
    pytest.importorskip("burnman")
    pytest.importorskip("cvxpy")
    from burnman.optimize.linear_fitting import weighted_constrained_least_squares

    A = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
    b = np.array([1, 2, 2], dtype=float)
    kwargs = {}
    if case == "correlated":
        kwargs["Cov_b"] = [[2, 0.3, 0], [0.3, 1, 0.2], [0, 0.2, 3]]
    if case in ["equality", "mixed"]:
        kwargs["equality_constraints"] = [[[1, 1]], [1]]
    if case in ["inequality", "mixed"]:
        kwargs["inequality_constraints"] = [-np.eye(2), [0, 0]]
        b = np.array([-1, -2, -2], dtype=float)
    if case == "singular":
        A = np.eye(2)
        b = np.array([1, 2], dtype=float)
        kwargs.update(Cov_b=np.diag([1, 0]), allow_rank_deficient=True)
    actual = fit(A, b, **kwargs)
    expected = weighted_constrained_least_squares(A, b, **kwargs)
    assert_allclose(A @ actual[0], A @ expected[0], rtol=1e-6, atol=1e-8)
    assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-12)
    assert_allclose(actual[2], expected[2], rtol=1e-7, atol=1e-12)
