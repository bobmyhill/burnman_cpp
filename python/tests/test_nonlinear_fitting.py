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

"""Core cases adapted from Python BurnMan's tests/test_fitting.py."""

import copy

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import burnman_cpp as bm
from burnman_cpp.optimize import nonlinear_fitting as fit


class PolynomialModel(fit.NonLinearModel):
    def __init__(self, data, cov, params, delta_params=None, tolerance=1.0e-10):
        self.data = np.array(data, dtype=float)
        self.data_covariances = np.array(cov, dtype=float)
        self.set_params(params)
        self.delta_params = (
            np.full(len(params), 1.0e-6)
            if delta_params is None
            else np.array(delta_params)
        )
        self.mle_tolerances = np.full(len(data), tolerance)

    def get_params(self):
        return self.params

    def set_params(self, param_values):
        self.params = np.array(param_values, dtype=float)

    def function(self, x, flag=None):
        return np.array([x[0], np.polyval(self.params, x[0])])

    def normal(self, x, flag=None):
        n = np.array([np.polyval(np.polyder(self.params), x[0]), -1.0])
        return n / np.linalg.norm(n)


def pearson_york(degree=1):
    # Pearson_York.dat: Neri et al., Meas. Sci. Technol. 1 (1990), 1007.
    x, wx, y, wy = np.array(
        [
            [0.0, 1000.0, 5.9, 1.0],
            [0.9, 1000.0, 5.4, 1.8],
            [1.8, 500.0, 4.4, 4.0],
            [2.6, 800.0, 4.6, 8.0],
            [3.3, 200.0, 3.5, 20.0],
            [4.4, 80.0, 3.7, 20.0],
            [5.2, 60.0, 2.8, 70.0],
            [6.1, 20.0, 2.8, 70.0],
            [6.5, 1.8, 2.4, 100.0],
            [7.4, 1.0, 1.5, 500.0],
        ]
    ).T
    cov = np.array([np.diag([1.0 / a, 1.0 / b]) for a, b in zip(wx, wy)])
    return PolynomialModel(
        np.column_stack([x, y]),
        cov,
        [-0.5, 5.5] if degree == 1 else [-1.2e-2, 0.161, -1.15, 6.142],
        delta_params=np.full(degree + 1, 1.0e-3 if degree == 1 else 1.0e-5),
        tolerance=1.0e-1 if degree == 1 else 1.0e-8,
    )


@pytest.mark.parametrize("degree,wss", [(1, 11.8663531941), (3, 10.486904577)])
def test_pearson_york_errors_in_both_coordinates(degree, wss):
    model = pearson_york(degree)
    assert bm.nonlinear_fitting is fit
    assert bm.nonlinear_least_squares_fit(model, param_tolerance=1.0e-5) is None
    assert_allclose(model.WSS, wss, rtol=1.0e-7)
    assert_allclose(model.popt, model.get_params(), rtol=0, atol=0)
    assert model.dof == len(model.data) - len(model.popt)
    assert model.n_iterations > 0


def weighted_line():
    # Same RandomState, observations and exact x coordinates as the baseline.
    rng = np.random.RandomState(0)
    x = np.linspace(0.0, 10.0, 100)
    y = 2.5 * x - 1.3 + rng.normal(scale=0.3, size=len(x))
    return PolynomialModel(
        np.column_stack([x, y]),
        np.tile(np.diag([0.0, 0.3**2]), (len(x), 1, 1)),
        [0.5, 0.5],
    )


@pytest.mark.parametrize("with_priors", [False, True])
@pytest.mark.parametrize("difference_sign", [1.0, -1.0])
def test_closed_form_weighted_fit_and_gaussian_prior(with_priors, difference_sign):
    model = weighted_line()
    model.delta_params *= difference_sign
    X = np.column_stack([model.data[:, 0], np.ones(len(model.data))])
    JTJ = X.T @ X / 0.3**2
    JTy = X.T @ model.data[:, 1] / 0.3**2
    kwargs = {}
    if with_priors:
        mean = np.array([2.0, -0.5])
        precision = np.array([[100.0, 3.0], [3.0, 4.0]])
        kwargs = dict(param_priors=mean, param_prior_inv_cov_matrix=precision)
        JTJ += precision
        JTy += precision @ mean
    fit.nonlinear_least_squares_fit(model, param_tolerance=1.0e-10, **kwargs)
    expected = np.linalg.solve(JTJ, JTy)
    expected_wss = np.sum(((X @ expected - model.data[:, 1]) / 0.3) ** 2)
    if with_priors:
        diff = expected - mean
        expected_wss += diff @ precision @ diff
    assert_allclose(model.popt, expected, rtol=1.0e-7)
    assert_allclose(model.WSS, expected_wss, rtol=1.0e-10)
    assert_allclose(model.goodness_of_fit, expected_wss / 98, rtol=1.0e-10)
    assert_allclose(model.pcov, np.linalg.inv(JTJ) * expected_wss / 98, rtol=1.0e-6)
    expected_noise = np.sum((X @ expected - model.data[:, 1]) ** 2) / (
        98 * (1 + expected[0] ** 2)
    )
    assert_allclose(model.noise_variance, expected_noise, rtol=1.0e-7)
    assert_allclose(model.weights, (1 + model.popt[0] ** 2) / 0.3**2)


def test_final_jacobian_and_residuals_match_popt():
    model = pearson_york()
    fit.nonlinear_least_squares_fit(model, param_tolerance=1.0e-5)
    cached = model.jacobian.copy()
    assert fit.calculate_jacobian(model) is None
    positions, residuals, weights = fit.find_mle(model)
    assert_array_equal(model.get_params(), model.popt)
    assert_array_equal(cached, model.jacobian)
    assert_array_equal(model.data_mle, positions)
    assert_array_equal(model.weighted_residuals, residuals)
    assert_array_equal(model.weights, weights)
    # Result arrays own their storage and remain independent of the adapter.
    model.popt[0] = 100
    assert model.get_params()[0] != 100


def test_correlated_tangent_plane_projection():
    model = pearson_york()
    x = np.array([2.0, 6.0])
    point = model.function(x)
    cov = np.array([[0.4, 0.1], [0.1, 0.7]])
    n = model.normal(point)
    variance = n @ cov @ n
    residual = (point - x) @ n
    estimate, actual_residual, actual_variance = fit.mle_estimate(model, x, point, cov)
    assert_allclose(fit.abs_line_project(cov, 10 * n), variance)
    assert_allclose(estimate, x + residual / variance * cov @ n)
    assert_allclose(actual_residual, residual)
    assert_allclose(actual_variance, variance)
    assert_allclose(model.function(estimate), estimate)


class BoundedLine(PolynomialModel):
    def set_params(self, params):
        self.params = np.clip(params, [-np.inf, -np.inf], [2.0, np.inf])


def test_model_enforces_parameter_bounds():
    model = weighted_line()
    model.__class__ = BoundedLine
    fit.nonlinear_least_squares_fit(model)
    assert model.popt[0] == 2.0
    assert np.isfinite(model.pcov).all()


class FlaggedLine(PolynomialModel):
    def function(self, x, flag):
        assert isinstance(flag, dict)
        return np.array([x[0], flag["scale"] * self.params[0] * x[0] + self.params[1]])

    def normal(self, x, flag):
        n = np.array([flag["scale"] * self.params[0], -1.0])
        return n / np.linalg.norm(n)


def flagged_line():
    model = weighted_line()
    model.__class__ = FlaggedLine
    model.flags = [{"scale": s} for s in np.tile([1.0, 2.0], 50)]
    model.data[1::2, 1] += 2.5 * model.data[1::2, 0]
    model.data_covariances[:, 0, 0] = 0.02
    model.data_covariances[:, 0, 1] = 0.01
    model.data_covariances[:, 1, 0] = 0.01
    return model


def test_per_observation_flags_are_forwarded_without_conversion():
    model = flagged_line()
    fit.nonlinear_least_squares_fit(model)
    assert_allclose(model.popt, [2.5, -1.3], atol=0.1)


def test_local_cosine_fit_matches_baseline_aliased_minimum():
    rng = np.random.RandomState(3)
    x = np.sort(rng.uniform(0.0, 6.0, 15))

    class FrequencyModel(PolynomialModel):
        def function(self, x, flag=None):
            return np.array([x[0], np.cos(self.params[0] * x[0])])

        def normal(self, x, flag=None):
            return np.array([0.0, 1.0])

        def set_params(self, params):
            self.params = np.clip(params, 0.1, 12.0)

    model = FrequencyModel(
        np.column_stack([x, np.cos(5.0 * x)]),
        np.tile(np.diag([0.0, 1.0e-4]), (len(x), 1, 1)),
        [9.5],
    )
    fit.nonlinear_least_squares_fit(model, param_tolerance=1.0e-8)
    assert_allclose(model.popt, [9.68239086], rtol=1.0e-4)


def test_adaptive_damping_rejects_an_overshooting_gauss_newton_step(capfd):
    class SquareModel(PolynomialModel):
        def function(self, x, flag=None):
            return np.array([x[0], self.params[0] ** 2])

        def normal(self, x, flag=None):
            return np.array([0.0, 1.0])

    model = SquareModel(
        [[0, 3.9], [1, 4.1], [2, 4.0]],
        np.tile(np.diag([0.0, 1.0]), (3, 1, 1)),
        [0.1],
    )
    fit.nonlinear_least_squares_fit(model, verbose=True)
    assert_allclose(model.popt, [2.0], rtol=1.0e-7)
    assert "step rejected" in capfd.readouterr().out


def test_jacobian_restores_parameters_when_model_raises():
    model = weighted_line()
    original = model.get_params().copy()
    model.function = lambda *args: (_ for _ in ()).throw(ValueError("bad EOS"))
    with pytest.raises(ValueError, match="bad EOS"):
        fit.calculate_jacobian(model)
    assert_array_equal(model.get_params(), original)


def test_failure_to_converge_is_reported():
    with pytest.raises(RuntimeError, match="Maximum number of LM iterations"):
        fit.nonlinear_least_squares_fit(pearson_york(3), max_lm_iterations=1)


def test_nonconverging_mle_projection_has_an_iteration_limit():
    model = weighted_line()
    model.function = lambda x, flag: np.array([x[0] + 1.0, 0.0])
    model.normal = lambda x, flag: np.array([0.0, 1.0])
    with pytest.raises(RuntimeError, match="MLE projection did not converge"):
        fit.find_mle(model)


def test_parameter_scales_do_not_trigger_a_relative_rank_cutoff():
    model = weighted_line()
    # The same line expressed in parameters spanning sixteen orders of magnitude.
    scale = np.array([1.0e-11, 1.0e5])
    model.params /= scale
    model.delta_params /= scale
    model.function = lambda x, flag: np.array(
        [x[0], np.polyval(model.params * scale, x[0])]
    )
    model.normal = lambda x, flag: np.array([0.0, 1.0])
    X = np.column_stack([model.data[:, 0], np.ones(len(model.data))])
    expected = np.linalg.lstsq(X, model.data[:, 1], rcond=None)[0]
    fit.nonlinear_least_squares_fit(model)
    assert_allclose(model.popt * scale, expected, rtol=1.0e-7)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda m: setattr(m, "delta_params", np.zeros(2)), "delta_params"),
        (lambda m: setattr(m, "mle_tolerances", [-1.0] * len(m.data)), "tolerances"),
        (lambda m: setattr(m, "flags", [None]), "flag"),
        (lambda m: m.data_covariances.__setitem__((0, 0, 0), -1.0), "semidefinite"),
        (lambda m: m.data_covariances.__setitem__((0, 0, 1), 0.1), "symmetric"),
        (lambda m: m.data_covariances.fill(0.0), "positive variance"),
    ],
)
def test_invalid_model_data_raise(change, match):
    model = weighted_line()
    change(model)
    with pytest.raises((ValueError, RuntimeError), match=match):
        fit.nonlinear_least_squares_fit(model)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lm_damping": -1.0},
        {"param_tolerance": 0.0},
        {"max_lm_iterations": 0},
        {"param_priors": [1.0, 2.0]},
        {"param_priors": [1.0], "param_prior_inv_cov_matrix": np.eye(2)},
        {"param_priors": [1.0, 2.0], "param_prior_inv_cov_matrix": -np.eye(2)},
    ],
)
def test_invalid_fit_options_raise(kwargs):
    with pytest.raises(ValueError):
        fit.nonlinear_least_squares_fit(weighted_line(), **kwargs)


def test_de_initialisation_is_explicitly_unsupported():
    with pytest.raises(NotImplementedError, match="Differential evolution"):
        fit.nonlinear_least_squares_fit(
            weighted_line(), first_run_differential_evolution=True
        )


@pytest.mark.parametrize("case", ["line", "cubic", "priors", "bounds", "flags"])
def test_optional_python_burnman_parity(case):
    pytest.importorskip("burnman")
    from burnman.optimize import nonlinear_fitting as reference

    if case in ("line", "cubic"):
        actual = pearson_york(1 if case == "line" else 3)
    elif case == "flags":
        actual = flagged_line()
    else:
        actual = weighted_line()
        if case == "bounds":
            actual.__class__ = BoundedLine
    expected = copy.deepcopy(actual)
    kwargs = {"param_tolerance": 1.0e-5}
    if case == "priors":
        kwargs.update(
            param_priors=[2.0, -0.5],
            param_prior_inv_cov_matrix=np.array([[100.0, 3.0], [3.0, 4.0]]),
        )
    fit.nonlinear_least_squares_fit(actual, **kwargs)
    reference.nonlinear_least_squares_fit(expected, **kwargs)
    for name in (
        "popt",
        "pcov",
        "WSS",
        "data_mle",
        "weighted_residuals",
        "weights",
        "jacobian",
        "goodness_of_fit",
        "noise_variance",
    ):
        assert_allclose(
            getattr(actual, name),
            getattr(expected, name),
            rtol=2.0e-5,
            atol=1.0e-7,
            err_msg=name,
        )
