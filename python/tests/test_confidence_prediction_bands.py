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

"""Analytic and unit-scaling checks for confidence and prediction intervals."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.stats import t

from burnman_cpp.optimize.nonlinear_fitting import (
    nonlinear_least_squares_fit,
    confidence_prediction_bands as bands,
)


class Line:
    def __init__(self, x_scale=1.0):
        x = np.linspace(-2.0, 3.0, 20)
        y = 2.5 * x - 1.3 + np.random.RandomState(7).normal(scale=0.3, size=len(x))
        self.data = np.column_stack([x * x_scale, y])
        self.data_covariances = np.tile(np.diag([0.0, 0.09]), (len(x), 1, 1))
        self.mle_tolerances = np.full(len(x), 1.0e-10)
        self.params = np.array([2.0 / x_scale, -1.0])
        self.delta_params = np.array([1.0e-5 / x_scale, 1.0e-5])

    def get_params(self):
        return self.params

    def set_params(self, p):
        self.params[:] = p  # Deliberately mutate the same array.

    def function(self, x, flag=None):
        return np.array([x[0], self.params[0] * x[0] + self.params[1]])

    def normal(self, x, flag=None):
        n = np.array([self.params[0], -1.0])
        return n / np.linalg.norm(n)


def fitted_line(x_scale=1.0):
    model = Line(x_scale)
    nonlinear_least_squares_fit(model, param_tolerance=1.0e-9)
    return model


def test_line_intervals_match_closed_form_regression():
    model = fitted_line()
    queries = np.array(
        [[-1, 999], [0, -999], [2, 0]]
    )  # Integers, deliberately off the curve.
    before = model.get_params().copy()
    actual = bands(model, queries, 0.95, lambda x: x[-1])
    X = np.column_stack([queries[:, 0], np.ones(len(queries))])
    mean = X @ before
    variance = np.einsum("ij,jk,ik->i", X, model.pcov, X)
    residuals = model.data[:, -1] - (model.data[:, 0] * before[0] + before[1])
    noise = residuals @ residuals / model.dof
    critical = t.ppf(0.975, model.dof)
    expected = np.array(
        [
            mean - critical * np.sqrt(variance),
            mean + critical * np.sqrt(variance),
            mean - critical * np.sqrt(variance + noise),
            mean + critical * np.sqrt(variance + noise),
        ]
    )
    assert_allclose(actual, expected, rtol=2.0e-9, atol=1.0e-10)
    assert (actual[2] <= actual[0]).all() and (actual[0] <= actual[1]).all()
    assert (actual[1] <= actual[3]).all()
    assert_array_equal(model.get_params(), before)


def test_coordinate_units_do_not_change_intervals():
    first = fitted_line()
    second = fitted_line(1.0e-9)
    x = np.array([[-1.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    other = x.copy()
    other[:, 0] *= 1.0e-9
    assert_allclose(
        bands(first, x, 0.95, lambda p: p[-1]),
        bands(second, other, 0.95, lambda p: p[-1]),
        rtol=2.0e-8,
        atol=1.0e-9,
    )


@pytest.mark.parametrize("factor", [2.0, 1000.0, -2.0])
def test_output_units_and_offsets_transform_all_intervals(factor):
    model = fitted_line()
    x = np.array([[0.0, 999.0], [2.0, 999.0]])
    original = bands(model, x, 0.9, lambda p: p[-1])
    scaled = bands(model, x, 0.9, lambda p: factor * p[-1] + 7.0)
    restored = (scaled - 7.0) / factor
    if factor < 0:
        restored = restored[[1, 0, 3, 2]]
    assert_allclose(restored, original, rtol=1.0e-8)


def test_unrelated_output_and_missing_response_noise_keep_confidence_limits():
    model = fitted_line()
    x = model.data[:2]
    unrelated = bands(model, x, 0.95, lambda p: model.get_params()[0])
    assert np.isfinite(unrelated[:2]).all()
    assert np.isnan(unrelated[2:]).all()
    model.flags = ["other"] * len(model.data)
    missing = bands(model, x, 0.95, lambda p: p[-1])
    assert np.isfinite(missing[:2]).all()
    assert np.isnan(missing[2:]).all()


def test_noise_uses_matching_flags_and_needs_no_legacy_noise_attribute():
    model = fitted_line()
    x = model.data[:2]
    expected = bands(model, x, 0.95, lambda p: p[-1])
    del model.noise_variance
    del model.flags
    assert_allclose(bands(model, x, 0.95, lambda p: p[-1]), expected)
    count = len(model.data)
    other = model.data.copy()
    other[:, -1] *= 1.0e9
    model.data = np.vstack([model.data, other])
    model.flags = [None] * count + ["other"] * count
    model.dof += count
    ratio = t.ppf(0.975, model.dof) / t.ppf(0.975, model.dof - count)
    centre = np.mean(expected[:2], axis=0)
    assert_allclose(
        bands(model, x, 0.95, lambda p: p[-1]), centre + (expected - centre) * ratio
    )


def test_optional_prediction_limits_match_pinned_python():
    reference = pytest.importorskip("burnman")
    model = fitted_line()
    x = model.data[:3]
    for output in (
        lambda p: p[-1],
        lambda p: -2.0 * p[-1] + 7.0,
        lambda p: model.get_params()[0],
    ):
        actual = bands(model, x, 0.95, output)
        expected = reference.nonlinear_fitting.confidence_prediction_bands(
            model, x, 0.95, output
        )
        assert_allclose(actual, expected, rtol=1.0e-8, equal_nan=True)


def test_differences_follow_original_query_coordinates():
    class MovingCoordinates(Line):
        def function(self, x, flag=None):
            return np.array(
                [x[0] + self.params[1], self.params[0] * x[0] + self.params[1]]
            )

    model = MovingCoordinates()
    model.params[:] = [2.0, 0.5]
    model.pcov = np.array([[0.04, 0.01], [0.01, 0.09]])
    model.dof = 18
    model.noise_variance = 0.0
    x = np.array([[1.0, 999.0], [2.0, -999.0]])
    actual = bands(model, x, 0.95, lambda p: p[-1])
    mean = 2.0 * x[:, 0] + 0.5
    G = np.column_stack([x[:, 0], np.ones(2)])
    sigma = np.sqrt(np.einsum("ij,jk,ik->i", G, model.pcov, G))
    assert_allclose(
        actual[:2],
        [mean - t.ppf(0.975, 18) * sigma, mean + t.ppf(0.975, 18) * sigma],
        rtol=1.0e-10,
    )


def test_parameters_are_restored_if_an_output_callback_fails():
    model = fitted_line()
    params = model.get_params().copy()

    def fail(point):
        if not np.array_equal(model.get_params(), params):
            raise ValueError("bad observable")
        return point[-1]

    with pytest.raises(ValueError, match="bad observable"):
        bands(model, model.data[:2], 0.95, fail)
    assert_array_equal(model.get_params(), params)


def test_one_sided_differences_at_a_bound_use_the_actual_step():
    class Bounded(Line):
        def set_params(self, p):
            self.params[:] = np.minimum(p, [2.0, np.inf])

    model = Bounded()
    model.params[:] = [2.0, -1.0]
    model.pcov = np.diag([0.04, 0.09])
    model.dof = 18
    model.noise_variance = 0.0
    x = np.array([[3.0, 0.0]])
    actual = bands(model, x, 0.95, lambda p: p[-1])
    width = t.ppf(0.975, 18) * np.sqrt(9.0 * 0.04 + 0.09)
    assert_allclose(actual[:2, 0], [5.0 - width, 5.0 + width], rtol=1.0e-10)
    assert (actual[2] <= actual[0]).all() and (actual[3] >= actual[1]).all()


@pytest.mark.parametrize("level", [0.0, 1.0, -1.0, np.nan, np.inf])
def test_invalid_confidence_levels_raise(level):
    model = fitted_line()
    with pytest.raises(ValueError):
        bands(model, model.data[:2], level, lambda p: p[-1])


def test_empty_queries_and_zero_parameter_covariance():
    model = fitted_line()
    assert bands(model, np.empty((0, 2)), 0.95, lambda p: p[-1]).shape == (4, 0)
    model.pcov = np.zeros((2, 2))
    result = bands(model, model.data[:2], 0.95, lambda p: p[-1])
    assert_array_equal(result[0], result[1])
    assert (result[2] < result[3]).all()


def test_native_material_bands_match_pinned_python():
    import burnman_cpp as bm

    m = bm.minerals.HP_2011_ds62.stv()
    p = np.linspace(1.0e9, 10.0e9, 8)
    rows = []
    for pressure in p:
        m.set_state(pressure, 300.0)
        rows.append([pressure, 300.0, m.molar_volume])
    rows = np.array(rows)
    rows[:, 2] *= 1.0 + np.linspace(-1.0e-4, 1.0e-4, 8)
    model = bm.eos_fitting.fit_PTV_data(
        m, ["V_0", "K_0", "Kprime_0"], rows, verbose=False
    )
    query = rows[:3].copy()
    query[:, 2] = 0.0
    result = bands(model, query, 0.95, lambda x: x[-1], flag="V")
    expected = np.array([model.function(row, "V")[-1] for row in query])
    assert_allclose((result[0] + result[1]) / 2, expected)
    assert (result[2] <= result[0]).all()

    reference = pytest.importorskip("burnman")
    expected = reference.nonlinear_fitting.confidence_prediction_bands(
        model, query, 0.95, lambda x: x[-1], "V"
    )
    assert_allclose(result, expected, rtol=1.0e-8, equal_nan=True)
