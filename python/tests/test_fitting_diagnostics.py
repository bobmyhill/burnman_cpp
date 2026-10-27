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

"""Diagnostics and Matplotlib output compared with the pinned Python routines."""

from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.stats import norm, genextreme

from burnman_cpp.optimize import nonlinear_fitting as fit


@pytest.mark.parametrize("count", [2, 11, 100, 1000])
@pytest.mark.parametrize("confidence", [0.0, 0.5, 0.9, 0.95, 1.0])
def test_extreme_values_match_scipy_reference(count, confidence):
    residuals = np.random.RandomState(3).normal(size=count)
    residuals[0] = -7.0
    residuals[-1] = 5.0
    location = norm.isf(1.0 / count)
    scale = 0.8 / np.sqrt(np.log(count))
    shape = 0.33 / np.log(count) ** 0.75
    expected_bound = genextreme.isf(
        1.0 - np.sqrt(confidence), shape, loc=location, scale=scale
    )
    expected_indices = [i for i, r in enumerate(residuals) if abs(r) > expected_bound]
    expected_probabilities = (
        1.0
        - (
            genextreme.sf(
                np.abs(residuals[expected_indices]), shape, loc=location, scale=scale
            )
            - 1.0
        )
        ** 2
    )
    bound, indices, probabilities = fit.extreme_values(residuals, confidence)
    assert_allclose(bound, expected_bound, rtol=1.0e-12, atol=1.0e-12)
    assert indices == expected_indices
    assert isinstance(indices, list)
    assert_allclose(probabilities, expected_probabilities, rtol=1.0e-10, atol=1.0e-14)


def test_extreme_values_strict_threshold_and_nonfinite_residuals():
    residuals = np.zeros(100)
    threshold, _, _ = fit.extreme_values(residuals, 0.95)
    residuals[3] = threshold
    residuals[9] = -threshold * 1.01
    residuals[7] = np.inf
    residuals[11] = np.nan
    bound, indices, probabilities = fit.extreme_values(residuals, 0.95)
    assert bound == threshold
    assert indices == [7, 9]
    assert probabilities[0] == 0.0
    assert 0.0 < probabilities[1] < 1.0
    assert fit.extreme_values(np.zeros(20), 0.95)[2].shape == (0,)
    with pytest.raises(ZeroDivisionError):
        fit.extreme_values(np.array([]), 0.95)
    assert np.isnan(fit.extreme_values(np.array([0.0]), 0.95)[0])


@pytest.mark.parametrize("confidence", [-0.1, 1.1, np.nan, np.inf])
def test_extreme_values_invalid_probability_preserves_python_nan(confidence):
    bound, indices, probabilities = fit.extreme_values(np.zeros(20), confidence)
    assert np.isnan(bound)
    assert indices == []
    assert probabilities.shape == (0,)


def test_optional_extreme_values_match_pinned_python():
    reference = pytest.importorskip("burnman").nonlinear_fitting
    residuals = np.linspace(-5.0, 5.0, 101)
    actual = fit.extreme_values(residuals, 0.9)
    expected = reference.extreme_values(residuals, 0.9)
    assert_allclose(actual[0], expected[0], rtol=1.0e-12)
    assert actual[1] == expected[1]
    assert_allclose(actual[2], expected[2], rtol=1.0e-10, atol=1.0e-14)


@pytest.fixture
def plt():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    yield pyplot
    pyplot.close("all")


def test_covariance_ellipse_matches_geometry_and_pinned_python(plt):
    reference = pytest.importorskip("burnman").nonlinear_fitting
    covariance = np.array([[4.0, 1.2], [1.2, 1.0]])
    position = np.array([2.0, -3.0])
    _, actual_axes = plt.subplots()
    _, expected_axes = plt.subplots()
    actual = fit.plot_cov_ellipse(
        covariance, position, nstd=2.5, ax=actual_axes, color="grey", alpha=0.4
    )
    expected = reference.plot_cov_ellipse(
        covariance, position, nstd=2.5, ax=expected_axes, color="grey", alpha=0.4
    )
    assert actual.axes is actual_axes
    assert_allclose(actual.get_center(), position)
    assert_allclose(
        [actual.width, actual.height, actual.angle],
        [expected.width, expected.height, expected.angle],
    )
    assert_allclose(
        actual.get_transform().get_matrix(), expected.get_transform().get_matrix()
    )
    assert actual.get_alpha() == expected.get_alpha()


def test_corner_plot_layout_and_ellipses_match_python(plt):
    reference = pytest.importorskip("burnman").nonlinear_fitting
    parameters = np.array([1.0e-5, 2.0e11, -4.0])
    errors = np.array([2.0e-7, 1.0e9, 0.2])
    covariance = np.outer(errors, errors) * np.array(
        [[1.0, 0.3, -0.1], [0.3, 1.0, 0.2], [-0.1, 0.2, 1.0]]
    )
    actual, axes = fit.corner_plot(parameters, covariance, ["V_0", "K_0", "q_0"], 2.0)
    expected, _ = reference.corner_plot(
        parameters, covariance, ["V_0", "K_0", "q_0"], 2.0
    )
    assert axes.shape == (2, 2)
    assert len(actual.axes) == len(expected.axes) == 3
    assert_allclose(actual.get_size_inches(), expected.get_size_inches())
    for a, e in zip(actual.axes, expected.axes):
        assert a.get_xlabel() == e.get_xlabel()
        assert a.get_ylabel() == e.get_ylabel()
        assert_allclose(a.get_xlim(), e.get_xlim())
        assert_allclose(a.get_ylim(), e.get_ylim())
        assert_allclose(a.patches[0].get_center(), e.patches[0].get_center())
        assert_allclose(
            [a.patches[0].width, a.patches[0].height, a.patches[0].angle],
            [e.patches[0].width, e.patches[0].height, e.patches[0].angle],
        )


def test_pinned_two_parameter_corner_plot_failure_is_preserved(plt):
    # The pin returns a scalar Axes for a 1x1 subplot grid, then indexes it.
    with pytest.raises(TypeError, match="not subscriptable"):
        fit.corner_plot(np.array([1.0, 2.0]), np.eye(2))


def test_weighted_residual_projection_and_colours_match_python(plt):
    reference = pytest.importorskip("burnman").nonlinear_fitting
    model = SimpleNamespace(
        data=np.array([[1.0e9, 300.0, 1.0], [2.0e9, 600.0, 2.0], [3.0e9, 900.0, 3.0]]),
        flags=["V", "H", "V"],
        weighted_residuals=np.array([-4.0, 0.0, 4.0]),
    )
    _, a = plt.subplots()
    _, e = plt.subplots()
    cmap = plt.cm.RdYlBu
    plt.sca(a)
    assert (
        fit.weighted_residual_plot(
            a, model, "V", plot_axes=[1, 0], scale_axes=[1.0, 1.0e-9], cmap=cmap
        )
        is None
    )
    plt.sca(e)
    assert (
        reference.weighted_residual_plot(
            e, model, "V", plot_axes=[1, 0], scale_axes=[1.0, 1.0e-9], cmap=cmap
        )
        is None
    )
    assert_allclose(a.collections[0].get_offsets(), e.collections[0].get_offsets())
    assert_array_equal(a.collections[0].get_array(), e.collections[0].get_array())
    assert_allclose(a.collections[0].norm.boundaries, e.collections[0].norm.boundaries)
    assert a.figure.axes[-1].get_ylabel() == e.figure.axes[-1].get_ylabel()
    assert cmap._rgba_under is None


def test_residual_histogram_and_reference_normal_match_python(plt):
    reference = pytest.importorskip("burnman").nonlinear_fitting
    residuals = np.array([-2.0, -1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    flags = ["V"] * 4 + ["H"] * 4
    _, a = plt.subplots()
    assert fit.plot_residuals(a, residuals, n_bins=6, flags=flags) is None
    _, e = plt.subplots()
    assert reference.plot_residuals(e, residuals, n_bins=6, flags=flags) is None
    assert len(a.patches) == len(e.patches) == 10
    for actual, expected in zip(a.patches, e.patches):
        assert_allclose(
            [actual.get_x(), actual.get_y(), actual.get_width(), actual.get_height()],
            [
                expected.get_x(),
                expected.get_y(),
                expected.get_width(),
                expected.get_height(),
            ],
        )
    for actual, expected in zip(a.lines, e.lines):
        assert_allclose(actual.get_xdata(), expected.get_xdata())
        assert_allclose(actual.get_ydata(), expected.get_ydata())
    assert a.get_legend_handles_labels()[1] == e.get_legend_handles_labels()[1]
    assert a.get_xlabel() == e.get_xlabel()
    assert a.get_ylabel() == e.get_ylabel()


def test_pinned_default_histogram_is_empty(plt):
    _, axes = plt.subplots()
    with pytest.warns(UserWarning, match="No artists"):
        fit.plot_residuals(axes, np.linspace(-2.0, 2.0, 20))
    assert len(axes.patches) == len(axes.lines) == 0


def test_eos_verbose_outlier_diagnostic_with_supplied_covariances(capfd):
    import burnman_cpp as bm

    mineral = bm.minerals.HP_2011_ds62.fo()
    rows = []
    for pressure in np.linspace(1.0e9, 2.0e9, 20):
        mineral.set_state(pressure, 298.15)
        rows.append([pressure, 298.15, mineral.molar_volume])
    data = np.array(rows)
    covariance = np.array([np.diag([0.0, 0.0, (row[2] * 1.0e-4) ** 2]) for row in data])
    data[-1, 2] *= 1.003
    model = bm.eos_fitting.fit_PTV_data(
        mineral, ["V_0"], data, covariance, verbose=True
    )
    output = capfd.readouterr().out
    bound, indices, probabilities = fit.extreme_values(model.weighted_residuals, 0.9)
    assert indices
    assert f"there are {len(indices)} data points" in output
    assert f"90.0% confidence level (> {bound:.1f} s.d." in output
    for index, probability in zip(indices, probabilities):
        assert f"[{index}]: {probability:.4f}" in output
