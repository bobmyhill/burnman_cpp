"""Solution fitting, following the pinned Python tests/test_fitting.py."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import burnman_cpp as bm
from burnman_cpp.optimize.eos_fitting import SolutionFit, fit_XPTp_data

FIT_PARAMS = [["V_0", 0], ["V_0", 1], ["V", 0, 1]]


def volume_data(solution):
    rows = []
    for i, fraction in enumerate(np.linspace(0.05, 0.95, 12)):
        solution.set_composition([1.0 - fraction, fraction])
        pressure, temperature = (i % 4) * 1.0e9, 500.0 + 60.0 * i
        solution.set_state(pressure, temperature)
        rows.append(
            [1.0 - fraction, fraction, pressure, temperature, solution.molar_volume]
        )
    return np.array(rows)


def direct_model(solution, fit_params, bounds=None):
    n = solution.n_endmembers + 3
    data = np.zeros((12, n))
    data[:, 0] = 1.0
    data[:, -3:] = [1.0e9, 1000.0, 1.0e-4]
    cov = np.zeros((12, n, n))
    cov[:, -3, -3] = 1.0
    return SolutionFit(
        solution,
        data,
        cov,
        ["V"] * 12,
        fit_params,
        np.full(12, 1.0e-10),
        bounds=bounds,
    )


def test_exact_solution_fit_and_parameter_updates():
    solution = bm.minerals.SLB_2011.mg_fe_olivine()
    data = volume_data(solution)
    model = fit_XPTp_data(solution, FIT_PARAMS, "V", data, verbose=False)
    assert isinstance(model, bm.nonlinear_fitting.NonLinearModel)
    assert model.m is solution
    assert model.fit_params == FIT_PARAMS
    assert model.fit_params_strings == ["V_0 (0)", "V_0 (1)", "V (0,1)"]
    assert model.flags == ["V"] * len(data)
    assert_array_equal(
        model.data_covariances, np.tile(np.diag([0.0, 0.0, 1.0, 0.0, 0.0]), (12, 1, 1))
    )
    assert_array_equal(model.popt, model.get_params())
    assert_array_equal(model.pcov, np.zeros((3, 3)))
    assert model.WSS == 0.0
    assert model.dof == 9
    assert_array_equal(model.function(data[4]), data[4])
    old_volume = solution.molar_volume
    trial = model.popt.copy()
    trial[0] *= 1.001
    trial[2] = 1.0e-7
    model.set_params(trial)
    assert_allclose(model.get_params(), trial, rtol=1.0e-15)
    assert solution.molar_volume != old_volume
    assert model.function(data[4])[-1] != old_volume
    bm.nonlinear_least_squares_fit(model, param_tolerance=1.0e-5)
    assert_allclose(model.popt[:2], trial[:2] / [1.001, 1.0], rtol=1.0e-8)
    assert abs(model.popt[2]) < 1.0e-12


@pytest.mark.parametrize("bounded", [False, True])
def test_solution_array_parameter_round_trip_from_python_baseline(bounded):
    solution = bm.minerals.JH_2015.garnet()
    fit_params = [["Cp", 0], ["V_0", 1], ["Cp", 1], ["E", 0, 1]]
    values = np.array([10.0, 20.0, 30.0, 40.0, 4.5e-5, 50.0, 60.0, 70.0, 80.0, 2000.0])
    expected = values.copy()
    bounds = None
    if bounded:
        bounds = np.tile([-np.inf, np.inf], (len(values), 1))
        bounds[2, 1], bounds[8, 1] = 25.0, 75.0
        expected[2], expected[8] = 25.0, 75.0
    model = direct_model(solution, fit_params, bounds)
    initial = model.get_params()
    assert_array_equal(model.delta_params, initial * 1.0e-5 + 1.0e-10)
    model.set_params(values)
    assert_array_equal(model.get_params(), expected)


def mixed_data(solution):
    # Same seven observations and uncertainties as test_bounded_solution_fitting.
    random = np.random.RandomState(10)
    data, covariances, flags = [], [], []
    for flag, error, count in [("V", 1.0e-3, 5), ("p_wave_velocity", 1.0e-2, 2)]:
        for _ in range(count):
            fraction = random.random()
            pressure = random.random() * 1.0e10
            temperature = random.random() * 1000.0 + 300.0
            solution.set_composition([1.0 - fraction, fraction])
            solution.set_state(pressure, temperature)
            prop = getattr(solution, "molar_volume" if flag == "V" else flag)
            measured = prop * (1.0 + (random.normal() - 0.5) * error)
            data.append([1.0 - fraction, fraction, pressure, temperature, measured])
            covariances.append(np.diag([0.0, 0.0, 0.0, 0.0, (prop * error) ** 2]))
            flags.append(flag)
    return np.array(data), np.array(covariances), flags


@pytest.mark.parametrize("with_priors", [False, True])
def test_mixed_bounded_solution_fit_matches_python_baseline(with_priors):
    reference = pytest.importorskip("burnman")
    native = bm.minerals.SLB_2011.mg_fe_olivine()
    python = reference.minerals.SLB_2011.mg_fe_olivine()
    data, covariances, flags = mixed_data(python)
    # The wrapper's default tolerances require an initial composition.
    native.set_composition(python.molar_fractions)
    kwargs = dict(
        data_covariances=covariances,
        delta_params=np.full(3, 1.0e-8),
        bounds=np.array([[0.0, np.inf], [0.0, np.inf], [-np.inf, np.inf]]),
        verbose=False,
    )
    if with_priors:
        kwargs.update(
            param_priors=[
                python.endmembers[0][0].params["V_0"],
                python.endmembers[1][0].params["V_0"],
                0.0,
            ],
            param_prior_inv_cov_matrix=np.diag([1.0e7] * 3),
        )
    expected = reference.eos_fitting.fit_XPTp_data(
        python, FIT_PARAMS, flags, data, **kwargs
    )
    actual = fit_XPTp_data(native, FIT_PARAMS, flags, data, **kwargs)
    for name in (
        "popt",
        "pcov",
        "WSS",
        "goodness_of_fit",
        "noise_variance",
        "data_mle",
        "weighted_residuals",
        "weights",
        "jacobian",
    ):
        assert_allclose(
            getattr(actual, name),
            getattr(expected, name),
            rtol=2.0e-5,
            atol=1.0e-18,
            err_msg=name,
        )


@pytest.mark.parametrize(
    "dataset,name", [("SLB_2011", "mg_fe_olivine"), ("JH_2015", "plagioclase")]
)
def test_binary_interactions_match_python_scaling_and_properties(dataset, name):
    reference = pytest.importorskip("burnman")
    native = getattr(getattr(bm.minerals, dataset), name)()
    python = getattr(getattr(reference.minerals, dataset), name)()
    params = [["E", 0, 1], ["S", 0, 1], ["V", 0, 1]]
    actual = direct_model(native, params)
    expected = reference.eos_fitting.SolutionFit(
        python,
        actual.data,
        actual.data_covariances,
        actual.flags,
        params,
        actual.mle_tolerances,
    )
    assert_allclose(actual.get_params(), expected.get_params(), rtol=1.0e-15)
    values = [1200.0, 1.5, 1.0e-7]
    actual.set_params(values)
    expected.set_params(values)
    assert_allclose(actual.get_params(), expected.get_params(), rtol=1.0e-15)
    point = np.array([0.4, 0.6, 2.0e9, 1000.0, 0.0])
    for flag in ["V", "H", "S", "gibbs"]:
        assert_allclose(
            actual.function(point, flag),
            expected.function(point, flag),
            rtol=1.0e-8,
            atol=1.0e-15,
        )


@pytest.mark.parametrize("flag", ["V", "H", "S", "gibbs", "p_wave_velocity"])
def test_solution_normals_follow_pin(flag):
    reference = pytest.importorskip("burnman")
    actual = direct_model(bm.minerals.SLB_2011.mg_fe_olivine(), FIT_PARAMS)
    expected = reference.eos_fitting.SolutionFit(
        reference.minerals.SLB_2011.mg_fe_olivine(),
        actual.data,
        actual.data_covariances,
        actual.flags,
        FIT_PARAMS,
        actual.mle_tolerances,
    )
    x = np.array([0.4, 0.6, 2.0e9, 1000.0, 0.0])
    assert_allclose(actual.normal(x, flag), expected.normal(x, flag), rtol=1.0e-8)
    assert_array_equal(actual.normal(x, flag)[:2], [0.0, 0.0])
    if flag == "p_wave_velocity":
        # The pinned fallback reads pressure at coordinate 2, not velocity.
        assert_allclose(actual.normal(x, flag)[-3:], [-(2**-0.5), 0.0, 2**-0.5])


def test_solution_fit_rejects_unimplemented_de():
    solution = bm.minerals.SLB_2011.mg_fe_olivine()
    data = volume_data(solution)
    with pytest.raises(NotImplementedError, match="Differential evolution"):
        fit_XPTp_data(
            solution, FIT_PARAMS, "V", data, first_run_differential_evolution=True
        )
