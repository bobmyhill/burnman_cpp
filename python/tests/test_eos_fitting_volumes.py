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

"""Volume-coordinate cases from Python BurnMan's test_fitting.py."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import burnman_cpp as bm
from burnman_cpp.optimize.eos_fitting import MineralFitV, fit_VTP_data, fit_VTp_data


def volume_data(mineral, property_name="pressure"):
    rows = []
    for pressure in np.linspace(1.0e9, 2.0e9, 8):
        mineral.set_state(pressure, mineral.params.T_0)
        volume, temperature = mineral.molar_volume, mineral.temperature
        mineral.set_state_with_volume(volume, temperature)
        rows.append([volume, temperature, getattr(mineral, property_name)])
    return np.array(rows)


@pytest.mark.parametrize("with_priors", [False, True])
def test_exact_vtp_fit_and_default_covariance(with_priors):
    mineral = bm.minerals.HP_2011_ds62.fo()
    data = volume_data(mineral)
    truth = np.array([mineral.params.V_0, mineral.params.K_0, mineral.params.Kprime_0])
    kwargs = {}
    if with_priors:
        kwargs.update(
            param_priors=truth,
            param_prior_inv_cov_matrix=np.diag(
                1.0 / np.array([truth[0] / 10.0, 1.0e9, 1.0]) ** 2
            ),
        )
    model = fit_VTP_data(
        mineral, ["V_0", "K_0", "Kprime_0"], data, verbose=False, **kwargs
    )
    assert isinstance(model, MineralFitV)
    assert isinstance(model, bm.nonlinear_fitting.NonLinearModel)
    assert model.m is mineral
    assert model.flags == ["P"] * 8
    assert_array_equal(
        model.data_covariances, np.tile(np.diag([0.0, 0.0, 1.0]), (8, 1, 1))
    )
    assert_array_equal(model.popt, truth)
    assert_array_equal(model.pcov, np.zeros((3, 3)))
    assert model.WSS == 0.0
    assert_array_equal(model.function(data[0]), data[0])
    assert_array_equal(model.function(data[0], "pressure"), data[0])
    bm.nonlinear_fitting.calculate_jacobian(model)
    assert_array_equal(model.get_params(), model.popt)
    _, residuals, weights = bm.nonlinear_fitting.find_mle(model)
    assert_array_equal(model.weighted_residuals, residuals)
    assert_array_equal(model.weights, weights)


@pytest.mark.parametrize(
    "flag,property_name,dataset,name",
    [
        ("helmholtz", "molar_helmholtz", "SLB_2011", "forsterite"),
        ("S", "molar_entropy", "HP_2011_ds62", "fo"),
        ("H", "molar_enthalpy", "HP_2011_ds62", "fo"),
    ],
)
def test_exact_volume_property_fits(flag, property_name, dataset, name):
    mineral = getattr(getattr(bm.minerals, dataset), name)()
    data = volume_data(mineral, property_name)
    params = ["V_0", "K_0", "Kprime_0"]
    if flag == "helmholtz":
        params.insert(0, "F_0")
    model = fit_VTp_data(mineral, params, flag, data, verbose=False)
    assert_array_equal(model.pcov, np.zeros((len(params), len(params))))
    assert model.WSS == 0.0
    assert model.flags == [flag] * 8


def test_volume_pressure_bounds_and_general_alias():
    mineral = bm.minerals.HP_2011_ds62.fo()
    rows = []
    for pressure in np.linspace(0.0, 10.0e9, 8):
        mineral.set_state(pressure, mineral.params.T_0)
        rows.append([mineral.molar_volume, mineral.temperature, pressure])
    data = np.array(rows)
    data[0, 0] *= 1.01
    data[-1, 0] *= 0.99
    bounds = [[0.0, np.inf], [0.0, np.inf], [3.0, 4.0]]
    model = fit_VTP_data(
        mineral, ["V_0", "K_0", "Kprime_0"], data, bounds=bounds, verbose=False
    )
    other = fit_VTp_data(
        bm.minerals.HP_2011_ds62.fo(),
        ["V_0", "K_0", "Kprime_0"],
        "pressure",
        data,
        bounds=bounds,
        verbose=False,
    )
    assert model.popt[2] == 3.0
    assert_array_equal(model.popt, other.popt)
    assert_array_equal(model.pcov, other.pcov)


def test_volume_states_include_property_modifier_volume():
    mineral = bm.minerals.SLB_2011.forsterite()
    mineral.set_property_modifiers(
        [("linear", {"delta_V": 2.0e-7, "delta_S": 0.5, "delta_E": 100.0})]
    )
    mineral.set_state(3.0e9, 700.0)
    volume, gibbs = mineral.molar_volume, mineral.molar_gibbs
    mineral.set_state_with_volume(volume, 700.0)
    assert mineral.molar_volume == volume
    assert_allclose(mineral.pressure, 3.0e9, rtol=1.0e-10)
    assert_allclose(mineral.molar_gibbs, gibbs, rtol=1.0e-12)


def test_noisy_helmholtz_fit_uses_python_example_initial_guesses():
    mineral = bm.minerals.SLB_2011.periclase()
    volumes, temperatures = np.meshgrid(
        np.linspace(0.7 * mineral.params.V_0, mineral.params.V_0, 4),
        np.linspace(500.0, 2000.0, 4),
    )
    values = []
    for volume, temperature in zip(volumes.ravel(), temperatures.ravel()):
        mineral.set_state_with_volume(volume, temperature)
        values.append(mineral.molar_helmholtz)
    energy = np.array(values) + np.random.RandomState(100).normal(scale=10.0, size=16)
    data = np.column_stack([volumes.ravel(), temperatures.ravel(), energy])
    cov = np.tile(np.diag([0.0, 0.0, 100.0]), (16, 1, 1))
    p = mineral.params
    p.F_0 += 2300.0
    p.V_0 *= 0.9
    p.K_0 *= 0.9
    model = fit_VTp_data(
        bm.Mineral(p),
        ["V_0", "K_0", "Kprime_0", "grueneisen_0", "q_0", "Debye_0", "F_0"],
        ["helmholtz"] * 16,
        data,
        cov,
        verbose=False,
    )
    assert model.dof == 9
    assert model.goodness_of_fit < 2.0
    assert np.isfinite(model.pcov).all()
    assert_array_equal(model.popt, model.get_params())


@pytest.mark.parametrize("case", ["pressure", "helmholtz", "entropy", "enthalpy"])
def test_optional_volume_fit_matches_python(case):
    reference = pytest.importorskip("burnman")
    from burnman_cpp.adapters import from_burnman

    original = reference.minerals.SLB_2011.periclase()
    native = from_burnman(original)
    prop = {
        "pressure": "pressure",
        "helmholtz": "helmholtz",
        "entropy": "S",
        "enthalpy": "H",
    }[case]
    params = ["V_0", "K_0", "Kprime_0"]
    if case == "helmholtz":
        params.insert(0, "F_0")
    if case == "entropy":
        params = ["V_0"]  # SLB entropy at fixed V,T is independent of K_0 and Kprime_0.
    rows = []
    sigmas = []
    for i, volume in enumerate(
        np.linspace(0.85 * original.params["V_0"], 0.99 * original.params["V_0"], 10)
    ):
        temperature = 500.0 + 50.0 * i
        original.set_state_with_volume(volume, temperature)
        value = getattr(original, prop)
        sigma = 1.0e-4 * abs(value)
        rows.append([volume, temperature, value + (-1) ** i * 0.5 * sigma])
        sigmas.append(sigma)
    data = np.array(rows)
    cov = np.array([np.diag([0.0, 0.0, sigma**2]) for sigma in sigmas])
    kwargs = dict(
        data_covariances=cov,
        verbose=False,
        param_tolerance=1.0e-5,
        mle_tolerances=np.full(10, 1.0e-10),
    )
    actual = fit_VTp_data(native, params, prop, data, **kwargs)
    expected = reference.eos_fitting.fit_VTp_data(
        original, params, prop, data, **kwargs
    )
    for name in (
        "popt",
        "pcov",
        "WSS",
        "data_mle",
        "weighted_residuals",
        "weights",
        "jacobian",
    ):
        assert_allclose(
            getattr(actual, name),
            getattr(expected, name),
            rtol=5.0e-4,
            atol=1.0e-16,
            err_msg=name,
        )
