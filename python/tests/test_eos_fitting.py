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

"""Default tolerances for material equation-of-state fitting."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import burnman_cpp as bm
from burnman_cpp.optimize.eos_fitting import (
    default_mle_tolerances,
    MineralFit,
    fit_PTV_data,
)


def ptv_data(mineral, pressures=None):
    if pressures is None:
        pressures = np.linspace(1.0e9, 2.0e9, 8)
    rows = []
    for pressure in pressures:
        mineral.set_state(pressure, mineral.params.T_0)
        rows.append([pressure, mineral.temperature, mineral.molar_volume])
    return np.array(rows)


def test_fixed_tolerances_and_standard_state():
    material = bm.minerals.HP_2011_ds62.fo()
    material.set_state(2.0e9, 1500.0)
    flags = [
        "H",
        "energy",
        "molar_internal_energy",
        "helmholtz",
        "molar_helmholtz",
        "gibbs",
        "molar_gibbs",
        "enthalpy",
        "molar_enthalpy",
        "P",
        "pressure",
    ]
    assert bm.eos_fitting.default_mle_tolerances is default_mle_tolerances
    assert_array_equal(default_mle_tolerances(material, flags), [1.0] * 9 + [100.0] * 2)
    assert material.pressure == 1.0e5
    assert material.temperature == 300.0


def test_relative_tolerances_preserve_order_and_duplicates():
    material = bm.minerals.SLB_2011.forsterite()
    flags = ["V", "T", "S", "alpha", "molar_volume", "T", "C_p", "K_T", "rho"]
    actual = default_mle_tolerances(material, flags)
    properties = [
        "molar_volume",
        "temperature",
        "molar_entropy",
        "thermal_expansivity",
        "molar_volume",
        "temperature",
        "molar_heat_capacity_p",
        "isothermal_bulk_modulus_reuss",
        "density",
    ]
    expected = [1.0e-5 * getattr(material, name) for name in properties]
    assert_array_equal(actual, expected)


def test_empty_flags_still_set_standard_state():
    material = bm.minerals.HP_2011_ds62.fo()
    material.set_state(2.0e9, 1500.0)
    actual = default_mle_tolerances(material, [])
    assert actual.shape == (0,)
    assert actual.dtype == np.float64
    assert material.pressure == 1.0e5
    assert material.temperature == 300.0


def test_nonpositive_relative_tolerance_raises():
    # Python's fixed energy flag list does not include the F alias.
    material = bm.minerals.HP_2011_ds62.fo()
    with pytest.raises(ValueError, match="All MLE tolerances should be > 0"):
        default_mle_tolerances(material, ["F"])
    assert material.molar_helmholtz < 0.0


@pytest.mark.parametrize(
    "dataset,name,composition",
    [
        ("HP_2011_ds62", "fo", None),
        ("SLB_2011", "forsterite", None),
        ("SLB_2011", "mg_fe_olivine", [0.8, 0.2]),
    ],
)
def test_optional_python_burnman_parity(dataset, name, composition):
    reference = pytest.importorskip("burnman")
    from burnman.optimize.eos_fitting import default_mle_tolerances as expected

    native_material = getattr(getattr(bm.minerals, dataset), name)()
    python_material = getattr(getattr(reference.minerals, dataset), name)()
    if composition is not None:
        native_material.set_composition(composition)
        python_material.set_composition(composition)
    flags = ["P", "H", "V", "T", "S", "alpha", "C_p", "K_T", "rho", "molar_volume"]
    assert_allclose(
        default_mle_tolerances(native_material, flags),
        expected(python_material, flags),
        rtol=1.0e-8,
        atol=0.0,
    )


@pytest.mark.parametrize("with_priors", [False, True])
def test_fit_ptv_exact_data_from_python_baseline(with_priors):
    mineral = bm.minerals.HP_2011_ds62.fo()
    data = ptv_data(mineral)
    params = ["V_0", "K_0", "Kprime_0"]
    original = np.array(
        [mineral.params.V_0, mineral.params.K_0, mineral.params.Kprime_0]
    )
    kwargs = {}
    if with_priors:
        kwargs.update(
            param_priors=original,
            param_prior_inv_cov_matrix=np.diag(
                1.0 / np.array([original[0] / 10.0, 1.0e9, 1.0]) ** 2
            ),
        )
    model = fit_PTV_data(mineral, params, data, verbose=False, **kwargs)
    assert isinstance(model, bm.nonlinear_fitting.NonLinearModel)
    assert model.m is mineral
    assert model.fit_params == params
    assert model.flags == ["V"] * len(data)
    assert_array_equal(model.popt, original)
    assert_array_equal(model.pcov, np.zeros((3, 3)))
    assert_array_equal(
        model.data_covariances, np.tile(np.diag([1.0, 0.0, 0.0]), (8, 1, 1))
    )
    assert model.dof == 5
    assert model.WSS == 0.0


def test_fit_ptv_recovers_parameters_and_updates_supplied_mineral():
    mineral = bm.minerals.HP_2011_ds62.fo()
    data = ptv_data(mineral, np.linspace(1.0e5, 20.0e9, 15))
    truth = np.array([mineral.params.V_0, mineral.params.K_0, mineral.params.Kprime_0])
    params = mineral.params
    params.V_0 *= 1.001
    params.K_0 *= 0.97
    params.Kprime_0 += 0.1
    mineral = bm.Mineral(params)
    model = fit_PTV_data(mineral, ["V_0", "K_0", "Kprime_0"], data, verbose=False)
    assert_allclose(model.popt, truth, rtol=1.0e-7)
    assert_array_equal(
        model.popt, [mineral.params.V_0, mineral.params.K_0, mineral.params.Kprime_0]
    )
    assert_allclose(model.function(data[0]), data[0], rtol=1.0e-12)
    bm.nonlinear_fitting.calculate_jacobian(model)
    assert_array_equal(model.popt, model.get_params())
    positions, residuals, weights = bm.nonlinear_fitting.find_mle(model)
    assert_array_equal(model.data_mle, positions)
    assert_array_equal(model.weighted_residuals, residuals)
    assert_array_equal(model.weights, weights)
    estimate = bm.nonlinear_fitting.mle_estimate(
        model, data[0], positions[0], model.data_covariances[0], "V"
    )
    assert np.isfinite(estimate[2])
    # Direct use of the native model in the public core fitter needs no adapter.
    bm.nonlinear_least_squares_fit(model, param_tolerance=1.0e-5)


def test_ptv_parameter_bounds_from_python_baseline():
    mineral = bm.minerals.HP_2011_ds62.fo()
    data = ptv_data(mineral, np.linspace(0.0, 10.0e9, 8))
    data[0, 2] *= 1.01
    data[-1, 2] *= 0.99
    bounds = np.array([[0.0, np.inf], [0.0, np.inf], [3.0, 4.0]])
    model = fit_PTV_data(
        mineral, ["V_0", "K_0", "Kprime_0"], data, bounds=bounds, verbose=False
    )
    assert model.popt[2] == 3.0
    assert_array_equal(model.bounds, bounds)


def test_mineral_fit_flattens_coefficients_and_invalidates_cached_properties():
    mineral = bm.minerals.HP_2011_ds62.fo()
    data = ptv_data(mineral)
    p = mineral.params
    model = MineralFit(
        mineral,
        data,
        np.tile(np.diag([1.0, 0.0, 0.0]), (8, 1, 1)),
        ["V"] * 8,
        ["V_0", "Cp"],
        np.full(8, 1.0e-10),
    )
    values = np.array([p.V_0, p.Cp.a, p.Cp.b, p.Cp.c, p.Cp.d])
    assert_array_equal(model.get_params(), values)
    assert_array_equal(model.delta_params, values * 1.0e-5 + 1.0e-10)
    assert (model.delta_params < 0.0).any()
    mineral.set_state(1.0e5, 300.0)
    old_volume = mineral.molar_volume
    values[0] *= 1.001
    model.set_params(values)
    assert mineral.molar_volume != old_volume
    assert_array_equal(model.get_params(), values)
    x = model.function(data[0], "V")
    n = np.array(
        [
            -1.0,
            mineral.thermal_expansivity * mineral.isothermal_bulk_modulus_reuss,
            -mineral.isothermal_bulk_modulus_reuss / mineral.molar_volume,
        ]
    )
    assert_allclose(model.normal(x, "V"), n / np.linalg.norm(n), rtol=1.0e-14)


@pytest.mark.parametrize(
    "dataset,name,max_pressure",
    [
        ("HP_2011_ds62", "fo", 2.0e9),
        ("SLB_2011", "forsterite", 100.0e9),
    ],
)
def test_ptv_bad_model_data_raise_as_in_python_baseline(dataset, name, max_pressure):
    mineral = getattr(getattr(bm.minerals, dataset), name)()
    data = ptv_data(mineral, np.linspace(1.0e9, 2.0e9, 8))
    volumes = np.linspace(data[0, 2], data[-1, 2], 8)
    data[:, 0] = np.linspace(1.0e9, max_pressure, 8)
    data[:, 2] = volumes
    with pytest.raises((ValueError, RuntimeError)):
        fit_PTV_data(mineral, ["V_0", "K_0", "Kprime_0"], data, verbose=False)


@pytest.mark.parametrize("case", ["weighted", "prior", "bounded"])
def test_optional_ptv_fit_matches_python(case):
    reference = pytest.importorskip("burnman")
    from burnman.optimize.eos_fitting import fit_PTV_data as reference_fit

    native = bm.minerals.HP_2011_ds62.fo()
    python = reference.minerals.HP_2011_ds62.fo()
    data = ptv_data(native, np.linspace(1.0e5, 10.0e9, 10))
    data[:, 2] *= 1.0 + np.array([1, -1, 2, -2, 1, -1, 2, -2, 1, -1]) * 1.0e-4
    kwargs = {"verbose": False}
    if case == "weighted":
        cov = np.zeros((len(data), 3, 3))
        cov[:, 0, 0] = (data[:, 0] * 0.01) ** 2
        cov[:, 2, 2] = (data[:, 2] * 1.0e-4) ** 2
        cov[:, 0, 2] = cov[:, 2, 0] = 0.2 * np.sqrt(cov[:, 0, 0] * cov[:, 2, 2])
        kwargs["data_covariances"] = cov
    if case == "prior":
        kwargs.update(
            param_priors=[native.params.V_0, native.params.K_0, 4.0],
            param_prior_inv_cov_matrix=np.diag([1.0e12, 1.0e-18, 1.0e15]),
        )
    if case == "bounded":
        kwargs["bounds"] = [[0.0, np.inf], [0.0, np.inf], [3.0, 4.0]]
    actual = fit_PTV_data(native, ["V_0", "K_0", "Kprime_0"], data, **kwargs)
    expected = reference_fit(python, ["V_0", "K_0", "Kprime_0"], data, **kwargs)
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
