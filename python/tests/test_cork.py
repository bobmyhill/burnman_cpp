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

"""CORK fluid properties, parameters and Python BurnMan comparisons."""

import copy
import math

import numpy as np
import pytest

import burnman_cpp as bm


def co2_params():
    return dict(
        name="carbon dioxide",
        equation_of_state="cork",
        formula={"C": 1.0, "O": 2.0},
        molar_mass=0.0440095,
        n=3.0,
        cork_params=[
            [54.5963, -8.63920],
            [0.918301],
            [-0.0330558, 0.00230524],
            [0.000693054, -0.0000838293],
        ],
        cork_T=304.2,
        cork_P=0.0738e8,
        H_0=-393510.0,
        S_0=213.7,
        Cp=[87.8, -0.002644, 706400.0, -998.9],
    )


PROPERTIES = (
    "molar_volume",
    "density",
    "isothermal_bulk_modulus_reuss",
    "thermal_expansivity",
    "molar_heat_capacity_p",
    "molar_heat_capacity_v",
    "molar_gibbs",
    "molar_entropy",
    "molar_enthalpy",
    "molar_helmholtz",
    "molar_internal_energy",
    "isentropic_bulk_modulus_reuss",
    "grueneisen_parameter",
    "shear_modulus",
)


def assert_properties_match(material, reference):
    energy_properties = {
        "molar_gibbs",
        "molar_enthalpy",
        "molar_helmholtz",
        "molar_internal_energy",
    }
    energy_scale = (
        abs(reference.molar_gibbs)
        + abs(reference.temperature * reference.molar_entropy)
        + abs(reference.pressure * reference.molar_volume)
    )
    for prop in PROPERTIES:
        # Derived energies can cancel to zero at the reference state.
        atol = (
            8.0 * np.finfo(float).eps * energy_scale
            if prop in energy_properties
            else 1e-15
        )
        np.testing.assert_allclose(
            getattr(material, prop),
            getattr(reference, prop),
            rtol=3e-11,
            atol=atol,
            err_msg=f"{reference.name}: {prop}",
        )


@pytest.mark.parametrize("method", ["cork", "CORK", bm.EOSType.CORK])
def test_cork_parameters_and_properties(method):
    params = dict(co2_params(), equation_of_state=method)
    original = copy.deepcopy(params)
    material = bm.Mineral(params)
    assert params == original
    validated = material.params
    assert validated.equation_of_state == bm.EOSType.CORK
    assert validated.T_0 == 298.15
    assert validated.P_0 == 0.0
    assert validated.dcorkdT == pytest.approx(
        [
            -8.63920 * 304.2**1.5 / 0.0738e8,
            0.0,
            0.00230524 / (0.0738e8) ** 1.5,
            -0.0000838293 / (0.0738e8) ** 2.0,
        ]
    )
    pressure, temperature = 1.0e8, 800.0
    material.set_state(pressure, temperature)
    assert material.molar_volume == pytest.approx(8.949082367226386e-5, rel=2e-12)
    assert material.molar_gibbs == pytest.approx(-529151.9636299624, rel=2e-12)
    assert material.shear_modulus == 0.0
    assert material.shear_wave_velocity == 0.0
    assert material.p_wave_velocity == pytest.approx(
        math.sqrt(material.isentropic_bulk_modulus_reuss / material.density)
    )
    eos = bm.make_eos(method)
    assert eos.compute_volume(pressure, temperature, validated) == material.molar_volume
    assert eos.compute_molar_heat_capacity_p(
        pressure, temperature, material.molar_volume, validated
    ) == pytest.approx(material.molar_heat_capacity_p)


@pytest.mark.parametrize("missing", ["cork_params", "cork_T", "cork_P", "Cp"])
def test_cork_required_parameters(missing):
    params = co2_params()
    del params[missing]
    with pytest.raises(ValueError, match=missing):
        bm.Mineral(params)


def test_cork_optional_reference_energies():
    params = co2_params()
    del params["H_0"], params["S_0"]
    material = bm.Mineral(params)
    assert math.isnan(material.params.H_0)
    assert math.isnan(material.params.S_0)
    material.set_state(1.0e8, 800.0)
    assert material.molar_volume > 0.0
    assert material.isothermal_bulk_modulus_reuss > 0.0
    assert material.molar_heat_capacity_p > 0.0
    assert math.isnan(material.molar_gibbs)
    assert math.isnan(material.molar_entropy)


@pytest.mark.parametrize("form", ["typed", "four_by_two", "numpy"])
def test_cork_coefficient_forms(form):
    params = co2_params()
    expected = bm.Mineral(params)
    if form == "typed":
        params["cork_params"] = bm.CorkParams(
            54.5963,
            -8.63920,
            0.918301,
            -0.0330558,
            0.00230524,
            0.000693054,
            -0.0000838293,
        )
    else:
        params["cork_params"][1].append(0.0)
        if form == "numpy":
            params["cork_params"] = np.array(params["cork_params"])
    actual = bm.Mineral(params)
    for material in (expected, actual):
        material.set_state(1.0e8, 800.0)
    for prop in PROPERTIES:
        assert getattr(actual, prop) == pytest.approx(
            getattr(expected, prop), rel=2e-12
        )


@pytest.mark.parametrize(
    "coefficients",
    [[], [[1.0, 2.0]] * 3, [[1.0]] * 4, [[1.0, 2.0, 3.0]] * 4],
)
def test_cork_rejects_malformed_coefficient_rows(coefficients):
    with pytest.raises(ValueError, match="cork_params"):
        bm.Mineral(dict(co2_params(), cork_params=coefficients))


def test_cork_thermodynamic_derivatives():
    eos = bm.make_eos("cork")
    params = bm.Mineral(co2_params()).params
    pressure, temperature = 1.0e8, 800.0
    volume = eos.compute_volume(pressure, temperature, params)
    dp, dt = 1000.0, 0.01
    dvdp = (
        eos.compute_volume(pressure + dp, temperature, params)
        - eos.compute_volume(pressure - dp, temperature, params)
    ) / (2.0 * dp)
    dvdt = (
        eos.compute_volume(pressure, temperature + dt, params)
        - eos.compute_volume(pressure, temperature - dt, params)
    ) / (2.0 * dt)
    assert eos.compute_isothermal_bulk_modulus_reuss(
        pressure, temperature, volume, params
    ) == pytest.approx(-volume / dvdp, rel=2e-8)
    assert eos.compute_thermal_expansivity(
        pressure, temperature, volume, params
    ) == pytest.approx(dvdt / volume, rel=2e-8)
    material = bm.Mineral(params)
    material.set_state(pressure, temperature)
    cp, entropy = material.molar_heat_capacity_p, material.molar_entropy
    material.set_state(pressure, temperature + dt)
    gp, sp = material.molar_gibbs, material.molar_entropy
    material.set_state(pressure, temperature - dt)
    gm, sm = material.molar_gibbs, material.molar_entropy
    assert cp == pytest.approx(temperature * (sp - sm) / (2.0 * dt), rel=2e-8)
    assert entropy == pytest.approx(-(gp - gm) / (2.0 * dt), rel=2e-8)
    material.set_state(pressure + dp, temperature)
    gp = material.molar_gibbs
    material.set_state(pressure - dp, temperature)
    gm = material.molar_gibbs
    assert volume == pytest.approx((gp - gm) / (2.0 * dp), rel=2e-8)


@pytest.mark.parametrize("name", ["CO2", "CH4", "O2", "H2", "S2", "H2S"])
@pytest.mark.parametrize("pressure", [1.0e5, 1.0e8, 1.0e9])
@pytest.mark.parametrize("temperature", [298.15, 800.0, 1600.0])
def test_cork_python_reference(name, pressure, temperature):
    fluids = pytest.importorskip("burnman.minerals.HP_2011_fluids")
    from burnman_cpp.adapters import from_burnman

    reference = getattr(fluids, name)()
    material = from_burnman(reference)
    reference.set_state(pressure, temperature)
    material.set_state(pressure, temperature)
    assert_properties_match(material, reference)


@pytest.mark.parametrize("critical_temperature", [0.0, 304.2])
@pytest.mark.parametrize("derivative_override", [False, True])
def test_cork_custom_reference_and_derivatives(
    critical_temperature, derivative_override
):
    burnman = pytest.importorskip("burnman")
    from burnman_cpp.adapters import from_burnman

    params = dict(co2_params(), T_0=350.0, P_0=2.0e7, cork_T=critical_temperature)
    if derivative_override:
        params["dcorkdT"] = [-0.008, 0.0, 2.0e-14, -1.0e-18]
    reference = burnman.Mineral(params)
    material = from_burnman(reference)
    assert material.params.T_0 == 350.0
    assert material.params.P_0 == 2.0e7
    assert material.params.dcorkdT == pytest.approx(reference.params["dcorkdT"])
    for pressure, temperature in ((5.0e7, 450.0), (2.0e8, 1100.0)):
        reference.set_state(pressure, temperature)
        material.set_state(pressure, temperature)
        assert_properties_match(material, reference)
