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

"""Modular MGD components and native/Python BurnMan thermodynamic comparisons."""

import gc

import numpy as np
import pytest

import burnman_cpp as bm
from burnman_cpp import debye_temperature_models as dt
from burnman_cpp import anharmonic_prefactor_models as ap
from burnman_cpp import anharmonic_thermal_models as at


def parameters(reference="bm3", debye="SLB"):
    return dict(
        equation_of_state="modular_mgd",
        reference_eos=reference,
        debye_temperature_model=getattr(dt, debye)(),
        V_0=4.3603e-5,
        K_0=1.279555e11,
        Kprime_0=4.21796,
        F_0=-2055403.0,
        P_0=1.0e5,
        T_0=298.15,
        n=7.0,
        molar_mass=0.1406931,
        Debye_0=1000.0,
        grueneisen_0=1.2,
        q_0=1.1,
        G_0=0.0,
        Gprime_0=0.0,
        Kprime_inf=2.0,
        Kdprime_0=-4.0e-11,
        c_1=0.2,
        c_2=0.1,
        q_1=1.0,
        q_2=2.0,
    )


def reference_parameters(native):
    pytest.importorskip("burnman")
    from burnman.eos.helper import create
    from burnman.eos import debye_temperature_models as py_dt
    from burnman.eos import anharmonic_prefactor_models as py_ap
    from burnman.eos import anharmonic_thermal_models as py_at

    result = dict(native)
    result["reference_eos"] = create(result["reference_eos"])
    for key, module in (
        ("debye_temperature_model", py_dt),
        ("anharmonic_prefactor_model", py_ap),
        ("anharmonic_thermal_model", py_at),
    ):
        if key in result:
            result[key] = getattr(module, type(result[key]).__name__)()
    return result


@pytest.mark.parametrize("reference", ["bm3", "morse", "rkprime"])
def test_modular_mgd_reference_state_and_component_lifetimes(reference):
    p = parameters(reference)
    mineral = bm.Mineral(p)
    del p
    gc.collect()
    mineral.set_state(1.0e5, 298.15)
    assert mineral.molar_volume == pytest.approx(mineral.params.V_0, rel=2.0e-13)
    assert mineral.molar_helmholtz == pytest.approx(mineral.params.F_0)
    assert mineral.isothermal_bulk_modulus_reuss == pytest.approx(mineral.params.K_0)
    assert mineral.params.reference_eos is not None
    assert mineral.params.debye_temperature_model.value(1.0, mineral.params) == 1000.0


@pytest.mark.parametrize(
    "key", ["reference_eos", "debye_temperature_model", "T_0", "n", "Debye_0"]
)
def test_modular_mgd_required_fields(key):
    p = parameters()
    del p[key]
    with pytest.raises(ValueError):
        bm.Mineral(p)


def test_modular_mgd_optional_component_validation():
    with pytest.raises(ValueError, match="gel"):
        bm.Mineral(dict(parameters(), bel_0=0.005))
    with pytest.raises(ValueError, match="anharmonic_prefactor_model"):
        bm.Mineral(dict(parameters(), anharmonic_thermal_model=at.Pade()))
    with pytest.raises(ValueError, match="c_anh"):
        bm.Mineral(
            dict(parameters(), equation_of_state="modular_mgd_with_anharmonicity")
        )
    with pytest.raises(ValueError, match="cannot itself be modular"):
        bm.Mineral(dict(parameters(), reference_eos="modular_mgd"))


@pytest.mark.parametrize("reference", ["bm3", "vinet", "spock"])
@pytest.mark.parametrize("debye", ["SLB", "PowerLawGammaSimple", "PowerLawGamma"])
@pytest.mark.parametrize("temperature", [0.0, 298.15, 2000.0])
@pytest.mark.parametrize("ratio", [0.8, 1.0])
def test_modular_mgd_python_reference(reference, debye, temperature, ratio):
    py_eos = pytest.importorskip(
        "burnman.eos.modular_mie_grueneisen_debye"
    ).ModularMGD()
    p = parameters(reference, debye)
    ref = reference_parameters(p)
    py_eos.validate_parameters(ref)
    volume = ratio * p["V_0"]
    pressure = py_eos.pressure(temperature, volume, ref)
    native = bm.Mineral(p)
    native.set_state(pressure, temperature)
    assert native.molar_volume == pytest.approx(volume, rel=2.0e-12)
    eos = bm.make_eos("modular_mgd")
    # The examples also evaluate properties directly along fixed-volume paths.
    for name in (
        "molar_heat_capacity_v",
        "molar_heat_capacity_p",
        "thermal_expansivity",
        "grueneisen_parameter",
        "isothermal_bulk_modulus_reuss",
    ):
        direct = getattr(eos, "compute_" + name)(
            pressure, temperature, volume, native.params
        )
        assert direct == pytest.approx(getattr(native, name), rel=2.0e-11, abs=2.0e-10)
    for actual, expected in (
        (
            native.isothermal_bulk_modulus_reuss,
            py_eos.isothermal_bulk_modulus_reuss(pressure, temperature, volume, ref),
        ),
        (
            native.molar_helmholtz,
            py_eos._helmholtz_energy(pressure, temperature, volume, ref),
        ),
        (native.molar_entropy, py_eos.entropy(pressure, temperature, volume, ref)),
        (
            native.molar_heat_capacity_v,
            py_eos._molar_heat_capacity_v(pressure, temperature, volume, ref),
        ),
        (
            native.molar_heat_capacity_p,
            py_eos.molar_heat_capacity_p(pressure, temperature, volume, ref),
        ),
        (
            native.thermal_expansivity,
            py_eos.thermal_expansivity(pressure, temperature, volume, ref),
        ),
        (
            native.grueneisen_parameter,
            py_eos._grueneisen_parameter(pressure, temperature, volume, ref),
        ),
    ):
        assert np.isfinite(actual)
        assert actual == pytest.approx(expected, rel=2.0e-11, abs=2.0e-10)
    assert bm.make_eos("modular_mgd").compute_pressure(
        temperature, volume, native.params
    ) == pytest.approx(pressure, rel=2.0e-11, abs=1.0e-3)


@pytest.mark.parametrize("thermal", [None, "Pade", "LogNormal"])
@pytest.mark.parametrize("prefactor", ["PowerLaw", "Sigmoid"])
def test_modular_mgd_electronic_and_additive_anharmonicity(thermal, prefactor):
    py_eos = pytest.importorskip(
        "burnman.eos.modular_mie_grueneisen_debye"
    ).ModularMGD()
    p = dict(parameters(), bel_0=0.005, gel=1.5)
    if thermal is not None:
        p.update(
            anharmonic_thermal_model=getattr(at, thermal)(),
            anharmonic_prefactor_model=getattr(ap, prefactor)(),
            a_anh=-0.2,
            m_anh=3.0,
            b_anh=0.2,
            c_anh=2.0,
            mu_anh=0.2,
            sigma_anh=1.2,
        )
    ref = reference_parameters(p)
    py_eos.validate_parameters(ref)
    t, v = 2000.0, 0.8 * p["V_0"]
    pressure = py_eos.pressure(t, v, ref)
    native = bm.Mineral(p)
    native.set_state(pressure, t)
    assert native.molar_volume == pytest.approx(v, rel=2.0e-12)
    for actual, expected in (
        (native.molar_helmholtz, py_eos._helmholtz_energy(pressure, t, v, ref)),
        (
            native.isothermal_bulk_modulus_reuss,
            py_eos.isothermal_bulk_modulus_reuss(pressure, t, v, ref),
        ),
        (
            native.molar_heat_capacity_v,
            py_eos._molar_heat_capacity_v(pressure, t, v, ref),
        ),
        (
            native.molar_heat_capacity_p,
            py_eos.molar_heat_capacity_p(pressure, t, v, ref),
        ),
        (native.molar_entropy, py_eos.entropy(pressure, t, v, ref)),
        (native.thermal_expansivity, py_eos.thermal_expansivity(pressure, t, v, ref)),
    ):
        assert actual == pytest.approx(expected, rel=2.0e-11, abs=1.0e-10)


@pytest.mark.parametrize("temperature", [298.15, 1000.0, 2000.0])
@pytest.mark.parametrize("electronic", [False, True])
def test_volume_rescaled_anharmonicity_matches_python(temperature, electronic):
    py_eos = pytest.importorskip(
        "burnman.eos.modular_mie_grueneisen_debye"
    ).ModularMGDWithAnharmonicity()
    p = dict(
        parameters(), equation_of_state="modular_mgd_with_anharmonicity", c_anh=0.05
    )
    if electronic:
        p.update(bel_0=0.005, gel=1.5)
    ref = reference_parameters(p)
    py_eos.validate_parameters(ref)
    v = 0.9 * p["V_0"]
    pressure = py_eos.pressure(temperature, v, ref)
    native = bm.Mineral(p)
    native.set_state(pressure, temperature)
    assert native.molar_volume == pytest.approx(v, rel=2.0e-11)
    # This model uses finite differences in both implementations; their second
    # derivatives amplify floating-point differences in the harmonic model.
    for actual, expected in (
        (
            native.molar_helmholtz,
            py_eos._helmholtz_energy(pressure, temperature, v, ref),
        ),
        (
            native.isothermal_bulk_modulus_reuss,
            py_eos.isothermal_bulk_modulus_reuss(pressure, temperature, v, ref),
        ),
        (native.molar_entropy, py_eos.entropy(pressure, temperature, v, ref)),
        (
            native.molar_heat_capacity_v,
            py_eos._molar_heat_capacity_v(pressure, temperature, v, ref),
        ),
        (
            native.molar_heat_capacity_p,
            py_eos.molar_heat_capacity_p(pressure, temperature, v, ref),
        ),
        (
            native.thermal_expansivity,
            py_eos.thermal_expansivity(pressure, temperature, v, ref),
        ),
    ):
        assert actual == pytest.approx(expected, rel=3.0e-6, abs=1.0e-8)


@pytest.mark.parametrize("debye", ["SLB", "PowerLawGammaSimple", "PowerLawGamma"])
def test_debye_component_derivatives_against_python(debye):
    native = getattr(dt, debye)()
    p = parameters(debye=debye)
    ref = reference_parameters(p)
    py_model = ref["debye_temperature_model"]
    typed = bm.mineral_params(p)
    for ratio in [0.7, 1.0, 1.1]:
        for method in ["value", "dVrel", "dVrel2"]:
            assert getattr(native, method)(ratio, typed) == pytest.approx(
                getattr(py_model, method)(ratio, ref), rel=2.0e-12
            )


@pytest.mark.parametrize("thermal", ["Pade", "LogNormal"])
def test_anharmonic_thermal_components_against_python(thermal):
    py_at = pytest.importorskip("burnman.eos.anharmonic_thermal_models")
    native, reference = getattr(at, thermal)(), getattr(py_at, thermal)()
    p = bm.mineral_params(mu_anh=0.2, sigma_anh=1.2)
    for temperature in [0.1, 300.0, 2000.0]:
        for name in (
            "helmholtz_energy",
            "entropy",
            "heat_capacity",
            "dhelmholtz_dTheta",
            "d2helmholtz_dTheta2",
            "dentropy_dTheta",
        ):
            method = "nondimensional_" + name
            expected = getattr(reference, method)(
                temperature, 1000.0, dict(mu_anh=0.2, sigma_anh=1.2)
            )
            assert getattr(native, method)(temperature, 1000.0, p) == pytest.approx(
                expected, rel=3.0e-11, abs=1.0e-12
            )


@pytest.mark.parametrize("method", ["modular_mgd", "modular_mgd_with_anharmonicity"])
@pytest.mark.parametrize("reference_eos", ["bm3", "murnaghan"])
def test_burnman_adapter_for_modular_mgd(method, reference_eos):
    reference = pytest.importorskip("burnman")
    from burnman_cpp.adapters import from_burnman

    p = dict(parameters(reference_eos), equation_of_state=method, c_anh=0.05)
    if reference_eos == "murnaghan":
        # Python Murnaghan defines its reference volume at zero pressure.
        p["P_0"] = 0.0
    original = reference.Mineral(reference_parameters(p))
    converted = from_burnman(original)
    converted.set_state(p["P_0"], 298.15)
    assert converted.molar_volume == pytest.approx(p["V_0"], rel=2.0e-12)
    assert converted.molar_helmholtz == pytest.approx(p["F_0"])
