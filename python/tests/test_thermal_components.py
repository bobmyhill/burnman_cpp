# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Low-level thermal APIs and their use in native MGD models."""

import importlib

import numpy as np
import pytest

import burnman_cpp as bm
from burnman_cpp.eos import debye, einstein, bukowinski_electronic as el
from burnman_cpp.eos import anharmonic_prefactor_models as ap
from burnman_cpp.eos import anharmonic_thermal_models as at
from burnman_cpp.eos.anharmonic_debye import AnharmonicDebye
from test_modular_mgd import parameters, reference_parameters

LATTICE_METHODS = (
    "thermal_energy",
    "molar_heat_capacity_v",
    "helmholtz_energy",
    "entropy",
    "dmolar_heat_capacity_v_dT",
    "dhelmholtz_dTheta",
    "d2helmholtz_dTheta2",
    "dentropy_dTheta",
)


def anharmonic_parameters(thermal="Pade", prefactor="PowerLaw"):
    return dict(
        parameters("bm3", "PowerLawGammaSimple"),
        P_0=0.0,
        anharmonic_prefactor_model=getattr(ap, prefactor)(),
        anharmonic_thermal_model=getattr(at, thermal)(),
        a_anh=-0.2,
        m_anh=3.0,
        b_anh=0.2,
        c_anh=2.0,
        mu_anh=0.2,
        sigma_anh=1.2,
    )


@pytest.mark.parametrize("model", [debye, einstein], ids=["debye", "einstein"])
def test_lattice_derivatives_and_broadcasting(model):
    t, theta, n = 500.0, 700.0, 0.5
    h = 0.01
    assert model.dhelmholtz_dTheta(t, theta, n) == pytest.approx(
        (
            model.helmholtz_energy(t, theta + h, n)
            - model.helmholtz_energy(t, theta - h, n)
        )
        / (2 * h),
        rel=2e-7,
    )
    assert model.d2helmholtz_dTheta2(t, theta, n) == pytest.approx(
        (
            model.dhelmholtz_dTheta(t, theta + h, n)
            - model.dhelmholtz_dTheta(t, theta - h, n)
        )
        / (2 * h),
        rel=2e-7,
    )
    assert model.dentropy_dTheta(t, theta, n) == pytest.approx(
        (model.entropy(t, theta + h, n) - model.entropy(t, theta - h, n)) / (2 * h),
        rel=2e-7,
    )
    temperatures = np.array([0.0, 100.0, 300.0, 2000.0])
    theta_grid = np.array([[400.0], [1000.0]])
    for name in LATTICE_METHODS:
        fn = getattr(model, name)
        values = fn(temperatures, theta_grid, n)
        assert values.shape == (2, 4)
        for i, theta in enumerate(theta_grid[:, 0]):
            np.testing.assert_allclose(
                values[i], [fn(t, theta, n) for t in temperatures], rtol=1e-14, atol=0.0
            )
        assert fn(0.0, 700.0, n) == 0.0
        assert np.isfinite(fn(1e-4, 700.0, n))
        assert np.isfinite(fn(1e8, 1.0, n))
    with pytest.raises(ValueError):
        model.thermal_energy(-1.0, 700.0, n)
    with pytest.raises(ValueError):
        model.dhelmholtz_dTheta(300.0, 0.0, n)
    with pytest.raises(ValueError):
        model.entropy(300.0, 700.0, -1.0)


def test_debye_function_and_component_import_paths():
    assert debye.debye_fn(0.0) == debye.debye_fn_cheb(0.0) == 1.0
    np.testing.assert_allclose(
        debye.debye_fn([0.1, 1.0, 10.0]),
        debye.debye_fn_cheb([0.1, 1.0, 10.0]),
        rtol=1e-12,
    )
    with pytest.raises(ValueError):
        debye.debye_fn_cheb(-1.0)
    assert (
        importlib.import_module("burnman_cpp.eos.debye_temperature_models")
        is bm.debye_temperature_models
    )
    assert (
        importlib.import_module("burnman_cpp.eos.anharmonic_prefactor_models")
        is bm.anharmonic_prefactor_models
    )
    assert (
        importlib.import_module("burnman_cpp.eos.anharmonic_thermal_models")
        is bm.anharmonic_thermal_models
    )


def test_electronic_component_derivatives_and_arrays():
    t, v, t0, v0, bel, gel = 1700.0, 0.8e-5, 300.0, 1e-5, 0.005, 1.5
    dv = v * 1e-5
    dt = 0.02

    def f(T, V):
        return el.helmholtz(T, V, t0, v0, bel, gel)

    assert el.pressure(t, v, t0, v0, bel, gel) == pytest.approx(
        -(f(t, v + dv) - f(t, v - dv)) / (2 * dv), rel=2e-7
    )
    assert el.entropy(t, v, v0, bel, gel) == pytest.approx(
        -(f(t + dt, v) - f(t - dt, v)) / (2 * dt), rel=2e-7
    )
    assert el.aKT(t, v, v0, bel, gel) == pytest.approx(
        (
            el.pressure(t + dt, v, t0, v0, bel, gel)
            - el.pressure(t - dt, v, t0, v0, bel, gel)
        )
        / (2 * dt),
        rel=2e-7,
    )
    assert el.KToverV(t, v, t0, v0, bel, gel) == pytest.approx(
        -(
            el.pressure(t, v + dv, t0, v0, bel, gel)
            - el.pressure(t, v - dv, t0, v0, bel, gel)
        )
        / (2 * dv),
        rel=2e-7,
    )
    np.testing.assert_allclose(
        el.CVoverT([v0, 0.8 * v0], v0, bel, gel),
        bel * np.array([1.0, 0.8]) ** gel,
        rtol=1e-14,
    )
    with pytest.raises(ValueError):
        el.entropy(t, 0.0, v0, bel, gel)


@pytest.mark.parametrize("thermal", ["Pade", "LogNormal"])
@pytest.mark.parametrize("prefactor", ["PowerLaw", "Sigmoid"])
def test_modular_mgd_adds_the_standalone_anharmonic_component(thermal, prefactor):
    p = anharmonic_parameters(thermal, prefactor)
    AnharmonicDebye.validate_parameters(p)
    typed = bm.mineral_params(p)
    bare = dict(p)
    del bare["anharmonic_prefactor_model"], bare["anharmonic_thermal_model"]
    bare = bm.mineral_params(bare)
    model = bm.make_eos("modular_mgd")
    t, v = 1500.0, 0.8 * p["V_0"]
    for method, name in (
        ("compute_helmholtz_free_energy", "helmholtz_energy"),
        ("compute_entropy", "entropy"),
        ("compute_molar_heat_capacity_v", "heat_capacity_v"),
        ("compute_isothermal_bulk_modulus_reuss", "isothermal_bulk_modulus"),
    ):
        difference = getattr(model, method)(0.0, t, v, typed) - getattr(model, method)(
            0.0, t, v, bare
        )
        assert difference == pytest.approx(
            getattr(AnharmonicDebye, name)(t, v, p), rel=2e-9, abs=1e-6
        )
    difference = model.compute_pressure(t, v, typed) - model.compute_pressure(
        t, v, bare
    )
    assert difference == pytest.approx(
        AnharmonicDebye.pressure(t, v, p), rel=2e-9, abs=1e-4
    )
    alpha_kt = lambda params: model.compute_thermal_expansivity(
        0.0, t, v, params
    ) * model.compute_isothermal_bulk_modulus_reuss(0.0, t, v, params)
    assert alpha_kt(typed) - alpha_kt(bare) == pytest.approx(
        AnharmonicDebye.dSdV(t, v, p), rel=2e-9, abs=1e-6
    )
    for name in ("helmholtz_energy", "pressure", "isothermal_bulk_modulus"):
        assert getattr(AnharmonicDebye, name)(p["T_0"], v, p) == 0.0
    assert AnharmonicDebye.entropy(t, v, typed) == AnharmonicDebye.entropy(t, v, p)
    missing = dict(p)
    del missing["anharmonic_prefactor_model"]
    with pytest.raises(ValueError, match="anharmonic_prefactor_model"):
        AnharmonicDebye.validate_parameters(missing)


@pytest.mark.parametrize("module", ["debye", "einstein"])
def test_optional_lattice_python_reference(module):
    reference = pytest.importorskip("burnman.eos." + module)
    native = importlib.import_module("burnman_cpp.eos." + module)
    for t in (0.0, 100.0, 300.0, 2000.0):
        for n in (0.5, 2.0):
            for method in LATTICE_METHODS:
                assert getattr(native, method)(t, 700.0, n) == pytest.approx(
                    getattr(reference, method)(t, 700.0, n), rel=3e-11, abs=1e-10
                ), method


def test_optional_electronic_python_reference():
    reference = pytest.importorskip("burnman.eos.bukowinski_electronic")
    for name in ("helmholtz", "pressure", "KToverV"):
        args = (1700.0, 0.8e-5, 300.0, 1e-5, 0.005, 1.5)
        assert getattr(el, name)(*args) == pytest.approx(
            getattr(reference, name)(*args), rel=1e-13
        )
    for name, args in (
        ("entropy", (1700.0, 0.8e-5, 1e-5, 0.005, 1.5)),
        ("aKT", (1700.0, 0.8e-5, 1e-5, 0.005, 1.5)),
        ("CVoverT", (0.8e-5, 1e-5, 0.005, 1.5)),
    ):
        assert getattr(el, name)(*args) == pytest.approx(
            getattr(reference, name)(*args), rel=1e-13
        )


@pytest.mark.parametrize("thermal", ["Pade", "LogNormal"])
@pytest.mark.parametrize("prefactor", ["PowerLaw", "Sigmoid"])
def test_optional_anharmonic_python_reference(thermal, prefactor):
    reference = pytest.importorskip("burnman.eos.anharmonic_debye").AnharmonicDebye
    p = anharmonic_parameters(thermal, prefactor)
    rp = reference_parameters(p)
    reference.validate_parameters(rp)
    # Python's LogNormal implementation does not define the zero-temperature limit.
    temperatures = (
        (0.1, 300.0, 1500.0) if thermal == "LogNormal" else (0.0, 300.0, 1500.0)
    )
    for t in temperatures:
        for ratio in (0.8, 1.0):
            v = ratio * p["V_0"]
            for name in (
                "helmholtz_energy",
                "pressure",
                "entropy",
                "heat_capacity_v",
                "isothermal_bulk_modulus",
                "dSdV",
            ):
                assert getattr(AnharmonicDebye, name)(t, v, p) == pytest.approx(
                    getattr(reference, name)(t, v, rp), rel=3e-10, abs=1e-8
                ), name
