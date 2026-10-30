# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""All published pressure standards against the pinned Python reference."""

import importlib
import inspect
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

import burnman_cpp as bm
from burnman_cpp.classes.calibrant import Calibrant
from burnman_cpp.calibrants.tools import pressure_to_pressure

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = json.loads((ROOT / "tests/reference/calibrants.json").read_text())
STANDARDS = REFERENCE["calibrants"]


def standard(identifier):
    module, name = identifier.split(".")
    return getattr(getattr(bm.calibrants, module), name)()


def test_complete_catalogue_and_python_structure():
    actual = set()
    for module_name in bm.calibrants.__all__:
        if module_name == "tools":
            continue
        module = importlib.import_module("burnman_cpp.calibrants." + module_name)
        assert module is getattr(bm.calibrants, module_name)
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if cls is not Calibrant and issubclass(cls, Calibrant):
                actual.add(module_name + "." + name)
    assert len(actual) == 52
    assert actual == set(STANDARDS)
    assert len(bm.calibrants.__all__) - 1 == 38
    assert bm.Calibrant is Calibrant
    assert REFERENCE["reference"] == "burnman.json"


@pytest.mark.parametrize("identifier", STANDARDS)
def test_all_calibrant_parameters_and_pressure_reference(identifier):
    cal = standard(identifier)
    expected = STANDARDS[identifier]
    assert isinstance(cal, Calibrant)
    assert cal.params == expected["params"]
    # Check Dorogokupets against a stable independent five-point derivative
    # of the Python energy expression; its tiny default step loses precision.
    differentiated = identifier.startswith("Dorogokupets_2007.")
    atol = 1.0 if differentiated else 0.02
    for volume, temperature, pressure in expected[
        "stable_pressure" if differentiated else "pressure"
    ]:
        assert_allclose(
            cal.pressure(volume, temperature),
            pressure,
            rtol=2.0e-11,
            atol=atol,
            err_msg=identifier,
        )
        assert_allclose(
            cal.pressure_function(volume, temperature, cal.params),
            cal.pressure(volume, temperature),
            rtol=0.0,
            atol=0.0,
        )
        assert_allclose(
            cal.calibrant_function(volume, temperature, cal.params),
            cal.pressure(volume, temperature),
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize("identifier", STANDARDS)
def test_all_calibrant_volume_reference_and_roundtrips(identifier):
    cal = standard(identifier)
    for pressure, temperature, volume in STANDARDS[identifier]["volume"]:
        actual = cal.volume(pressure, temperature)
        # The reference calibrant uses scipy.brentq's default absolute xtol
        # of 2e-12 m3/mol. The native inversion uses a tighter scaled solve.
        assert_allclose(actual, volume, atol=2.1e-12, rtol=0.0)
        assert_allclose(
            cal.pressure(actual, temperature), pressure, atol=20.0, rtol=1.0e-10
        )
        assert_allclose(
            cal.volume_function(pressure, temperature, cal.params),
            actual,
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize("identifier", STANDARDS)
def test_all_calibrant_covariance_reference(identifier):
    cal = standard(identifier)
    for method, key in [
        ("pressure", "pressure_covariance"),
        ("volume", "volume_covariance"),
    ]:
        value, temperature, covariance, result, expected = STANDARDS[identifier][key]
        actual, C = getattr(cal, method)(value, temperature, np.array(covariance))
        assert_allclose(
            actual, result, rtol=2.0e-7, atol=5.0 if method == "pressure" else 2.1e-12
        )
        if identifier.startswith("Dorogokupets_2007."):
            expected = STANDARDS[identifier]["stable_" + method + "_covariance"]
        expected = np.array(expected)
        scales = np.sqrt(np.diag(expected))
        normalized_error = (C - expected) / np.outer(scales, scales)
        assert_allclose(
            normalized_error, 0.0, atol=5.0e-4, err_msg=identifier + " " + method
        )
        assert_allclose(C, C.T, rtol=1.0e-13, atol=1.0e-25)
        assert (
            np.linalg.eigvalsh(
                C / np.outer(np.sqrt(np.diag(C)), np.sqrt(np.diag(C)))
            ).min()
            > -1.0e-10
        )


@pytest.mark.parametrize("identifier", STANDARDS)
def test_all_calibrant_mutable_parameter_dictionaries(identifier):
    cal = standard(identifier)
    volume = 0.9 * cal.params["V_0"]
    before = cal.pressure(volume, 1000.0)
    cal.params["V_0"] *= 1.001
    after = cal.pressure(volume, 1000.0)
    assert after != before
    assert_allclose(after, cal.pressure_function(volume, 1000.0, cal.params))


def custom(return_type="pressure", k=150.0e9, alpha=2.0e6):
    params = {"V_0": 1.0e-5, "k": k, "alpha": alpha, "extra": np.array([1.0, 2.0])}
    if return_type == "pressure":
        function = lambda v, t, p: p["k"] * (p["V_0"] / v - 1) + p["alpha"] * (
            t - 300.0
        )
    else:
        function = lambda p, t, a: a["V_0"] / (
            1 + (p - a["alpha"] * (t - 300.0)) / a["k"]
        )
    return Calibrant(function, return_type, params)


@pytest.mark.parametrize("return_type", ["pressure", "volume"])
def test_custom_calibrants_and_covariance_with_analytic_inverse(return_type):
    cal = custom(return_type)
    p, t = 20.0e9, 1000.0
    v = 1.0e-5 / (1 + (p - 2.0e6 * (t - 300.0)) / 150.0e9)
    assert_allclose(cal.volume(p, t), v, rtol=1.0e-12)
    assert_allclose(cal.pressure(v, t), p, rtol=1.0e-12)
    VT = np.array([[(v * 1.0e-4) ** 2, v * 2.0e-4], [v * 2.0e-4, 100.0]])
    _, C = cal.pressure(v, t, VT_covariance=VT)
    J = np.array([[-150.0e9 * 1.0e-5 / v**2, 2.0e6], [1.0, 0.0], [0.0, 1.0]])
    assert_allclose(C, J @ VT @ J.T, rtol=2.0e-6, atol=1.0e-24)
    PT = np.array([[1.0e16, 2.0e8], [2.0e8, 100.0]])
    _, C = cal.volume(p, t, PT_covariance=PT)
    dvdp = -(v**2) / (150.0e9 * 1.0e-5)
    J = np.array([[dvdp, -2.0e6 * dvdp], [1.0, 0.0], [0.0, 1.0]])
    expected = J @ PT @ J.T
    assert_allclose(
        (C - expected) / np.sqrt(np.outer(np.diag(expected), np.diag(expected))),
        0.0,
        atol=2.0e-5,
    )
    cal.params = dict(cal.params, k=160.0e9)
    v_new = cal.volume(p, t)
    assert v_new != v
    assert_allclose(cal.pressure(v_new, t), p, rtol=1.0e-12)


def test_pressure_conversion_covariance_against_analytic_jacobian():
    old, new = custom(), custom(k=160.0e9, alpha=2.5e6)
    p, t = 20.0e9, 1000.0
    PT = np.array([[1.0e16, 2.0e8], [2.0e8, 100.0]])
    ratio = 160.0 / 150.0
    thermal = 2.5e6 - ratio * 2.0e6
    expected = ratio * p + thermal * (t - 300.0)
    J = np.array([[ratio, thermal], [0.0, 1.0]])
    assert_allclose(pressure_to_pressure(old, new, p, t), expected, rtol=1.0e-12)
    p_new, C = pressure_to_pressure(old, new, p, t, PT_covariance=PT)
    assert_allclose(p_new, expected, rtol=1.0e-12)
    assert_allclose(C, J @ PT @ J.T, rtol=2.0e-5)


@pytest.mark.parametrize(
    "old,new",
    [
        ("Anderson_1989.Au", "Fei_2007.Au"),
        ("Holmes_1989.Pt", "Matsui_2009.Pt"),
        ("Decker_1971.NaCl_B1", "Matsui_2012.NaCl_B1"),
        ("Litasov_2013.Mo_bcc", "Huang_2016.Mo"),
        ("Tsuchiya_2003.Au", "Matsui_2010.Au"),
    ],
)
def test_pressure_conversion_roundtrips(old, new):
    old, new = standard(old), standard(new)
    p, t = 20.0e9, 1000.0
    p_new = pressure_to_pressure(old, new, p, t)
    recovered = pressure_to_pressure(new, old, p_new, t)
    assert_allclose(recovered, p, rtol=1.0e-10, atol=1.0)
    PT = np.array([[1.0e18, 1.0e8], [1.0e8, 10.0]])
    p_new, C = pressure_to_pressure(old, new, p, t, PT)
    p_back, C_back = pressure_to_pressure(new, old, p_new, t, C)
    assert_allclose(p_back, p, rtol=1.0e-9)
    assert_allclose(C_back, PT, rtol=2.0e-4)


@pytest.mark.parametrize(
    "C",
    [
        np.zeros((3, 3)),
        np.array([[1.0, 2.0], [2.0, 1.0]]),
        [[1.0, 0.0], [1.0, 1.0]],
        [[1.0, 0.0], [0.0, -1.0]],
        [[np.nan, 0.0], [0.0, 1.0]],
    ],
)
def test_invalid_covariance_rejected(C):
    cal = custom()
    with pytest.raises(ValueError):
        cal.pressure(1.0e-5, 1000.0, C)
    with pytest.raises(ValueError):
        cal.volume(1.0e9, 1000.0, C)


def test_invalid_inputs_and_return_type():
    with pytest.raises(ValueError):
        Calibrant(lambda x, t, p: x, "energy", {})
    cal = custom()
    for volume, temperature in [
        (0.0, 1000.0),
        (-1.0, 1000.0),
        (1.0e-5, -1.0),
        (np.nan, 1000.0),
    ]:
        with pytest.raises(ValueError):
            cal.pressure(volume, temperature)
    assert_allclose(cal.pressure(1.0e-5, 1000.0, np.zeros((2, 2)))[1], 0.0)


def test_volume_inversion_keeps_high_compression_brackets_positive():
    cal = custom()
    pressure, temperature = 1.2e12, 1000.0
    volume = cal.volume(pressure, temperature)
    assert 0.0 < volume < 0.2 * cal.params["V_0"]
    assert_allclose(cal.pressure(volume, temperature), pressure, rtol=1.0e-12)
